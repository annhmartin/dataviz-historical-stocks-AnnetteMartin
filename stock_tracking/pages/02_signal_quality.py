import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_signals, load_price, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD, apply_chart_style

st.set_page_config(page_title="Signal Quality", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(t):
    return load_signals(t)

daily_signals = get_signals(token)
if daily_signals.empty:
    st.error("No signal data.")
    st.stop()

all_tickers = sorted(daily_signals["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
if not selected:
    st.warning("No tickers for selected sector.")
    st.stop()

sig_col = get_sig_col(daily_signals)
apply_chart_style()

st.header("Signal Quality")
st.markdown(
    "Each dot = one day where sentiment crossed the neutral threshold. "
    "The quadrant it lands in shows whether the prediction was correct.\n\n"
    "- True Positive (top-right, green): positive buzz, stock went UP\n"
    "- False Positive (bottom-right, red): positive buzz, stock went DOWN\n"
    "- True Negative (bottom-left, blue): negative buzz, stock went DOWN\n"
    "- False Negative (top-left, orange): negative buzz, stock went UP\n\n"
    "Dot size = number of stories that day. Orange line = overall trend."
)

col1, col2 = st.columns(2)
ticker    = col1.selectbox("Ticker", selected)
hold_days = col2.slider("Hold Period (Trading Days)", 1, 21, 5)

sig   = daily_signals[(daily_signals["ticker"] == ticker) & (daily_signals["date"] >= start) & (daily_signals["date"] <= end)].copy()
price = load_price(ticker, token)
if price.empty or sig.empty:
    st.warning("No data for selected ticker.")
    st.stop()

price = price.set_index("Date").sort_index()
rows  = []
for _, srow in sig.iterrows():
    sv = srow.get(sig_col, np.nan)
    if pd.isna(sv) or abs(sv) < SENTIMENT_THRESHOLD:
        continue
    future = price.index[price.index > srow["date"]]
    if len(future) < hold_days:
        continue
    ep  = price.loc[future[0], "Close"]
    xp  = price.loc[future[hold_days - 1], "Close"]
    ret = (xp - ep) / ep * 100
    if sv >= SENTIMENT_THRESHOLD and ret > 0:
        q, c = "True Positive",  "#27ae60"
    elif sv >= SENTIMENT_THRESHOLD and ret <= 0:
        q, c = "False Positive", "#e74c3c"
    elif sv <= -SENTIMENT_THRESHOLD and ret < 0:
        q, c = "True Negative",  "#2980b9"
    else:
        q, c = "False Negative", "#e67e22"
    rows.append({"sent": sv, "ret": ret, "q": q, "c": c, "stories": srow.get("story_count", 1)})

if not rows:
    st.info("Not enough signal days for this ticker and date range.")
    st.stop()

df_plot = pd.DataFrame(rows)
counts  = df_plot["q"].value_counts()
total   = len(df_plot)
tp = int(counts.get("True Positive",  0))
fp = int(counts.get("False Positive", 0))
tn = int(counts.get("True Negative",  0))
fn = int(counts.get("False Negative", 0))
acc = (tp + tn) / total

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",       str(round(acc * 100, 1)) + "%")
c2.metric("True Positive",  str(tp),  str(round(tp / total * 100)) + "%")
c3.metric("False Positive", str(fp),  str(round(fp / total * 100)) + "%")
c4.metric("True Negative",  str(tn),  str(round(tn / total * 100)) + "%")
c5.metric("False Negative", str(fn),  str(round(fn / total * 100)) + "%")

fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
ax.scatter(df_plot["sent"], df_plot["ret"], c=df_plot["c"], alpha=0.65,
           s=df_plot["stories"].clip(1, 50) * 8 + 15, edgecolors="white", linewidths=0.4, zorder=3)
ax.axhline(0, color="#888888", linewidth=1)
ax.axvline(0, color="#888888", linewidth=1)
xmax = float(df_plot["sent"].abs().max()) * 1.1
ymax = float(df_plot["ret"].abs().max())  * 1.1
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)
ax.text( xmax * .55,  ymax * .88, "True Positive\n"  + str(tp) + " (" + str(round(tp/total*100)) + "%)", ha="center", fontsize=13, color="#27ae60", fontweight="bold")
ax.text(-xmax * .55,  ymax * .88, "False Negative\n" + str(fn) + " (" + str(round(fn/total*100)) + "%)", ha="center", fontsize=13, color="#e67e22", fontweight="bold")
ax.text( xmax * .55, -ymax * .88, "False Positive\n" + str(fp) + " (" + str(round(fp/total*100)) + "%)", ha="center", fontsize=13, color="#e74c3c", fontweight="bold")
ax.text(-xmax * .55, -ymax * .88, "True Negative\n"  + str(tn) + " (" + str(round(tn/total*100)) + "%)", ha="center", fontsize=13, color="#2980b9", fontweight="bold")
if len(df_plot) > 5:
    z  = np.polyfit(df_plot["sent"], df_plot["ret"], 1)
    xs = np.linspace(-xmax, xmax, 100)
    ax.plot(xs, np.poly1d(z)(xs), color="#f39c12", linewidth=2, zorder=4)
ax.set_xlabel("Negative Buzz  <--  |  -->  Positive Buzz")
ax.set_ylabel("Price Change " + str(hold_days) + " Days Later (%)")
ax.set_title(ticker + " - Signal Quality | Accuracy: " + str(round(acc * 100, 1)) + "% | n=" + str(total))
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()
