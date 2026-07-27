
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_signals, load_price, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD

st.set_page_config(page_title="Signal Quality", page_icon="🎯", layout="wide")

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(token):
    return load_signals(token)

token = st.secrets.get("GITHUB_TOKEN", None)
daily_signals = get_signals(token)
if daily_signals.empty:
    st.error("No signal data. Run A_sentiment_engine.ipynb first."); st.stop()

all_tickers = sorted(daily_signals["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
if not selected:
    st.warning("Select at least one ticker."); st.stop()

sig_col = get_sig_col(daily_signals)

st.header("🎯 Signal Quality — Four Quadrant")
st.markdown("""
Each dot = one day where sentiment was above/below neutral.
The quadrant it lands in shows whether the prediction was right.

**True Positive** (top-right 🟢): positive buzz → stock UP ✓
**False Positive** (bottom-right 🔴): positive buzz → stock DOWN ✗  
**True Negative** (bottom-left 🔵): negative buzz → stock DOWN ✓
**False Negative** (top-left 🟠): negative buzz → stock UP ✗

**Dot size** = stories that day. **Orange trend line** = overall direction.
""")

col1, col2 = st.columns(2)
ticker    = col1.selectbox("Ticker", selected)
hold_days = col2.slider("Hold period (trading days)", 1, 21, 5)

sig   = daily_signals[(daily_signals["ticker"]==ticker) &
                       (daily_signals["date"] >= start) &
                       (daily_signals["date"] <= end)].copy()
price = load_price(ticker, token)

if price.empty or sig.empty:
    st.warning("No data for selected ticker."); st.stop()

price = price.set_index("Date").sort_index()
rows  = []
for _, srow in sig.iterrows():
    sv = srow.get(sig_col, np.nan)
    if pd.isna(sv) or abs(sv) < SENTIMENT_THRESHOLD: continue
    future = price.index[price.index > srow["date"]]
    if len(future) < hold_days: continue
    ep  = price.loc[future[0],  "Close"]
    xp  = price.loc[future[hold_days-1], "Close"]
    ret = (xp - ep) / ep * 100
    if sv >= SENTIMENT_THRESHOLD and ret > 0:    q,c = "True Positive",  "#27ae60"
    elif sv >= SENTIMENT_THRESHOLD and ret <= 0: q,c = "False Positive", "#e74c3c"
    elif sv <= -SENTIMENT_THRESHOLD and ret < 0: q,c = "True Negative",  "#2980b9"
    else:                                         q,c = "False Negative", "#e67e22"
    rows.append({"sent": sv, "ret": ret, "q": q, "c": c,
                 "stories": srow.get("story_count", 1)})

if not rows:
    st.info("Not enough signal days for the selected ticker and date range."); st.stop()

df_plot = pd.DataFrame(rows)
counts  = df_plot["q"].value_counts()
total   = len(df_plot)
tp = counts.get("True Positive",0); fp = counts.get("False Positive",0)
tn = counts.get("True Negative",0); fn = counts.get("False Negative",0)
acc = (tp+tn)/total

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy",        f"{acc:.1%}")
c2.metric("True Positive",   f"{tp}",  f"{tp/total:.0%}")
c3.metric("False Positive",  f"{fp}",  f"{fp/total:.0%}")
c4.metric("True Negative",   f"{tn}",  f"{tn/total:.0%}")
c5.metric("False Negative",  f"{fn}",  f"{fn/total:.0%}")

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
ax.scatter(df_plot["sent"], df_plot["ret"], c=df_plot["c"], alpha=0.65,
           s=df_plot["stories"].clip(1,50)*8+15, edgecolors="white", linewidths=0.4, zorder=3)
ax.axhline(0, color="#888888", linewidth=1)
ax.axvline(0, color="#888888", linewidth=1)
xmax = df_plot["sent"].abs().max()*1.1; ymax = df_plot["ret"].abs().max()*1.1
ax.set_xlim(-xmax,xmax); ax.set_ylim(-ymax,ymax)
for txt, x, y, col in [
    (f"True Positive\n{tp} ({tp/total:.0%})",   xmax*.55,  ymax*.88, "#27ae60"),
    (f"False Negative\n{fn} ({fn/total:.0%})", -xmax*.55,  ymax*.88, "#e67e22"),
    (f"False Positive\n{fp} ({fp/total:.0%})",  xmax*.55, -ymax*.88, "#e74c3c"),
    (f"True Negative\n{tn} ({tn/total:.0%})",  -xmax*.55, -ymax*.88, "#2980b9"),
]:
    ax.text(x, y, txt, ha="center", fontsize=9, color=col, fontweight="bold")
if len(df_plot) > 5:
    z = np.polyfit(df_plot["sent"], df_plot["ret"], 1)
    xs = np.linspace(-xmax, xmax, 100)
    ax.plot(xs, np.poly1d(z)(xs), color="#f39c12", linewidth=2, zorder=4)
ax.set_xlabel("← Negative buzz  |  Positive buzz →", fontsize=10)
ax.set_ylabel(f"Price change {hold_days}d later (%)", fontsize=10)
ax.set_title(f"{ticker} — Signal Quality | Accuracy: {acc:.1%} | n={total}", fontsize=12, fontweight="bold")
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()
