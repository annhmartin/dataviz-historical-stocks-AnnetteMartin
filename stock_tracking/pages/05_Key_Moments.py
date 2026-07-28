import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_signals, load_price, load_csv, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD, CORR_PREFIX, apply_chart_style

st.set_page_config(page_title="Key Moments", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

df_key_moves = load_csv(CORR_PREFIX + "/key_moves.csv", token)

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(t):
    return load_signals(t)

daily_signals = get_signals(token)
if df_key_moves.empty or daily_signals.empty:
    st.warning("No key moves data. Run B_correlation_engine.ipynb first.")
    st.stop()

for col in ["move_date", "sent_date"]:
    if col in df_key_moves.columns:
        df_key_moves[col] = pd.to_datetime(df_key_moves[col])

all_tickers = sorted(df_key_moves["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
if not selected:
    st.warning("Select tickers.")
    st.stop()

sig_col = get_sig_col(daily_signals)
apply_chart_style()

st.header("Key Moments")
st.markdown("Shows 30 days before and 15 days after a major price move where sentiment preceded it. Orange line = day of big price move. Purple dashed = when sentiment fired beforehand.")

col1, col2 = st.columns(2)
ticker_km      = col1.selectbox("Ticker", selected)
show_predicted = col2.checkbox("Correctly Predicted Only", value=True)

km = df_key_moves[df_key_moves["ticker"] == ticker_km].copy()
if show_predicted and "predicted" in km.columns:
    km = km[km["predicted"] == True]
km = km[(km["move_date"] >= start) & (km["move_date"] <= end)]
km = km.sort_values("zscore", key=abs, ascending=False).head(10)

if km.empty:
    st.info("No key moves found. Try unchecking Correctly Predicted Only or expanding the date range.")
    st.stop()

option_labels = []
for r in km.itertuples():
    option_labels.append(r.move_date.strftime("%b %Y") + " (" + "{:+.1f}%".format(float(r.return_pct)) + ", z=" + "{:+.1f}".format(float(r.zscore)) + ")")

sel_idx = st.selectbox("Select Event", range(len(option_labels)), format_func=lambda i: option_labels[i])
move    = km.iloc[sel_idx]
win_start = move["move_date"] - pd.Timedelta(days=30)
win_end   = move["move_date"] + pd.Timedelta(days=15)

sig_w   = daily_signals[(daily_signals["ticker"] == ticker_km) & (daily_signals["date"] >= win_start) & (daily_signals["date"] <= win_end)].copy()
price_w = load_price(ticker_km, token)
if not price_w.empty:
    price_w = price_w[(price_w["Date"] >= win_start) & (price_w["Date"] <= win_end)]

if price_w.empty:
    st.warning("No price data for " + ticker_km)
    st.stop()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="white")
ref   = float(price_w["Close"].iloc[0])
pct_w = (price_w["Close"] - ref) / ref * 100
ax1.plot(price_w["Date"], pct_w, color="#2980b9", linewidth=2)
ax1.fill_between(price_w["Date"], 0, pct_w, where=pct_w >= 0, color="#a9dfbf", alpha=0.4)
ax1.fill_between(price_w["Date"], 0, pct_w, where=pct_w < 0,  color="#f5b7b1", alpha=0.4)
ax1.axhline(0, color="#aaaaaa", linewidth=0.8)
ax1.axvline(move["move_date"], color="#f39c12", linewidth=3, zorder=5, label="Big Move")
ax1.axvline(move["sent_date"], color="#8e44ad", linewidth=2, linestyle="--", zorder=4, label="Sentiment " + str(int(move["days_before"])) + "d Before")
ax1.set_ylabel("% Change From Window Start")
ax1.set_title(ticker_km + " - " + move["move_date"].strftime("%B %Y") + " | Move: " + "{:+.1f}%".format(float(move["return_pct"])) + " | Sentiment: " + "{:+.3f}".format(float(move["sentiment"])))
ax1.legend(loc="upper left")
ax1.set_facecolor("white")
if not sig_w.empty:
    sv = sig_w[sig_col].fillna(0)
    bc = ["#27ae60" if v >= SENTIMENT_THRESHOLD else ("#e74c3c" if v <= -SENTIMENT_THRESHOLD else "#bdc3c7") for v in sv]
    ax2.bar(sig_w["date"], sv, color=bc, alpha=0.8, width=1.5)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.7)
    ax2.axvline(move["move_date"], color="#f39c12", linewidth=3, zorder=5)
    ax2.axvline(move["sent_date"], color="#8e44ad", linewidth=2, linestyle="--", zorder=4)
ax2.set_ylabel("Sentiment Score")
ax2.set_title("Sentiment - 30 Days Before and 15 Days After The Move")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()
