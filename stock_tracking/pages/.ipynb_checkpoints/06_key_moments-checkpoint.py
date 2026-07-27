
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_signals, load_price, load_csv, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD, CORR_PREFIX

st.set_page_config(page_title="Key Moments", page_icon="🔍", layout="wide")

token        = st.secrets.get("GITHUB_TOKEN", None)
df_key_moves = load_csv(f"{CORR_PREFIX}/key_moves.csv", token)

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(token): return load_signals(token)
daily_signals = get_signals(token)

if df_key_moves.empty or daily_signals.empty:
    st.warning("No key moves data. Run B_correlation_engine.ipynb first."); st.stop()

for col in ["move_date","sent_date"]:
    if col in df_key_moves.columns:
        df_key_moves[col] = pd.to_datetime(df_key_moves[col])

all_tickers = sorted(df_key_moves["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
if not selected: st.warning("Select tickers."); st.stop()

sig_col = get_sig_col(daily_signals)

st.header("🔍 Key Moment Zoom")
st.markdown("""
Shows the 30 days before and 15 days after a major price move.
- **Orange line** = day of big price move
- **Purple dashed** = when sentiment fired before the move
- Top panel = % price change from window start
- Bottom panel = sentiment bars in same window
""")

col1, col2 = st.columns(2)
ticker_km      = col1.selectbox("Ticker", selected)
show_predicted = col2.checkbox("Correctly predicted only", value=True)

km = df_key_moves[df_key_moves["ticker"]==ticker_km].copy()
if show_predicted and "predicted" in km.columns:
    km = km[km["predicted"]==True]
km = km[(km["move_date"] >= start) & (km["move_date"] <= end)]
km = km.sort_values("zscore", key=abs, ascending=False).head(10)

if km.empty:
    st.info("No key moves found. Try unchecking predicted only or expanding the date range."); st.stop()

options = [f"{r.move_date.strftime('%b %Y')} ({r.return_pct:+.1f}%, z={r.zscore:+.1f})"
           for r in km.itertuples()]
sel_idx = st.selectbox("Select event", range(len(options)), format_func=lambda i: options[i])
move    = km.iloc[sel_idx]

win_start = move["move_date"] - pd.Timedelta(days=30)
win_end   = move["move_date"] + pd.Timedelta(days=15)

sig_w   = daily_signals[(daily_signals["ticker"]==ticker_km) &
                         (daily_signals["date"] >= win_start) &
                         (daily_signals["date"] <= win_end)].copy()
price_w = load_price(ticker_km, token)
price_w = price_w[(price_w["Date"] >= win_start) & (price_w["Date"] <= win_end)] if not price_w.empty else pd.DataFrame()

if price_w.empty:
    st.warning(f"No price data for {ticker_km}."); st.stop()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), facecolor="white")

ref = price_w["Close"].iloc[0]
pct_w = (price_w["Close"] - ref) / ref * 100
ax1.plot(price_w["Date"], pct_w, color="#2980b9", linewidth=2)
ax1.fill_between(price_w["Date"], 0, pct_w, where=pct_w>=0, color="#a9dfbf", alpha=0.4)
ax1.fill_between(price_w["Date"], 0, pct_w, where=pct_w<0,  color="#f5b7b1", alpha=0.4)
ax1.axhline(0, color="#aaaaaa", linewidth=0.8)
ax1.axvline(move["move_date"],  color="#f39c12", linewidth=3, zorder=5, label="Big move")
ax1.axvline(move["sent_date"],  color="#8e44ad", linewidth=2, linestyle="--", zorder=4,
            label=f"Sentiment {move['days_before']}d before")
ax1.set_ylabel("% change from window start", fontsize=9)
ax1.set_title(
    f"{ticker_km} — {move['move_date'].strftime('%B %Y')} | "
    f"Move: {move['return_pct']:+.1f}% (z={move['zscore']:+.1f})
"
    f"Sentiment {move['days_before']}d before: {move['sentiment']:+.3f} | "
    f"Sources: {move.get('sources_active','unknown')}",
    fontsize=11, fontweight="bold"
)
ax1.legend(fontsize=8, facecolor="white")
ax1.set_facecolor("white")

if not sig_w.empty:
    sv = sig_w[sig_col].fillna(0)
    bc = ["#27ae60" if v >= SENTIMENT_THRESHOLD else
          ("#e74c3c" if v <= -SENTIMENT_THRESHOLD else "#bdc3c7") for v in sv]
    ax2.bar(sig_w["date"], sv, color=bc, alpha=0.8, width=1.5)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.7)
    ax2.axvline(move["move_date"], color="#f39c12", linewidth=3, zorder=5)
    ax2.axvline(move["sent_date"], color="#8e44ad", linewidth=2, linestyle="--", zorder=4)
ax2.set_ylabel("Sentiment score", fontsize=9)
ax2.set_title("Sentiment in 30 days before and 15 days after the move", fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.set_facecolor("white")

plt.tight_layout()
st.pyplot(fig)
plt.close()
