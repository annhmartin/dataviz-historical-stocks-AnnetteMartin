
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, load_signals, sidebar_filters, get_sig_col, CORR_PREFIX, OUTPUT_PREFIX, SENTIMENT_THRESHOLD, apply_chart_style

st.set_page_config(page_title="Correlation", layout="wide")
token = None
try: token = st.secrets["GITHUB_TOKEN"]
except Exception: pass

best_per_ticker = load_csv(CORR_PREFIX + "/best_per_ticker.csv", token)

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(t): return load_signals(t)
daily_signals = get_signals(token)

if best_per_ticker.empty: st.warning("No correlation data. Run B_correlation_engine.ipynb first."); st.stop()

selected, start, end, token = sidebar_filters()
apply_chart_style()

st.header("Correlation")
st.markdown("How strongly does sentiment predict future price moves? Green = positive buzz predicted price UP. Red = positive buzz predicted price DOWN (contrarian signal).")

plot_df = best_per_ticker.copy()
if "n" in plot_df.columns: plot_df = plot_df[plot_df["n"] >= 200]
plot_df = plot_df.sort_values("corr")

if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df)*0.35)), facecolor="white")
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
    labels = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
    ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation (sentiment -> forward price return)")
    ax.set_title("Best Sentiment Signal Per Ticker")
    ax.set_facecolor("white"); plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("---")
st.subheader("Ticker Definitions")
if not plot_df.empty:
    ticker_defs = []
    for _, row in plot_df.iterrows():
        t = row["ticker"]
        corr_str = "{:+.3f}".format(float(row["corr"]))
        direction = "Follow positive buzz" if float(row["corr"]) > 0 else "Contrarian - negative buzz more predictive"
        ticker_defs.append({"Ticker": t, "Correlation": corr_str, "Signal Direction": direction, "Signal Type": row["signal"], "Best Horizon": "T+" + str(int(row["horizon"]))})
    st.dataframe(pd.DataFrame(ticker_defs), width='stretch', hide_index=True)

st.markdown("---")
st.subheader("How To Read This Chart")
st.markdown("""
| Term | Definition |
|------|-----------|
| **Pearson Correlation** | -1.0 to +1.0. 0 = no relationship. Further from 0 = stronger link between sentiment and price. |
| **Positive bar (green)** | Positive buzz predicted price going UP. Follow the signal. |
| **Negative bar (red)** | Positive buzz predicted price going DOWN. Contrarian - negative buzz may be more useful for this ticker. |
| **Signal type** | Which sentiment score worked best: norm_sentiment (raw), adaptive_sentiment (window-adjusted), roll_Nd (N-day rolling average). |
| **T+N horizon** | How many trading days forward the correlation was measured. T+1 = next day, T+21 = one month later. |
| **n** | Number of data point pairs used. All bars shown have n >= 200 for statistical reliability. |
""")

st.markdown("---")
st.subheader("Signal Quality Over Time")
if not daily_signals.empty:
    valid = [t for t in selected if t in daily_signals["ticker"].unique()]
    if not valid:
        st.info("No signal data for tickers in selected sector.")
    else:
        ticker_t = st.selectbox("Ticker for time series", valid, key="corr_ts")
        sig = daily_signals[(daily_signals["ticker"]==ticker_t) & (daily_signals["date"]>=start) & (daily_signals["date"]<=end)].copy()
        sig_col = get_sig_col(daily_signals)
        if not sig.empty and sig[sig_col].notna().sum() > 10:
            sig = sig.sort_values("date")
            sig["roll_acc"] = sig[sig_col].rolling(90, min_periods=20).apply(
                lambda x: float((x > SENTIMENT_THRESHOLD).sum()) / float(len(x[x.abs() > SENTIMENT_THRESHOLD])) if float(len(x[x.abs() > SENTIMENT_THRESHOLD])) > 0 else 0.5
            )
            fig2, ax2 = plt.subplots(figsize=(14,4), facecolor="white")
            ax2.plot(sig["date"], sig["roll_acc"], color="#8e44ad", linewidth=2)
            ax2.axhline(0.5, color="#aaaaaa", linewidth=1, linestyle="--")
            ax2.fill_between(sig["date"], 0.5, sig["roll_acc"], where=sig["roll_acc"]>=0.5, color="#a9dfbf", alpha=0.4)
            ax2.fill_between(sig["date"], 0.5, sig["roll_acc"], where=sig["roll_acc"]<0.5,  color="#f5b7b1", alpha=0.4)
            ax2.set_ylabel("90-Day Rolling Positive Accuracy")
            ax2.set_ylim(0,1)
            ax2.set_title(ticker_t + " - Sentiment Signal Quality Over Time (above 50% = signal beating random chance)")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax2.set_facecolor("white"); plt.tight_layout(); st.pyplot(fig2); plt.close()
        else:
            st.info("Not enough signal days to compute rolling quality.")
