import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, sidebar_filters, CORR_PREFIX, apply_chart_style

st.set_page_config(page_title="Correlation", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

best_per_ticker = load_csv(CORR_PREFIX + "/best_per_ticker.csv", token)
if best_per_ticker.empty:
    st.warning("No correlation data. Run B_correlation_engine.ipynb first.")
    st.stop()

all_tickers = sorted(best_per_ticker["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
apply_chart_style()

st.header("Correlation")
st.markdown("How strongly does sentiment predict future price moves for each ticker? Green bar = positive buzz predicted price UP. Red bar = positive buzz predicted price DOWN (contrarian signal).")

n_show  = st.slider("Number of Tickers to Show", 10, 50, 20)
min_n   = st.slider("Minimum Data Points (n)", 50, 500, 200)

plot_df = best_per_ticker.copy()
if "n" in plot_df.columns:
    plot_df = plot_df[plot_df["n"] >= min_n]
plot_df = plot_df.sort_values("corr").tail(n_show)

if plot_df.empty:
    st.info("No tickers match the current filters. Try lowering the minimum n.")
else:
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.4)), facecolor="white")
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
    labels = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
    ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation (sentiment -> forward price return)")
    ax.set_title("Best Sentiment Signal Per Ticker | " + str(len(plot_df)) + " shown")
    ax.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.subheader("How to Read This Chart")
st.markdown("""
| Term | Definition |
|------|-----------|
| **Pearson Correlation** | A number from -1.0 to +1.0 measuring how closely sentiment and price move together. 0 = no relationship. |
| **Positive bar (green)** | When sentiment was positive, the stock tended to go UP over the following days. |
| **Negative bar (red)** | When sentiment was positive, the stock tended to go DOWN. This is a contrarian signal - negative buzz may actually be more predictive. |
| **Signal type** | Which version of the sentiment score worked best: norm_sentiment (raw), adaptive_sentiment (window-adjusted), roll_3d/5d/7d (rolling averages). |
| **T+N horizon** | How many trading days forward the correlation was measured. T+1 = next day, T+21 = three weeks later. |
| **n** | Number of data point pairs used to compute the correlation. Higher n = more reliable. |
| **Min n filter** | Only show tickers with at least this many data points to avoid spurious correlations from tiny samples. |
""")

with st.expander("View Raw Correlation Table"):
    cols = [c for c in ["ticker","signal","horizon","corr","pval","n"] if c in plot_df.columns]
    st.dataframe(plot_df[cols].reset_index(drop=True), width='stretch')
