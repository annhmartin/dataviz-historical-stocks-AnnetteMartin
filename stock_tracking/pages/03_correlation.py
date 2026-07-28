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

selected, start, end, token = sidebar_filters()
apply_chart_style()

st.header("Correlation")
st.markdown("How strongly does sentiment predict future price moves? Green = positive buzz predicted price UP. Red = positive buzz predicted price DOWN (contrarian signal).")

plot_df = best_per_ticker.copy()
if "n" in plot_df.columns:
    plot_df = plot_df[plot_df["n"] >= 200]
plot_df = plot_df.sort_values("corr")

if plot_df.empty:
    st.info("No tickers match the current filters.")
else:
    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.35)), facecolor="white")
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
    labels = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
    ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation (sentiment -> forward price return)")
    ax.set_title("Best Sentiment Signal Per Ticker")
    ax.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.subheader("How To Read This Chart")
st.markdown("""
| Term | Definition |
|------|-----------|
| **Pearson Correlation** | A number from -1.0 to +1.0. 0 = no relationship. The further from 0, the stronger the link between sentiment and price. |
| **Positive bar (green)** | When sentiment was positive, the stock tended to go UP over the following days. Follow the buzz. |
| **Negative bar (red)** | When sentiment was positive, the stock tended to go DOWN. Contrarian signal - negative buzz may be more predictive for this ticker. |
| **Signal type** | Which sentiment score worked best: norm_sentiment (raw daily), adaptive_sentiment (window-adjusted), roll_3d/5d/7d (rolling averages). |
| **T+N horizon** | How many trading days forward the correlation was measured. T+1 = next day, T+5 = one week, T+21 = one month. |
| **n** | Number of data point pairs used. Higher n = more reliable. All bars shown have n >= 200. |
""")

with st.expander("View Raw Correlation Table"):
    cols = [c for c in ["ticker","signal","horizon","corr","pval","n"] if c in plot_df.columns]
    st.dataframe(plot_df[cols].reset_index(drop=True), width="stretch")
