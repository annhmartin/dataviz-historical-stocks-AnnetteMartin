
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_csv, sidebar_filters, CORR_PREFIX

st.set_page_config(page_title="Spillover", page_icon="🌊", layout="wide")

token       = st.secrets.get("GITHUB_TOKEN", None)
df_spillover = load_csv(f"{CORR_PREFIX}/spillover_pairs.csv", token)

if df_spillover.empty:
    st.warning("No spillover data. Run B_correlation_engine.ipynb first."); st.stop()

all_tickers = sorted(set(df_spillover["sent_ticker"].tolist() + df_spillover["price_ticker"].tolist()))
selected, start, end, token = sidebar_filters(all_tickers)

st.header("🌊 Cross-Company Sentiment Spillover")
st.markdown("""
When Company A gets buzz, does Company B's stock move?
Pairs shown here held consistently across 90-day rolling windows.

- **Positive pair**: A's positive buzz → B's price goes UP
- **Negative pair**: A's positive buzz → B's price goes DOWN
""")

min_cons = st.slider("Min consistency (fraction of windows correlated)", 0.20, 0.70, 0.35, step=0.05)
filtered  = df_spillover[df_spillover["consistency"] >= min_cons]

st.metric("Confirmed pairs", len(filtered))

if filtered.empty:
    st.info("No pairs at this consistency level. Try lowering the threshold."); st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Positive spillover pairs")
    pos = filtered[filtered["direction"]=="positive"].nlargest(15,"avg_corr")
    st.dataframe(pos[["sent_ticker","price_ticker","avg_corr","consistency"]].reset_index(drop=True), use_container_width=True)
with col2:
    st.subheader("Negative spillover pairs")
    neg = filtered[filtered["direction"]=="negative"].nsmallest(15,"avg_corr")
    st.dataframe(neg[["sent_ticker","price_ticker","avg_corr","consistency"]].reset_index(drop=True), use_container_width=True)

try:
    pivot = filtered.pivot_table(index="sent_ticker", columns="price_ticker", values="avg_corr")
    if not pivot.empty and len(pivot) <= 40:
        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.7+3),
                                        max(8, len(pivot)*0.5+3)), facecolor="white")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    linewidths=0.3, ax=ax, annot_kws={"size": 7},
                    cbar_kws={"label": "Avg rolling correlation"})
        ax.set_title("Spillover Heatmap\nRow = source of buzz | Col = stock that moved",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
except Exception:
    pass
