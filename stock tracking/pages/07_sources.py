
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_csv, sidebar_filters, OUTPUT_PREFIX

st.set_page_config(page_title="Sources", page_icon="📰", layout="wide")

token       = st.secrets.get("GITHUB_TOKEN", None)
source_attr = load_csv(f"{OUTPUT_PREFIX}/source_attribution.csv", token)

if source_attr.empty:
    st.warning("No source data. Run A_sentiment_engine.ipynb first."); st.stop()

all_tickers = sorted(source_attr["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)

st.header("📰 Source Attribution")
st.markdown("""
Which data source contributes the most articles per ticker?
This shows **volume** (number of articles), not signal quality.

| Source | Audience | Best for |
|--------|----------|----------|
| **GDELT** | Mainstream news (Reuters, FT, Bloomberg) | Macro events, earnings |
| **HN** | Tech experts / engineers | Tech company signals |
| **Reddit WSB** | Retail momentum traders | Meme stocks, momentum |
| **Reddit stocks/investing** | General investors | Broader coverage |
| **StockTwits** | Direct investor posts | High-relevance, ticker-tagged |
| **EDGAR 8-K** | SEC filings | Material events |
""")

n_tickers = st.slider("Tickers to show", 10, 50, 25)
top = (source_attr.groupby("ticker")["item_count"].sum()
       .sort_values(ascending=False).head(n_tickers).index.tolist())
pivot = source_attr[source_attr["ticker"].isin(top)].pivot_table(
    index="ticker", columns="source", values="item_count", fill_value=0
)
pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot = pivot.sort_values(pivot.columns[0], ascending=False)

src_colors = {
    "hn":"#e67e22",
    "reddit_wallstreetbets":"#e74c3c",
    "reddit_stocks":"#c0392b",
    "reddit_investing":"#922b21",
    "reddit_technology":"#7b241c",
    "reddit_SecurityAnalysis":"#641e16",
    "gdelt":"#2980b9",
    "stocktwits":"#27ae60",
    "edgar_8k":"#8e44ad",
}

fig, ax = plt.subplots(figsize=(13, max(7, len(pivot)*0.4)), facecolor="white")
bottom = np.zeros(len(pivot))
for col in pivot.columns:
    ax.bar(pivot.index, pivot[col], bottom=bottom, label=col,
           color=src_colors.get(col,"#888888"), edgecolor="white", alpha=0.85)
    bottom += pivot[col].values
ax.set_ylabel("Share of articles (%)", fontsize=10)
ax.set_title("Source Contribution per Ticker (% of total articles)",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelrotation=45)
ax.legend(fontsize=8, facecolor="white", bbox_to_anchor=(1.01,1), loc="upper left")
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()

with st.expander("View dominant source per ticker"):
    dom = (source_attr.sort_values("item_count", ascending=False)
           .drop_duplicates(subset="ticker")
           [["ticker","source","item_count","avg_sentiment"]]
           .rename(columns={"source":"dominant_source","item_count":"articles"})
           .sort_values("articles", ascending=False))
    st.dataframe(dom.reset_index(drop=True), use_container_width=True)
