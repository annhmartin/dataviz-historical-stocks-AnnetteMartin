import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, OUTPUT_PREFIX, apply_chart_style,
                   POS, NEG, INK, AXIS, MUTED, GRID, CANVAS, SOURCE_COLORS)

st.header("Sources")

selected, start, end, token = sidebar_filters()
apply_chart_style()

source_attr = load_csv(OUTPUT_PREFIX + "/source_attribution.csv", token)
if source_attr.empty:
    st.warning("No source data. Run A_sentiment_engine.ipynb first.")
    st.stop()

st.markdown("Which data source contributes the most articles per ticker?")

scope = st.radio("Show", ["Selected sector", "Highest volume overall"],
                 horizontal=True, key="src_scope")

if scope == "Selected sector":
    source_attr = source_attr[source_attr["ticker"].isin(selected)]
    if source_attr.empty:
        st.info("No source data for tickers in this sector.")
        st.stop()
    st.caption("Scoped to the " + str(source_attr["ticker"].nunique())
               + " tickers in this sector that have source data.")

src_colors = SOURCE_COLORS

top = (source_attr.groupby("ticker")["item_count"].sum()
       .sort_values(ascending=False).head(25).index.tolist())
if not top:
    st.info("No tickers to display.")
    st.stop()
pivot_vol = source_attr[source_attr["ticker"].isin(top)].pivot_table(
    index="ticker", columns="source", values="item_count", fill_value=0
)
pivot_vol = pivot_vol.div(pivot_vol.sum(axis=1), axis=0) * 100
pivot_vol = pivot_vol.sort_values(pivot_vol.columns[0], ascending=False)

fig1, ax1 = plt.subplots(figsize=(14, min(10, max(6, len(pivot_vol) * 0.4))),
                         facecolor=CANVAS, dpi=100)
bottom = np.zeros(len(pivot_vol))
for col in pivot_vol.columns:
    ax1.bar(pivot_vol.index, pivot_vol[col], bottom=bottom, label=col,
            color=src_colors.get(col, MUTED), edgecolor="white", alpha=0.85)
    bottom += pivot_vol[col].values
ax1.set_ylabel("Share of Articles (%)")
ax1.set_title("Volume: Source Contribution Per Ticker")
ax1.tick_params(axis="x", labelrotation=45)
ax1.legend(fontsize=10, facecolor=CANVAS, bbox_to_anchor=(1.01, 1), loc="upper left")
ax1.set_facecolor(CANVAS)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

source_quality = (
    source_attr.groupby("source")
    .agg(
        avg_abs_sentiment=("avg_sentiment", lambda x: float(x.abs().mean())),
        avg_sentiment=("avg_sentiment", "mean"),
        total_items=("item_count", "sum"),
    )
    .reset_index()
    .sort_values("avg_abs_sentiment", ascending=True)
)

fig2, ax2 = plt.subplots(figsize=(14, min(8, max(5, len(source_quality) * 0.5))),
                         facecolor=CANVAS, dpi=100)
q_colors = [POS if v >= 0 else NEG for v in source_quality["avg_sentiment"]]
bars = ax2.barh(source_quality["source"], source_quality["avg_abs_sentiment"],
                color=q_colors, edgecolor="white")
for bar, val in zip(bars, source_quality["avg_abs_sentiment"]):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             "{:.3f}".format(val), va="center", fontsize=11)
ax2.axvline(0, color=AXIS, linewidth=0.8)
ax2.set_xlabel("Average Absolute Sentiment Score (signal strength)")
ax2.set_title("Quality: Signal Strength Per Source (green = net positive, red = net negative)")
ax2.set_facecolor(CANVAS)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")
st.markdown("""
| Source | Audience | Best For | Notes |
|--------|----------|----------|-------|
| **GDELT** | Mainstream news (Reuters, FT, Bloomberg) | Macro events, earnings | Broad market coverage |
| **HN (Hacker News)** | Tech experts and engineers | Tech company signals | High quality, lower volume |
| **Reddit WSB** | Retail momentum traders | Meme stocks, momentum | High volume, noisy |
| **Reddit stocks** | General investors | Broader market coverage | Medium quality |
| **Reddit investing** | Value investors | Fundamental discussion | Medium quality |
| **Reddit technology** | Tech enthusiasts | Product launches | Medium quality |
| **Reddit SecurityAnalysis** | Serious analysts | Deep company analysis | High quality |
| **StockTwits** | Direct investor posts | Ticker-specific sentiment | High relevance, pre-tagged |
| **EDGAR 8-K** | SEC filings | Material corporate events | Factual, highest weight in model |
""")

with st.expander("Dominant Source Per Ticker"):
    dom = (source_attr.sort_values("item_count", ascending=False)
           .drop_duplicates(subset="ticker")
           [["ticker", "source", "item_count", "avg_sentiment"]]
           .rename(columns={"ticker": "Ticker", "source": "Dominant Source",
                            "item_count": "Articles", "avg_sentiment": "Avg Sentiment"})
           .sort_values("Articles", ascending=False))
    st.dataframe(dom.reset_index(drop=True), width="stretch", hide_index=True)
