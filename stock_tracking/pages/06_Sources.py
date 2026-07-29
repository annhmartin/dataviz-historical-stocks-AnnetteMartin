
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, sidebar_filters, OUTPUT_PREFIX, apply_chart_style

st.header("Sources")

token = None
try: token = st.secrets["GITHUB_TOKEN"]
except Exception: pass

source_attr = load_csv(OUTPUT_PREFIX + "/source_attribution.csv", token)
if source_attr.empty:
    st.warning("No source data. Run A_sentiment_engine.ipynb first.")
    st.stop()

selected, start, end, token = sidebar_filters()
apply_chart_style()

st.markdown("Which data source contributes the most articles per ticker? Left = volume share. Right = average sentiment strength (signal quality) per source.")

top = source_attr.groupby("ticker")["item_count"].sum().sort_values(ascending=False).head(25).index.tolist()
pivot_vol = source_attr[source_attr["ticker"].isin(top)].pivot_table(
    index="ticker", columns="source", values="item_count", fill_value=0
)
pivot_vol = pivot_vol.div(pivot_vol.sum(axis=1), axis=0) * 100
pivot_vol = pivot_vol.sort_values(pivot_vol.columns[0], ascending=False)

src_colors = {
    "hn"                      : "#e67e22",
    "reddit_wallstreetbets"   : "#e74c3c",
    "reddit_stocks"           : "#c0392b",
    "reddit_investing"        : "#922b21",
    "reddit_technology"       : "#7b241c",
    "reddit_SecurityAnalysis" : "#641e16",
    "gdelt"                   : "#2980b9",
    "stocktwits"              : "#27ae60",
    "edgar_8k"                : "#8e44ad",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, min(14, max(7, len(pivot_vol) * 0.4))), dpi=100, facecolor="white")

# Left: volume chart
bottom = np.zeros(len(pivot_vol))
for col in pivot_vol.columns:
    ax1.bar(pivot_vol.index, pivot_vol[col], bottom=bottom, label=col,
            color=src_colors.get(col, "#888888"), edgecolor="white", alpha=0.85)
    bottom += pivot_vol[col].values
ax1.set_ylabel("Share of Articles (%)")
ax1.set_title("Volume: Source Contribution Per Ticker")
ax1.tick_params(axis="x", labelrotation=45)
ax1.legend(fontsize=9, facecolor="white", bbox_to_anchor=(0, -0.35), loc="upper left", ncol=2)
ax1.set_facecolor("white")

# Right: quality chart — avg absolute sentiment per source
source_quality = (
    source_attr.groupby("source")
    .agg(
        avg_abs_sentiment = ("avg_sentiment", lambda x: float(x.abs().mean())),
        avg_sentiment     = ("avg_sentiment", "mean"),
        total_items       = ("item_count",    "sum"),
    )
    .reset_index()
    .sort_values("avg_abs_sentiment", ascending=True)
)
q_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in source_quality["avg_sentiment"]]
bars = ax2.barh(source_quality["source"], source_quality["avg_abs_sentiment"],
                color=q_colors, edgecolor="white")
for bar, val in zip(bars, source_quality["avg_abs_sentiment"]):
    ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
             "{:.3f}".format(val), va="center", fontsize=11)
ax2.axvline(0, color="#aaaaaa", linewidth=0.8)
ax2.set_xlabel("Avg Absolute Sentiment Score (signal strength)")
ax2.set_title("Quality: Signal Strength Per Source\n(green = net positive avg, red = net negative avg)")
ax2.set_facecolor("white")

plt.tight_layout()
st.pyplot(fig)
plt.close()

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
    dom = (
        source_attr.sort_values("item_count", ascending=False)
        .drop_duplicates(subset="ticker")
        [["ticker", "source", "item_count", "avg_sentiment"]]
        .rename(columns={"ticker": "Ticker", "source": "Dominant Source",
                          "item_count": "Articles", "avg_sentiment": "Avg Sentiment"})
        .sort_values("Articles", ascending=False)
    )
    st.dataframe(dom.reset_index(drop=True), use_container_width=True)
