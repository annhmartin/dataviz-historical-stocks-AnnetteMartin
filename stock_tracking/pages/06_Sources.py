import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, OUTPUT_PREFIX, SOURCE_COLORS,
                   apply_chart_style, titled, show, POS, NEG, CONTEXT, MUTED, INK)

st.header("Sources")

selected, start, end, token = sidebar_filters()
apply_chart_style()

source_attr = load_csv(OUTPUT_PREFIX + "/source_attribution.csv", token)
if source_attr.empty:
    st.warning("No source data. Run A_sentiment_engine.ipynb first.")
    st.stop()

st.markdown("Six independent feeds contribute to every sentiment reading. They differ "
            "enormously in how much they publish, and separately in how strongly they lean.")

scope = st.radio("Show", ["Selected sector", "Highest volume market-wide"],
                 horizontal=True, key="src_scope")

sa = source_attr.copy()
if scope == "Selected sector":
    sa = sa[sa["ticker"].isin(selected)]
    if sa.empty:
        st.info("No source data for tickers in this sector.")
        st.stop()

top = (sa.groupby("ticker")["item_count"].sum()
       .sort_values(ascending=False).head(25).index.tolist())
pivot = sa[sa["ticker"].isin(top)].pivot_table(
    index="ticker", columns="source", values="item_count", fill_value=0)
if pivot.empty:
    st.info("Nothing to display.")
    st.stop()

share = pivot.div(pivot.sum(axis=1), axis=0) * 100
share = share.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig = go.Figure()
for col in share.columns:
    fig.add_trace(go.Bar(
        x=share.index, y=share[col], name=col,
        marker_color=SOURCE_COLORS.get(col, CONTEXT), marker_line_width=0,
        hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.1f}}% of coverage<extra></extra>"))

fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.22, x=0),
                  bargap=0.22)
fig.update_yaxes(title="Share of that ticker's coverage (%)", ticksuffix="%")
fig.update_xaxes(title="", tickangle=-45)

dominant = share.mean().idxmax()
titled(fig,
       f"{dominant} supplies most of the coverage across these companies",
       f"Each bar is one ticker's coverage split by source, ordered by total volume. "
       f"Showing {len(share)} tickers",
       height=560)
show(fig)

st.markdown("---")

quality = (sa.groupby("source")
           .agg(avg_abs=("avg_sentiment", lambda x: float(x.abs().mean())),
                avg_signed=("avg_sentiment", "mean"),
                items=("item_count", "sum"))
           .reset_index().sort_values("avg_abs"))

fig2 = go.Figure()
# Both entries are always drawn so the reader can see that a category exists
# even when no source currently falls into it
for lean, colour, label in [(True,  POS, "Leans net positive"),
                            (False, NEG, "Leans net negative")]:
    d = quality[(quality["avg_signed"] >= 0) == lean]
    if d.empty:
        # Placeholder trace: keeps the legend entry without drawing a bar
        fig2.add_trace(go.Bar(
            y=[None], x=[None], orientation="h",
            name=f"{label} (none)", marker_color=colour, marker_line_width=0,
            showlegend=True, hoverinfo="skip"))
        continue
    fig2.add_trace(go.Bar(
        y=d["source"], x=d["avg_abs"], orientation="h",
        name=label, marker_color=colour, marker_line_width=0,
        text=[f"{v:.3f}" for v in d["avg_abs"]],
        textposition="outside", textfont=dict(size=12, color=MUTED),
        customdata=np.stack([d["items"], d["avg_signed"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>average strength %{x:.3f}"
                      "<br>net lean %{customdata[1]:+.3f}"
                      "<br>%{customdata[0]:,} items<extra></extra>"))

fig2.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0,
                itemwidth=40, itemsizing="constant"),
    margin=dict(b=110))
fig2.update_xaxes(title="Average absolute sentiment score")
fig2.update_yaxes(title="")

strongest = quality.iloc[-1]
titled(fig2,
       f"{strongest['source']} expresses the most strongly worded sentiment",
       "Bar length is how forcefully a source phrases things, not how often it is right",
       height=500)
show(fig2)

st.info(
    "Strength of language and accuracy of prediction are separate properties. "
    "A source can be consistently emphatic and consistently wrong."
)

with st.expander("Dominant source per ticker"):
    dom = (sa.sort_values("item_count", ascending=False)
           .drop_duplicates(subset="ticker")
           [["ticker", "source", "item_count", "avg_sentiment"]]
           .rename(columns={"ticker": "Ticker", "source": "Dominant source",
                            "item_count": "Articles", "avg_sentiment": "Average sentiment"})
           .sort_values("Articles", ascending=False))
    st.dataframe(dom.reset_index(drop=True), width="stretch", hide_index=True)
