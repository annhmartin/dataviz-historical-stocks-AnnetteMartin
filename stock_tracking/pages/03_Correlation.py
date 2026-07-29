import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, load_signals, sidebar_filters, get_sig_col,
                   CORR_PREFIX, SENTIMENT_THRESHOLD, apply_chart_style, titled, show,
                   POS, NEG, CONTEXT, INK, MUTED, ACCENT)

st.header("Correlation")

selected, start, end, token = sidebar_filters()
apply_chart_style()

best_per_ticker = load_csv(CORR_PREFIX + "/best_per_ticker.csv", token)
if best_per_ticker.empty:
    st.warning("No correlation data. Run B_correlation_engine.ipynb first.")
    st.stop()

st.markdown(
    "How strongly does sentiment lead price for each company? Bars to the right mean "
    "positive buzz preceded gains. Bars to the left are contrarian — there, negative "
    "buzz was the more useful signal."
)

MIN_N = 200
qualified = best_per_ticker.copy()
if "n" in qualified.columns:
    qualified = qualified[qualified["n"] >= MIN_N]
qualified = qualified.dropna(subset=["corr"])

if qualified.empty:
    st.info("No tickers have enough data points for a reliable correlation.")
    st.stop()

scope = st.radio("Show", ["Selected sector", "Strongest signals market-wide"],
                 horizontal=True, key="corr_scope")

if scope == "Selected sector":
    plot_df = qualified[qualified["ticker"].isin(selected)].sort_values("corr")
    caption = (f"{len(plot_df)} of {len(selected)} tickers in this sector clear "
               f"the {MIN_N}-observation minimum")
else:
    plot_df = (pd.concat([qualified.nsmallest(20, "corr"), qualified.nlargest(20, "corr")])
               .drop_duplicates(subset="ticker").sort_values("corr"))
    caption = (f"Strongest {len(plot_df)} signals from {len(qualified):,} tickers "
               f"clearing the {MIN_N}-observation minimum")

if plot_df.empty:
    st.info("No tickers in this sector have enough data. Try the market-wide view.")
else:
    strongest = plot_df["corr"].abs().idxmax()
    colors = [
        (POS if v > 0 else NEG) if i == strongest else CONTEXT
        for i, v in zip(plot_df.index, plot_df["corr"])
    ]
    labels = plot_df["ticker"] + "  " + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str)

    fig = go.Figure(go.Bar(
        y=labels, x=plot_df["corr"], orientation="h",
        marker_color=colors, marker_line_width=0,
        customdata=plot_df["n"] if "n" in plot_df.columns else None,
        hovertemplate="<b>%{y}</b><br>correlation %{x:+.4f}"
                      + ("<br>%{customdata:,} observations" if "n" in plot_df.columns else "")
                      + "<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1.5)

    top = plot_df.loc[strongest]
    fig.add_annotation(
        x=float(top["corr"]), y=f"{top['ticker']}  {top['signal']} T+{int(top['horizon'])}",
        text="  strongest  ", showarrow=False,
        xanchor="left" if top["corr"] > 0 else "right",
        font=dict(size=12, color=POS if top["corr"] > 0 else NEG))

    fig.update_xaxes(title="Correlation between sentiment and forward return",
                     tickformat="+.3f")
    fig.update_yaxes(title="")
    titled(fig,
           f"{top['ticker']} shows the strongest sentiment-to-price relationship here",
           caption,
           height=max(380, len(plot_df) * 26 + 150))
    show(fig)

st.markdown("---")
st.subheader("Signal quality over time")
st.markdown("Does the signal still work, or has it decayed as more people traded on it?")

with st.spinner("Loading sentiment signals..."):
    daily_signals = load_signals(token, tickers=selected,
                                 start_year=start.year, end_year=end.year)

if daily_signals.empty:
    st.info("No signal data for this sector and date range.")
    st.stop()

valid = [t for t in selected if t in set(daily_signals["ticker"])]
if not valid:
    st.info("No signal data for tickers in the selected sector.")
    st.stop()

ticker_t = st.selectbox("Ticker", valid, key="corr_ts")
sig_col  = get_sig_col(daily_signals)
sig = daily_signals[
    (daily_signals["ticker"] == ticker_t)
    & (daily_signals["date"] >= start)
    & (daily_signals["date"] <= end)
].copy().sort_values("date")

if sig[sig_col].notna().sum() <= 20:
    st.info("Not enough signal days to compute a rolling measure for this ticker.")
    st.stop()

def positive_share(x):
    active = x[np.abs(x) > SENTIMENT_THRESHOLD]
    if len(active) == 0:
        return 0.5
    return float((active > SENTIMENT_THRESHOLD).sum()) / float(len(active))

sig["roll"] = sig[sig_col].rolling(90, min_periods=20).apply(positive_share, raw=True)
d = sig.dropna(subset=["roll"])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=d["date"], y=d["roll"], mode="lines",
    line=dict(color=POS, width=2.5), name="90-day share",
    hovertemplate="%{x|%b %Y}<br>%{y:.1%} positive<extra></extra>"))
fig2.add_trace(go.Scatter(
    x=d["date"], y=np.where(d["roll"] < 0.5, d["roll"], 0.5),
    fill="tonexty", mode="none", fillcolor="rgba(213,94,0,0.18)",
    showlegend=False, hoverinfo="skip"))
fig2.add_hline(y=0.5, line_color=MUTED, line_width=1.5, line_dash="dot")
fig2.add_annotation(x=d["date"].max(), y=0.5, text="even split ",
                    showarrow=False, xanchor="right", yshift=11,
                    font=dict(size=12, color=MUTED))

fig2.update_yaxes(title="Share of active signals that were positive",
                  tickformat=".0%", range=[0, 1])
fig2.update_xaxes(title="")

recent = float(d["roll"].tail(60).mean())
early  = float(d["roll"].head(60).mean())
drift  = "grown more bullish" if recent > early else "cooled"
titled(fig2,
       f"{ticker_t} discussion has {drift} over this period",
       "Rolling 90-day share of active sentiment days that were positive. "
       "Shading marks stretches where negative days dominated",
       height=420)
show(fig2)
