import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, load_signals, load_price, get_token, get_sig_col,
                   SENTIMENT_THRESHOLD, STRAT_PREFIX, SECTOR_MAP,
                   apply_chart_style, titled, show,
                   POS, NEG, NEU, ACCENT, HIGHLIGHT, CONTEXT, MUTED, INK, GOLD)

st.title("Does online chatter predict what a stock does next?")
st.markdown(
    "Six sources of public discussion — mainstream news, Hacker News, five subreddits, "
    "investor posts and SEC filings — scored for sentiment across eleven years and "
    "matched against daily prices for roughly two thousand companies.\n\n"
    "**The short answer: yes, but modestly, and not in the direction you would expect.**"
)

token = get_token()
apply_chart_style()

ALL = sorted({t for ts in SECTOR_MAP.values() for t in ts})

# ── Compute the headline numbers from saved outputs ──────────────────────────
@st.cache_data(ttl=3600, show_spinner="Measuring signal accuracy...")
def headline_stats(_token):
    """Directional hit rate for positive vs negative sentiment, and by year."""
    sig = load_signals(_token, tickers=ALL, start_year=2015)
    if sig.empty:
        return None
    col = get_sig_col(sig)
    sig = sig[sig[col].notna() & (sig[col].abs() >= SENTIMENT_THRESHOLD)]
    if sig.empty:
        return None

    HOLD = 5
    rows = []
    for ticker in sig.ticker.unique():
        p = load_price(ticker, _token)
        if p.empty:
            continue
        p = p.sort_values("Date")
        closes, dates = p["Close"].values, p["Date"].values
        s = sig[sig.ticker == ticker]
        entry = np.searchsorted(dates, s["date"].values, side="right")
        ex = entry + HOLD - 1
        ok = (ex < len(closes)) & (entry < len(closes))
        if ok.sum() == 0:
            continue
        rows.append(pd.DataFrame({
            "date": s["date"].values[ok],
            "sent": s[col].values[ok],
            "fwd": (closes[ex[ok]] - closes[entry[ok]]) / closes[entry[ok]],
        }))
    if not rows:
        return None

    ev = pd.concat(rows, ignore_index=True)
    ev["correct"] = ((ev.sent > 0) & (ev.fwd > 0)) | ((ev.sent < 0) & (ev.fwd < 0))
    ev["dir"] = np.where(ev.sent > 0, "Positive", "Negative")
    ev["year"] = pd.to_datetime(ev.date).dt.year

    by_dir = ev.groupby("dir")["correct"].agg(["mean", "size"])
    by_year = ev.groupby("year")["correct"].agg(["mean", "size"]).reset_index()
    by_year = by_year[by_year["size"] >= 100]
    return dict(overall=float(ev.correct.mean()), n=len(ev),
                by_dir=by_dir, by_year=by_year, hold=HOLD)

stats = headline_stats(token)
equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)

if stats is None:
    st.warning("Signal data not available yet. Run A_sentiment_engine.ipynb first.")
    st.stop()

pos_rate = float(stats["by_dir"].loc["Positive", "mean"]) if "Positive" in stats["by_dir"].index else np.nan
neg_rate = float(stats["by_dir"].loc["Negative", "mean"]) if "Negative" in stats["by_dir"].index else np.nan
pos_n    = int(stats["by_dir"].loc["Positive", "size"]) if "Positive" in stats["by_dir"].index else 0
neg_n    = int(stats["by_dir"].loc["Negative", "size"]) if "Negative" in stats["by_dir"].index else 0

# ── Top-line metrics ────────────────────────────────────────────────────────
st.markdown("---")
m = st.columns(4)
m[0].metric("Signals analysed", f"{stats['n']:,}",
            help=f"Days where sentiment crossed the neutral threshold, judged against "
                 f"the following {stats['hold']} trading days")
m[1].metric("Positive buzz accuracy", f"{pos_rate*100:.1f}%",
            f"{(pos_rate-0.5)*100:+.1f} pts vs chance")
m[2].metric("Negative buzz accuracy", f"{neg_rate*100:.1f}%",
            f"{(neg_rate-0.5)*100:+.1f} pts vs chance",
            delta_color="inverse" if neg_rate < 0.5 else "normal")

if not equity.empty and "Sector_Rotation" in equity.columns and "S&P_500_SPY" in equity.columns:
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index().ffill()
    rot = float(eq["Sector_Rotation"].dropna().iloc[-1])
    spy = float(eq["S&P_500_SPY"].dropna().iloc[-1])
    m[3].metric("Best strategy vs S&P 500", f"{rot/spy:.1f}×",
                f"${rot:,.0f} vs ${spy:,.0f}")
else:
    m[3].metric("Best strategy vs S&P 500", "—")

# ── Finding 1: the asymmetry ────────────────────────────────────────────────
st.markdown("---")
st.subheader("1 · Pessimism is a buy signal")

contrarian = neg_rate < 0.5 < pos_rate
if contrarian:
    st.markdown(
        f"Positive chatter predicts gains **{pos_rate*100:.1f}%** of the time — a real "
        f"if modest edge. Negative chatter predicts correctly only **{neg_rate*100:.1f}%** "
        f"of the time, which means prices usually *rose* after it. Public pessimism looks "
        f"closer to a contrarian buy indicator than a warning."
    )
else:
    st.markdown(
        f"Positive chatter is right {pos_rate*100:.1f}% of the time and negative chatter "
        f"{neg_rate*100:.1f}%, a gap of {abs(pos_rate-neg_rate)*100:.1f} percentage points. "
        f"The two directions are not mirror images."
    )

fig1 = go.Figure()
for label, rate, n, colour in [("Positive buzz", pos_rate, pos_n, POS),
                               ("Negative buzz", neg_rate, neg_n, NEG)]:
    fig1.add_trace(go.Bar(
        x=[label], y=[rate], name=label, marker_color=colour, marker_line_width=0,
        width=0.45, text=[f"{rate*100:.1f}%"], textposition="outside",
        textfont=dict(size=17, color=colour),
        customdata=[n],
        hovertemplate="<b>%{x}</b><br>correct %{y:.1%}<br>%{customdata:,} signals<extra></extra>"))

fig1.add_hline(y=0.5, line_color=MUTED, line_width=2, line_dash="dot")
fig1.add_annotation(x=1.45, y=0.5, text="coin flip", showarrow=False,
                    xanchor="right", yshift=13, font=dict(size=13, color=MUTED))
fig1.update_yaxes(title="Share of signals that called direction correctly",
                  tickformat=".0%",
                  range=[min(0.35, neg_rate - 0.08), max(0.62, pos_rate + 0.08)])
fig1.update_xaxes(title="")
fig1.update_layout(showlegend=False)
titled(fig1,
       ("Negative sentiment points the wrong way more often than chance"
        if contrarian else "Positive and negative sentiment differ in reliability"),
       f"Judged over the following {stats['hold']} trading days · "
       f"{pos_n:,} positive and {neg_n:,} negative signals",
       height=420)
show(fig1)

# ── Finding 2: decay over time ──────────────────────────────────────────────
st.markdown("---")
st.subheader("2 · The edge has faded since the retail trading boom")

by_year = stats["by_year"]
if len(by_year) >= 4:
    early = by_year[by_year.year < 2021]["mean"].mean()
    late  = by_year[by_year.year >= 2021]["mean"].mean()
    st.markdown(
        f"Accuracy averaged **{early*100:.1f}%** before 2021 and **{late*100:.1f}%** "
        f"afterwards. Commission-free trading, a surge in retail participation and funds "
        f"mining social data directly all arrived in that window — consistent with an edge "
        f"being competed away as more people traded on the same public signals."
    )

    colours = [CONTEXT if y < 2021 else POS for y in by_year.year]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=by_year.year, y=by_year["mean"], marker_color=colours, marker_line_width=0,
        customdata=by_year["size"],
        hovertemplate="%{x}<br>accuracy %{y:.1%}<br>%{customdata:,} signals<extra></extra>",
        showlegend=False))
    fig2.add_hline(y=0.5, line_color=MUTED, line_width=1.5, line_dash="dot")
    fig2.add_hline(y=early, line_color=CONTEXT, line_width=2, line_dash="dash")
    fig2.add_hline(y=late, line_color=POS, line_width=2, line_dash="dash")
    fig2.add_annotation(x=by_year.year.min(), y=early, text=f"pre-2021 average {early:.1%}",
                        showarrow=False, xanchor="left", yshift=13,
                        font=dict(size=12, color=MUTED))
    fig2.add_annotation(x=by_year.year.max(), y=late, text=f"2021 onward {late:.1%} ",
                        showarrow=False, xanchor="right", yshift=-15,
                        font=dict(size=12, color=POS))
    fig2.update_yaxes(title="Directional accuracy", tickformat=".0%")
    fig2.update_xaxes(title="", dtick=1)
    titled(fig2,
           f"Accuracy fell {abs(late-early)*100:.1f} points after 2021",
           "Annual hit rate. Years with fewer than 100 signals are excluded",
           height=430)
    show(fig2)
else:
    st.info("Not enough years with sufficient signal volume to show the trend.")

# ── Finding 3: does it survive execution ────────────────────────────────────
st.markdown("---")
st.subheader("3 · It beats the market, but not comfortably")

if not equity.empty:
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    curves = eq.set_index("date").sort_index().ffill()

    SERIES = [("S&P_500_SPY", "S&P 500", CONTEXT, "dash", 2.0),
              ("Buy_&_Hold", "Buy & Hold", MUTED, "solid", 2.0),
              ("Sector_Rotation", "Sector Rotation", ACCENT, "solid", 3.2),
              ("Position_Trader", "Position Trader", HIGHLIGHT, "solid", 2.2)]
    SERIES = [s for s in SERIES if s[0] in curves.columns]

    if "Sector_Rotation" in curves.columns and "S&P_500_SPY" in curves.columns:
        r12 = curves.pct_change(252)
        margin = (r12["Sector_Rotation"] - r12["S&P_500_SPY"]).dropna()
        behind = float((margin < 0).mean())
        st.markdown(
            f"A buzz-driven sector rotation turned $10,000 into "
            f"**${float(curves['Sector_Rotation'].dropna().iloc[-1]):,.0f}** against "
            f"**${float(curves['S&P_500_SPY'].dropna().iloc[-1]):,.0f}** for the index, "
            f"after transaction costs. The catch is in the lower panel: it trailed the "
            f"index in **{behind:.0%}** of rolling twelve-month windows. The gains arrive "
            f"in bursts separated by long stretches of underperformance."
        )
    else:
        margin, behind = None, None

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                         row_heights=[0.68, 0.32],
                         subplot_titles=("", "Rolling 12-month margin over the S&P 500"))
    for col, label, colour, dash, width in SERIES:
        s = curves[col].dropna()
        fig3.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(color=colour, width=width, dash=dash),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"),
            row=1, col=1)

    if margin is not None:
        fig3.add_trace(go.Scatter(
            x=margin.index, y=margin.values, mode="lines",
            line=dict(color=ACCENT, width=2), name="Rotation minus S&P 500",
            hovertemplate="%{x|%b %Y}<br>margin %{y:+.1%}<extra></extra>"), row=2, col=1)
        fig3.add_trace(go.Scatter(
            x=margin.index, y=np.where(margin.values < 0, margin.values, 0),
            fill="tozeroy", mode="none", fillcolor="rgba(213,94,0,0.28)",
            name="Trailing the index", hoverinfo="skip"), row=2, col=1)
        fig3.add_hline(y=0, line_color=MUTED, line_width=1.5, row=2, col=1)

    fig3.update_yaxes(title="Portfolio value", tickprefix="$", row=1, col=1)
    fig3.update_yaxes(title="Margin", tickformat="+.0%", row=2, col=1)
    fig3.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                    itemwidth=40, itemsizing="constant"),
        margin=dict(t=150), hovermode="x unified")

    best = max(((c, l) for c, l, *_ in SERIES),
               key=lambda t: float(curves[t[0]].dropna().iloc[-1]))
    titled(fig3,
           f"{best[1]} finished ahead of both passive benchmarks",
           "Starting from $10,000 in 2015, including transaction costs",
           height=700)
    show(fig3)
else:
    st.info("Strategy results not available yet. Run C_strategy_engine.ipynb.")

# ── Where to go next ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Explore the data yourself")
st.markdown(
    "Use the sector and date filters in the sidebar, then work through the pages:\n\n"
    "| Page | What it answers |\n"
    "|------|-----------------|\n"
    "| **Overview** | How does sentiment track price for one company? |\n"
    "| **Signal Quality** | When sentiment fires, how often is it right? |\n"
    "| **Correlation** | Which companies have the most predictive chatter? |\n"
    "| **Strategies** | Would trading these signals have beaten the market? |\n"
    "| **Key Moments** | What was being said before a large price move? |\n"
    "| **Sources** | Which feeds carry the most, and the strongest, signal? |\n"
    "| **What If** | What would your own investment have returned? |"
)
st.caption("Built for coursework and research. Nothing here is investment advice.")
