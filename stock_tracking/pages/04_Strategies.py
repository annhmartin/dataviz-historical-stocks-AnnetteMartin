import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style,
                   titled, show, POS, NEG, ACCENT, HIGHLIGHT, CONTEXT, MUTED, INK)

st.header("Strategies")

selected, start, end, token = sidebar_filters(show_sector=False)
apply_chart_style()

st.markdown(
    "**S&P 500 (SPY)** — passive benchmark, never touched. &nbsp;"
    "**Buy & Hold** — equal weight across all tickers.<br>"
    "**Sector Rotation** — weekly rotation into the sector with the strongest buzz. &nbsp;"
    "**Position Trader** — 21-day holds on high-conviction signals.",
    unsafe_allow_html=True)
st.markdown("---")

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
df_sector = load_csv(STRAT_PREFIX + "/equity_curves_by_sector.csv", token)

if df_equity.empty and df_sector.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

sector_name = st.session_state.get("sector", "")
has_sector = (not df_sector.empty and "sector" in df_sector.columns
              and sector_name in set(df_sector["sector"]))

if has_sector:
    scope = st.radio("Scope", ["Whole portfolio", "Selected sector only"],
                     horizontal=True, key="strat_scope")
else:
    scope = "Whole portfolio"
    st.caption("Per-sector curves are not in the data yet. Run section 10 of "
               "C_strategy_engine.ipynb to enable the sector view.")

if scope == "Selected sector only" and has_sector:
    sub = df_sector[df_sector["sector"] == sector_name].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    df_equity = sub.pivot_table(index="date", columns="strategy",
                                values="value", aggfunc="last").reset_index()
    df_equity = df_equity.rename(columns={
        "S&P 500 (SPY)": "S&P_500_SPY", "Buy & Hold": "Buy_&_Hold",
        "Position Trader": "Position_Trader", "Sector Rotation": "Sector_Rotation"})
else:
    df_equity["date"] = pd.to_datetime(df_equity["date"])

SERIES = [
    ("S&P_500_SPY",     "S&P 500 (SPY)",   CONTEXT,   "dash",  2.0),
    ("Buy_&_Hold",      "Buy & Hold",      MUTED,     "solid", 2.0),
    ("Sector_Rotation", "Sector Rotation", ACCENT,    "solid", 3.2),
    ("Position_Trader", "Position Trader", HIGHLIGHT, "solid", 2.2),
]
SERIES = [s for s in SERIES if s[0] in df_equity.columns]
if not SERIES:
    st.warning("No recognised strategy columns in the equity file.")
    st.stop()

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)]
curves = eq.set_index("date").sort_index()[[s[0] for s in SERIES]].ffill()
curves = curves.dropna(how="all")
if curves.empty:
    st.warning("No strategy data in this date range.")
    st.stop()

mc = st.columns(len(SERIES))
for (col, label, *_), box in zip(SERIES, mc):
    s = curves[col].dropna()
    if s.empty:
        continue
    final, first = float(s.iloc[-1]), float(s.iloc[0])
    box.metric(label, f"${final:,.0f}", f"{(final/first-1)*100:+.1f}%")

st.markdown("---")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("", "Rolling 12-month margin over the S&P 500"))

for col, label, colour, dash, width in SERIES:
    s = curves[col].dropna()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=label,
        line=dict(color=colour, width=width, dash=dash),
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"),
        row=1, col=1)
    fig.add_annotation(x=s.index[-1], y=float(s.iloc[-1]),
                       text=f"  {label}", showarrow=False, xanchor="left",
                       font=dict(size=11, color=colour), row=1, col=1)

if "S&P_500_SPY" in curves.columns and "Sector_Rotation" in curves.columns:
    r12 = curves.pct_change(252)
    margin = (r12["Sector_Rotation"] - r12["S&P_500_SPY"]).dropna()
    if not margin.empty:
        fig.add_trace(go.Scatter(
            x=margin.index, y=margin.values, mode="lines",
            line=dict(color=ACCENT, width=2),
            name="Rotation minus S&P 500", legendgroup="margin",
            hovertemplate="%{x|%b %Y}<br>margin %{y:+.1%}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=margin.index, y=np.where(margin.values < 0, margin.values, 0),
            fill="tozeroy", mode="none", fillcolor="rgba(213,94,0,0.28)",
            name="Trailing the index", legendgroup="margin",
            hoverinfo="skip"), row=2, col=1)
        fig.add_hline(y=0, line_color=MUTED, line_width=1.5, row=2, col=1)

fig.update_yaxes(title="Portfolio value", tickprefix="$", row=1, col=1)
fig.update_yaxes(title="Margin", tickformat="+.0%", row=2, col=1)
fig.update_layout(
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    margin=dict(r=170, t=140),
    hovermode="x unified")

finals = {label: float(curves[col].dropna().iloc[-1]) for col, label, *_ in SERIES}
best = max(finals, key=finals.get)
spy  = finals.get("S&P 500 (SPY)")
sub  = (f"Best strategy finished at ${finals[best]:,.0f}"
        + (f", about {finals[best]/spy:.1f}× the index" if spy else ""))
titled(fig, f"Which strategy came out ahead? {best}", sub, height=720)
show(fig)

st.markdown("---")
st.subheader("Monthly leader")

monthly = curves.resample("ME").last().dropna(how="all")
if not monthly.empty:
    rets = monthly.pct_change()
    winners = monthly.idxmax(axis=1)
    label_for = {c: l for c, l, *_ in SERIES}
    color_for = {c: col for c, _, col, _, _ in SERIES}
    bar_c, bar_t = [], []
    for i, (month, win) in enumerate(winners.items()):
        if i > 0 and month in rets.index and rets.loc[month].notna().any() and (rets.loc[month].dropna() < 0).all():
            bar_c.append(CONTEXT); bar_t.append("all lost money")
        else:
            bar_c.append(color_for.get(win, CONTEXT)); bar_t.append(label_for.get(win, win))
    figm = go.Figure()
    # One trace per leader so Plotly draws a real legend
    seen = set()
    for col, label, colour, _dash, _w in SERIES:
        mask = [t == label for t in bar_t]
        if not any(mask):
            continue
        seen.add(label)
        figm.add_trace(go.Bar(
            x=winners.index[mask], y=[1]*sum(mask),
            name=label, marker_color=colour, marker_line_width=0,
            hovertemplate="%{x|%b %Y}<br>" + label + "<extra></extra>"))
    mask_none = [t == "all lost money" for t in bar_t]
    if any(mask_none):
        figm.add_trace(go.Bar(
            x=winners.index[mask_none], y=[1]*sum(mask_none),
            name="Every strategy lost", marker_color=CONTEXT, marker_line_width=0,
            hovertemplate="%{x|%b %Y}<br>every strategy lost money<extra></extra>"))
    figm.update_yaxes(showticklabels=False, showgrid=False, title="")
    figm.update_xaxes(title="")
    figm.update_layout(
        bargap=0.05, barmode="stack",
        legend=dict(orientation="h", yanchor="top", y=-0.45,
                    xanchor="left", x=0),
        margin=dict(b=120))
    titled(figm, "Which strategy was ahead in any given month?",
           "Grey marks months where every strategy lost money", height=320)
    show(figm)

if not df_trades.empty and "strategy" in df_trades.columns:
    st.markdown("---")
    st.subheader("Position trade log")

    t = df_trades[df_trades["strategy"].astype(str).str.strip() == "Position Trader"].copy()
    if t.empty:
        st.info("No Position Trader trades recorded.")
    else:
        if st.checkbox("Selected sector only", value=False) and "ticker" in t.columns:
            t = t[t["ticker"].isin(selected)]
        if "entry_date" in t.columns:
            t["entry_date"] = pd.to_datetime(t["entry_date"])
            t = t[(t["entry_date"] >= start) & (t["entry_date"] <= end)]

        total = len(t)
        wins = int((t["return_pct"] > 0).sum()) if total else 0
        k = st.columns(4)
        k[0].metric("Trades", f"{total}")
        k[1].metric("Win rate", f"{wins/total*100:.1f}%" if total else "N/A")
        k[2].metric("Average return", f"{t['return_pct'].mean():+.2f}%" if total else "N/A")
        k[3].metric("Best trade", f"{t['return_pct'].max():+.1f}%" if total else "N/A")

        if total:
            cols = {"ticker":"Ticker","entry_date":"Entry date","exit_date":"Exit date",
                    "entry_price":"Entry price","exit_price":"Exit price",
                    "return_pct":"Return %","portfolio_val":"Portfolio value",
                    "trigger_source":"Trigger source"}
            keep = [c for c in cols if c in t.columns]
            disp = t[keep].rename(columns=cols).sort_values("Entry date", ascending=False)
            for dc in ("Entry date","Exit date"):
                if dc in disp.columns:
                    disp[dc] = pd.to_datetime(disp[dc]).dt.strftime("%Y-%m-%d")
            for pc in ("Entry price","Exit price"):
                if pc in disp.columns:
                    disp[pc] = disp[pc].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if "Portfolio value" in disp.columns:
                disp["Portfolio value"] = disp["Portfolio value"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            if "Return %" in disp.columns:
                disp["Return %"] = disp["Return %"].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
            st.dataframe(disp.reset_index(drop=True), width="stretch", hide_index=True)

    st.caption("Sector Rotation reallocates weekly rather than opening discrete "
               "positions, so it produces an equity curve rather than trade rows.")
