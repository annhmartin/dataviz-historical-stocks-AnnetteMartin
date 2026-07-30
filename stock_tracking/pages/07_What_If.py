import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style,
                   titled, show, POS, NEG, ACCENT, HIGHLIGHT, CONTEXT, MUTED, INK)

st.header("What If")

selected, start, end, token = sidebar_filters(show_sector=False)
apply_chart_style()

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
df_sector = load_csv(STRAT_PREFIX + "/equity_curves_by_sector.csv", token)

if df_equity.empty and df_sector.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

st.markdown("Put in an amount and see how each strategy would have handled it over the "
            "date range set in the sidebar.")

sector_name = st.session_state.get("sector", "")
has_sector = (not df_sector.empty and "sector" in df_sector.columns
              and sector_name in set(df_sector["sector"]))

c1, c2 = st.columns([1, 1])
with c1:
    amount = st.number_input("Starting investment ($)", min_value=100,
                             max_value=10_000_000, value=10_000, step=1000,
                             help="Every curve is rescaled to this starting amount "
                                  "across the sidebar date range.")
with c2:
    if has_sector:
        scope = st.radio("Scope", ["Whole portfolio", "Selected sector only"],
                         horizontal=True, key="whatif_scope")
    else:
        scope = "Whole portfolio"

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
eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)]
if eq.empty or not SERIES:
    st.warning("No data in this date range.")
    st.stop()

curves = eq.set_index("date").sort_index()[[s[0] for s in SERIES]].ffill().dropna(how="all")

results = {}
for col, label, colour, dash, width in SERIES:
    s = curves[col].dropna()
    if s.empty:
        continue
    scaled = s * (amount / float(s.iloc[0]))
    results[label] = dict(series=scaled, colour=colour, dash=dash, width=width,
                          final=float(scaled.iloc[-1]),
                          gain=float(scaled.iloc[-1]) - amount,
                          pct=(float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100)

if not results:
    st.warning("No strategy data in this date range.")
    st.stop()

st.markdown("---")
mc = st.columns(len(results))
for (label, d), box in zip(results.items(), mc):
    sign = "+" if d["gain"] >= 0 else "−"
    box.metric(label, f"${d['final']:,.0f}",
               f"{sign}${abs(d['gain']):,.0f}  ({d['pct']:+.1f}%)")

fig = go.Figure()
for label, d in results.items():
    fig.add_trace(go.Scatter(
        x=d["series"].index, y=d["series"].values, mode="lines", name=label,
        line=dict(color=d["colour"], width=d["width"], dash=d["dash"]),
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>$%{{y:,.0f}}<extra></extra>"))
    fig.add_annotation(x=d["series"].index[-1], y=d["final"],
                       text=f"  ${d['final']:,.0f}", showarrow=False, xanchor="left",
                       font=dict(size=12, color=d["colour"]))

fig.add_hline(y=amount, line_color=MUTED, line_width=1.5, line_dash="dot")
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="lines",
    line=dict(color=MUTED, width=2, dash="dot"),
    name=f"Starting amount (${amount:,.0f})"))
fig.add_annotation(x=curves.index[0], y=amount, text=f"started with ${amount:,.0f} ",
                   showarrow=False, xanchor="left", yshift=-16,
                   font=dict(size=12, color=MUTED))

fig.update_yaxes(title="Portfolio value", tickprefix="$")
fig.update_xaxes(title="")
fig.update_layout(
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                itemwidth=40, itemsizing="constant"),
    margin=dict(r=150, t=140),
    hovermode="x unified")

best = max(results, key=lambda k: results[k]["final"])
titled(fig,
       f"What would ${amount:,.0f} have become? ${results[best]['final']:,.0f} in {best}",
       f"From {start:%B %Y} to {end:%B %Y}, including transaction costs",
       height=580)
show(fig)

st.markdown("---")
st.subheader("Every trade, win and loss")

if not df_trades.empty and {"entry_date", "return_pct"} <= set(df_trades.columns):
    t = df_trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t = t[(t["entry_date"] >= start) & (t["entry_date"] <= end)]
    if scope == "Selected sector only" and "ticker" in t.columns:
        t = t[t["ticker"].isin(selected)]

    if t.empty:
        st.info("No trades in this date range.")
    else:
        wins, losses = t[t.return_pct > 0], t[t.return_pct <= 0]
        figt = go.Figure()
        for d, colour, name in [(wins, POS, "Winning trade"), (losses, NEG, "Losing trade")]:
            if d.empty:
                continue
            figt.add_trace(go.Bar(
                x=d["entry_date"], y=d["return_pct"], name=name,
                marker_color=colour, marker_line_width=0,
                customdata=d["ticker"] if "ticker" in d.columns else None,
                hovertemplate=("%{customdata}<br>%{x|%d %b %Y}"
                               "<br>return %{y:+.1f}%<extra></extra>")))
        figt.add_hline(y=0, line_color=MUTED, line_width=1.5)
        figt.update_yaxes(title="Trade return (%)", tickformat="+.0f")
        figt.update_xaxes(title="")
        figt.update_layout(legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
                           bargap=0.3)

        win_rate = len(wins) / len(t) * 100
        titled(figt,
               f"How many trades actually made money? {len(wins)} of {len(t)}, a {win_rate:.0f}% win rate",
               f"Average return {t.return_pct.mean():+.2f}% per trade",
               height=420)
        show(figt)

        k = st.columns(4)
        k[0].metric("Trades", f"{len(t)}")
        k[1].metric("Wins", f"{len(wins)}", f"{win_rate:.0f}%")
        k[2].metric("Losses", f"{len(losses)}", f"{100-win_rate:.0f}%")
        k[3].metric("Average", f"{t.return_pct.mean():+.2f}%")
else:
    st.info("No trade log available.")

st.markdown("---")
summary = pd.DataFrame([{
    "Strategy": label,
    "Started with": f"${amount:,.0f}",
    "Ended with": f"${d['final']:,.0f}",
    "Gain or loss": ("+" if d["gain"] >= 0 else "−") + f"${abs(d['gain']):,.0f}",
    "Total return": f"{d['pct']:+.1f}%",
} for label, d in sorted(results.items(), key=lambda x: -x[1]["final"])])
st.dataframe(summary, width="stretch", hide_index=True)
