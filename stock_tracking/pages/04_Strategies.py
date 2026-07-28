import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style

st.set_page_config(page_title="Strategies", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv",     token)

if df_equity.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

df_equity["date"] = pd.to_datetime(df_equity["date"])
if not df_trades.empty and "entry_date" in df_trades.columns:
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])

ticker_list = df_trades["ticker"].unique().tolist() if not df_trades.empty and "ticker" in df_trades.columns else ["NVDA"]
selected, start, end, token = sidebar_filters(ticker_list)
apply_chart_style()

st.header("Strategies")

SHOW_COLS  = ["SP_500_SPY_", "Buy__Hold", "Sector_Rotation", "Position_Trader"]
COL_COLORS = {"SP_500_SPY_": "#f39c12", "Buy__Hold": "#1a1a2e", "Sector_Rotation": "#e74c3c", "Position_Trader": "#8e44ad"}
COL_LABELS = {"SP_500_SPY_": "S&P 500 (SPY)", "Buy__Hold": "Buy & Hold", "Sector_Rotation": "Sector Rotation", "Position_Trader": "Position Trader"}

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)].copy()
strat_cols = [c for c in SHOW_COLS if c in eq.columns]

mcols = st.columns(len(strat_cols))
for col_name, mcol in zip(strat_cols, mcols):
    series = eq.set_index("date")[col_name].dropna()
    if series.empty:
        continue
    final = float(series.iloc[-1])
    s0    = float(series.iloc[0])
    ret   = (final / s0 - 1) * 100
    mcol.metric(COL_LABELS.get(col_name, col_name), "$" + "{:,.0f}".format(final), "{:+.1f}%".format(ret))

st.markdown("---")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor="white", gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
spy_series = None
for col_name in strat_cols:
    series = eq.set_index("date")[col_name].dropna()
    if series.empty:
        continue
    color = COL_COLORS.get(col_name, "#888888")
    label = COL_LABELS.get(col_name, col_name)
    ls    = "--" if "SPY" in col_name else "-"
    lw    = 2.5 if col_name in ("Sector_Rotation", "Buy__Hold", "SP_500_SPY_") else 1.8
    ax.plot(series.index, series.values, color=color, linestyle=ls, linewidth=lw,
            label=label + ": $" + "{:,.0f}".format(float(series.iloc[-1])), alpha=0.9)
    if "SPY" in col_name:
        spy_series = series

rot_col = [c for c in strat_cols if "Rotation" in c]
if rot_col and spy_series is not None:
    rot_s = eq.set_index("date")[rot_col[0]].dropna()
    spy_a = spy_series.reindex(rot_s.index).interpolate("time")
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s >= spy_a, color="#a9dfbf", alpha=0.2, label="Rotation beating SPY")
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s < spy_a,  color="#f5b7b1", alpha=0.2, label="SPY beating Rotation")

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
ax.set_title("Portfolio Growth - All Strategies vs S&P 500")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_facecolor("white")

ax2 = axes[1]
monthly = eq.set_index("date").resample("ME").last().dropna(how="all")
if not monthly.empty:
    valid = [c for c in strat_cols if c in monthly.columns and not monthly[c].isna().all()]
    if valid:
        winner_each = monthly[valid].idxmax(axis=1)
        for month, winner in winner_each.items():
            ax2.bar(month, 1, width=20, color=COL_COLORS.get(winner, "#cccccc"), alpha=0.9)
        ax2.set_yticks([])
        ax2.set_ylabel("Leader")
        ax2.set_title("Leading Strategy Each Month")
        patches = [mpatches.Patch(color=v, label=COL_LABELS.get(k, k)) for k, v in COL_COLORS.items() if k in strat_cols]
        ax2.legend(handles=patches, loc="upper left", ncol=4)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.set_facecolor("white")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")
st.subheader("Strategy Definitions")
st.markdown(
    "**S&P 500 (SPY)** - Passive benchmark. Money sits in SPY ETF from day one, never touched. The baseline to beat.\n\n"
    "**Buy & Hold** - Equal weight across all tickers, held from start to end with no changes. Shows what happens if you buy everything and wait.\n\n"
    "**Sector Rotation** - Each week, buzz sentiment determines which sector has the strongest signal. Money rotates into that sector, weighted by each company's share of the sentiment. Between signals, money stays in the previous sector.\n\n"
    "**Position Trader** - Waits for a high-conviction sentiment signal then holds for 21 trading days. Money stays in sector rotation between positions rather than sitting in cash."
)

if not df_trades.empty:
    st.markdown("---")
    st.subheader("Trade Log")
    strat_options = ["Sector Rotation", "Position Trader"]
    strat_filter  = st.multiselect("Filter by Strategy", options=strat_options, default=strat_options)
    t = df_trades.copy()
    strat_map = {"Sector Rotation": "Sector_Rotation", "Position Trader": "Position_Trader"}
    internal  = [strat_map[s] for s in strat_filter if s in strat_map]
    if "strategy" in t.columns and internal:
        t = t[t["strategy"].isin(internal)]
    if "entry_date" in t.columns:
        t = t[(t["entry_date"] >= pd.Timestamp(start)) & (t["entry_date"] <= pd.Timestamp(end))]
    col_map = {"ticker": "Ticker", "strategy": "Strategy", "entry_date": "Entry Date",
               "exit_date": "Exit Date", "entry_price": "Entry Price", "exit_price": "Exit Price",
               "return_pct": "Return %", "portfolio_val": "Portfolio Value", "trigger_source": "Trigger Source"}
    present = [k for k in col_map if k in t.columns]
    t_disp  = t[present].rename(columns=col_map).copy()
    if "Return %"        in t_disp.columns: t_disp["Return %"]        = t_disp["Return %"].apply(lambda x: "{:+.2f}%".format(x) if pd.notna(x) else "")
    if "Entry Price"     in t_disp.columns: t_disp["Entry Price"]     = t_disp["Entry Price"].apply(lambda x: "${:,.2f}".format(x) if pd.notna(x) else "")
    if "Exit Price"      in t_disp.columns: t_disp["Exit Price"]      = t_disp["Exit Price"].apply(lambda x: "${:,.2f}".format(x) if pd.notna(x) else "")
    if "Portfolio Value" in t_disp.columns: t_disp["Portfolio Value"] = t_disp["Portfolio Value"].apply(lambda x: "${:,.0f}".format(x) if pd.notna(x) else "")
    if "Strategy"        in t_disp.columns: t_disp["Strategy"]        = t_disp["Strategy"].str.replace("_", " ")
    if "Entry Date"      in t_disp.columns: t_disp["Entry Date"]      = pd.to_datetime(t_disp["Entry Date"]).dt.date
    if "Exit Date"       in t_disp.columns: t_disp["Exit Date"]       = pd.to_datetime(t_disp["Exit Date"]).dt.date
    wins    = sum(1 for x in t["return_pct"] if pd.notna(x) and x > 0) if "return_pct" in t.columns else 0
    total_t = len(t)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", str(total_t))
    c2.metric("Win Rate",     "{:.1f}%".format(wins / total_t * 100) if total_t else "N/A")
    c3.metric("Avg Return",   "{:+.2f}%".format(float(t["return_pct"].mean())) if "return_pct" in t.columns and total_t else "N/A")
    c4.metric("Best Trade",   "{:+.1f}%".format(float(t["return_pct"].max()))  if "return_pct" in t.columns and total_t else "N/A")
    sort_col = "Entry Date" if "Entry Date" in t_disp.columns else t_disp.columns[0]
    st.dataframe(t_disp.sort_values(sort_col, ascending=False).head(200).reset_index(drop=True), use_container_width=True)
