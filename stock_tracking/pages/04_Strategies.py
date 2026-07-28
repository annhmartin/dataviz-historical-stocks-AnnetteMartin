
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
try: token = st.secrets["GITHUB_TOKEN"]
except Exception: pass

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
if df_equity.empty: st.warning("No strategy data. Run C_strategy_engine.ipynb first."); st.stop()

df_equity["date"] = pd.to_datetime(df_equity["date"])
if not df_trades.empty and "entry_date" in df_trades.columns:
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])

selected, start, end, token = sidebar_filters()
apply_chart_style()

st.header("Strategies")

# Debug: show actual column names so we can confirm what's in the file
eq_cols = [c for c in df_equity.columns if c != "date"]

# Map whatever columns exist to display labels
# Try multiple possible column name formats
COL_COLORS = {}
COL_LABELS = {}

name_variants = {
    "S&P 500 (SPY)"   : ["SP_500_SPY_", "SPY", "SP500", "S&P_500", "sp_500_spy"],
    "Buy & Hold"      : ["Buy__Hold", "Buy_Hold", "BuyHold", "buy_hold", "equal_weight"],
    "Sector Rotation" : ["Sector_Rotation", "sector_rotation", "SectorRotation"],
    "Position Trader" : ["Position_Trader", "position_trader", "PositionTrader"],
}

color_map = {
    "S&P 500 (SPY)"   : "#f39c12",
    "Buy & Hold"      : "#1a1a2e",
    "Sector Rotation" : "#e74c3c",
    "Position Trader" : "#8e44ad",
}

# Build mapping from actual col name -> display label
col_to_label = {}
for label, variants in name_variants.items():
    for v in variants:
        for actual_col in eq_cols:
            if actual_col.lower() == v.lower() or v.lower() in actual_col.lower():
                col_to_label[actual_col] = label
                COL_COLORS[actual_col]   = color_map[label]
                COL_LABELS[actual_col]   = label

strat_cols = list(col_to_label.keys())

# Show all columns if mapping failed
if not strat_cols:
    strat_cols = eq_cols
    for c in strat_cols:
        COL_COLORS[c] = "#888888"
        COL_LABELS[c] = c.replace("_", " ").strip()

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)].copy()

# Metrics row
mcols = st.columns(len(strat_cols))
for col_name, mcol in zip(strat_cols, mcols):
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    final = float(series.iloc[-1]); s0 = float(series.iloc[0])
    ret = (final / s0 - 1) * 100
    mcol.metric(COL_LABELS.get(col_name, col_name), "$" + "{:,.0f}".format(final), "{:+.1f}%".format(ret))

st.markdown("---")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor="white", gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
spy_series = None

for col_name in strat_cols:
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    color = COL_COLORS.get(col_name, "#888888")
    label = COL_LABELS.get(col_name, col_name)
    ls    = "--" if "SPY" in col_name or "SP_500" in col_name or "S&P" in label else "-"
    lw    = 2.5 if "SPY" in col_name or "Hold" in label or "Rotation" in label or "SP_500" in col_name else 1.8
    ax.plot(series.index, series.values, color=color, linestyle=ls, linewidth=lw,
            label=label + ": $" + "{:,.0f}".format(float(series.iloc[-1])), alpha=0.9)
    if "SPY" in col_name or "SP_500" in col_name or "S&P" in label:
        spy_series = series

rot_col = [c for c in strat_cols if "Rotation" in c or "rotation" in c]
if rot_col and spy_series is not None:
    rot_s = eq.set_index("date")[rot_col[0]].dropna()
    spy_a = spy_series.reindex(rot_s.index).interpolate("time")
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s >= spy_a, color="#a9dfbf", alpha=0.2, label="Rotation beating SPY")
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s <  spy_a, color="#f5b7b1", alpha=0.2, label="SPY beating Rotation")

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
ax.set_title("Portfolio Growth - All Strategies vs S&P 500")
ax.legend(loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.set_facecolor("white")

# Monthly winner bar with gray for all-losing months
ax2 = axes[1]
monthly = eq.set_index("date").resample("ME").last().dropna(how="all")
if not monthly.empty:
    valid_c = [c for c in strat_cols if c in monthly.columns and not monthly[c].isna().all()]
    if valid_c:
        monthly_ret = monthly[valid_c].pct_change()
        winner_each = monthly[valid_c].idxmax(axis=1)
        for i, (month, winner) in enumerate(winner_each.items()):
            if i > 0 and month in monthly_ret.index:
                rets = monthly_ret.loc[month, valid_c]
                if not rets.isna().all() and (rets < 0).all():
                    ax2.bar(month, 1, width=20, color="#aaaaaa", alpha=0.7)
                    continue
            ax2.bar(month, 1, width=20, color=COL_COLORS.get(winner, "#cccccc"), alpha=0.9)
        ax2.set_yticks([])
        ax2.set_ylabel("Leader")
        ax2.set_title("Leading Strategy Each Month (gray = all strategies lost money)")
        patches = [mpatches.Patch(color=COL_COLORS.get(k, "#888888"), label=COL_LABELS.get(k, k)) for k in strat_cols]
        patches.append(mpatches.Patch(color="#aaaaaa", label="All lost"))
        ax2.legend(handles=patches, loc="upper left", ncol=5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax2.set_facecolor("white")

plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("---")
st.subheader("Strategy Definitions")
st.markdown(
    "**S&P 500 (SPY)** - Passive benchmark. Money sits in SPY ETF from day one, never touched.\n\n"
    "**Buy & Hold** - Equal weight across all tickers, held from start to end with no changes.\n\n"
    "**Sector Rotation** - Each week, sentiment determines which sector has the strongest buzz. Money rotates into that sector weighted by sentiment share.\n\n"
    "**Position Trader** - Waits for high-conviction sentiment then holds for 21 trading days. Money stays in sector rotation between positions."
)

if not df_trades.empty:
    st.markdown("---")
    st.subheader("Trade Log")

    # Show what strategy names actually exist in the data
    actual_strategies = df_trades["strategy"].unique().tolist() if "strategy" in df_trades.columns else []

    # Map display names to actual strategy column values
    display_to_actual = {}
    for actual in actual_strategies:
        clean = actual.replace("_", " ").strip()
        display_to_actual[clean] = actual

    strat_options = list(display_to_actual.keys())

    if strat_options:
        strat_filter = st.multiselect(
            "Filter by Strategy",
            options=strat_options,
            default=strat_options  # default = all selected = show all
        )
    else:
        strat_filter = []

    t = df_trades.copy()

    # Only filter if user has made a selection
    if strat_filter and "strategy" in t.columns:
        actual_values = [display_to_actual[s] for s in strat_filter if s in display_to_actual]
        if actual_values:
            t = t[t["strategy"].isin(actual_values)]

    if "entry_date" in t.columns:
        t = t[(t["entry_date"] >= pd.Timestamp(start)) & (t["entry_date"] <= pd.Timestamp(end))]

    col_map = {
        "ticker"        : "Ticker",
        "strategy"      : "Strategy",
        "entry_date"    : "Entry Date",
        "exit_date"     : "Exit Date",
        "entry_price"   : "Entry Price",
        "exit_price"    : "Exit Price",
        "return_pct"    : "Return %",
        "portfolio_val" : "Portfolio Value",
        "trigger_source": "Trigger Source",
    }
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
    st.dataframe(t_disp.sort_values(sort_col, ascending=False).head(200).reset_index(drop=True), width='stretch')

    # Debug expander so we can see what column names are actually in the equity file
    with st.expander("Debug: actual column names in equity file"):
        st.write("Equity columns:", eq_cols)
        st.write("Mapped to:", col_to_label)
        st.write("Strategy values in trade log:", actual_strategies)
