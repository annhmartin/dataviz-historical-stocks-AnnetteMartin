
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style,
                   INK, AXIS, MUTED, GRID, CANVAS, WIN, LOSS, STRAT_COLORS, STRAT_LABELS)

st.header("What If")

token = None
try: token = st.secrets["GITHUB_TOKEN"]
except Exception: pass

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
df_sector = load_csv(STRAT_PREFIX + "/equity_curves_by_sector.csv", token)

if df_equity.empty and df_sector.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

sector_name = st.session_state.get("sector", "")
has_sector_data = (not df_sector.empty
                   and "sector" in df_sector.columns
                   and sector_name in set(df_sector["sector"]))

if has_sector_data:
    scope = st.radio("Scope", ["Whole portfolio", "Selected sector only"],
                     horizontal=True, key="whatif_scope")
else:
    scope = "Whole portfolio"
    st.caption(
        "Per-sector curves are not in the data yet. Run Section 10 of "
        "C_strategy_engine.ipynb to generate equity_curves_by_sector.csv."
    )

if scope == "Selected sector only" and has_sector_data:
    sub = df_sector[df_sector["sector"] == sector_name].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    df_equity = sub.pivot_table(index="date", columns="strategy",
                                values="value", aggfunc="last").reset_index()
    df_equity = df_equity.rename(columns={"S&P 500 (SPY)": "S&P_500_SPY",
                                          "Buy & Hold": "Buy_&_Hold",
                                          "Position Trader": "Position_Trader",
                                          "Sector Rotation": "Sector_Rotation"})
    st.caption("Backtested using only the tickers in " + sector_name + ".")
else:
    df_equity["date"] = pd.to_datetime(df_equity["date"])
if not df_trades.empty and "entry_date" in df_trades.columns:
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])

# Date range comes from sidebar only
selected, start, end, token = sidebar_filters()
apply_chart_style()

st.markdown("Enter a starting investment amount to see how each strategy would have performed. Date range is controlled by the sidebar.")

starting_amount = st.number_input("Starting Investment ($)", min_value=100, max_value=10000000, value=10000, step=1000)

SHOW_COLS  = ["S&P_500_SPY", "Buy_&_Hold", "Sector_Rotation", "Position_Trader"]
COL_COLORS = STRAT_COLORS
COL_LABELS = STRAT_LABELS

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)].copy()

# Try flexible column name matching
name_variants = {
    "S&P 500 (SPY)"   : ["S&P_500_SPY", "SPY", "SP500", "sp_500_spy"],
    "Buy & Hold"      : ["Buy_&_Hold", "Buy_Hold", "BuyHold", "buy_hold"],
    "Sector Rotation" : ["Sector_Rotation", "sector_rotation"],
    "Position Trader" : ["Position_Trader", "position_trader"],
}
color_map = {v: STRAT_COLORS.get(k, MUTED) for k, v in STRAT_LABELS.items()}
eq_cols = [c for c in df_equity.columns if c != "date"]
col_to_label = {}
for label, variants in name_variants.items():
    for v in variants:
        for actual in eq_cols:
            if actual.lower() == v.lower() or v.lower() in actual.lower():
                if actual not in col_to_label:
                    col_to_label[actual] = label
strat_cols = list(col_to_label.keys()) or eq_cols

if eq.empty or not strat_cols:
    st.warning("No data for selected date range.")
    st.stop()

results = {}
for col_name in strat_cols:
    if col_name not in eq.columns: continue
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    s0 = float(series.iloc[0]); final = float(series.iloc[-1])
    scale = starting_amount / s0 if s0 > 0 else 1.0
    scaled = series * scale
    label = col_to_label.get(col_name, col_name)
    results[col_name] = {
        "label"     : label,
        "series"    : scaled,
        "final"     : float(scaled.iloc[-1]),
        "gain"      : float(scaled.iloc[-1]) - starting_amount,
        "return_pct": (final / s0 - 1) * 100 if s0 > 0 else 0,
    }

if not results:
    st.warning("No strategy data for selected date range.")
    st.stop()

# Metrics
st.markdown("---")
mcols = st.columns(len(results))
for (col_name, data), mcol in zip(results.items(), mcols):
    gain_str = ("+$" if data["gain"] >= 0 else "-$") + "{:,.0f}".format(abs(data["gain"]))
    mcol.metric(data["label"], "$" + "{:,.0f}".format(data["final"]),
                gain_str + " ({:+.1f}%)".format(data["return_pct"]))

# Equity curve
fig, ax = plt.subplots(figsize=(14, 6), facecolor=CANVAS)
for col_name, data in results.items():
    raw_color = color_map.get(data["label"], MUTED)
    ls = "--" if "SPY" in col_name or "S&P" in data["label"] else "-"
    lw = 2.5 if data["label"] in ("Sector Rotation", "Buy & Hold", "S&P 500 (SPY)") else 1.8
    ax.plot(data["series"].index, data["series"].values, color=raw_color, linestyle=ls,
            linewidth=lw, label=data["label"] + ": $" + "{:,.0f}".format(data["final"]), alpha=0.9)
ax.axhline(starting_amount, color=AXIS, linewidth=1, linestyle=":", label="Starting amount")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
ax.set_title("What If You Had Invested $" + "{:,.0f}".format(starting_amount) + "?")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_facecolor(CANVAS)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Good vs bad trades chart
st.markdown("---")
st.subheader("Good vs Bad Trades Over Time")
if not df_trades.empty and "entry_date" in df_trades.columns and "return_pct" in df_trades.columns:
    t = df_trades[(df_trades["entry_date"] >= start) & (df_trades["entry_date"] <= end)].copy()
    if scope == "Selected sector only" and "ticker" in t.columns:
        t = t[t["ticker"].isin(selected)]
    if not t.empty:
        wins   = t[t["return_pct"] >  0]
        losses = t[t["return_pct"] <= 0]
        fig2, ax2 = plt.subplots(figsize=(14, 5), facecolor=CANVAS)
        ax2.bar(wins["entry_date"],   wins["return_pct"],   color=WIN, alpha=0.85, width=5, label="Winning trade")
        ax2.bar(losses["entry_date"], losses["return_pct"], color=LOSS, alpha=0.85, width=5, label="Losing trade")
        ax2.axhline(0, color=AXIS, linewidth=0.8)
        ax2.set_ylabel("Trade Return (%)")
        ax2.set_title("Individual Trades Over Time (green = win, red = loss)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.legend(loc="upper left")
        ax2.set_facecolor(CANVAS)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        w = len(wins); l = len(losses); tot = len(t)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", str(tot))
        c2.metric("Wins",   str(w), str(round(w / tot * 100, 1)) + "%" if tot else "")
        c3.metric("Losses", str(l), str(round(l / tot * 100, 1)) + "%" if tot else "")
        c4.metric("Avg Return", "{:+.2f}%".format(float(t["return_pct"].mean())))
    else:
        st.info("No trades in selected date range.")
else:
    st.info("No trade log data available.")

# Summary table
st.markdown("---")
st.subheader("Summary")
summary_rows = []
for col_name, data in results.items():
    summary_rows.append({
        "Strategy"       : data["label"],
        "Starting Amount": "$" + "{:,.0f}".format(starting_amount),
        "Final Value"    : "$" + "{:,.0f}".format(data["final"]),
        "Total Gain/Loss": ("+$" if data["gain"] >= 0 else "-$") + "{:,.0f}".format(abs(data["gain"])),
        "Total Return"   : "{:+.1f}%".format(data["return_pct"]),
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
