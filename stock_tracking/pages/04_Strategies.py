import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style,
                   INK, AXIS, MUTED, GRID, CANVAS, POS_FILL, NEG_FILL, STRAT_COLORS, STRAT_LABELS, STRAT_NEUTRAL)

st.header("Strategies")

st.markdown(
    "**S&P 500 (SPY)** - Passive benchmark. Money sits in the SPY ETF from day one and is never touched.\n\n"
    "**Buy & Hold** - Equal weight across all tickers, held start to end with no changes.\n\n"
    "**Sector Rotation** - Each week sentiment picks the sector with the strongest buzz, and money "
    "rotates into that sector weighted by each company's share of the sentiment.\n\n"
    "**Position Trader** - Waits for a high-conviction sentiment signal, then holds for 21 trading days. "
    "Money sits in sector rotation between positions."
)
st.markdown("---")

selected, start, end, token = sidebar_filters()
apply_chart_style()

df_equity  = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
df_trades  = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
df_sector  = load_csv(STRAT_PREFIX + "/equity_curves_by_sector.csv", token)

if df_equity.empty and df_sector.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

sector_name = st.session_state.get("sector", "")
has_sector_data = (not df_sector.empty
                   and "sector" in df_sector.columns
                   and sector_name in set(df_sector["sector"]))

if has_sector_data:
    scope = st.radio("Scope", ["Whole portfolio", "Selected sector only"],
                     horizontal=True, key="strat_scope")
else:
    scope = "Whole portfolio"
    st.caption(
        "Per-sector curves are not in the data yet. Run Section 10 of "
        "C_strategy_engine.ipynb to generate equity_curves_by_sector.csv, "
        "and the sector filter will apply to these charts."
    )

if scope == "Selected sector only" and has_sector_data:
    sub = df_sector[df_sector["sector"] == sector_name].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    df_equity = sub.pivot_table(index="date", columns="strategy",
                                values="value", aggfunc="last").reset_index()
    rename_map = {"S&P 500 (SPY)": "S&P_500_SPY",
                  "Buy & Hold": "Buy_&_Hold",
                  "Position Trader": "Position_Trader",
                  "Sector Rotation": "Sector_Rotation"}
    df_equity = df_equity.rename(columns=rename_map)
    st.caption("Backtested using only the tickers in " + sector_name + ".")
else:
    df_equity["date"] = pd.to_datetime(df_equity["date"])

# Column names exactly as written by C_strategy_engine
STRATEGIES = [
    ("S&P_500_SPY",     "S&P 500 (SPY)",   MARKER_EVENT, "--", 2.2),
    ("Buy_&_Hold",      "Buy & Hold",      STRAT_COLORS["Buy_&_Hold"],      "-",  2.0),
    ("Sector_Rotation", "Sector Rotation", NEG, "-",  2.5),
    ("Position_Trader", "Position Trader", MARKER_SIGNAL, "-",  1.8),
]
STRATEGIES = [s for s in STRATEGIES if s[0] in df_equity.columns]

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)].copy()
if eq.empty or not STRATEGIES:
    st.warning("No strategy data in the selected date range.")
    st.stop()

# Each strategy records on its own dates, so forward-fill to align them
eq_idx = eq.set_index("date").sort_index()
aligned = eq_idx[[s[0] for s in STRATEGIES]].ffill()

mcols = st.columns(len(STRATEGIES))
for (col, label, color, ls, lw), mcol in zip(STRATEGIES, mcols):
    series = aligned[col].dropna()
    if series.empty:
        continue
    final = float(series.iloc[-1])
    s0    = float(series.iloc[0])
    ret   = (final / s0 - 1) * 100 if s0 else 0
    mcol.metric(label, "$" + "{:,.0f}".format(final), "{:+.1f}%".format(ret))

st.markdown("---")

def quarter_fmt(x, _pos=None):
    d = mdates.num2date(x)
    return str(d.year) + " Q" + str((d.month - 1) // 3 + 1)

fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor=CANVAS, dpi=100,
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]

spy_series = None
for col, label, color, ls, lw in STRATEGIES:
    series = aligned[col].dropna()
    if series.empty:
        continue
    ax.plot(series.index, series.values, color=color, linestyle=ls, linewidth=lw,
            label=label + ": $" + "{:,.0f}".format(float(series.iloc[-1])), alpha=0.9)
    if col == "S&P_500_SPY":
        spy_series = series

if spy_series is not None and "Sector_Rotation" in aligned.columns:
    rot_s = aligned["Sector_Rotation"].dropna()
    spy_a = spy_series.reindex(rot_s.index).ffill()
    ok = rot_s.notna() & spy_a.notna()
    ax.fill_between(rot_s.index[ok], spy_a[ok], rot_s[ok],
                    where=rot_s[ok] >= spy_a[ok], color=POS_FILL, alpha=0.20,
                    label="Rotation ahead of SPY")
    ax.fill_between(rot_s.index[ok], spy_a[ok], rot_s[ok],
                    where=rot_s[ok] < spy_a[ok], color=NEG_FILL, alpha=0.20,
                    label="SPY ahead of Rotation")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
ax.set_title("Portfolio Growth - All Strategies vs S&P 500")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mticker.FuncFormatter(quarter_fmt))
ax.set_facecolor(CANVAS)

# Leading strategy each month, gray when every strategy lost money
ax2 = axes[1]
monthly = aligned.resample("ME").last().dropna(how="all")
if not monthly.empty:
    monthly_ret = monthly.pct_change()
    label_for  = {s[0]: s[1] for s in STRATEGIES}
    color_for  = {s[0]: s[2] for s in STRATEGIES}
    winners = monthly.idxmax(axis=1)
    for i, (month, winner) in enumerate(winners.items()):
        color = color_for.get(winner, GRID)
        if i > 0 and month in monthly_ret.index:
            rets = monthly_ret.loc[month].dropna()
            if len(rets) > 0 and (rets < 0).all():
                color = AXIS
        ax2.bar(month, 1, width=20, color=color, alpha=0.9)
    ax2.set_yticks([])
    ax2.set_ylabel("Leader")
    ax2.set_title("Leading Strategy Each Month (gray = every strategy lost money)")
    patches = [mpatches.Patch(color=c, label=l) for _, l, c, _, _ in STRATEGIES]
    patches.append(mpatches.Patch(color=AXIS, label="All lost"))
    ax2.legend(handles=patches, loc="upper left", ncol=5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(quarter_fmt))
    ax2.set_facecolor(CANVAS)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

if not df_trades.empty and "strategy" in df_trades.columns:
    st.markdown("---")
    st.subheader("Position Trade Log")

    # Sector Rotation reallocates weekly rather than opening discrete positions,
    # and the optimal-stopping variants are not shown on this page, so Position
    # Trader is the only strategy with individual trades to list.
    t = df_trades[df_trades["strategy"].astype(str).str.strip() == "Position Trader"].copy()

    if t.empty:
        st.info("No Position Trader trades recorded.")
    else:
        sector_only = st.checkbox("Selected sector only", value=False)
        if sector_only and "ticker" in t.columns:
            t = t[t["ticker"].isin(selected)]

        if "entry_date" in t.columns:
            t["entry_date"] = pd.to_datetime(t["entry_date"])
            t = t[(t["entry_date"] >= start) & (t["entry_date"] <= end)]

        total_t = len(t)
        wins = int((t["return_pct"] > 0).sum()) if total_t and "return_pct" in t.columns else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", str(total_t))
        c2.metric("Win Rate",   "{:.1f}%".format(wins / total_t * 100) if total_t else "N/A")
        c3.metric("Avg Return", "{:+.2f}%".format(float(t["return_pct"].mean())) if total_t else "N/A")
        c4.metric("Best Trade", "{:+.1f}%".format(float(t["return_pct"].max())) if total_t else "N/A")

        if total_t:
            col_map = {
                "ticker": "Ticker",
                "entry_date": "Entry Date", "exit_date": "Exit Date",
                "entry_price": "Entry Price", "exit_price": "Exit Price",
                "return_pct": "Return %", "portfolio_val": "Portfolio Value",
                "trigger_source": "Trigger Source",
            }
            keep = [x for x in col_map if x in t.columns]
            disp = t[keep].rename(columns=col_map)
            disp = disp.sort_values("Entry Date", ascending=False)
            if "Entry Date" in disp.columns:
                disp["Entry Date"] = pd.to_datetime(disp["Entry Date"]).dt.strftime("%Y-%m-%d")
            if "Exit Date" in disp.columns:
                disp["Exit Date"] = pd.to_datetime(disp["Exit Date"]).dt.strftime("%Y-%m-%d")
            for cc in ("Entry Price", "Exit Price"):
                if cc in disp.columns:
                    disp[cc] = disp[cc].apply(lambda x: "${:,.2f}".format(x) if pd.notna(x) else "")
            if "Portfolio Value" in disp.columns:
                disp["Portfolio Value"] = disp["Portfolio Value"].apply(
                    lambda x: "${:,.0f}".format(x) if pd.notna(x) else "")
            if "Return %" in disp.columns:
                disp["Return %"] = disp["Return %"].apply(
                    lambda x: "{:+.2f}%".format(x) if pd.notna(x) else "")
            st.dataframe(disp.reset_index(drop=True), width="stretch", hide_index=True)
        else:
            st.info("No trades in this date range.")

    st.caption(
        "Only Position Trader opens and closes discrete positions. Sector Rotation "
        "reallocates across a whole sector each week, so it is recorded as an equity "
        "curve rather than individual trades."
    )
