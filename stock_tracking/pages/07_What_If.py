import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_csv, sidebar_filters, STRAT_PREFIX, apply_chart_style

st.set_page_config(page_title="What If", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
if df_equity.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    st.stop()

df_equity["date"] = pd.to_datetime(df_equity["date"])

selected, start, end, token = sidebar_filters(["NVDA"])
apply_chart_style()

st.header("What If")
st.markdown("Enter a starting investment amount and date range to see how each strategy would have performed with your money.")

col1, col2, col3 = st.columns(3)
starting_amount = col1.number_input("Starting Investment ($)", min_value=100, max_value=10000000, value=10000, step=1000)
wi_start = col2.date_input("Start Date", value=pd.Timestamp("2018-01-01").date(), min_value=pd.Timestamp("2015-01-01").date(), max_value=pd.Timestamp.today().date())
wi_end   = col3.date_input("End Date",   value=pd.Timestamp.today().date(),        min_value=pd.Timestamp("2015-01-01").date(), max_value=pd.Timestamp.today().date())

wi_start_ts = pd.Timestamp(wi_start)
wi_end_ts   = pd.Timestamp(wi_end)

SHOW_COLS  = ["SP_500_SPY_", "Buy__Hold", "Sector_Rotation", "Position_Trader"]
COL_COLORS = {"SP_500_SPY_": "#f39c12", "Buy__Hold": "#1a1a2e", "Sector_Rotation": "#e74c3c", "Position_Trader": "#8e44ad"}
COL_LABELS = {"SP_500_SPY_": "S&P 500 (SPY)", "Buy__Hold": "Buy & Hold", "Sector_Rotation": "Sector Rotation", "Position_Trader": "Position Trader"}

eq = df_equity[(df_equity["date"] >= wi_start_ts) & (df_equity["date"] <= wi_end_ts)].copy()
strat_cols = [c for c in SHOW_COLS if c in eq.columns]

if eq.empty:
    st.warning("No data for the selected date range.")
    st.stop()

results = {}
for col_name in strat_cols:
    series = eq.set_index("date")[col_name].dropna()
    if series.empty:
        continue
    s0    = float(series.iloc[0])
    final = float(series.iloc[-1])
    scale = starting_amount / s0 if s0 > 0 else 1.0
    scaled_series = series * scale
    results[col_name] = {
        "series"     : scaled_series,
        "final"      : float(scaled_series.iloc[-1]),
        "gain"       : float(scaled_series.iloc[-1]) - starting_amount,
        "return_pct" : (final / s0 - 1) * 100 if s0 > 0 else 0,
    }

if not results:
    st.warning("No strategy data available for the selected date range.")
    st.stop()

st.markdown("---")
mcols = st.columns(len(results))
for (col_name, data), mcol in zip(results.items(), mcols):
    label = COL_LABELS.get(col_name, col_name)
    gain_str = ("+" if data["gain"] >= 0 else "") + "${:,.0f}".format(data["gain"])
    mcol.metric(label, "${:,.0f}".format(data["final"]), gain_str + " ({:+.1f}%)".format(data["return_pct"]))

fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
for col_name, data in results.items():
    color = COL_COLORS.get(col_name, "#888888")
    label = COL_LABELS.get(col_name, col_name)
    ls    = "--" if "SPY" in col_name else "-"
    lw    = 2.5 if col_name in ("Sector_Rotation", "Buy__Hold", "SP_500_SPY_") else 1.8
    ax.plot(data["series"].index, data["series"].values, color=color, linestyle=ls,
            linewidth=lw, label=label + ": $" + "{:,.0f}".format(data["final"]), alpha=0.9)

ax.axhline(starting_amount, color="#aaaaaa", linewidth=1, linestyle=":", label="Starting amount")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "$" + "{:,.0f}".format(x)))
ax.set_title("What If You Had Invested $" + "{:,.0f}".format(starting_amount) + " on " + str(wi_start) + "?")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")
st.subheader("Summary")
summary_rows = []
for col_name, data in results.items():
    label = COL_LABELS.get(col_name, col_name)
    summary_rows.append({
        "Strategy"       : label,
        "Starting Amount": "${:,.0f}".format(starting_amount),
        "Final Value"    : "${:,.0f}".format(data["final"]),
        "Total Gain/Loss": ("+" if data["gain"] >= 0 else "") + "${:,.0f}".format(data["gain"]),
        "Total Return"   : "{:+.1f}%".format(data["return_pct"]),
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
st.caption("Note: Past performance does not guarantee future results. This is for educational purposes only.")
