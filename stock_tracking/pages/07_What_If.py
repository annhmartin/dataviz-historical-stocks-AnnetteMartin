
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
try: token = st.secrets["GITHUB_TOKEN"]
except Exception: pass

df_equity = load_csv(STRAT_PREFIX + "/equity_curves.csv", token)
if df_equity.empty: st.warning("No strategy data. Run C_strategy_engine.ipynb first."); st.stop()

df_equity["date"] = pd.to_datetime(df_equity["date"])
selected, start, end, token = sidebar_filters()
apply_chart_style()

st.header("What If")
st.markdown("Enter a starting investment amount to see how each strategy would have performed. Date range is controlled by the sidebar.")

starting_amount = st.number_input("Starting Investment ($)", min_value=100, max_value=10000000, value=10000, step=1000)

SHOW_COLS  = ["SP_500_SPY_","Buy__Hold","Sector_Rotation","Position_Trader"]
COL_COLORS = {"SP_500_SPY_":"#f39c12","Buy__Hold":"#1a1a2e","Sector_Rotation":"#e74c3c","Position_Trader":"#8e44ad"}
COL_LABELS = {"SP_500_SPY_":"S&P 500 (SPY)","Buy__Hold":"Buy & Hold","Sector_Rotation":"Sector Rotation","Position_Trader":"Position Trader"}

eq = df_equity[(df_equity["date"]>=start) & (df_equity["date"]<=end)].copy()
strat_cols = [c for c in SHOW_COLS if c in eq.columns]
if eq.empty or not strat_cols: st.warning("No data for the selected date range."); st.stop()

results = {}
for col_name in strat_cols:
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    s0 = float(series.iloc[0]); final = float(series.iloc[-1])
    scale = starting_amount/s0 if s0 > 0 else 1.0
    scaled = series*scale
    results[col_name] = {"series":scaled,"final":float(scaled.iloc[-1]),"gain":float(scaled.iloc[-1])-starting_amount,"return_pct":(final/s0-1)*100 if s0>0 else 0}

if not results: st.warning("No strategy data for the selected date range."); st.stop()

st.markdown("---")
mcols = st.columns(len(results))
for (col_name,data), mcol in zip(results.items(), mcols):
    gain_str = ("+$" if data["gain"]>=0 else "-$")+"{:,.0f}".format(abs(data["gain"]))
    mcol.metric(COL_LABELS.get(col_name,col_name), "$"+"{:,.0f}".format(data["final"]), gain_str+" ({:+.1f}%)".format(data["return_pct"]))

# Main equity curve
fig, ax = plt.subplots(figsize=(14,6), facecolor="white")
for col_name, data in results.items():
    color = COL_COLORS.get(col_name,"#888888"); label = COL_LABELS.get(col_name,col_name)
    ls = "--" if "SPY" in col_name else "-"
    lw = 2.5 if col_name in ("Sector_Rotation","Buy__Hold","SP_500_SPY_") else 1.8
    ax.plot(data["series"].index, data["series"].values, color=color, linestyle=ls, linewidth=lw,
            label=label+": $"+"{:,.0f}".format(data["final"]), alpha=0.9)
ax.axhline(starting_amount, color="#aaaaaa", linewidth=1, linestyle=":", label="Starting amount")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: "$"+"{:,.0f}".format(x)))
ax.set_title("What If You Had Invested $"+"{:,.0f}".format(starting_amount)+"?")
ax.legend(loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.set_facecolor("white")
plt.tight_layout(); st.pyplot(fig); plt.close()

# Good vs bad trades chart
st.markdown("---")
st.subheader("Good vs Bad Trades Over Time")
df_trades = load_csv(STRAT_PREFIX + "/trade_log.csv", token)
if not df_trades.empty and "entry_date" in df_trades.columns and "return_pct" in df_trades.columns:
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])
    t = df_trades[(df_trades["entry_date"]>=start) & (df_trades["entry_date"]<=end)].copy()
    if not t.empty:
        fig2, ax2 = plt.subplots(figsize=(14,5), facecolor="white")
        wins = t[t["return_pct"]>0]; losses = t[t["return_pct"]<=0]
        ax2.bar(wins["entry_date"],   wins["return_pct"],   color="#27ae60", alpha=0.8, width=5, label="Winning trade")
        ax2.bar(losses["entry_date"], losses["return_pct"], color="#e74c3c", alpha=0.8, width=5, label="Losing trade")
        ax2.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax2.set_ylabel("Trade Return (%)")
        ax2.set_title("Individual Trades Over Time (green=win, red=loss)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.legend(loc="upper left"); ax2.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        # Summary
        w = len(wins); l = len(losses); tot = len(t)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Trades", str(tot))
        c2.metric("Wins", str(w), str(round(w/tot*100,1))+"%" if tot else "")
        c3.metric("Losses", str(l), str(round(l/tot*100,1))+"%" if tot else "")
        c4.metric("Avg Return", "{:+.2f}%".format(float(t["return_pct"].mean())))
    else:
        st.info("No trades in selected date range.")
else:
    st.info("No trade log data available.")

st.markdown("---")
st.subheader("Summary Table")
summary_rows = []
for col_name, data in results.items():
    summary_rows.append({"Strategy":COL_LABELS.get(col_name,col_name),"Starting Amount":"$"+"{:,.0f}".format(starting_amount),"Final Value":"$"+"{:,.0f}".format(data["final"]),"Total Gain/Loss":("+$" if data["gain"]>=0 else "-$")+"{:,.0f}".format(abs(data["gain"])),"Total Return":"{:+.1f}%".format(data["return_pct"])})
st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
st.caption("Past performance does not guarantee future results. For educational purposes only.")
