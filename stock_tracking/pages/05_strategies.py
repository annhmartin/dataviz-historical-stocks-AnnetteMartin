
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_csv, sidebar_filters, STRAT_PREFIX, STRAT_COLORS, STRAT_LABELS

st.set_page_config(page_title="Strategies", page_icon="💰", layout="wide")

token     = st.secrets.get("GITHUB_TOKEN", None)
df_equity = load_csv(f"{STRAT_PREFIX}/equity_curves.csv", token)
df_trades = load_csv(f"{STRAT_PREFIX}/trade_log.csv",     token)

if df_equity.empty:
    st.warning("No strategy data. Run C_strategy_engine.ipynb first."); st.stop()

df_equity["date"] = pd.to_datetime(df_equity["date"])
if not df_trades.empty and "entry_date" in df_trades.columns:
    df_trades["entry_date"] = pd.to_datetime(df_trades["entry_date"])

all_tickers = df_trades["ticker"].unique().tolist() if not df_trades.empty and "ticker" in df_trades.columns else []
selected, start, end, token = sidebar_filters(all_tickers or ["NVDA"])

st.header("💰 Portfolio Strategy Comparison")

eq = df_equity[(df_equity["date"] >= start) & (df_equity["date"] <= end)]

# Metrics row
strat_cols = [c for c in df_equity.columns if c != "date"]
mcols = st.columns(len(strat_cols))
for col_name, mcol in zip(strat_cols, mcols):
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    final = series.iloc[-1]; s0 = series.iloc[0]
    ret   = (final/s0 - 1) * 100
    label = STRAT_LABELS.get(col_name, col_name.replace("_"," ").strip())
    mcol.metric(label, f"${final:,.0f}", f"{ret:+.1f}%")

st.markdown("---")

fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor="white",
                          gridspec_kw={"height_ratios": [3,1]})
ax = axes[0]
spy_series = None
for col_name in strat_cols:
    series = eq.set_index("date")[col_name].dropna()
    if series.empty: continue
    color = STRAT_COLORS.get(col_name, "#888888")
    label = STRAT_LABELS.get(col_name, col_name.replace("_"," ").strip())
    ls    = "--" if "SPY" in col_name else "-"
    lw    = 2.5 if col_name in ("Sector_Rotation","Buy__Hold","SP_500_SPY_") else 1.8
    ax.plot(series.index, series.values, color=color, linestyle=ls,
            linewidth=lw, label=f"{label}: ${series.iloc[-1]:,.0f}", alpha=0.9)
    if "SPY" in col_name: spy_series = series

rot_col = [c for c in strat_cols if "Rotation" in c]
if rot_col and spy_series is not None:
    rot_s = eq.set_index("date")[rot_col[0]].dropna()
    spy_a = spy_series.reindex(rot_s.index).interpolate("time")
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s>=spy_a, color="#a9dfbf", alpha=0.2)
    ax.fill_between(rot_s.index, spy_a, rot_s, where=rot_s<spy_a,  color="#f5b7b1", alpha=0.2)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.set_title("Portfolio Growth: All Strategies vs S&P 500", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, facecolor="white", loc="upper left", ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_facecolor("white")

ax2 = axes[1]
monthly = eq.set_index("date").resample("ME").last().dropna(how="all")
if not monthly.empty:
    valid_cols = [c for c in monthly.columns if not monthly[c].isna().all()]
    winner_each = monthly[valid_cols].idxmax(axis=1)
    for month, winner in winner_each.items():
        ax2.bar(month, 1, width=20, color=STRAT_COLORS.get(winner,"#cccccc"), alpha=0.9)
    ax2.set_yticks([])
    ax2.set_ylabel("Leader", fontsize=8)
    ax2.set_title("Leading strategy each month", fontsize=10, fontweight="bold")
    patches = [mpatches.Patch(color=v, label=STRAT_LABELS.get(k,k)) for k,v in STRAT_COLORS.items()]
    ax2.legend(handles=patches, fontsize=7, loc="upper left", ncol=3, facecolor="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_facecolor("white")

plt.tight_layout()
st.pyplot(fig)
plt.close()

if not df_trades.empty:
    st.markdown("---")
    st.subheader("Trade Log")
    strat_filter = st.multiselect("Filter by strategy",
        options=df_trades["strategy"].unique().tolist() if "strategy" in df_trades.columns else [],
        default=df_trades["strategy"].unique().tolist() if "strategy" in df_trades.columns else [])
    t = df_trades.copy()
    if strat_filter and "strategy" in t.columns:
        t = t[t["strategy"].isin(strat_filter)]
    if "entry_date" in t.columns:
        t = t[(t["entry_date"] >= start) & (t["entry_date"] <= end)]
    wins  = (t["return_pct"] > 0).sum() if "return_pct" in t.columns else 0
    total_t = len(t)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trades", total_t)
    c2.metric("Win rate", f"{wins/total_t:.1%}" if total_t else "N/A")
    c3.metric("Avg return", f"{t['return_pct'].mean():.2f}%" if "return_pct" in t.columns else "N/A")
    c4.metric("Best trade", f"{t['return_pct'].max():.1f}%" if "return_pct" in t.columns else "N/A")
    st.dataframe(t.sort_values("entry_date", ascending=False).head(200).reset_index(drop=True),
                 use_container_width=True)
