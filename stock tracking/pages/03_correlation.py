
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_csv, sidebar_filters, CORR_PREFIX

st.set_page_config(page_title="Correlation", page_icon="🔗", layout="wide")

token = st.secrets.get("GITHUB_TOKEN", None)
best_per_ticker = load_csv(f"{CORR_PREFIX}/best_per_ticker.csv", token)
df_corr         = load_csv(f"{CORR_PREFIX}/corr_matrix.csv", token)

if best_per_ticker.empty:
    st.warning("No correlation data. Run B_correlation_engine.ipynb first."); st.stop()

all_tickers = sorted(best_per_ticker["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)

st.header("🔗 Sentiment-to-Price Correlation")
st.markdown("""
**How to read:** Bar length = strength. Green = positive buzz predicted price UP.
Red = positive buzz predicted price DOWN (contrarian — consider using negative buzz instead).
★ = statistically significant (p < 0.05). Label shows which signal type and time horizon worked best.
""")

col1, col2, col3 = st.columns(3)
sig_only = col1.checkbox("Significant only (p<0.05)", value=True)
n_show   = col2.slider("Tickers to show", 10, 50, 20)
min_n    = col3.slider("Min data points (n)", 50, 500, 200)

plot_df = best_per_ticker.copy()
if "n" in plot_df.columns:
    plot_df = plot_df[plot_df["n"] >= min_n]
if sig_only and "significant" in plot_df.columns:
    plot_df = plot_df[plot_df["significant"]==True]
plot_df = plot_df.sort_values("corr").tail(n_show)

if plot_df.empty:
    st.info("No tickers match filters. Try lowering min n or unchecking significant only."); st.stop()

fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df)*0.4)), facecolor="white")
colors  = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
labels  = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
bars    = ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
ax.axvline(0, color="#aaaaaa", linewidth=0.8)
if "significant" in plot_df.columns:
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        if row.get("significant"):
            ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                    "★", va="center", fontsize=9, color="gold")
ax.set_xlabel("Pearson Correlation", fontsize=10)
ax.set_title(f"Best Sentiment Signal per Ticker | {len(plot_df)} shown", fontsize=12, fontweight="bold")
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close()

with st.expander("Raw correlation table"):
    cols = [c for c in ["ticker","signal","horizon","corr","pval","n","significant"] if c in plot_df.columns]
    st.dataframe(plot_df[cols].reset_index(drop=True), use_container_width=True)
