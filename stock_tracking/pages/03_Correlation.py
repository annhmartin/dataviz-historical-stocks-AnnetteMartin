import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_csv, load_signals, sidebar_filters, get_sig_col,
                   CORR_PREFIX, SENTIMENT_THRESHOLD, apply_chart_style)

st.header("Correlation")

selected, start, end, token = sidebar_filters()
apply_chart_style()

# ── Signal quality over time ─────────────────────────────────────────────────
st.subheader("Signal Quality Over Time")

with st.spinner("Loading sentiment signals..."):
    daily_signals = load_signals(token, tickers=selected,
                                 start_year=start.year, end_year=end.year)

if daily_signals.empty:
    st.info("No signal data for this sector and date range.")
else:
    valid = [t for t in selected if t in set(daily_signals["ticker"])]
    if not valid:
        st.info("No signal data for tickers in the selected sector.")
    else:
        ticker_t = st.selectbox("Ticker", valid, key="corr_ts")
        sig_col  = get_sig_col(daily_signals)
        sig = daily_signals[
            (daily_signals["ticker"] == ticker_t)
            & (daily_signals["date"] >= start)
            & (daily_signals["date"] <= end)
        ].copy().sort_values("date")

        if sig[sig_col].notna().sum() > 20:
            def rolling_accuracy(x):
                active = x[np.abs(x) > SENTIMENT_THRESHOLD]
                if len(active) == 0:
                    return 0.5
                return float((active > SENTIMENT_THRESHOLD).sum()) / float(len(active))

            sig["roll_acc"] = sig[sig_col].rolling(90, min_periods=20).apply(rolling_accuracy, raw=True)
            fig2, ax2 = plt.subplots(figsize=(14, 4), facecolor="white", dpi=100)
            ax2.plot(sig["date"], sig["roll_acc"], color="#8e44ad", linewidth=2)
            ax2.axhline(0.5, color="#aaaaaa", linewidth=1, linestyle="--")
            ax2.fill_between(sig["date"], 0.5, sig["roll_acc"],
                             where=sig["roll_acc"] >= 0.5, color="#a9dfbf", alpha=0.4)
            ax2.fill_between(sig["date"], 0.5, sig["roll_acc"],
                             where=sig["roll_acc"] < 0.5, color="#f5b7b1", alpha=0.4)
            ax2.set_ylabel("90-Day Rolling Positive Accuracy")
            ax2.set_ylim(0, 1)
            ax2.set_title(ticker_t + " - Signal Quality Over Time (above 50% beats random chance)")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax2.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Not enough signal days to compute rolling quality for this ticker.")

# ── Correlation strength by ticker ───────────────────────────────────────────
st.markdown("---")
st.subheader("Sentiment to Price Correlation")
st.markdown("How strongly does sentiment predict future price moves? "
            "Green = positive buzz preceded price going UP. "
            "Red = positive buzz preceded price going DOWN.")

best_per_ticker = load_csv(CORR_PREFIX + "/best_per_ticker.csv", token)
if best_per_ticker.empty:
    st.warning("No correlation data. Run B_correlation_engine.ipynb first.")
    st.stop()

MIN_N = 200

qualified = best_per_ticker.copy()
if "n" in qualified.columns:
    qualified = qualified[qualified["n"] >= MIN_N]
qualified = qualified.dropna(subset=["corr"])

scope = st.radio("Show", ["Selected sector", "Strongest signals overall"],
                 horizontal=True, key="corr_scope")

if scope == "Selected sector":
    plot_df = qualified[qualified["ticker"].isin(selected)].sort_values("corr")
    if plot_df.empty:
        st.info("None of the tickers in this sector have at least "
                + str(MIN_N) + " data points. Switch to "
                '"Strongest signals overall" to see the market-wide view.')
        st.stop()
    st.caption("Showing " + str(len(plot_df)) + " of " + str(len(selected))
               + " tickers in this sector that have at least " + str(MIN_N) + " data points.")
else:
    TOP_N_SIDE = 20
    plot_df = (pd.concat([qualified.nsmallest(TOP_N_SIDE, "corr"),
                          qualified.nlargest(TOP_N_SIDE, "corr")])
               .drop_duplicates(subset="ticker")
               .sort_values("corr"))
    st.caption("Showing the " + str(len(plot_df)) + " strongest signals out of "
               + "{:,}".format(len(qualified)) + " tickers market-wide with at least "
               + str(MIN_N) + " data points.")

fig_height = min(14, max(4, len(plot_df) * 0.35))
fig, ax = plt.subplots(figsize=(11, fig_height), facecolor="white", dpi=100)
colors = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
labels = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
ax.axvline(0, color="#aaaaaa", linewidth=0.8)
ax.set_xlabel("Pearson Correlation (sentiment -> forward price return)")
ax.set_title("Best Sentiment Signal Per Ticker")
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
