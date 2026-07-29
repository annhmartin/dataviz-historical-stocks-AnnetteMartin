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

best_per_ticker = load_csv(CORR_PREFIX + "/best_per_ticker.csv", token)
if best_per_ticker.empty:
    st.warning("No correlation data. Run B_correlation_engine.ipynb first.")
    st.stop()

st.markdown("How strongly does sentiment predict future price moves? "
            "Green = positive buzz preceded price going UP. "
            "Red = positive buzz preceded price going DOWN.")

MIN_N      = 200   # minimum data points for a correlation to be trustworthy
TOP_N_SIDE = 20    # strongest signals shown per direction

qualified = best_per_ticker.copy()
if "n" in qualified.columns:
    qualified = qualified[qualified["n"] >= MIN_N]
qualified = qualified.dropna(subset=["corr"])

if qualified.empty:
    st.info("No tickers have enough data points to show a reliable correlation.")
    st.stop()

# Showing every qualifying ticker produces a figure tens of thousands of pixels
# tall, so display the strongest signals in each direction instead.
strongest_pos = qualified.nlargest(TOP_N_SIDE, "corr")
strongest_neg = qualified.nsmallest(TOP_N_SIDE, "corr")
plot_df = (pd.concat([strongest_neg, strongest_pos])
           .drop_duplicates(subset="ticker")
           .sort_values("corr"))

st.caption(
    "Showing the " + str(len(plot_df)) + " strongest signals out of "
    + "{:,}".format(len(qualified)) + " tickers with at least " + str(MIN_N) + " data points."
)

# Height is capped so the rendered image stays within image-size limits
fig_height = min(14, max(6, len(plot_df) * 0.35))
fig, ax = plt.subplots(figsize=(11, fig_height), facecolor="white", dpi=100)
colors = ["#e74c3c" if v < 0 else "#27ae60" for v in plot_df["corr"]]
labels = plot_df["ticker"] + " (" + plot_df["signal"] + " T+" + plot_df["horizon"].astype(str) + ")"
ax.barh(labels, plot_df["corr"], color=colors, edgecolor="white")
ax.axvline(0, color="#aaaaaa", linewidth=0.8)
ax.set_xlabel("Pearson Correlation (sentiment -> forward price return)")
ax.set_title("Strongest Sentiment Signals")
ax.set_facecolor("white")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")
st.subheader("Ticker Definitions")

TICKER_INFO = {
    "NVDA": "NVIDIA - AI chips and GPUs",
    "AMD":  "Advanced Micro Devices - CPUs and GPUs",
    "TSM":  "Taiwan Semiconductor - chip foundry",
    "INTC": "Intel - semiconductors and processors",
    "QCOM": "Qualcomm - mobile chips and wireless",
    "GOOGL":"Alphabet / Google - search, cloud, AI",
    "MSFT": "Microsoft - software, Azure, OpenAI partner",
    "AAPL": "Apple - iPhone, Mac, services",
    "META": "Meta Platforms - Facebook, Instagram, WhatsApp",
    "AMZN": "Amazon - e-commerce and AWS",
    "SNOW": "Snowflake - cloud data platform",
    "DDOG": "Datadog - cloud monitoring",
    "CRM":  "Salesforce - CRM and enterprise software",
    "NOW":  "ServiceNow - workflow automation",
    "MDB":  "MongoDB - NoSQL database platform",
    "CRWD": "CrowdStrike - endpoint cybersecurity",
    "PANW": "Palo Alto Networks - network security",
    "OKTA": "Okta - identity and access management",
    "PLTR": "Palantir - enterprise and government data analytics",
    "COIN": "Coinbase - cryptocurrency exchange",
    "TSLA": "Tesla - electric vehicles and energy",
    "INCY": "Incyte - biopharmaceuticals",
    "KGC":  "Kinross Gold - gold mining",
    "NVO":  "Novo Nordisk - pharma, Ozempic and Wegovy",
    "PM":   "Philip Morris International - tobacco and IQOS",
    "WPM":  "Wheaton Precious Metals - metals streaming",
    "NFLX": "Netflix - streaming entertainment",
    "SPOT": "Spotify - music and podcast streaming",
    "PINS": "Pinterest - visual discovery platform",
    "PYPL": "PayPal - digital payments",
    "GFS":  "GlobalFoundries - semiconductor foundry",
    "DNUT": "Krispy Kreme - baked goods",
    "SPY":  "SPDR S&P 500 ETF - index benchmark",
    "QQQ":  "Invesco QQQ - Nasdaq 100 index fund",
}

rows = []
for _, row in plot_df.iterrows():
    corr_val = float(row["corr"])
    rows.append({
        "Ticker"          : row["ticker"],
        "Company"         : TICKER_INFO.get(row["ticker"], "See SEC EDGAR for full name"),
        "Correlation"     : "{:+.3f}".format(corr_val),
        "Signal Direction": "Follow positive buzz" if corr_val > 0 else "Contrarian signal",
        "Best Signal"     : row["signal"],
        "Best Horizon"    : "T+" + str(int(row["horizon"])),
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.markdown("---")
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
        ticker_t = st.selectbox("Ticker for time series", valid, key="corr_ts")
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
