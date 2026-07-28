import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_signals, load_price, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD, apply_chart_style

st.set_page_config(page_title="Overview", layout="wide")
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

@st.cache_data(ttl=3600, show_spinner="Loading signals...")
def get_signals(t):
    return load_signals(t)

daily_signals = get_signals(token)
if daily_signals.empty:
    st.error("No signal data found. Check that sentiment_outputs folder exists in stock_tracking/.")
    st.stop()

all_tickers = sorted(daily_signals["ticker"].unique().tolist())
selected, start, end, token = sidebar_filters(all_tickers)
if not selected:
    st.warning("No tickers available for selected sector.")
    st.stop()

sig_col = get_sig_col(daily_signals)
apply_chart_style()

st.header("Overview")
st.markdown("Bars = daily sentiment (green=positive, red=negative, gray=neutral) | Dark line = rolling average | Blue shading = 7-day price change %")

view = st.radio("View Mode", ["One Ticker", "All Stacked"], horizontal=True)
roll = st.slider("Rolling Window (days)", 7, 90, 30)
tickers_to_plot = [st.selectbox("Ticker", selected)] if view == "One Ticker" else selected

for ticker in tickers_to_plot:
    sig = daily_signals[
        (daily_signals["ticker"] == ticker) &
        (daily_signals["date"] >= start) &
        (daily_signals["date"] <= end)
    ].copy()
    price = load_price(ticker, token)
    if price.empty:
        st.warning(ticker + ": no price data")
        continue
    price = price[(price["Date"] >= start) & (price["Date"] <= end)]
    has_signal = int(sig[sig_col].notna().sum())
    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor="white")
    ax2 = ax1.twinx()
    pct = price["pct_7d"].fillna(0)
    ax2.fill_between(price["Date"], 0, pct, where=pct >= 0, color="#27ae60", alpha=0.12)
    ax2.fill_between(price["Date"], 0, pct, where=pct < 0,  color="#e74c3c", alpha=0.12)
    ax2.plot(price["Date"], pct, color="#2980b9", linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax2.set_ylabel("7-Day % Price Change", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.grid(False)
    if has_signal >= 5:
        sv = sig[sig_col].fillna(0)
        smooth = sig[sig_col].rolling(roll, min_periods=1).mean()
        colors_s = ["#27ae60" if v >= SENTIMENT_THRESHOLD else ("#e74c3c" if v <= -SENTIMENT_THRESHOLD else "#bdc3c7") for v in sv]
        ax1.bar(sig["date"], sv, color=colors_s, alpha=0.5, width=2)
        ax1.plot(sig["date"], smooth, color="#2c3e50", linewidth=2, alpha=0.9, label=str(roll) + "d rolling avg")
    else:
        ax1.text(0.5, 0.5, "Only " + str(has_signal) + " signal days", ha="center", va="center", transform=ax1.transAxes, color="#888888")
    ax1.axhline(0, color="#aaaaaa", linewidth=0.5)
    ax1.set_ylabel("Sentiment Score")
    ax1.set_facecolor("white")
    ax1.set_title(ticker + " - Sentiment + 7-Day % Price Change | " + str(has_signal) + " signal days")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if has_signal >= 5:
        ax1.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
