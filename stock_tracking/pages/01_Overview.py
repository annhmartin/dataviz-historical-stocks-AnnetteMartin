import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_signals, load_price, sidebar_filters, get_sig_col, SENTIMENT_THRESHOLD, apply_chart_style,
                   POS, NEG, NEU, INK, PRICE, AXIS, MUTED, GRID, CANVAS, POS_FILL, NEG_FILL)

st.header("Overview")

selected, start, end, token = sidebar_filters()
apply_chart_style()

with st.spinner("Loading sentiment signals..."):
    daily_signals = load_signals(token, tickers=selected,
                                 start_year=start.year, end_year=end.year)

if daily_signals.empty:
    st.error("No signal data found for this sector and date range.")
    st.stop()

sig_col = get_sig_col(daily_signals)

view = st.radio("View Mode", ["One Ticker", "All Stacked"], horizontal=True)

ctrl, legend = st.columns([2, 3])
with ctrl:
    roll = st.slider("Rolling Window (days)", 7, 90, 30)
with legend:
    st.markdown(
        "<div style='padding-top:1.9rem; font-size:0.95rem;'>"
        "<span style='display:inline-block;width:14px;height:14px;background:' + POS + ';"
        "border-radius:3px;vertical-align:middle;margin-right:6px;'></span>Positive"
        "<span style='display:inline-block;width:14px;height:14px;background:' + NEG + ';"
        "border-radius:3px;vertical-align:middle;margin:0 6px 0 18px;'></span>Negative"
        "<span style='display:inline-block;width:14px;height:14px;background:' + NEU + ';"
        "border-radius:3px;vertical-align:middle;margin:0 6px 0 18px;'></span>Neutral"
        "<span style='display:inline-block;width:22px;height:3px;background:' + INK + ';"
        "vertical-align:middle;margin:0 6px 0 18px;'></span>Rolling average"
        "<span style='display:inline-block;width:22px;height:3px;background:' + PRICE + ';"
        "vertical-align:middle;margin:0 6px 0 18px;'></span>7-day price change"
        "</div>",
        unsafe_allow_html=True,
    )

valid = [t for t in selected if t in set(daily_signals["ticker"])]
if not valid:
    st.warning("No signal data for tickers in this sector.")
    st.stop()

tickers_to_plot = [st.selectbox("Ticker", valid)] if view == "One Ticker" else valid

for ticker in tickers_to_plot:
    sig = daily_signals[
        (daily_signals["ticker"] == ticker)
        & (daily_signals["date"] >= start)
        & (daily_signals["date"] <= end)
    ].copy()

    price = load_price(ticker, token)
    if price.empty:
        st.warning(ticker + ": no price data")
        continue
    price = price[(price["Date"] >= start) & (price["Date"] <= end)]

    has_signal = int(sig[sig_col].notna().sum())

    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor=CANVAS, dpi=100)
    ax2 = ax1.twinx()

    pct = price["pct_7d"].fillna(0)
    ax2.fill_between(price["Date"], 0, pct, where=pct >= 0, color=POS, alpha=0.12)
    ax2.fill_between(price["Date"], 0, pct, where=pct < 0,  color=NEG, alpha=0.12)
    ax2.plot(price["Date"], pct, color=PRICE, linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color=AXIS, linewidth=0.5)
    ax2.set_ylabel("7-Day % Price Change", color=PRICE)
    ax2.tick_params(axis="y", labelcolor=PRICE)
    ax2.grid(False)

    if has_signal >= 5:
        sv = sig[sig_col].fillna(0)
        smooth = sig[sig_col].rolling(roll, min_periods=1).mean()
        colors_s = [
            POS if v >= SENTIMENT_THRESHOLD
            else (NEG if v <= -SENTIMENT_THRESHOLD else NEU)
            for v in sv
        ]
        ax1.bar(sig["date"], sv, color=colors_s, alpha=0.5, width=2)
        ax1.plot(sig["date"], smooth, color=INK, linewidth=2, alpha=0.9)
    else:
        ax1.text(0.5, 0.5, "Only " + str(has_signal) + " signal days",
                 ha="center", va="center", transform=ax1.transAxes, color=MUTED)

    ax1.axhline(0, color=AXIS, linewidth=0.5)
    ax1.set_ylabel("Sentiment Score")
    ax1.set_facecolor(CANVAS)
    ax1.set_title(ticker + " - Sentiment and 7-Day % Price Change | "
                  + str(has_signal) + " signal days")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
