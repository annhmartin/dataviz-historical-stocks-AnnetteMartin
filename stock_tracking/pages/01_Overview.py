import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_signals, load_price, sidebar_filters, get_sig_col,
                   SENTIMENT_THRESHOLD, apply_chart_style, titled, show,
                   POS, NEG, NEU, INK, PRICE, MUTED, CONTEXT, GRID)

st.header("Sentiment vs Price")

selected, start, end, token = sidebar_filters()
apply_chart_style()

with st.spinner("Loading sentiment signals..."):
    daily_signals = load_signals(token, tickers=selected,
                                 start_year=start.year, end_year=end.year)

if daily_signals.empty:
    st.error("No signal data found for this sector and date range.")
    st.stop()

sig_col = get_sig_col(daily_signals)

st.markdown(
    "Sentiment scored from six sources, shown against how the price actually moved. "
    "Blue bars are positive days, orange negative, grey inside the neutral threshold."
)

c1, c2 = st.columns([1, 1])
with c1:
    view = st.radio("View", ["One ticker", "All in sector"], horizontal=True)
with c2:
    roll = st.slider("Rolling window (days)", 7, 90, 30,
                     help="How many days to average the sentiment line over. "
                          "Wider smooths daily noise but reacts more slowly.")

valid = [t for t in selected if t in set(daily_signals["ticker"])]
if not valid:
    st.warning("No signal data for tickers in this sector.")
    st.stop()

tickers_to_plot = [st.selectbox("Ticker", valid)] if view == "One ticker" else valid

for ticker in tickers_to_plot:
    sig = daily_signals[
        (daily_signals["ticker"] == ticker)
        & (daily_signals["date"] >= start)
        & (daily_signals["date"] <= end)
    ].copy().sort_values("date")

    price = load_price(ticker, token)
    if price.empty:
        st.warning(f"{ticker}: no price data")
        continue
    price = price[(price["Date"] >= start) & (price["Date"] <= end)]

    n_signal = int(sig[sig_col].notna().sum())
    if n_signal < 5:
        st.info(f"{ticker}: only {n_signal} signal days in this range.")
        continue

    sv = sig[sig_col].fillna(0)
    smooth = sig[sig_col].rolling(roll, min_periods=1).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # One trace per category so each gets its own legend entry
    for mask, colour, label in [
        (sv >= SENTIMENT_THRESHOLD,                              POS, "Positive sentiment"),
        (sv <= -SENTIMENT_THRESHOLD,                             NEG, "Negative sentiment"),
        ((sv > -SENTIMENT_THRESHOLD) & (sv < SENTIMENT_THRESHOLD), NEU, "Neutral"),
    ]:
        if not mask.any():
            continue
        fig.add_trace(go.Bar(
            x=sig["date"][mask], y=sv[mask],
            marker_color=colour, marker_line_width=0,
            name=label, opacity=0.7,
            hovertemplate="%{x|%d %b %Y}<br>sentiment %{y:.3f}<extra></extra>"),
            secondary_y=False)

    fig.add_trace(go.Scatter(
        x=sig["date"], y=smooth, mode="lines",
        line=dict(color=INK, width=2.5), name=f"{roll}-day average",
        hovertemplate="%{x|%d %b %Y}<br>average %{y:.3f}<extra></extra>"),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=price["Date"], y=price["pct_7d"], mode="lines",
        line=dict(color=PRICE, width=1.6), name="7-day price change",
        hovertemplate="%{x|%d %b %Y}<br>price %{y:+.1f}%<extra></extra>"),
        secondary_y=True)

    fig.add_hline(y=0, line_color=MUTED, line_width=1, line_dash="dot")

    fig.update_yaxes(title_text="Sentiment score", secondary_y=False)
    fig.update_yaxes(title_text="7-day price change (%)", secondary_y=True,
                     showgrid=False, tickformat="+.0f")
    fig.update_layout(
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
        barmode="relative", bargap=0.1, hovermode="x unified")

    avg_sent = float(sig[sig_col].mean())
    mood = "leaned positive" if avg_sent > 0.02 else ("leaned negative" if avg_sent < -0.02 else "stayed close to neutral")
    titled(fig,
           f"How did sentiment toward {ticker} behave? It {mood}",
           f"{n_signal:,} days carried a sentiment reading | "
           f"average score {avg_sent:+.3f}",
           height=460)
    show(fig)
