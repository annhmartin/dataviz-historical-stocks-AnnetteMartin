import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_signals, load_price, load_csv, sidebar_filters, get_sig_col,
                   SENTIMENT_THRESHOLD, CORR_PREFIX, apply_chart_style, titled, show,
                   POS, NEG, NEU, PRICE, MUTED, INK, GOLD, HIGHLIGHT)

st.header("Key Moments")

selected, start, end, token = sidebar_filters()
apply_chart_style()

df_key_moves = load_csv(CORR_PREFIX + "/key_moves.csv", token)
if df_key_moves.empty:
    st.warning("No key moves data. Run B_correlation_engine.ipynb first.")
    st.stop()

for col in ["move_date", "sent_date"]:
    if col in df_key_moves.columns:
        df_key_moves[col] = pd.to_datetime(df_key_moves[col])

st.markdown(
    "Zooming in on days when a stock moved abnormally — beyond two standard deviations — "
    "to see what the chatter looked like in the weeks beforehand."
)

valid = [t for t in selected if t in set(df_key_moves["ticker"])]
if not valid:
    st.info("No key moves found for tickers in this sector.")
    st.stop()

c1, c2 = st.columns(2)
ticker_km = c1.selectbox("Ticker", valid)
predicted_only = c2.checkbox("Correctly predicted only", value=True,
                             help="Show only moves where sentiment pointed the same "
                                  "way as the price move that followed.")

km = df_key_moves[df_key_moves["ticker"] == ticker_km].copy()
if predicted_only and "predicted" in km.columns:
    km = km[km["predicted"] == True]
km = km[(km["move_date"] >= start) & (km["move_date"] <= end)]
km = km.sort_values("move_date", ascending=False).head(20)

if km.empty:
    st.info("No key moves here. Try unchecking the filter or widening the date range.")
    st.stop()

labels = [f"{r.move_date:%d %b %Y}   {r.return_pct:+.1f}%   (z {r.zscore:+.1f})"
          for r in km.itertuples()]
idx = st.selectbox("Event", range(len(labels)), format_func=lambda i: labels[i])
move = km.iloc[idx]

win_start = move["move_date"] - pd.Timedelta(days=30)
win_end   = move["move_date"] + pd.Timedelta(days=15)

with st.spinner("Loading sentiment..."):
    daily_signals = load_signals(token, tickers=[ticker_km],
                                 start_year=win_start.year, end_year=win_end.year)

price_w = load_price(ticker_km, token)
if not price_w.empty:
    price_w = price_w[(price_w["Date"] >= win_start) & (price_w["Date"] <= win_end)]
if price_w.empty:
    st.warning(f"No price data for {ticker_km}.")
    st.stop()

sig_w, sig_col = pd.DataFrame(), None
if not daily_signals.empty:
    sig_col = get_sig_col(daily_signals)
    sig_w = daily_signals[
        (daily_signals["ticker"] == ticker_km)
        & (daily_signals["date"] >= win_start)
        & (daily_signals["date"] <= win_end)
    ].copy().sort_values("date")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                    row_heights=[0.56, 0.44],
                    subplot_titles=("", "Sentiment across the same window"))

ref = float(price_w["Close"].iloc[0])
pct = (price_w["Close"] - ref) / ref * 100
fig.add_trace(go.Scatter(
    x=price_w["Date"], y=pct, mode="lines",
    line=dict(color=PRICE, width=2.5), name="Price",
    hovertemplate="%{x|%d %b %Y}<br>%{y:+.1f}% from window start<extra></extra>"),
    row=1, col=1)
fig.add_trace(go.Scatter(
    x=price_w["Date"], y=np.where(pct >= 0, pct, 0), fill="tozeroy", mode="none",
    fillcolor="rgba(0,114,178,0.16)", showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=price_w["Date"], y=np.where(pct < 0, pct, 0), fill="tozeroy", mode="none",
    fillcolor="rgba(213,94,0,0.16)", showlegend=False, hoverinfo="skip"), row=1, col=1)

if not sig_w.empty and sig_col:
    sv = sig_w[sig_col].fillna(0)
    colors = np.where(sv >= SENTIMENT_THRESHOLD, POS,
              np.where(sv <= -SENTIMENT_THRESHOLD, NEG, NEU))
    fig.add_trace(go.Bar(
        x=sig_w["date"], y=sv, marker_color=colors, marker_line_width=0,
        showlegend=False,
        hovertemplate="%{x|%d %b %Y}<br>sentiment %{y:.3f}<extra></extra>"),
        row=2, col=1)

for row in (1, 2):
    fig.add_vline(x=move["move_date"], line_color=GOLD, line_width=3, row=row, col=1)
    fig.add_vline(x=move["sent_date"], line_color=HIGHLIGHT, line_width=2,
                  line_dash="dash", row=row, col=1)

fig.add_annotation(x=move["move_date"], y=1.0, yref="paper",
                   text="price move", showarrow=False, xanchor="left",
                   xshift=5, font=dict(size=12, color=GOLD))
fig.add_annotation(x=move["sent_date"], y=1.0, yref="paper",
                   text="signal ", showarrow=False, xanchor="right",
                   xshift=-5, font=dict(size=12, color=HIGHLIGHT))

fig.add_hline(y=0, line_color=MUTED, line_width=1, line_dash="dot", row=2, col=1)
fig.update_yaxes(title="% change from window start", tickformat="+.0f", row=1, col=1)
fig.update_yaxes(title="Sentiment score", row=2, col=1)
fig.update_xaxes(tickformat="%d %b", row=1, col=1)
fig.update_xaxes(tickformat="%d %b", title="", row=2, col=1)
fig.update_layout(showlegend=False, bargap=0.15)

days_before = int(move["days_before"])
titled(fig,
       f"Sentiment moved {days_before} day{'s' if days_before != 1 else ''} "
       f"before {ticker_km} jumped {move['return_pct']:+.1f}%",
       f"{move['move_date']:%d %B %Y} · signal strength {move['sentiment']:+.3f} · "
       f"sources: {move.get('sources_active', 'unknown')}",
       height=680)
show(fig)

st.markdown(
    f"<span style='color:{POS}'>▌</span> positive sentiment &nbsp;&nbsp;"
    f"<span style='color:{NEG}'>▌</span> negative sentiment &nbsp;&nbsp;"
    f"<span style='color:{NEU}'>▌</span> neutral &nbsp;&nbsp;&nbsp;"
    f"<span style='color:{GOLD}'>▬</span> the price move &nbsp;&nbsp;"
    f"<span style='color:{HIGHLIGHT}'>▬</span> when sentiment fired",
    unsafe_allow_html=True)
