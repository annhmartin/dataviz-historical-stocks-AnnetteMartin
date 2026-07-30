import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (load_signals, load_price, sidebar_filters, get_sig_col,
                   SENTIMENT_THRESHOLD, apply_chart_style, titled, show,
                   QUAD_TP, QUAD_TN, QUAD_FP, QUAD_FN, INK, MUTED, GOLD, CONTEXT)

st.header("Signal Quality")

selected, start, end, token = sidebar_filters()
apply_chart_style()

with st.spinner("Loading sentiment signals..."):
    daily_signals = load_signals(token, tickers=selected,
                                 start_year=start.year, end_year=end.year)

if daily_signals.empty:
    st.error("No signal data for this sector and date range.")
    st.stop()

sig_col = get_sig_col(daily_signals)

st.markdown(
    "Every dot is one day where sentiment crossed the neutral threshold, placed by how "
    "strong the signal was and what the price did next. Dot size reflects how many "
    "stories drove that reading."
)

valid = [t for t in selected if t in set(daily_signals["ticker"])]
if not valid:
    st.warning("No signal data for tickers in this sector.")
    st.stop()

c1, c2 = st.columns(2)
ticker    = c1.selectbox("Ticker", valid)
hold_days = c2.slider("Hold period (trading days)", 1, 21, 5,
                      help="How long a position is assumed held before judging "
                           "whether the signal was right.")

sig = daily_signals[
    (daily_signals["ticker"] == ticker)
    & (daily_signals["date"] >= start)
    & (daily_signals["date"] <= end)
].copy()

price = load_price(ticker, token)
if price.empty or sig.empty:
    st.warning("No data for selected ticker.")
    st.stop()

price = price.set_index("Date").sort_index()
active = sig[sig[sig_col].notna() & (sig[sig_col].abs() >= SENTIMENT_THRESHOLD)]
if active.empty:
    st.info("No days crossed the sentiment threshold here.")
    st.stop()

pdates, closes = price.index.values, price["Close"].values
entry = np.searchsorted(pdates, active["date"].values, side="right")
exit_ = entry + hold_days - 1
ok    = (exit_ < len(closes)) & (entry < len(closes))
active, entry, exit_ = active[ok], entry[ok], exit_[ok]

if len(active) == 0:
    st.info("Not enough forward price history for this hold period.")
    st.stop()

rets  = (closes[exit_] - closes[entry]) / closes[entry] * 100
sents = active[sig_col].values
pos_buzz, went_up = sents >= SENTIMENT_THRESHOLD, rets > 0

quad = np.where(pos_buzz & went_up,  "True positive",
       np.where(pos_buzz & ~went_up, "False positive",
       np.where(~pos_buzz & ~went_up,"True negative", "False negative")))

QCOL = {"True positive": QUAD_TP, "True negative": QUAD_TN,
        "False positive": QUAD_FP, "False negative": QUAD_FN}

df = pd.DataFrame({
    "sent": sents, "ret": rets, "quad": quad,
    "stories": active["story_count"].values if "story_count" in active.columns else 1,
    "date": pd.to_datetime(active["date"].values),
})

counts = df.quad.value_counts()
tp = int(counts.get("True positive", 0));  fp = int(counts.get("False positive", 0))
tn = int(counts.get("True negative", 0));  fn = int(counts.get("False negative", 0))
total = len(df)
acc = (tp + tn) / total

m = st.columns(5)
m[0].metric("Accuracy", f"{acc*100:.1f}%")
m[1].metric("True positive",  f"{tp}", f"{tp/total*100:.0f}%")
m[2].metric("False positive", f"{fp}", f"{fp/total*100:.0f}%")
m[3].metric("True negative",  f"{tn}", f"{tn/total*100:.0f}%")
m[4].metric("False negative", f"{fn}", f"{fn/total*100:.0f}%")

fig = go.Figure()
for q in ["True positive", "True negative", "False positive", "False negative"]:
    d = df[df.quad == q]
    if d.empty:
        continue
    fig.add_trace(go.Scatter(
        x=d.sent, y=d.ret, mode="markers", name=q,
        marker=dict(color=QCOL[q], size=np.clip(d.stories, 1, 40) * 0.7 + 7,
                    line=dict(color="white", width=1), opacity=0.75),
        customdata=np.stack([d.stories, d.date.dt.strftime("%d %b %Y")], axis=-1),
        hovertemplate=("<b>" + q + "</b><br>%{customdata[1]}"
                       "<br>sentiment %{x:.3f}<br>return %{y:+.1f}%"
                       "<br>%{customdata[0]} stories<extra></extra>")))

xmax = float(np.abs(df.sent).max()) * 1.12
ymax = float(np.abs(df.ret).max()) * 1.12
fig.add_hline(y=0, line_color=MUTED, line_width=1.5)
fig.add_vline(x=0, line_color=MUTED, line_width=1.5)

if len(df) > 5:
    z = np.polyfit(df.sent, df.ret, 1)
    xs = np.linspace(-xmax, xmax, 60)
    fig.add_trace(go.Scatter(x=xs, y=np.poly1d(z)(xs), mode="lines",
                             line=dict(color=GOLD, width=2.5),
                             name="Overall trend", hoverinfo="skip"))

for txt, xa, ya, col in [
    (f"{tp} ({tp/total*100:.0f}%)",  xmax*0.62,  ymax*0.88, QUAD_TP),
    (f"{fn} ({fn/total*100:.0f}%)", -xmax*0.62,  ymax*0.88, QUAD_FN),
    (f"{fp} ({fp/total*100:.0f}%)",  xmax*0.62, -ymax*0.88, QUAD_FP),
    (f"{tn} ({tn/total*100:.0f}%)", -xmax*0.62, -ymax*0.88, QUAD_TN)]:
    fig.add_annotation(x=xa, y=ya, text=f"<b>{txt}</b>", showarrow=False,
                       font=dict(size=15, color=col))

fig.update_xaxes(title="Negative buzz  ←     |     →  Positive buzz",
                 range=[-xmax, xmax])
fig.update_yaxes(title=f"Price change {hold_days} days later (%)",
                 range=[-ymax, ymax], tickformat="+.0f")
fig.update_layout(legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))

verdict = "beats" if acc > 0.5 else "trails"
titled(fig,
       f"Does {ticker} sentiment beat a coin flip? It {verdict} it, at {acc*100:.1f}%",
       f"Cool colours mark correct calls, warm colours incorrect. n = {total:,} signals",
       height=620)
show(fig)

st.markdown("---")
st.markdown(
    f"<span style='color:{QUAD_TP}'><b>True positive</b></span> — positive buzz, price rose &nbsp;·&nbsp; "
    f"<span style='color:{QUAD_FP}'><b>False positive</b></span> — positive buzz, price fell<br>"
    f"<span style='color:{QUAD_TN}'><b>True negative</b></span> — negative buzz, price fell &nbsp;·&nbsp; "
    f"<span style='color:{QUAD_FN}'><b>False negative</b></span> — negative buzz, price rose",
    unsafe_allow_html=True)
