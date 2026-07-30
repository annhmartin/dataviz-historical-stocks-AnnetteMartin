import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio

GITHUB_REPO   = "annhmartin/dataviz-historical-stocks-AnnetteMartin"
FOLDER        = "stock_tracking"
OUTPUT_PREFIX = FOLDER + "/sentiment_outputs"
CORR_PREFIX   = FOLDER + "/correlation_outputs"
STRAT_PREFIX  = FOLDER + "/strategy_outputs"
STOCKS_PREFIX = FOLDER + "/stocks"
SENTIMENT_THRESHOLD = 0.05

SECTOR_MAP = {
    "AI Accelerators"      : ["NVDA","AMD"],
    "Semiconductor Supply" : ["TSM","INTC","QCOM"],
    "Big Tech"             : ["GOOGL","MSFT","AAPL","META"],
    "Cloud / SaaS"         : ["AMZN","SNOW","DDOG","CRM","NOW","MDB"],
    "Cybersecurity"        : ["CRWD","PANW","OKTA"],
    "Enterprise AI"        : ["PLTR"],
    "Macro Risk"           : ["COIN","TSLA"],
    "Portfolio"            : ["INCY","KGC","NVO","PM","WPM"],
    "Consumer Tech"        : ["NFLX","SPOT","PINS"],
    "Enterprise Fintech"   : ["PYPL"],
}


# Only these columns are ever used by the app. Loading all ~20 columns
# across 47 quarterly files is what exhausts memory on Streamlit Cloud.
SIGNAL_COLS = [
    "ticker", "date", "norm_sentiment", "adaptive_sentiment",
    "story_count", "sources_active",
]


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(path, token=None):
    url = "https://raw.githubusercontent.com/" + GITHUB_REPO + "/main/" + path
    hdrs = {"Authorization": "Bearer " + token} if token else {}
    try:
        resp = requests.get(url, headers=hdrs, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        content = resp.text.strip()
        if not content:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(content), low_memory=False)
    except Exception:
        return pd.DataFrame()

def _load_signal_quarter(year, quarter, tickers, token):
    """Load one quarterly file, keeping only needed columns and tickers."""
    path = OUTPUT_PREFIX + "/daily_signals_" + str(year) + "_Q" + str(quarter) + ".csv"
    url  = "https://raw.githubusercontent.com/" + GITHUB_REPO + "/main/" + path
    hdrs = {"Authorization": "Bearer " + token} if token else {}
    try:
        resp = requests.get(url, headers=hdrs, timeout=60)
        if resp.status_code != 200:
            return None
        text = resp.text.strip()
        if not text:
            return None
        header = pd.read_csv(io.StringIO(text), nrows=0)
        usecols = [c for c in SIGNAL_COLS if c in header.columns]
        df = pd.read_csv(io.StringIO(text), usecols=usecols, low_memory=False)
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        for c in ("norm_sentiment", "adaptive_sentiment"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], downcast="float")
        if "story_count" in df.columns:
            df["story_count"] = pd.to_numeric(df["story_count"], downcast="integer")
        df["ticker"] = df["ticker"].astype("category")
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def load_signals(token=None, tickers=None, start_year=2015, end_year=None):
    """
    Load sentiment signals.

    Prefers dashboard_signals.csv, a single pre-filtered file written by
    section 7 of A_sentiment_engine. Falls back to reading the quarterly files
    one by one if that bundle has not been generated yet.
    """
    if end_year is None:
        end_year = datetime.now().year
    tickers = tuple(tickers) if tickers else None

    bundle = load_csv(OUTPUT_PREFIX + "/dashboard_signals.csv", token)
    if not bundle.empty:
        bundle["date"] = pd.to_datetime(bundle["date"])
        if tickers:
            bundle = bundle[bundle["ticker"].isin(tickers)]
        bundle = bundle[(bundle["date"].dt.year >= start_year)
                        & (bundle["date"].dt.year <= end_year)]
        if not bundle.empty:
            return bundle.reset_index(drop=True)

    frames = []
    for year in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            df = _load_signal_quarter(year, q, tickers, token)
            if df is not None:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str)
    return out

@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def load_price(ticker, token=None):
    df = load_csv(STOCKS_PREFIX + "/prices_" + ticker + ".csv", token)
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in ["Date", "Close"] if c in df.columns]
    df = df[keep].copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], downcast="float")
    df["daily_return"] = df["Close"].pct_change(fill_method=None)
    df["pct_7d"] = df["Close"].pct_change(7, fill_method=None) * 100
    df["cumret"] = (1 + df["daily_return"].fillna(0)).cumprod() * 100 - 100
    return df.sort_values("Date").reset_index(drop=True)

def get_sig_col(ds):
    return "adaptive_sentiment" if "adaptive_sentiment" in ds.columns else "norm_sentiment"

def get_token():
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return None

DATE_PRESETS = {
    "All data": None,
    "Last 1 year": 365,
    "Last 3 years": 365 * 3,
    "Last 5 years": 365 * 5,
    "2018 onward": "2018",
    "Custom range": "custom",
}

def sidebar_filters(show_sector=True, show_date=True):
    """
    Render whichever sidebar controls a page actually uses and return
    (tickers, start, end, token). Pages that ignore a dimension pass the
    matching flag as False so the control does not appear at all.
    """
    token = get_token()

    if "sector" not in st.session_state:
        st.session_state["sector"] = "Big Tech"
    if "date_preset" not in st.session_state:
        st.session_state["date_preset"] = "All data"
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = pd.Timestamp("2015-01-01").date()
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = pd.Timestamp.today().date()

    if show_sector:
        sector = st.sidebar.radio(
            "Sector",
            options=list(SECTOR_MAP.keys()),
            index=list(SECTOR_MAP.keys()).index(st.session_state["sector"]),
            key="sector_radio",
            help="Which group of tickers the charts are built from. "
                 "Carries across pages.",
        )
        st.session_state["sector"] = sector
    selected = list(SECTOR_MAP[st.session_state["sector"]])

    if show_date:
        preset = st.sidebar.selectbox(
            "Date range",
            options=list(DATE_PRESETS.keys()),
            index=list(DATE_PRESETS.keys()).index(st.session_state["date_preset"]),
            key="date_preset_select",
            help="Choose a preset window or set your own dates.",
        )
        st.session_state["date_preset"] = preset
        rule = DATE_PRESETS[preset]
        today = pd.Timestamp.today().normalize()

        if rule is None:                       # All data
            start = pd.Timestamp("2015-01-01")
            end   = today
        elif rule == "2018":
            start = pd.Timestamp("2018-01-01")
            end   = today
        elif rule == "custom":
            picked = st.sidebar.date_input(
                "Custom dates",
                value=[st.session_state["start_date"], st.session_state["end_date"]],
                min_value=pd.Timestamp("2015-01-01").date(),
                max_value=today.date(),
                key="date_range_input",
            )
            if len(picked) == 2:
                st.session_state["start_date"], st.session_state["end_date"] = picked
            start = pd.Timestamp(st.session_state["start_date"])
            end   = pd.Timestamp(st.session_state["end_date"])
        else:                                  # rolling window in days
            end   = today
            start = today - pd.Timedelta(days=int(rule))

        st.session_state["start_date"] = start.date()
        st.session_state["end_date"]   = end.date()
    else:
        start = pd.Timestamp("2015-01-01")
        end   = pd.Timestamp.today().normalize()

    st.sidebar.markdown("---")
    if show_sector:
        st.sidebar.caption("**Tickers in view**")
        st.sidebar.caption(", ".join(selected) if selected else "none")

    latest, days_old = data_freshness(token)
    if latest is not None:
        if days_old is not None and days_old > 7:
            st.sidebar.warning(f"Data last updated {latest} ({days_old} days ago).")
        else:
            st.sidebar.caption(f"Data current to {latest}")

    return selected, start, end, token


@st.cache_data(ttl=86400, show_spinner=False)
def load_company_names():
    """
    Ticker -> official company name, taken from the SEC's public company list.
    Cached for a day. Returns {} if the request fails so callers can fall back.
    """
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "TechPulse research contact@example.com"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        names = {}
        for entry in data.values():
            t = str(entry.get("ticker", "")).upper().strip()
            n = str(entry.get("title", "")).strip()
            if t and n:
                names[t] = n.title()
        return names
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Colour system
#
# Built on the Okabe-Ito palette, which is designed to stay distinguishable
# under deuteranopia and protanopia. Colour carries meaning in three separate
# roles here, and the roles deliberately do not share hues:
#
#   SENTIMENT  diverging   direction is the meaning (blue up / orange down)
#   STRATEGY   categorical colour is only a label; benchmarks are muted grey
#              so the two signal-driven strategies carry the saturated hues
#   QUADRANT   cool = the prediction was right, warm = it was wrong
# ─────────────────────────────────────────────────────────────────────────────

# Canvas
INK          = "#22262E"   # primary text and the rolling-average line
MUTED        = "#6B7280"   # secondary text
GRID         = "#E4E7EC"   # gridlines
AXIS         = "#C7CCD6"   # zero lines and axis spines
CANVAS       = "#FBFBFD"   # figure and axes background

# Sentiment, diverging
POS          = "#0072B2"   # positive sentiment
NEG          = "#D55E00"   # negative sentiment
NEU          = "#AEB4BF"   # inside the neutral threshold
POS_FILL     = "#CFE3F3"   # light wash under a positive area
NEG_FILL     = "#F7DDC4"   # light wash under a negative area

# Price and event markers
PRICE        = "#4A4E57"   # price line, kept neutral so it never reads as sentiment
MARKER_EVENT = "#B8860B"   # the big price move
MARKER_SIGNAL= "#7B2D8E"   # when sentiment fired
TREND        = "#7B2D8E"   # regression line

# Strategies, categorical
STRAT_COLORS = {
    "S&P_500_SPY"     : "#9AA4B2",   # benchmark, muted on purpose
    "Buy_&_Hold"      : "#4A4E57",   # benchmark, muted on purpose
    "Sector_Rotation" : "#CC79A7",   # signal-driven, saturated
    "Position_Trader" : "#009E73",   # signal-driven, saturated
}
STRAT_LABELS = {
    "S&P_500_SPY"     : "S&P 500 (SPY)",
    "Buy_&_Hold"      : "Buy & Hold",
    "Sector_Rotation" : "Sector Rotation",
    "Position_Trader" : "Position Trader",
}
STRAT_NEUTRAL = "#AEB4BF"   # months where every strategy lost

# Signal quality quadrants: cool = correct, warm = incorrect
QUAD_TP = "#0072B2"   # positive buzz, price rose
QUAD_TN = "#009E73"   # negative buzz, price fell
QUAD_FP = "#D55E00"   # positive buzz, price fell
QUAD_FN = "#CC79A7"   # negative buzz, price rose

# Outcome states
WIN  = POS
LOSS = NEG

# Data sources.
# Nine sources, nine genuinely distinct hues spanning the wheel. Drawn from the
# Okabe-Ito and Paul Tol qualitative schemes, both built and tested to stay
# separable under red-green colour blindness.
SOURCE_COLORS = {
    "gdelt"                   : "#0072B2",   # blue
    "hn"                      : "#E69F00",   # amber
    "stocktwits"              : "#009E73",   # emerald
    "edgar_8k"                : "#332288",   # indigo
    "reddit_wallstreetbets"   : "#D55E00",   # vermillion
    "reddit_stocks"           : "#88CCEE",   # cyan
    "reddit_investing"        : "#999933",   # olive
    "reddit_technology"       : "#AA4499",   # magenta
    "reddit_SecurityAnalysis" : "#661100",   # brick
}

def sentiment_color(value, threshold=SENTIMENT_THRESHOLD):
    """Colour for a single sentiment score."""
    if value >= threshold:
        return POS
    if value <= -threshold:
        return NEG
    return NEU

def sentiment_colors(values, threshold=SENTIMENT_THRESHOLD):
    """Colour list for a sequence of sentiment scores."""
    return [sentiment_color(v, threshold) for v in values]


@st.cache_data(ttl=300, show_spinner=False)
def data_freshness(token=None):
    """
    How current the underlying data is. Returns (label, days_old) so the sidebar
    can warn when the dashboard is showing stale numbers.
    """
    try:
        df = load_csv(STOCKS_PREFIX + "/prices_SPY.csv", token)
        if df.empty or "Date" not in df.columns:
            return None, None
        latest = pd.to_datetime(df["Date"], errors="coerce").max()
        if pd.isna(latest):
            return None, None
        days = (pd.Timestamp.today().normalize() - latest.normalize()).days
        return latest.strftime("%d %b %Y"), int(days)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Plotly presentation layer
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT  = "#BFC5CF"   # muted grey for anything not being emphasised
ACCENT   = "#009E73"
HIGHLIGHT= "#CC79A7"
GOLD     = "#E69F00"

_TEMPLATE_NAME = "techpulse"

def apply_chart_style():
    """Register and activate the shared Plotly template."""
    pio.templates[_TEMPLATE_NAME] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                      size=14, color=INK),
            title=dict(font=dict(size=19, color=INK), x=0.01, xanchor="left",
                       y=0.96, yanchor="top"),
            paper_bgcolor=CANVAS, plot_bgcolor=CANVAS,
            xaxis=dict(showgrid=False, zeroline=False, showline=True,
                       linecolor=GRID, linewidth=1, ticks="outside",
                       tickcolor=GRID, ticklen=5,
                       tickfont=dict(color=MUTED, size=12),
                       title=dict(font=dict(size=13, color=MUTED))),
            yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                       showline=False, tickfont=dict(color=MUTED, size=12),
                       title=dict(font=dict(size=13, color=MUTED))),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                        font=dict(size=12, color=INK)),
            margin=dict(l=70, r=40, t=95, b=60),
            hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=GRID),
            colorway=[POS, NEG, ACCENT, HIGHLIGHT, GOLD, MUTED],
        )
    )
    pio.templates.default = _TEMPLATE_NAME

def titled(fig, takeaway, subtitle=None, height=520):
    """Title states the finding; subtitle carries the mechanics."""
    text = f"<b>{takeaway}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(title=dict(text=text), height=height)
    return fig

def show(fig):
    """Render a figure full width with the modebar kept minimal."""
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False})
