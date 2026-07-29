import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from datetime import datetime
import matplotlib as mpl

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

def apply_chart_style():
    """Matplotlib defaults matching the app theme."""
    mpl.rcParams.update({
        "figure.facecolor" : CANVAS,
        "axes.facecolor"   : CANVAS,
        "savefig.facecolor": CANVAS,
        "axes.edgecolor"   : AXIS,
        "axes.labelcolor"  : INK,
        "text.color"       : INK,
        "xtick.color"      : MUTED,
        "ytick.color"      : MUTED,
        "axes.grid"        : True,
        "axes.axisbelow"   : True,
        "grid.color"       : GRID,
        "grid.linewidth"   : 0.8,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "font.size"        : 14,
        "axes.titlesize"   : 16,
        "axes.titleweight" : "bold",
        "axes.titlecolor"  : INK,
        "axes.labelsize"   : 14,
        "xtick.labelsize"  : 12,
        "ytick.labelsize"  : 12,
        "legend.fontsize"  : 12,
        "legend.framealpha": 0.95,
        "legend.facecolor" : CANVAS,
        "legend.edgecolor" : GRID,
    })


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

    Only the columns in SIGNAL_COLS are read, and if `tickers` is given only
    those rows are kept. This keeps memory well under the Streamlit Cloud
    limit instead of materialising all ~9M rows at once.
    """
    if end_year is None:
        end_year = datetime.now().year
    tickers = tuple(tickers) if tickers else None

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

def sidebar_filters():
    token = get_token()

    if "sector" not in st.session_state:
        st.session_state["sector"] = "Big Tech"
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = pd.Timestamp("2018-01-01").date()
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = pd.Timestamp.today().date()

    sector = st.sidebar.radio(
        "Sector",
        options=list(SECTOR_MAP.keys()),
        index=list(SECTOR_MAP.keys()).index(st.session_state["sector"]),
        key="sector_radio",
    )
    st.session_state["sector"] = sector
    selected = list(SECTOR_MAP[sector])

    date_range = st.sidebar.date_input(
        "Date Range",
        value=[st.session_state["start_date"], st.session_state["end_date"]],
        min_value=pd.Timestamp("2015-01-01").date(),
        max_value=pd.Timestamp.today().date(),
        key="date_range_input",
    )
    if len(date_range) == 2:
        st.session_state["start_date"] = date_range[0]
        st.session_state["end_date"]   = date_range[1]

    start = pd.Timestamp(st.session_state["start_date"])
    end   = pd.Timestamp(st.session_state["end_date"])
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

# Data sources, grouped by family so related feeds read as related
SOURCE_COLORS = {
    "gdelt"                   : "#0072B2",
    "hn"                      : "#E69F00",
    "stocktwits"              : "#009E73",
    "edgar_8k"                : "#7B2D8E",
    "reddit_wallstreetbets"   : "#CC79A7",
    "reddit_stocks"           : "#B05C86",
    "reddit_investing"        : "#8E4A6B",
    "reddit_technology"       : "#6D3A53",
    "reddit_SecurityAnalysis" : "#4F2A3C",
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
