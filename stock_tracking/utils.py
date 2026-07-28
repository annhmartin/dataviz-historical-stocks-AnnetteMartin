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
    "All Tickers"          : [],
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

DEFAULT_TICKERS = ["NVDA","AAPL","MSFT","GOOGL","META","CRWD","PANW","PLTR","NVO","PM"]

STRAT_COLORS = {
    "SP_500_SPY_"     : "#f39c12",
    "Buy__Hold"       : "#1a1a2e",
    "Sector_Rotation" : "#e74c3c",
    "Position_Trader" : "#8e44ad",
}

STRAT_LABELS = {
    "SP_500_SPY_"     : "S&P 500 (SPY)",
    "Buy__Hold"       : "Buy & Hold",
    "Sector_Rotation" : "Sector Rotation",
    "Position_Trader" : "Position Trader",
}

def apply_chart_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor"  : "white",
        "axes.edgecolor"  : "#cccccc",
        "axes.grid"       : True,
        "grid.color"      : "#e8e8e8",
        "grid.linewidth"  : 0.7,
        "axes.spines.top" : False,
        "axes.spines.right": False,
        "font.size"       : 14,
        "axes.titlesize"  : 16,
        "axes.labelsize"  : 14,
        "xtick.labelsize" : 12,
        "ytick.labelsize" : 12,
        "legend.fontsize" : 12,
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_signals(token=None):
    frames = []
    for year in range(2015, datetime.now().year + 1):
        for q in [1, 2, 3, 4]:
            path = OUTPUT_PREFIX + "/daily_signals_" + str(year) + "_Q" + str(q) + ".csv"
            df = load_csv(path, token)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_price(ticker, token=None):
    df = load_csv(STOCKS_PREFIX + "/prices_" + ticker + ".csv", token)
    if df.empty:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["daily_return"] = df["Close"].pct_change(fill_method=None)
    df["pct_7d"]  = df["Close"].pct_change(7, fill_method=None) * 100
    df["cumret"]  = (1 + df["daily_return"].fillna(0)).cumprod() * 100 - 100
    return df.sort_values("Date").reset_index(drop=True)

def get_sig_col(daily_signals):
    if "adaptive_sentiment" in daily_signals.columns:
        return "adaptive_sentiment"
    return "norm_sentiment"

def get_token():
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return None

def sidebar_filters(all_tickers):
    token = get_token()
    st.sidebar.title("Tech Pulse")
    st.sidebar.markdown("---")
    sector = st.sidebar.radio("Sector", options=list(SECTOR_MAP.keys()), index=0)
    if sector == "All Tickers":
        selected = [t for t in DEFAULT_TICKERS if t in all_tickers]
    else:
        selected = [t for t in SECTOR_MAP[sector] if t in all_tickers]
    if not selected:
        selected = [t for t in DEFAULT_TICKERS if t in all_tickers]
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date()],
        min_value=pd.Timestamp("2015-01-01").date(),
        max_value=pd.Timestamp.today().date(),
    )
    start = pd.Timestamp(date_range[0])
    end   = pd.Timestamp(date_range[1]) if len(date_range) > 1 else pd.Timestamp.today()
    st.sidebar.markdown("---")
    st.sidebar.caption("Selected: " + ", ".join(selected))
    return selected, start, end, token
