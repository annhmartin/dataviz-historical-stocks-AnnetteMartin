
import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
from datetime import datetime

# ── GitHub repo config ────────────────────────────────────────────────────────
GITHUB_REPO   = "annhmartin/dataviz-historical-stocks-AnnetteMartin"

# Data folder paths — relative to repo root
# Based on current structure: all data is inside "stock tracking/" folder
FOLDER        = "stock tracking"   # parent folder in repo
OUTPUT_PREFIX = f"{FOLDER}/sentiment_outputs"
CORR_PREFIX   = f"{FOLDER}/correlation_outputs"
STRAT_PREFIX  = f"{FOLDER}/strategy_outputs"
STOCKS_PREFIX = f"{FOLDER}/stocks"

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

DEFAULT_TICKERS = [
    "NVDA","AAPL","MSFT","GOOGL","META",
    "CRWD","PANW","PLTR","NVO","PM"
]

STRAT_COLORS = {
    "SP_500_SPY_"         : "#f39c12",
    "Buy__Hold"           : "#1a1a2e",
    "Sector_Rotation"     : "#e74c3c",
    "Position_Trader"     : "#8e44ad",
    "Opt_Stop_Global_"    : "#27ae60",
    "Opt_Stop_Per_Sector_": "#00b4d8",
}

STRAT_LABELS = {
    "SP_500_SPY_"         : "S&P 500 (SPY)",
    "Buy__Hold"           : "Buy & Hold",
    "Sector_Rotation"     : "Sector Rotation",
    "Position_Trader"     : "Position Trader",
    "Opt_Stop_Global_"    : "Optimal Stop (Global)",
    "Opt_Stop_Per_Sector_": "Optimal Stop (Sector)",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(path, token=None):
    """Load a CSV file from GitHub raw URL."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}"
    hdrs = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(url, headers=hdrs, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        content = resp.text.strip()
        if not content:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(content), low_memory=False)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_signals(token=None):
    """Load all quarterly sentiment signal files."""
    frames = []
    for year in range(2015, datetime.now().year + 1):
        for q in [1, 2, 3, 4]:
            path = f"{OUTPUT_PREFIX}/daily_signals_{year}_Q{q}.csv"
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
    """Load price data for a single ticker."""
    df = load_csv(f"{STOCKS_PREFIX}/prices_{ticker}.csv", token)
    if df.empty:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["daily_return"] = df["Close"].pct_change(fill_method=None)
    df["pct_7d"]  = df["Close"].pct_change(7, fill_method=None) * 100
    df["cumret"]  = (1 + df["daily_return"].fillna(0)).cumprod() * 100 - 100
    return df.sort_values("Date").reset_index(drop=True)

def get_sig_col(daily_signals):
    """Return the best available sentiment column name."""
    return "adaptive_sentiment" if "adaptive_sentiment" in daily_signals.columns else "norm_sentiment"

def get_token():
    """Get GitHub token from Streamlit secrets."""
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return None

def sidebar_filters(all_tickers):
    """
    Render sidebar controls and return (selected_tickers, start_date, end_date, token).
    """
    token = get_token()

    st.sidebar.title("📡 Tech Pulse")
    st.sidebar.markdown("---")

    sector_filter = st.sidebar.selectbox(
        "Filter by sector", ["All"] + list(SECTOR_MAP.keys())
    )

    if sector_filter != "All":
        default = [t for t in SECTOR_MAP[sector_filter] if t in all_tickers]
    else:
        default = [t for t in DEFAULT_TICKERS if t in all_tickers]

    selected = st.sidebar.multiselect(
        "Select tickers",
        options=all_tickers,
        default=default,
        help="Choose companies to display across all charts"
    )

    date_range = st.sidebar.date_input(
        "Date range",
        value=[pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date()],
        min_value=pd.Timestamp("2015-01-01").date(),
        max_value=pd.Timestamp.today().date(),
    )
    start = pd.Timestamp(date_range[0])
    end   = pd.Timestamp(date_range[1]) if len(date_range) > 1 else pd.Timestamp.today()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Tickers selected:** {len(selected)}")
    st.sidebar.caption(f"**Total available:** {len(all_tickers):,}")

    return selected, start, end, token
