
import streamlit as st

st.set_page_config(
    page_title="Tech Pulse — Sentiment Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Tech Pulse — Sentiment Analysis Dashboard")
st.markdown("""
Welcome to Tech Pulse. Use the sidebar to filter tickers and date ranges,
then navigate between pages using the menu on the left.

| Page | What it shows |
|------|--------------|
| **01 Overview** | Sentiment overlaid with % price change |
| **02 Signal Quality** | Four-quadrant True/False Positive/Negative |
| **03 Correlation** | Sentiment-to-price correlation rankings |
| **04 Spillover** | Cross-company sentiment contagion |
| **05 Strategies** | Portfolio comparison vs S&P 500 |
| **06 Key Moments** | Zoom into big price moves with preceding sentiment |
| **07 Sources** | Which data source drives each ticker |

---
*Data sources: Hacker News · Reddit · GDELT · Alpha Vantage · SEC EDGAR*
""")

# Quick data health check
import requests, io
import pandas as pd

token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

GITHUB_REPO = "annhmartin/dataviz-historical-stocks-AnnetteMartin"
FOLDER = "stock tracking"

with st.expander("Data health check"):
    checks = {
        "Sentiment signals": f"{FOLDER}/sentiment_outputs/daily_signals_2024_Q1.csv",
        "Correlation matrix": f"{FOLDER}/correlation_outputs/best_per_ticker.csv",
        "Strategy outputs": f"{FOLDER}/strategy_outputs/equity_curves.csv",
        "Stock prices (SPY)": f"{FOLDER}/stocks/prices_SPY.csv",
    }
    for label, path in checks.items():
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}"
        hdrs = {"Authorization": f"Bearer {token}"} if token else {}
        resp = requests.get(url, headers=hdrs, timeout=10)
        if resp.status_code == 200:
            st.success(f"✓ {label}")
        else:
            st.error(f"✗ {label} — not found at {path}")
