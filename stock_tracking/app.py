import streamlit as st
import requests

st.set_page_config(
    page_title="Tech Pulse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Tech Pulse - Sentiment Analysis Dashboard")
st.markdown("""
Welcome to Tech Pulse. Use the sidebar to filter by sector and date range.

| Page | What it shows |
|------|--------------|
| **Overview** | Sentiment overlaid with price change |
| **Signal Quality** | Four-quadrant prediction accuracy |
| **Correlation** | Sentiment-to-price correlation rankings |
| **Strategies** | Portfolio comparison vs S&P 500 |
| **Key Moments** | Zoom into big price moves |
| **Sources** | Which data source drives each ticker |
| **What If** | Enter a dollar amount and see projected returns |
""")

GITHUB_REPO = "annhmartin/dataviz-historical-stocks-AnnetteMartin"
FOLDER = "stock_tracking"
token = None
try:
    token = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

with st.expander("Data Health Check"):
    checks = {
        "Sentiment signals" : FOLDER + "/sentiment_outputs/daily_signals_2024_Q1.csv",
        "Correlation matrix": FOLDER + "/correlation_outputs/best_per_ticker.csv",
        "Strategy outputs"  : FOLDER + "/strategy_outputs/equity_curves.csv",
        "Stock prices (SPY)": FOLDER + "/stocks/prices_SPY.csv",
    }
    for label, path in checks.items():
        url  = "https://raw.githubusercontent.com/" + GITHUB_REPO + "/main/" + path
        hdrs = {"Authorization": "Bearer " + token} if token else {}
        resp = requests.get(url, headers=hdrs, timeout=10)
        if resp.status_code == 200:
            st.success("Found: " + label)
        else:
            st.error("Not found: " + label + " at " + path)
