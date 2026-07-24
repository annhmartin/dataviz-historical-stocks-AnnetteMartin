
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
| **01 Overview** | Sentiment vs price overlaid chart |
| **02 Signal Quality** | Four-quadrant True/False Positive/Negative |
| **03 Correlation** | Sentiment-to-price correlation rankings |
| **04 Spillover** | Cross-company sentiment contagion |
| **05 Strategies** | Portfolio comparison vs S&P 500 |
| **06 Key Moments** | Zoom into big price moves with preceding sentiment |
| **07 Sources** | Which data source drives each ticker |

---
*Data sources: Hacker News · Reddit · GDELT · Alpha Vantage · SEC EDGAR*
""")
