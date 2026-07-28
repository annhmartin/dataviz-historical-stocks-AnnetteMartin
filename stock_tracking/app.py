
import streamlit as st

st.set_page_config(
    page_title="Tech Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Tech Pulse")
st.markdown("""
Welcome. Select a page from the left sidebar to get started.

| Page | Description |
|------|-------------|
| Overview | Sentiment overlaid with price change |
| Signal Quality | Four-quadrant prediction accuracy |
| Correlation | Sentiment-to-price correlation rankings |
| Strategies | Portfolio comparison vs S&P 500 |
| Key Moments | Zoom into big price moves |
| Sources | Which data source drives each ticker |
| What If | Projected returns on a custom investment amount |
""")
