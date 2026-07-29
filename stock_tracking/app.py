
import streamlit as st

st.set_page_config(
    page_title="Tech Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

overview = st.Page("stock_tracking/pages/01_Overview.py",       title="Overview")
signal   = st.Page("stock_tracking/pages/02_Signal_Quality.py", title="Signal Quality")
corr     = st.Page("stock_tracking/pages/03_Correlation.py",    title="Correlation")
strats   = st.Page("stock_tracking/pages/04_Strategies.py",     title="Strategies")
moments  = st.Page("stock_tracking/pages/05_Key_Moments.py",    title="Key Moments")
sources  = st.Page("stock_tracking/pages/06_Sources.py",        title="Sources")
whatif   = st.Page("stock_tracking/pages/07_What_If.py",        title="What If")

pg = st.navigation(
    [overview, signal, corr, strats, moments, sources, whatif],
    position="sidebar"
)
pg.run()
