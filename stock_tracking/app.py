
import streamlit as st

st.set_page_config(
    page_title="Tech Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

overview = st.Page("pages/01_Overview.py",       title="Overview",       icon="📊")
signal   = st.Page("pages/02_Signal_Quality.py", title="Signal Quality", icon="🎯")
corr     = st.Page("pages/03_Correlation.py",    title="Correlation",    icon="🔗")
strats   = st.Page("pages/04_Strategies.py",     title="Strategies",     icon="💰")
moments  = st.Page("pages/05_Key_Moments.py",    title="Key Moments",    icon="🔍")
sources  = st.Page("pages/06_Sources.py",        title="Sources",        icon="📰")
whatif   = st.Page("pages/07_What_If.py",        title="What If",        icon="💡")

pg = st.navigation(
    [overview, signal, corr, strats, moments, sources, whatif],
    position="sidebar"
)
pg.run()
