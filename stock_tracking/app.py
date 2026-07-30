import streamlit as st
import os, traceback

st.set_page_config(
    page_title="Tech Pulse",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = os.path.dirname(os.path.abspath(__file__))

PAGES = [
    ("pages/00_Key_Findings.py",   "Key Findings"),
    ("pages/01_Overview.py",       "Overview"),
    ("pages/02_Signal_Quality.py", "Signal Quality"),
    ("pages/03_Correlation.py",    "Correlation"),
    ("pages/04_Strategies.py",     "Strategies"),
    ("pages/05_Key_Moments.py",    "Key Moments"),
    ("pages/06_Sources.py",        "Sources"),
    ("pages/07_What_If.py",        "What If"),
]

missing = [rel for rel, _ in PAGES if not os.path.isfile(os.path.join(HERE, rel))]
if missing:
    st.error("Some page files are missing from the app folder:")
    for m in missing:
        st.code(m)
    st.stop()

try:
    pg = st.navigation([st.Page(rel, title=title) for rel, title in PAGES],
                       position="sidebar")
    pg.run()
except Exception:
    st.error("Something went wrong loading this page.")
    with st.expander("Technical details"):
        st.code(traceback.format_exc(), language="text")
