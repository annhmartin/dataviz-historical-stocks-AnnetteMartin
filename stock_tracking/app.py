import streamlit as st
import os, sys, traceback

st.set_page_config(
    page_title="Tech Pulse",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = os.path.dirname(os.path.abspath(__file__))

PAGES = [
    ("pages/01_Overview.py",       "Overview"),
    ("pages/02_Signal_Quality.py", "Signal Quality"),
    ("pages/03_Correlation.py",    "Correlation"),
    ("pages/04_Strategies.py",     "Strategies"),
    ("pages/05_Key_Moments.py",    "Key Moments"),
    ("pages/06_Sources.py",        "Sources"),
    ("pages/07_What_If.py",        "What If"),
]

# Verify every page file exists before handing them to st.navigation
missing = [rel for rel, _ in PAGES if not os.path.isfile(os.path.join(HERE, rel))]

if missing:
    st.error("These page files were not found next to app.py:")
    for m in missing:
        st.code(os.path.join(HERE, m))
    st.write("Contents of", HERE)
    st.code("\n".join(sorted(os.listdir(HERE))))
    pages_dir = os.path.join(HERE, "pages")
    if os.path.isdir(pages_dir):
        st.write("Contents of pages/")
        st.code("\n".join(sorted(os.listdir(pages_dir))))
    st.stop()

try:
    nav_pages = [st.Page(rel, title=title) for rel, title in PAGES]
    pg = st.navigation(nav_pages, position="sidebar")
    pg.run()
except Exception as e:
    st.error("The app raised an exception. Full traceback below.")
    st.code(traceback.format_exc(), language="text")
    st.write("Python:", sys.version)
    st.write("Working dir:", os.getcwd())
    st.write("app.py dir:", HERE)
