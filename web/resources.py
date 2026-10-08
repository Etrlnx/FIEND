"""Process-wide resources shared by the Streamlit script (app.py) and its server launcher (serve.py)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from finrag.pipeline import load_production_pipeline


@st.cache_resource(show_spinner="Loading FIEND pipeline...")
def get_pipeline():
    return load_production_pipeline()
