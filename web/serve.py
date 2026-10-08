"""Streamlit server entry point: loads the FIEND pipeline at server start, before
the first browser connects, instead of making the first visitor wait (~30s).

    streamlit run web/serve.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import streamlit as st

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Import after st.App has started the runtime so the real st.cache_resource store is used.
    from resources import get_pipeline

    try:
        await anyio.to_thread.run_sync(get_pipeline)
    except Exception:
        # Keep serving: app.py retries the load and shows the error in the UI.
        logger.exception("Pipeline preload failed")
    yield


app = st.App(str(WEB / "app.py"), lifespan=lifespan)
