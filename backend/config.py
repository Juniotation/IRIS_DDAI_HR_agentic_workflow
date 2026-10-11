from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

# Local default keeps the prototype self-contained.  DATA_DIR / DATABASE_PATH can
# be overridden for hosted environments without changing application code.
load_dotenv(BASE_DIR / ".env")


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read environment variables first, then Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        # This module is also imported by tests and non-Streamlit scripts.
        pass
    return default


_data_dir_setting = get_setting("DATA_DIR")
DATA_DIR = Path(_data_dir_setting).expanduser() if _data_dir_setting else (BASE_DIR / "data")
DB_PATH = Path(get_setting("DATABASE_PATH") or (DATA_DIR / "recruiting_workbench.db")).expanduser()
DEFAULT_CANDIDATE_CSV = BASE_DIR / "data" / "thara_energy_candidates_db.csv"

GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_SCREEN_MODEL = get_setting("GEMINI_SCREEN_MODEL", "gemini-3.5-flash-lite")
GEMINI_THINKING_LEVEL = get_setting("GEMINI_THINKING_LEVEL", "minimal")
GEMINI_SCREEN_THINKING_LEVEL = get_setting("GEMINI_SCREEN_THINKING_LEVEL", "minimal")
GEMINI_MAX_OUTPUT_TOKENS = int(get_setting("GEMINI_MAX_OUTPUT_TOKENS", "1800") or 1800)
SCREENING_WORKERS = max(1, min(3, int(get_setting("SCREENING_WORKERS", "2") or 2)))
SCREENING_DELAY_SECONDS = max(0.0, float(get_setting("SCREENING_DELAY_SECONDS", "0.15") or 0.15))
