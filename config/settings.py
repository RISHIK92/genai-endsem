"""
Centralized configuration for EduAgent-OS.
Loads environment variables and provides constants.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (local dev)
load_dotenv()

logger = logging.getLogger("eduagent.settings")

def _get(key: str, default: str = "") -> str:
    """Read from Streamlit secrets (cloud) first, then os.environ (local)."""
    try:
        import streamlit as st
        value = st.secrets.get(key, os.getenv(key, default))
    except Exception:
        value = os.getenv(key, default)

    if value and value != default:
        logger.debug("Config key '%s' loaded from environment", key)
    else:
        logger.warning("Config key '%s' not found — using default: '%s'", key, default)
    return value

# ─── Base Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ML_MODELS_DIR = PROJECT_ROOT / "ml" / "models"
RAG_DATA_DIR = PROJECT_ROOT / "rag" / "data"
PDF_DIR = RAG_DATA_DIR / "pdfs"
FAISS_INDEX_DIR = RAG_DATA_DIR / "faiss_index"

# ─── API Keys ────────────────────────────────────────────────
GROQ_API_KEY = _get("GROQ_API_KEY")

# ─── Model Configuration ────────────────────────────────────
LLM_MODEL = _get("LLM_MODEL", "llama-3.1-70b-versatile")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ─── XGBoost Model Paths ────────────────────────────────────
# Model A: Text-only features (25 features)
XGB_CLF_MODEL_A = ML_MODELS_DIR / "xgb_clf_model_A.json"
XGB_REG_MODEL_A = ML_MODELS_DIR / "xgb_reg_model_A.json"

# Model B: All features (30 features)
XGB_CLF_MODEL_B = ML_MODELS_DIR / "xgb_all_clf_model_B.json"
XGB_REG_MODEL_B = ML_MODELS_DIR / "xgb_all_reg_model_B.json"

# Pickle models (sklearn pipelines)
XGB_ALL_MODEL_PKL = ML_MODELS_DIR / "xgb_all_model.pkl"
XGB_TEXT_MODEL_PKL = ML_MODELS_DIR / "xgb_text_model.pkl"

# ─── Feature Extraction Config ──────────────────────────────
# Model A expects 25 text-based features
MODEL_A_NUM_FEATURES = 25
# Model B expects 30 features (text + statistical)
MODEL_B_NUM_FEATURES = 30

# Difficulty classes (3-class classification)
DIFFICULTY_CLASSES = ["Easy", "Medium", "Hard"]

# ─── RAG Configuration ──────────────────────────────────────
CHUNK_SIZE = 1000           # characters per chunk
CHUNK_OVERLAP = 200         # overlap between chunks
RAG_TOP_K = 5               # number of chunks to retrieve
EMBEDDING_DIMENSION = 384   # all-MiniLM-L6-v2 output dimension

# ─── Agent Configuration ────────────────────────────────────
MAX_REFINEMENT_ITERATIONS = 3
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048

# ─── Feature Audit Thresholds ───────────────────────────────
MAX_STEM_LENGTH = 200       # characters
MAX_LATEX_DENSITY = 0.30    # ratio
MAX_WORD_COUNT = 150        # words
MIN_OPTION_COUNT = 4
MAX_OPTION_COUNT = 5
MIN_READABILITY_SCORE = 30  # Flesch-Kincaid
