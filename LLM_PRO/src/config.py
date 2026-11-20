import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Provider choice: groq (default), grok/xai, or gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq (GroqCloud: gsk_...) / xAI Grok (xai-...) Configuration
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY", "").strip()
    or os.getenv("GROK_API_KEY", "").strip()
    or os.getenv("XAI_API_KEY", "").strip()
)

# Auto-detect endpoint based on key prefix
if GROQ_API_KEY.startswith("gsk_"):
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    DEFAULT_PROVIDER = "groq"
else:
    GROQ_BASE_URL = "https://api.x.ai/v1"
    GROQ_MODEL = os.getenv("GROK_MODEL", "grok-2-latest").strip()
    DEFAULT_PROVIDER = "xai"

# Gemini Configuration (backup/alternative)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip()

# Generation parameters matching the research paper & clinical precision
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 8192

# Dataset paths
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_FILE = DATASET_DIR / "diabetes_cases_train.csv"
VAL_FILE = DATASET_DIR / "diabetes_cases_val.csv"
TEST_FILE = DATASET_DIR / "diabetes_cases_test.csv"
ACI_BENCH_FILE = DATASET_DIR / "aci_bench_diabetes_subset.csv"
KNOWLEDGE_FILE = DATASET_DIR / "knowledge_store" / "diabetes_clinical_guidelines.json"

# Cache paths for embeddings
CACHE_DIR = ROOT_DIR / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_EMBEDDINGS_CACHE = CACHE_DIR / "train_dialogue_embeddings.npy"

def validate_api_key(provider: str = None):
    """Verify that the API key for the selected provider is configured."""
    p = (provider or LLM_PROVIDER).lower()
    if p in ["groq", "grok", "xai"]:
        key = GROQ_API_KEY
        if not key or key.startswith("your_"):
            raise ValueError(
                "GROQ_API_KEY is not set in .env file.\n"
                "Please open .env and add your valid key:\n"
                "GROQ_API_KEY=gsk_..."
            )
        return key
    else:
        key = GEMINI_API_KEY
        if not key or key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set in .env file.\n"
                "Please open .env and add your valid Gemini API key:\n"
                "GEMINI_API_KEY=AIzaSy..."
            )
        return key
