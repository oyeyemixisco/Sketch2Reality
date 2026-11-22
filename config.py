"""Phase 3: local UI + Kaggle GPU inference API."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

ROOT_DIR = Path(__file__).resolve().parent
MEDIA_DIR = ROOT_DIR / "instance" / "media"
HISTORY_FILE = ROOT_DIR / "instance" / "history.json"
MAX_UPLOAD_MB = 10
MAX_SOURCE_DIMENSION = 1600
MAX_HISTORY_ITEMS = 35

HOST = os.environ.get("SKETCH2FUTURE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SKETCH2FUTURE_PORT", "5000"))
DEBUG = os.environ.get("SKETCH2FUTURE_DEBUG", "0") == "1"

BACKEND_MODE = os.getenv("SKETCH2FUTURE_BACKEND", "kaggle").strip().lower()
if BACKEND_MODE not in {"mock", "kaggle"}:
    raise ValueError("SKETCH2FUTURE_BACKEND must be 'mock' or 'kaggle'")
KAGGLE_API_URL = os.getenv("KAGGLE_API_URL", "").strip()
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY", "").strip()
KAGGLE_CONNECT_TIMEOUT = int(os.getenv("KAGGLE_CONNECT_TIMEOUT", "10"))
KAGGLE_READ_TIMEOUT = int(os.getenv("KAGGLE_READ_TIMEOUT", "1800"))
