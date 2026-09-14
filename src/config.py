from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "raw"

PROCESSED_DATA_PATH = DATA_DIR / "processed"

DATABASE_PATH = BASE_DIR / "database" / "churn.db"

MODELS_PATH = BASE_DIR / "models"

REPORTS_PATH = BASE_DIR / "reports"