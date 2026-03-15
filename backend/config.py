"""Application configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    # Default lookback window (days) for ML features
    LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "60"))
    # Number of future days to predict
    PREDICTION_DAYS = int(os.getenv("PREDICTION_DAYS", "30"))
    # Minimum history needed (days)
    MIN_HISTORY_DAYS = int(os.getenv("MIN_HISTORY_DAYS", "365"))
