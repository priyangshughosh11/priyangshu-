"""Backend unit tests – run with pytest from the backend/ directory."""
import math
import types
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Ensure backend package is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# data_service helpers
# ---------------------------------------------------------------------------

class TestAddTechnicalIndicators:
    """Tests for add_technical_indicators()."""

    @pytest.fixture
    def sample_df(self):
        """Synthetic OHLCV data with 200 rows."""
        from services.data_service import add_technical_indicators  # noqa: F401 – import check
        np.random.seed(0)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n))
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        }, index=dates)
        return df

    def test_columns_added(self, sample_df):
        from services.data_service import add_technical_indicators
        result = add_technical_indicators(sample_df)
        expected_cols = [
            "SMA_20", "SMA_50", "EMA_20",
            "MACD", "MACD_Signal", "MACD_Hist",
            "RSI_14",
            "BB_Upper", "BB_Lower", "BB_Middle",
            "ATR_14",
            "Daily_Return", "Log_Return",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_after_drop(self, sample_df):
        from services.data_service import add_technical_indicators
        result = add_technical_indicators(sample_df)
        assert not result.isnull().any().any(), "NaN values remain after dropna"

    def test_rsi_bounds(self, sample_df):
        from services.data_service import add_technical_indicators
        result = add_technical_indicators(sample_df)
        assert result["RSI_14"].between(0, 100).all(), "RSI out of [0, 100] range"


class TestPrepareFeatures:
    """Tests for prepare_features()."""

    @pytest.fixture
    def enriched_df(self):
        from services.data_service import add_technical_indicators
        np.random.seed(1)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n))
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        }, index=dates)
        return add_technical_indicators(df)

    def test_shapes_consistent(self, enriched_df):
        from services.data_service import prepare_features
        X, y, feature_names = prepare_features(enriched_df)
        assert X.shape[0] == y.shape[0], "X and y row counts must match"
        assert X.shape[1] == len(feature_names), "Feature matrix columns must match feature_names"

    def test_no_nans(self, enriched_df):
        from services.data_service import prepare_features
        X, y, _ = prepare_features(enriched_df)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


# ---------------------------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------------------------

class TestRandomForestModel:
    """Tests for RandomForestModel."""

    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 10 + 100
        feature_names = [f"feat_{i}" for i in range(10)]
        return X, y, feature_names

    def test_train_returns_metrics(self, training_data):
        from models.rf_model import RandomForestModel
        X, y, feature_names = training_data
        rf = RandomForestModel(n_estimators=10)
        metrics = rf.train(X, y, feature_names)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert "directional_accuracy" in metrics

    def test_predict_shape(self, training_data):
        from models.rf_model import RandomForestModel
        X, y, feature_names = training_data
        rf = RandomForestModel(n_estimators=10)
        rf.train(X, y, feature_names)
        preds = rf.predict(X)
        assert preds.shape == (len(X),)

    def test_feature_importance_sorted(self, training_data):
        from models.rf_model import RandomForestModel
        X, y, feature_names = training_data
        rf = RandomForestModel(n_estimators=10)
        rf.train(X, y, feature_names)
        fi = rf.get_feature_importance()
        importances = [item["importance"] for item in fi]
        assert importances == sorted(importances, reverse=True)


# ---------------------------------------------------------------------------
# LSTM model (lightweight smoke test – avoids full training)
# ---------------------------------------------------------------------------

class TestLSTMModel:
    """Smoke tests for LSTMModel (minimal epochs to keep CI fast)."""

    @pytest.fixture
    def training_data(self):
        np.random.seed(7)
        n = 150
        X = np.random.randn(n, 10)
        y = np.cumsum(np.random.randn(n)) + 100
        return X, y

    def test_train_and_predict(self, training_data):
        from models.lstm_model import LSTMModel
        X, y = training_data
        lstm = LSTMModel(lookback=20, units=16)
        metrics = lstm.train(X, y, epochs=2, batch_size=16)
        assert "mae" in metrics
        preds = lstm.predict(X)
        # predict returns (n - lookback) predictions
        assert preds.shape[0] == len(X) - 20


# ---------------------------------------------------------------------------
# Flask routes (mocked network I/O)
# ---------------------------------------------------------------------------

class TestFlaskRoutes:
    """Integration-style tests for Flask API endpoints using mocked data."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Return a Flask test client with network calls mocked out."""
        # Build a fake DataFrame returned by fetch_history
        np.random.seed(99)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n))
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        fake_df = pd.DataFrame({
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        }, index=dates)

        monkeypatch.setattr("services.data_service.yf.Ticker", _make_mock_ticker(fake_df))

        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_history_endpoint(self, client):
        resp = client.get("/api/stock/AAPL/history?period=1y")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "date" in data[0]
        assert "close" in data[0]

    def test_compare_endpoint(self, client):
        resp = client.get("/api/stock/compare?tickers=AAPL,MSFT&period=1y")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "AAPL" in data
        assert "MSFT" in data

    def test_compare_no_tickers(self, client):
        resp = client.get("/api/stock/compare")
        assert resp.status_code == 400

    def test_predict_invalid_model(self, client):
        resp = client.get("/api/predict/AAPL?model=xgboost")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ticker(fake_df: pd.DataFrame):
    """Return a constructor that mimics yf.Ticker and returns fake data."""
    class MockTicker:
        def __init__(self, symbol):
            self.info = {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "marketCap": 3_000_000_000_000,
                "trailingPE": 28.5,
                "longBusinessSummary": "Makes great products.",
                "currency": "USD",
            }

        def history(self, period=None, interval=None, auto_adjust=True):
            return fake_df.copy()

    return MockTicker
