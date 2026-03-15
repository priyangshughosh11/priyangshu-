# StockAI – ML-Powered Stock Market Prediction Application

A full-stack, AI-driven stock market prediction platform built with **Python (Flask)** on the backend and **React** on the frontend. It uses machine learning models (Random Forest and LSTM) to forecast stock prices, combined with interactive charts and technical indicators.

---

## Features

- **Real-time & historical data** via Yahoo Finance (yfinance)
- **Machine learning models**
  - Random Forest – ensemble method with feature importance
  - LSTM Neural Network – time-series deep learning
- **Technical indicators**: SMA 20/50, EMA 20, MACD, RSI, Bollinger Bands, ATR
- **Interactive price chart** with volume sub-chart and MA overlays
- **30/60/90-day price forecasts** with confidence metrics
- **Multi-stock comparison** chart
- **Model evaluation metrics**: MAE, RMSE, R², MAPE, Directional Accuracy
- Dark-themed, responsive React UI

---

## Architecture

```
priyangshu-/
├── backend/                  # Flask REST API
│   ├── app.py                # Application factory & entry point
│   ├── config.py             # Configuration
│   ├── requirements.txt      # Python dependencies
│   ├── models/
│   │   ├── rf_model.py       # Random Forest model
│   │   └── lstm_model.py     # LSTM (TensorFlow/Keras) model
│   ├── services/
│   │   ├── data_service.py   # yfinance fetching + indicators
│   │   └── prediction_service.py  # Orchestration layer
│   ├── routes/
│   │   ├── stock_routes.py   # /api/stock/* endpoints
│   │   └── prediction_routes.py   # /api/predict/* endpoints
│   └── tests/
│       └── test_backend.py   # Pytest unit + integration tests
└── frontend/                 # React single-page application
    ├── src/
    │   ├── App.js             # Root component & routing
    │   ├── App.css            # Dark-theme global styles
    │   ├── components/
    │   │   ├── StockChart.js          # OHLCV + MA overlay chart
    │   │   ├── TechnicalIndicators.js # RSI / MACD sub-charts
    │   │   ├── PredictionPanel.js     # ML prediction UI
    │   │   ├── StockComparison.js     # Multi-ticker comparison
    │   │   └── StockInfoCard.js       # Company metadata
    │   ├── hooks/
    │   │   └── useStock.js    # Data-fetching React hooks
    │   └── services/
    │       └── api.js         # Typed API client (fetch wrapper)
    └── package.json
```

---

## How the AI Models Work

### Random Forest
- Trains a `RandomForestRegressor` on 18+ features: OHLCV + SMA, EMA, MACD, RSI, Bollinger Bands, ATR, and return metrics.
- Predicts the **next-day closing price**.
- Uses `TimeSeriesSplit` cross-validation to produce honest evaluation metrics without data leakage.
- Returns **feature importance scores** showing which indicators influenced the prediction.

### LSTM (Long Short-Term Memory)
- A stacked LSTM network that learns temporal dependencies in price sequences.
- Input: a **60-day sliding window** of all 18 features.
- Architecture: `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(32) → Dense(1)`.
- Trained with `EarlyStopping` and `ReduceLROnPlateau` for robust convergence.
- Iterative **multi-step forecasting**: each predicted price is appended to the history and indicators are recomputed before the next step.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm | ≥ 9 |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/priyangshughosh11/priyangshu-.git
cd priyangshu-
```

### 2. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask dev server
python app.py
```

The API will be available at `http://localhost:5000`.

> **Note**: The first run downloads TensorFlow (~600 MB). Subsequent starts are fast.

### 3. Frontend

```bash
cd frontend
npm install
npm start
```

The React app opens at `http://localhost:3000` and proxies `/api/*` requests to Flask automatically.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stock/{ticker}/history` | OHLCV bars (`?period=1y&interval=1d`) |
| `GET` | `/api/stock/{ticker}/indicators` | OHLCV + technical indicators |
| `GET` | `/api/stock/{ticker}/info` | Company metadata |
| `GET` | `/api/stock/compare` | Multi-ticker comparison (`?tickers=AAPL,MSFT&period=1y`) |
| `GET` | `/api/predict/{ticker}` | ML forecast (`?model=rf&days=30`) |

---

## Running Tests

```bash
cd backend
pip install pytest
pytest tests/ -v
```

---

## Production Deployment

### Backend (Gunicorn + Nginx)

```bash
cd backend
gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

### Frontend (Static build)

```bash
cd frontend
REACT_APP_API_URL=https://your-api-domain.com npm run build
# Serve the build/ folder with Nginx or any static host
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `SECRET_KEY` | `dev-secret-key` | Flask session secret |
| `LOOKBACK_WINDOW` | `60` | LSTM lookback days |
| `PREDICTION_DAYS` | `30` | Default forecast horizon |
| `REACT_APP_API_URL` | `http://localhost:5000` | Backend URL for the React build |

---

## Disclaimer

Stock price predictions generated by this application are **for educational and research purposes only**. They do not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
