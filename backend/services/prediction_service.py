"""Prediction service: orchestrates data fetching, training, and inference."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.data_service import fetch_history, add_technical_indicators, prepare_features
from models.rf_model import RandomForestModel
from models.lstm_model import LSTMModel


def run_prediction(ticker: str, model_type: str = "rf", prediction_days: int = 30) -> dict:
    """Train a model on historical data and generate future price predictions.

    This function:
    1. Fetches 2 years of daily OHLCV data for *ticker*.
    2. Computes technical indicators.
    3. Prepares feature matrix X and target y.
    4. Trains the requested model.
    5. Produces *prediction_days* forward price estimates by iteratively
       appending the last predicted value and re-running inference.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        model_type: One of 'rf' (Random Forest) or 'lstm'.
        prediction_days: Number of future trading days to project.

    Returns:
        Dictionary with keys:
            - predictions: list of {date, predicted_price}
            - metrics: model evaluation metrics dict
            - model_type: echo of the requested model
            - last_price: most recent closing price
            - feature_importance: (RF only) list of feature importance dicts
    """
    # 1. Data
    df = fetch_history(ticker, period="2y")
    df = add_technical_indicators(df)
    X, y, feature_names = prepare_features(df)

    if len(X) < 100:
        raise ValueError(
            f"Not enough data for '{ticker}' to train a model "
            f"(need ≥100 rows, got {len(X)})."
        )

    last_price = float(df["Close"].iloc[-1])
    last_date = df.index[-1].to_pydatetime()

    # 2. Train
    if model_type == "lstm":
        model = LSTMModel(lookback=min(60, len(X) // 4))
        metrics = model.train(X, y, epochs=30, batch_size=32)
    else:
        model = RandomForestModel()
        metrics = model.train(X, y, feature_names)

    # 3. Future predictions via iterative single-step forecasting
    predictions = _generate_future_predictions(
        model=model,
        df=df,
        feature_names=feature_names,
        model_type=model_type,
        prediction_days=prediction_days,
        last_date=last_date,
    )

    result = {
        "predictions": predictions,
        "metrics": metrics,
        "model_type": model_type,
        "last_price": last_price,
    }

    if model_type == "rf":
        result["feature_importance"] = model.get_feature_importance()

    return result


def _generate_future_predictions(
    model,
    df: pd.DataFrame,
    feature_names: list,
    model_type: str,
    prediction_days: int,
    last_date: datetime,
) -> list:
    """Iteratively predict *prediction_days* future prices.

    For each step, the most recent feature row is used to predict the next
    close; the DataFrame is then extended with that predicted close so the
    next prediction reflects the new state.

    Args:
        model: Trained RF or LSTM model instance.
        df: Full historical DataFrame with indicators already added.
        feature_names: Feature columns used by the model.
        model_type: 'rf' or 'lstm'.
        prediction_days: Number of business days to project.
        last_date: Date of the last historical bar.

    Returns:
        List of dicts with 'date' (ISO string) and 'predicted_price' (float).
    """
    from services.data_service import add_technical_indicators

    working_df = df.copy()
    predictions = []
    current_date = last_date

    for _ in range(prediction_days):
        # Advance by one business day
        current_date = _next_business_day(current_date)

        # Build feature row from the most recent data
        X_latest = working_df[feature_names].values[-1:] if model_type == "rf" else working_df[feature_names].values

        try:
            if model_type == "lstm":
                pred = float(model.predict(X_latest)[-1])
            else:
                pred = float(model.predict(X_latest)[0])
        except Exception:
            # Fallback: use last known close
            pred = float(working_df["Close"].iloc[-1])

        predictions.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred, 4),
        })

        # Extend the DataFrame with the new predicted row so next iteration
        # has an updated context (use last known OHLV ratios scaled by pred/close)
        last_row = working_df.iloc[-1].copy()
        scale = pred / last_row["Close"] if last_row["Close"] != 0 else 1.0
        new_row = last_row.copy()
        new_row["Open"] = round(last_row["Close"], 4)
        new_row["High"] = round(pred * 1.005, 4)
        new_row["Low"] = round(pred * 0.995, 4)
        new_row["Close"] = round(pred, 4)
        new_row["Volume"] = last_row["Volume"]
        new_row.name = pd.Timestamp(current_date)
        working_df = pd.concat([working_df, pd.DataFrame([new_row])])

        # Recompute indicators on the extended frame (needed for accurate features)
        try:
            working_df = add_technical_indicators(working_df)
        except Exception:
            pass

    return predictions


def _next_business_day(dt: datetime) -> datetime:
    """Return the next Monday-Friday date after *dt*."""
    next_day = dt + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day
