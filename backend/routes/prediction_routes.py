"""Prediction routes: train models and return future price forecasts."""
from flask import Blueprint, jsonify, request
from services.prediction_service import run_prediction

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api/predict")

VALID_MODELS = {"rf", "lstm"}


@prediction_bp.route("/<ticker>", methods=["GET"])
def predict(ticker: str):
    """Train a model and return future price predictions for a ticker.

    Query params:
        model (str): Model type – 'rf' or 'lstm' (default 'rf').
        days (int): Number of future trading days to predict (default 30, max 90).

    Returns:
        JSON with predictions list, evaluation metrics, and last known price.
    """
    model_type = request.args.get("model", "rf").lower()
    if model_type not in VALID_MODELS:
        return jsonify({"error": f"Invalid model '{model_type}'. Choose from: {sorted(VALID_MODELS)}"}), 400

    try:
        days = int(request.args.get("days", 30))
        days = max(1, min(days, 90))
    except ValueError:
        return jsonify({"error": "'days' must be an integer."}), 400

    try:
        result = run_prediction(ticker.upper(), model_type=model_type, prediction_days=days)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)
