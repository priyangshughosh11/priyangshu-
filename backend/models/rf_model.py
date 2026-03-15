"""Random Forest model for stock price prediction."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class RandomForestModel:
    """Wraps a scikit-learn RandomForestRegressor with time-series aware training.

    The model predicts the next-day closing price given a feature vector of
    OHLCV data and technical indicators.  A MinMaxScaler is applied to the
    features before training/inference.

    Attributes:
        model: Underlying RandomForestRegressor instance.
        scaler: Fitted MinMaxScaler for features.
        feature_names: List of feature column names used during training.
        metrics: Dictionary of evaluation metrics computed on the hold-out fold.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = MinMaxScaler()
        self.feature_names: list = []
        self.metrics: dict = {}

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        """Train the model using time-series cross-validation.

        The last fold of a 5-fold TimeSeriesSplit is used as the validation set
        to compute evaluation metrics; the model is then retrained on the full
        dataset so that it has seen as much history as possible.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            feature_names: Names corresponding to columns in X.

        Returns:
            Dictionary with MAE, RMSE, R2, MAPE, and directional accuracy metrics.
        """
        self.feature_names = feature_names

        # Normalise features
        X_scaled = self.scaler.fit_transform(X)

        # Time-series aware cross-validation for honest evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X_scaled))
        train_idx, val_idx = splits[-1]  # use the last fold as hold-out

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.model.fit(X_train, y_train)
        y_pred_val = self.model.predict(X_val)

        self.metrics = _compute_metrics(y_val, y_pred_val)

        # Retrain on the full dataset
        self.model.fit(X_scaled, y)
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return next-day price predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted prices as a 1-D numpy array.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> list:
        """Return feature importances sorted in descending order.

        Returns:
            List of dicts with 'feature' and 'importance' keys.
        """
        importances = self.model.feature_importances_
        result = sorted(
            [{"feature": name, "importance": float(imp)}
             for name, imp in zip(self.feature_names, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )
        return result

    def save(self, path: str) -> None:
        """Persist the model and scaler to disk.

        Args:
            path: Directory where model files are saved.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "rf_model.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "rf_scaler.joblib"))

    def load(self, path: str) -> None:
        """Load a previously saved model and scaler.

        Args:
            path: Directory where model files are stored.
        """
        self.model = joblib.load(os.path.join(path, "rf_model.joblib"))
        self.scaler = joblib.load(os.path.join(path, "rf_scaler.joblib"))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression and directional accuracy metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys: mae, rmse, r2, mape, directional_accuracy.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE – guard against zeros
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # Directional accuracy: did the model predict the right direction?
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    directional_acc = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "directional_accuracy": directional_acc,
    }
