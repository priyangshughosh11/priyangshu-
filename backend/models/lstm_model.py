"""LSTM model for time-series stock price prediction.

Uses TensorFlow/Keras to build a stacked LSTM network that learns sequential
patterns from a sliding window of past closing prices and technical indicators.
"""
import numpy as np
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info/warning logs

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LSTMModel:
    """Stacked LSTM for next-step stock price prediction.

    The model accepts a 3-D input of shape (samples, lookback, features) and
    outputs a scalar prediction (the next day's closing price).

    Attributes:
        lookback: Number of historical time steps fed into the network.
        model: Compiled Keras model (set after calling build).
        scaler: MinMaxScaler fitted on the training features.
        metrics: Evaluation metrics computed on the validation split.
    """

    def __init__(self, lookback: int = 60, units: int = 64, dropout: float = 0.2):
        """Initialise hyper-parameters.

        Args:
            lookback: Size of the sliding window (time steps).
            units: Number of LSTM units per layer.
            dropout: Dropout rate applied after each LSTM layer.
        """
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model: tf.keras.Model | None = None
        self.scaler = MinMaxScaler()
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        validation_split: float = 0.15,
    ) -> dict:
        """Train the LSTM on the provided feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features) – scaled internally.
            y: Target array (n_samples,) of next-day closing prices.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction of training data withheld for validation.

        Returns:
            Dictionary of evaluation metrics (MAE, RMSE, R2, MAPE, directional accuracy).
        """
        n_features = X.shape[1]

        # Fit scaler on the feature matrix (X only; y is handled separately)
        X_scaled = self.scaler.fit_transform(X)

        # Scale y to [0, 1] using the Close feature column (index 3)
        close_idx = 3
        y_min = self.scaler.data_min_[close_idx]
        y_max = self.scaler.data_max_[close_idx]
        y_scaled = (y - y_min) / (y_max - y_min + 1e-8)
        self._y_min = y_min
        self._y_max = y_max

        # Build sliding-window sequences
        X_seq, y_seq = _create_sequences(X_scaled, y_scaled, self.lookback)

        # Build and compile model if needed
        if self.model is None:
            self.model = _build_model(self.lookback, n_features, self.units, self.dropout)

        # Time-ordered train/val split (no shuffling)
        val_size = max(1, int(len(X_seq) * validation_split))
        X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
        y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
            ),
        ]

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on validation split (inverse-transform before metrics)
        y_pred_scaled = self.model.predict(X_val, verbose=0).flatten()
        y_pred = y_pred_scaled * (y_max - y_min) + y_min
        y_true = y_val * (y_max - y_min) + y_min

        self.metrics = _compute_metrics(y_true, y_pred)
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate next-day predictions from a feature matrix.

        The function creates sequences internally – only the last
        (n_samples - lookback) outputs are returned.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predicted closing prices as a 1-D numpy array.
        """
        X_scaled = self.scaler.transform(X)
        X_seq, _ = _create_sequences(X_scaled, np.zeros(len(X_scaled)), self.lookback)
        y_pred_scaled = self.model.predict(X_seq, verbose=0).flatten()
        return y_pred_scaled * (self._y_max - self._y_min) + self._y_min

    def save(self, path: str) -> None:
        """Save Keras model weights and scaler.

        Args:
            path: Directory to store files.
        """
        import joblib

        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "lstm_model.keras"))
        joblib.dump(self.scaler, os.path.join(path, "lstm_scaler.joblib"))
        np.save(os.path.join(path, "lstm_yscale.npy"), np.array([self._y_min, self._y_max]))

    def load(self, path: str) -> None:
        """Load a previously saved model and scaler.

        Args:
            path: Directory containing saved files.
        """
        import joblib

        self.model = tf.keras.models.load_model(os.path.join(path, "lstm_model.keras"))
        self.scaler = joblib.load(os.path.join(path, "lstm_scaler.joblib"))
        yscale = np.load(os.path.join(path, "lstm_yscale.npy"))
        self._y_min, self._y_max = yscale[0], yscale[1]


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Convert a flat feature matrix into overlapping sequences.

    Args:
        X: Scaled feature matrix (n_samples, n_features).
        y: Target array (n_samples,).
        lookback: Number of time steps per sequence.

    Returns:
        Tuple (X_seq, y_seq) with shapes
        (n_samples - lookback, lookback, n_features) and (n_samples - lookback,).
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _build_model(lookback: int, n_features: int, units: int, dropout: float) -> tf.keras.Model:
    """Construct and compile the stacked LSTM model.

    Architecture:
        Input → LSTM(units, return_sequences=True) → Dropout
              → LSTM(units // 2) → Dropout
              → Dense(32, relu) → Dense(1, linear)

    Args:
        lookback: Number of input time steps.
        n_features: Number of features per time step.
        units: LSTM units in the first layer.
        dropout: Dropout fraction.

    Returns:
        Compiled Keras model.
    """
    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = tf.keras.layers.LSTM(units, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(units // 2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression and directional accuracy metrics.

    Args:
        y_true: Ground-truth closing prices.
        y_pred: Predicted closing prices.

    Returns:
        Dictionary with mae, rmse, r2, mape, directional_accuracy.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

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
