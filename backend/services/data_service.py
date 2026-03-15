"""Data service: fetch and preprocess historical stock data."""
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        period: Data period string understood by yfinance (e.g. '1y', '2y', '5y').
        interval: Bar interval ('1d', '1wk', '1mo').

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume and a DatetimeIndex.

    Raises:
        ValueError: If no data is returned for the given ticker.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators and append them as new columns.

    Indicators added:
        SMA_20, SMA_50, EMA_20
        MACD, MACD_Signal, MACD_Hist
        RSI_14
        BB_Upper, BB_Lower, BB_Middle
        ATR_14
        Daily_Return, Log_Return

    Args:
        df: DataFrame with at least Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with additional indicator columns; rows with NaN dropped.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Moving averages
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=close, window=20).ema_indicator()

    # MACD
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # RSI
    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()

    # ATR (volatility)
    df["ATR_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Return metrics
    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))

    df = df.dropna()
    return df


def get_stock_info(ticker: str) -> dict:
    """Return basic company information for a ticker.

    Args:
        ticker: Stock symbol.

    Returns:
        Dictionary with keys: name, sector, industry, market_cap, pe_ratio, description.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", None),
        "pe_ratio": info.get("trailingPE", None),
        "description": info.get("longBusinessSummary", ""),
        "currency": info.get("currency", "USD"),
    }


def prepare_features(df: pd.DataFrame, target_col: str = "Close") -> tuple:
    """Build feature matrix X and target vector y for ML training.

    The target is the next-day closing price (shifted by -1).

    Args:
        df: DataFrame with OHLCV + indicator columns.
        target_col: Column to predict.

    Returns:
        Tuple (X, y, feature_names) where X and y are numpy arrays and
        feature_names is the list of feature column names used.
    """
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI_14",
        "BB_Upper", "BB_Lower", "BB_Middle",
        "ATR_14",
        "Daily_Return", "Log_Return",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    data = df[feature_cols].copy()
    target = df[target_col].shift(-1)  # next day price

    # Align and drop last row (no target for it)
    data = data.iloc[:-1]
    target = target.iloc[:-1]

    X = data.values
    y = target.values
    return X, y, feature_cols
