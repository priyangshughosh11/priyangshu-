"""Stock data routes: OHLCV history, indicators, and company info."""
from flask import Blueprint, jsonify, request
from services.data_service import fetch_history, add_technical_indicators, get_stock_info

stock_bp = Blueprint("stock", __name__, url_prefix="/api/stock")


@stock_bp.route("/<ticker>/history")
def history(ticker: str):
    """Return historical OHLCV bars for a ticker.

    Query params:
        period (str): yfinance period string (default '1y').
        interval (str): Bar interval (default '1d').

    Returns:
        JSON list of {date, open, high, low, close, volume} objects.
    """
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")

    try:
        df = fetch_history(ticker.upper(), period=period, interval=interval)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch data: {exc}"}), 500

    records = []
    for ts, row in df.iterrows():
        records.append({
            "date": ts.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return jsonify(records)


@stock_bp.route("/<ticker>/indicators")
def indicators(ticker: str):
    """Return OHLCV data enriched with technical indicators.

    Query params:
        period (str): yfinance period string (default '1y').

    Returns:
        JSON list of bars with all indicator fields appended.
    """
    period = request.args.get("period", "1y")

    try:
        df = fetch_history(ticker.upper(), period=period)
        df = add_technical_indicators(df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to compute indicators: {exc}"}), 500

    records = []
    for ts, row in df.iterrows():
        record = {"date": ts.strftime("%Y-%m-%d")}
        for col in df.columns:
            val = row[col]
            record[col.lower()] = None if (isinstance(val, float) and not __import__("math").isfinite(val)) else round(float(val), 4)
        records.append(record)
    return jsonify(records)


@stock_bp.route("/<ticker>/info")
def info(ticker: str):
    """Return company information for a ticker.

    Returns:
        JSON object with name, sector, industry, market_cap, pe_ratio, description.
    """
    try:
        data = get_stock_info(ticker.upper())
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch company info: {exc}"}), 500
    return jsonify(data)


@stock_bp.route("/compare")
def compare():
    """Compare historical closing prices for multiple tickers.

    Query params:
        tickers (str): Comma-separated list of stock symbols.
        period (str): yfinance period string (default '1y').

    Returns:
        JSON object mapping ticker → list of {date, close}.
    """
    raw = request.args.get("tickers", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "Provide at least one ticker via ?tickers=AAPL,MSFT"}), 400

    period = request.args.get("period", "1y")
    result = {}

    for ticker in tickers:
        try:
            df = fetch_history(ticker, period=period)
            result[ticker] = [
                {"date": ts.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 4)}
                for ts, row in df.iterrows()
            ]
        except Exception as exc:
            result[ticker] = {"error": str(exc)}

    return jsonify(result)
