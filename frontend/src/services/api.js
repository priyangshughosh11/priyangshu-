/**
 * API service layer for communicating with the Flask backend.
 * Base URL defaults to http://localhost:5000 but can be overridden via
 * REACT_APP_API_URL environment variable.
 */

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

/**
 * Fetch JSON from an endpoint, throwing on non-2xx responses.
 * @param {string} path - API path (e.g. "/api/stock/AAPL/history").
 * @returns {Promise<any>} Parsed JSON response body.
 */
async function apiFetch(path) {
  const resp = await fetch(`${BASE_URL}${path}`);
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data;
}

/**
 * Fetch OHLCV history for a stock ticker.
 * @param {string} ticker
 * @param {string} [period="1y"]
 * @param {string} [interval="1d"]
 */
export function fetchHistory(ticker, period = "1y", interval = "1d") {
  return apiFetch(
    `/api/stock/${encodeURIComponent(ticker)}/history?period=${period}&interval=${interval}`
  );
}

/**
 * Fetch OHLCV + technical indicators for a ticker.
 * @param {string} ticker
 * @param {string} [period="1y"]
 */
export function fetchIndicators(ticker, period = "1y") {
  return apiFetch(
    `/api/stock/${encodeURIComponent(ticker)}/indicators?period=${period}`
  );
}

/**
 * Fetch company metadata for a ticker.
 * @param {string} ticker
 */
export function fetchStockInfo(ticker) {
  return apiFetch(`/api/stock/${encodeURIComponent(ticker)}/info`);
}

/**
 * Compare multiple tickers (closing prices).
 * @param {string[]} tickers
 * @param {string} [period="1y"]
 */
export function fetchComparison(tickers, period = "1y") {
  const params = new URLSearchParams({ tickers: tickers.join(","), period });
  return apiFetch(`/api/stock/compare?${params}`);
}

/**
 * Run a prediction model for a ticker.
 * @param {string} ticker
 * @param {string} [model="rf"]  - "rf" or "lstm"
 * @param {number} [days=30]
 */
export function fetchPrediction(ticker, model = "rf", days = 30) {
  const params = new URLSearchParams({ model, days });
  return apiFetch(`/api/predict/${encodeURIComponent(ticker)}?${params}`);
}
