/**
 * useStockData – custom hook for fetching stock history + indicators.
 *
 * Returns { data, loading, error } where data is the array of bar objects
 * returned by the API.
 */
import { useState, useEffect, useCallback } from "react";
import { fetchIndicators } from "../services/api";

export function useStockData(ticker, period = "1y") {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetchIndicators(ticker, period);
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, period]);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, error, reload: load };
}

/**
 * usePrediction – custom hook for fetching ML predictions.
 *
 * Returns { prediction, loading, error, run }.
 * Call `run()` to (re-)execute the prediction.
 */
export function usePrediction(ticker, model = "rf", days = 30) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const { fetchPrediction } = await import("../services/api");
      const result = await fetchPrediction(ticker, model, days);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, model, days]);

  return { prediction, loading, error, run };
}
