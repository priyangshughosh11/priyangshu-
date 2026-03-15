/**
 * PredictionPanel – runs ML prediction and visualises the forecast.
 *
 * Props:
 *   ticker {string}       – stock symbol
 *   historicalData {Array} – OHLCV bars from useStockData
 */
import React, { useState } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { fetchPrediction } from "../services/api";

const MODEL_OPTIONS = [
  { value: "rf", label: "Random Forest" },
  { value: "lstm", label: "LSTM Neural Network" },
];

export default function PredictionPanel({ ticker, historicalData = [] }) {
  const [model, setModel] = useState("rf");
  const [days, setDays] = useState(30);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handlePredict() {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchPrediction(ticker, model, days);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Build combined chart data: last 60 historical bars + predictions
  const chartData = React.useMemo(() => {
    const hist = historicalData.slice(-60).map((d) => ({
      date: d.date,
      actual: d.close,
      predicted: null,
    }));

    const preds = (result?.predictions || []).map((d) => ({
      date: d.date,
      actual: null,
      predicted: d.predicted_price,
    }));

    return [...hist, ...preds];
  }, [historicalData, result]);

  const metrics = result?.metrics;
  const lastHistDate = historicalData.length ? historicalData[historicalData.length - 1].date : null;

  return (
    <div className="prediction-panel card">
      <h2 className="section-title">Price Prediction</h2>

      {/* Controls */}
      <div className="controls-row">
        <div className="control-group">
          <label>Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Forecast Days</label>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {[7, 14, 30, 60, 90].map((d) => (
              <option key={d} value={d}>{d} days</option>
            ))}
          </select>
        </div>

        <button
          className="btn-primary"
          onClick={handlePredict}
          disabled={loading || !ticker}
        >
          {loading ? "Running Model…" : "Run Prediction"}
        </button>
      </div>

      {loading && (
        <div className="loading-bar">
          <div className="loading-fill" />
        </div>
      )}

      {error && <p className="error-msg">⚠ {error}</p>}

      {result && (
        <>
          {/* Forecast Chart */}
          <div className="chart-wrapper" style={{ marginTop: "1.25rem" }}>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  tickFormatter={(v) => v?.slice(5)}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  width={70}
                  tickFormatter={(v) => `$${v.toFixed(0)}`}
                />
                <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`]} />
                <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />

                {lastHistDate && (
                  <ReferenceLine
                    x={lastHistDate}
                    stroke="#f59e0b"
                    strokeDasharray="4 2"
                    label={{ value: "Today", fill: "#f59e0b", fontSize: 11 }}
                  />
                )}

                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#38bdf8"
                  dot={false}
                  strokeWidth={2}
                  name="Historical"
                  connectNulls={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#f472b6"
                  dot={false}
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  name="Forecast"
                  connectNulls={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Metrics */}
          {metrics && <MetricsDisplay metrics={metrics} model={model} lastPrice={result.last_price} />}

          {/* Feature Importance (RF only) */}
          {result.feature_importance && (
            <FeatureImportance data={result.feature_importance.slice(0, 8)} />
          )}
        </>
      )}
    </div>
  );
}

function MetricsDisplay({ metrics, model, lastPrice }) {
  return (
    <div className="metrics-grid">
      <MetricCard label="MAE" value={`$${metrics.mae?.toFixed(2)}`} hint="Mean Absolute Error" />
      <MetricCard label="RMSE" value={`$${metrics.rmse?.toFixed(2)}`} hint="Root Mean Square Error" />
      <MetricCard label="R²" value={metrics.r2?.toFixed(3)} hint="Coefficient of determination" />
      <MetricCard label="MAPE" value={`${metrics.mape?.toFixed(1)}%`} hint="Mean Absolute % Error" />
      <MetricCard
        label="Dir. Accuracy"
        value={`${metrics.directional_accuracy?.toFixed(1)}%`}
        hint="% of days model predicted correct direction"
        highlight={metrics.directional_accuracy > 55}
      />
      <MetricCard label="Last Price" value={`$${lastPrice?.toFixed(2)}`} hint="Most recent closing price" />
    </div>
  );
}

function MetricCard({ label, value, hint, highlight }) {
  return (
    <div className={`metric-card ${highlight ? "highlight" : ""}`} title={hint}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value ?? "—"}</span>
    </div>
  );
}

function FeatureImportance({ data }) {
  const max = data[0]?.importance || 1;
  return (
    <div className="feature-importance">
      <h3 className="sub-title">Top Feature Importances</h3>
      {data.map((item) => (
        <div key={item.feature} className="fi-row">
          <span className="fi-name">{item.feature}</span>
          <div className="fi-bar-bg">
            <div
              className="fi-bar-fill"
              style={{ width: `${(item.importance / max) * 100}%` }}
            />
          </div>
          <span className="fi-pct">{(item.importance * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
