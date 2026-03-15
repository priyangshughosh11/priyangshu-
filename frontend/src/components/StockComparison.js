/**
 * StockComparison – compare closing prices of multiple tickers on one chart.
 *
 * Props:
 *   initialTickers {string[]} – seed tickers (default [])
 */
import React, { useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchComparison } from "../services/api";

const PALETTE = [
  "#38bdf8", "#f472b6", "#34d399", "#f59e0b", "#a78bfa",
  "#fb923c", "#22d3ee", "#e879f9", "#4ade80", "#facc15",
];

const PERIOD_OPTIONS = ["6mo", "1y", "2y", "5y"];

export default function StockComparison({ initialTickers = [] }) {
  const [tickers, setTickers] = useState(initialTickers);
  const [input, setInput] = useState("");
  const [period, setPeriod] = useState("1y");
  const [chartData, setChartData] = useState([]);
  const [seriesKeys, setSeriesKeys] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const addTicker = () => {
    const sym = input.trim().toUpperCase();
    if (sym && !tickers.includes(sym)) {
      setTickers((prev) => [...prev, sym]);
    }
    setInput("");
  };

  const removeTicker = (sym) => setTickers((prev) => prev.filter((t) => t !== sym));

  const handleCompare = useCallback(async () => {
    if (tickers.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const raw = await fetchComparison(tickers, period);

      // Normalise: index by date, each ticker as a column
      const dateMap = {};
      const keys = [];

      for (const [sym, bars] of Object.entries(raw)) {
        if (Array.isArray(bars)) {
          keys.push(sym);
          for (const bar of bars) {
            if (!dateMap[bar.date]) dateMap[bar.date] = { date: bar.date };
            dateMap[bar.date][sym] = bar.close;
          }
        }
      }

      const merged = Object.values(dateMap).sort((a, b) => a.date.localeCompare(b.date));
      setSeriesKeys(keys);
      setChartData(merged);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [tickers, period]);

  return (
    <div className="comparison-panel card">
      <h2 className="section-title">Multi-Stock Comparison</h2>

      <div className="controls-row">
        <input
          className="ticker-input"
          type="text"
          placeholder="Add ticker (e.g. GOOG)"
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === "Enter" && addTicker()}
        />
        <button className="btn-secondary" onClick={addTicker}>Add</button>

        <select
          className="period-select"
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
        >
          {PERIOD_OPTIONS.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>

        <button
          className="btn-primary"
          onClick={handleCompare}
          disabled={loading || tickers.length === 0}
        >
          {loading ? "Loading…" : "Compare"}
        </button>
      </div>

      {/* Ticker chips */}
      <div className="chip-row">
        {tickers.map((t, i) => (
          <span
            key={t}
            className="ticker-chip"
            style={{ borderColor: PALETTE[i % PALETTE.length] }}
          >
            {t}
            <button className="chip-remove" onClick={() => removeTicker(t)}>✕</button>
          </span>
        ))}
      </div>

      {error && <p className="error-msg">⚠ {error}</p>}

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickFormatter={(v) => v?.slice(5)}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              width={70}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`]} />
            <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
            {seriesKeys.map((sym, i) => (
              <Line
                key={sym}
                type="monotone"
                dataKey={sym}
                stroke={PALETTE[i % PALETTE.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
