/**
 * StockChart – candlestick + moving average chart for a single ticker.
 *
 * Renders:
 *   - A candlestick bar for each OHLCV session.
 *   - SMA 20 and SMA 50 line overlays.
 *   - Bollinger Band shaded region.
 *   - Volume bar chart.
 *
 * Data shape (each element):
 *   { date, open, high, low, close, volume, sma_20, sma_50, bb_upper, bb_lower }
 */
import React, { useMemo } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const COLORS = {
  up: "#26a69a",
  down: "#ef5350",
  sma20: "#f59e0b",
  sma50: "#3b82f6",
  bbUpper: "#a78bfa",
  bbLower: "#a78bfa",
  bbBand: "rgba(167,139,250,0.12)",
  volume: "#94a3b8",
};

/** Custom tooltip for candlestick data. */
const CandlestickTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const isUp = d.close >= d.open;
  return (
    <div className="tooltip-box">
      <p className="tooltip-date">{label}</p>
      <p style={{ color: isUp ? COLORS.up : COLORS.down }}>
        O: {d.open?.toFixed(2)}  H: {d.high?.toFixed(2)}  L: {d.low?.toFixed(2)}  C: {d.close?.toFixed(2)}
      </p>
      {d.sma_20 && <p style={{ color: COLORS.sma20 }}>SMA 20: {d.sma_20.toFixed(2)}</p>}
      {d.sma_50 && <p style={{ color: COLORS.sma50 }}>SMA 50: {d.sma_50.toFixed(2)}</p>}
      <p style={{ color: COLORS.volume }}>Vol: {(d.volume / 1_000_000).toFixed(2)}M</p>
    </div>
  );
};

export default function StockChart({ data }) {
  // Slim the dataset to last 120 bars to avoid overcrowding
  const visible = useMemo(() => (data ? data.slice(-120) : []), [data]);

  if (!visible.length) return <p className="placeholder">No data to display.</p>;

  return (
    <div className="chart-wrapper">
      {/* Price + Indicators */}
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={visible} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
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
          <Tooltip content={<CandlestickTooltip />} />
          <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />

          {/* Bollinger Band area */}
          <Area
            type="monotone"
            dataKey="bb_upper"
            stroke={COLORS.bbUpper}
            strokeWidth={1}
            fill={COLORS.bbBand}
            dot={false}
            name="BB Upper"
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="bb_lower"
            stroke={COLORS.bbLower}
            strokeWidth={1}
            fill="#0f172a"
            dot={false}
            name="BB Lower"
            legendType="none"
          />

          {/* Candle bars – represented as close price bars coloured by direction */}
          <Bar
            dataKey="close"
            name="Close"
            fill={COLORS.up}
            opacity={0.85}
            maxBarSize={8}
            cell={(entry) => (
              <rect
                fill={entry.close >= entry.open ? COLORS.up : COLORS.down}
              />
            )}
          />

          {/* Moving averages */}
          <Line
            type="monotone"
            dataKey="sma_20"
            stroke={COLORS.sma20}
            dot={false}
            strokeWidth={1.5}
            name="SMA 20"
          />
          <Line
            type="monotone"
            dataKey="sma_50"
            stroke={COLORS.sma50}
            dot={false}
            strokeWidth={1.5}
            name="SMA 50"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Volume sub-chart */}
      <ResponsiveContainer width="100%" height={80}>
        <ComposedChart data={visible} margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
          <XAxis dataKey="date" hide />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} width={70} tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`} />
          <Tooltip formatter={(v) => [`${(v / 1e6).toFixed(2)}M`, "Volume"]} />
          <Bar dataKey="volume" name="Volume" fill={COLORS.volume} maxBarSize={8} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
