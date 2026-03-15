/**
 * TechnicalIndicators – RSI, MACD, and ATR mini-charts beneath the main price chart.
 *
 * Props:
 *   data {Array} – enriched indicator data from useStockData
 */
import React, { useMemo } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

export default function TechnicalIndicators({ data }) {
  const visible = useMemo(() => (data ? data.slice(-120) : []), [data]);

  if (!visible.length) return null;

  return (
    <div className="indicator-section">
      {/* RSI */}
      <SubChart title="RSI (14)" height={120}>
        <ComposedChart data={visible} margin={{ top: 5, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="date" hide />
          <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} width={40} />
          <Tooltip formatter={(v) => [Number(v).toFixed(1), "RSI"]} />
          <ReferenceLine y={70} stroke="#ef5350" strokeDasharray="3 3" />
          <ReferenceLine y={30} stroke="#26a69a" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="rsi_14" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        </ComposedChart>
      </SubChart>

      {/* MACD */}
      <SubChart title="MACD" height={120}>
        <ComposedChart data={visible} margin={{ top: 5, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="date" hide />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} width={55} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip formatter={(v) => [Number(v).toFixed(3)]} />
          <ReferenceLine y={0} stroke="#475569" />
          <Bar dataKey="macd_hist" name="Histogram" fill="#38bdf8" opacity={0.7} maxBarSize={6} />
          <Line type="monotone" dataKey="macd" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="MACD" />
          <Line type="monotone" dataKey="macd_signal" stroke="#f472b6" dot={false} strokeWidth={1.5} name="Signal" />
        </ComposedChart>
      </SubChart>
    </div>
  );
}

function SubChart({ title, height, children }) {
  return (
    <div className="sub-chart">
      <p className="sub-chart-title">{title}</p>
      <ResponsiveContainer width="100%" height={height}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}
