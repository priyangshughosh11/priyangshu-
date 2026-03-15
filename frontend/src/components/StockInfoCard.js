/**
 * StockInfoCard – displays company metadata for a ticker.
 *
 * Props:
 *   ticker {string}
 */
import React, { useState, useEffect } from "react";
import { fetchStockInfo } from "../services/api";

function formatMarketCap(val) {
  if (!val) return "N/A";
  if (val >= 1e12) return `$${(val / 1e12).toFixed(2)}T`;
  if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
  if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
  return `$${val}`;
}

export default function StockInfoCard({ ticker }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    fetchStockInfo(ticker)
      .then(setInfo)
      .catch(() => setInfo(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading) return <div className="info-card card skeleton" />;
  if (!info) return null;

  return (
    <div className="info-card card">
      <h2 className="info-name">{info.name}</h2>
      <div className="info-row">
        <InfoItem label="Sector" value={info.sector} />
        <InfoItem label="Industry" value={info.industry} />
        <InfoItem label="Market Cap" value={formatMarketCap(info.market_cap)} />
        <InfoItem label="P/E Ratio" value={info.pe_ratio ? info.pe_ratio.toFixed(1) : "N/A"} />
        <InfoItem label="Currency" value={info.currency} />
      </div>
      {info.description && (
        <p className="info-desc">{info.description.slice(0, 280)}…</p>
      )}
    </div>
  );
}

function InfoItem({ label, value }) {
  return (
    <div className="info-item">
      <span className="info-label">{label}</span>
      <span className="info-value">{value || "N/A"}</span>
    </div>
  );
}
