/**
 * Root application component.
 *
 * Layout:
 *   Header → Ticker search bar
 *   Main content (tabs):
 *     - Dashboard  : StockChart + TechnicalIndicators + StockInfoCard
 *     - Prediction : PredictionPanel
 *     - Compare    : StockComparison
 */
import React, { useState } from "react";
import StockChart from "./components/StockChart";
import TechnicalIndicators from "./components/TechnicalIndicators";
import PredictionPanel from "./components/PredictionPanel";
import StockComparison from "./components/StockComparison";
import StockInfoCard from "./components/StockInfoCard";
import { useStockData } from "./hooks/useStock";
import "./App.css";

const TABS = ["Dashboard", "Prediction", "Compare"];
const DEFAULT_TICKER = "AAPL";
const PERIOD_OPTIONS = [
  { label: "6 Months", value: "6mo" },
  { label: "1 Year", value: "1y" },
  { label: "2 Years", value: "2y" },
  { label: "5 Years", value: "5y" },
];

export default function App() {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [inputVal, setInputVal] = useState(DEFAULT_TICKER);
  const [period, setPeriod] = useState("1y");
  const [activeTab, setActiveTab] = useState("Dashboard");

  const { data, loading, error } = useStockData(ticker, period);

  const handleSearch = (e) => {
    e.preventDefault();
    const sym = inputVal.trim().toUpperCase();
    if (sym) setTicker(sym);
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon">📈</span>
          <span className="brand-name">StockAI</span>
          <span className="brand-sub">ML-Powered Market Predictions</span>
        </div>

        <form className="search-form" onSubmit={handleSearch}>
          <input
            className="search-input"
            type="text"
            placeholder="Ticker symbol (e.g. TSLA)"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value.toUpperCase())}
          />
          <button className="btn-primary" type="submit">Search</button>
        </form>

        <nav className="tab-nav">
          {TABS.map((t) => (
            <button
              key={t}
              className={`tab-btn ${activeTab === t ? "active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {/* ── Content ── */}
      <main className="app-main">
        {activeTab === "Dashboard" && (
          <DashboardTab
            ticker={ticker}
            data={data}
            loading={loading}
            error={error}
            period={period}
            onPeriodChange={setPeriod}
          />
        )}
        {activeTab === "Prediction" && (
          <PredictionPanel ticker={ticker} historicalData={data} />
        )}
        {activeTab === "Compare" && (
          <StockComparison initialTickers={[ticker, "SPY"]} />
        )}
      </main>

      <footer className="app-footer">
        <p>
          Data provided by Yahoo Finance · Predictions are for educational purposes only and
          should not be used as financial advice.
        </p>
      </footer>
    </div>
  );
}

function DashboardTab({ ticker, data, loading, error, period, onPeriodChange }) {
  return (
    <div className="dashboard">
      {/* Period selector */}
      <div className="period-bar">
        <span className="current-ticker">{ticker}</span>
        <div className="period-btns">
          {PERIOD_OPTIONS.map((p) => (
            <button
              key={p.value}
              className={`period-btn ${period === p.value ? "active" : ""}`}
              onClick={() => onPeriodChange(p.value)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Company info */}
      <StockInfoCard ticker={ticker} />

      {/* Main price chart */}
      <div className="card">
        <h2 className="section-title">Price Chart &amp; Moving Averages</h2>
        {loading && <LoadingSpinner />}
        {error && <p className="error-msg">⚠ {error}</p>}
        {!loading && !error && <StockChart data={data} />}
      </div>

      {/* Technical indicators */}
      {!loading && !error && data.length > 0 && (
        <div className="card">
          <h2 className="section-title">Technical Indicators</h2>
          <TechnicalIndicators data={data} />
        </div>
      )}
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div className="spinner-wrapper">
      <div className="spinner" />
      <p className="spinner-label">Fetching market data…</p>
    </div>
  );
}
