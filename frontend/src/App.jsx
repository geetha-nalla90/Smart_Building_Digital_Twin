import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = import.meta.env.VITE_API_BASE_URL || "";
const SENSOR_FIELDS = [
  { key: "Occupancy", label: "Occupancy", min: 0, max: 100, unit: "%" },
  { key: "Temperature", label: "Temperature", min: 18, max: 42, unit: "°C" },
  { key: "Humidity", label: "Humidity", min: 25, max: 95, unit: "%" },
  { key: "Lighting_Load", label: "Lighting load", min: 0, max: 100, unit: "dataset units" },
  { key: "HVAC_Load", label: "HVAC load", min: 0, max: 100, unit: "dataset units" },
  { key: "Computer_Load", label: "Computer load", min: 0, max: 100, unit: "dataset units" },
  { key: "Other_Load", label: "Other load", min: 0, max: 100, unit: "dataset units" },
];

async function requestJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(Number(value))
    ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
}

function MetricCard({ label, value, unit, accent }) {
  return (
    <article className={`metric-card ${accent || ""}`}>
      <span>{label}</span>
      <strong>{value == null ? "—" : formatNumber(value)}</strong>
      <small>{unit}</small>
    </article>
  );
}

function ChartPanel({ title, subtitle, children, className = "" }) {
  return (
    <section className={`panel chart-panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function App() {
  const [building, setBuilding] = useState(null);
  const [liveReadings, setLiveReadings] = useState([]);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [scenario, setScenario] = useState(null);
  const [scenarioResult, setScenarioResult] = useState(null);
  const [loadingScenario, setLoadingScenario] = useState(false);
  const [error, setError] = useState("");
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    let active = true;
    let pollTimer;
    let liveRequest;

    async function loadDashboard() {
      const results = await Promise.allSettled([
        requestJson("/api/energy/history?source=historical&limit=120"),
        requestJson("/api/model/metrics"),
      ]);
      if (!active) return;
      if (results[0].status === "fulfilled") {
        setHistory(results[0].value.rows);
      } else {
        setError(`Historical analytics: ${results[0].reason.message}`);
      }
      if (results[1].status === "fulfilled") {
        setMetrics(results[1].value);
      } else {
        setError(`Model metrics: ${results[1].reason.message}`);
      }
    }

    async function pollLiveTwin() {
      liveRequest = new AbortController();
      try {
        const current = await requestJson("/api/building/current", {
          signal: liveRequest.signal,
        });
        if (active) {
          setIsLive(true);
          setBuilding((previous) =>
            previous?.last_updated === current.last_updated ? previous : current,
          );
          setLiveReadings((previous) => {
            if (previous.at(-1)?.timestamp === current.current.timestamp) {
              return previous;
            }
            return [
              ...previous,
              {
                ...current.current,
                predicted_energy_consumption:
                  current.predicted_energy_consumption,
              },
            ].slice(-60);
          });
          setScenario((previous) => previous || { ...current.current });
          setError((previous) =>
            previous.startsWith("Live twin:") ? "" : previous,
          );
        }
      } catch (requestError) {
        if (active && requestError.name !== "AbortError") {
          setIsLive(false);
          setError(`Live twin: ${requestError.message}`);
        }
      } finally {
        if (active) {
          pollTimer = window.setTimeout(pollLiveTwin, 1000);
        }
      }
    }

    loadDashboard();
    pollLiveTwin();
    return () => {
      active = false;
      window.clearTimeout(pollTimer);
      liveRequest?.abort();
    };
  }, []);

  const hourlyEnergy = useMemo(() => {
    const buckets = new Map();
    history.forEach((row) => {
      const hour = Number(row.Hour);
      if (!buckets.has(hour)) buckets.set(hour, { hour, total: 0, count: 0 });
      const bucket = buckets.get(hour);
      bucket.total += Number(row.Total_Energy_Consumption);
      bucket.count += 1;
    });
    return [...buckets.values()]
      .map((bucket) => ({ hour: `${String(bucket.hour).padStart(2, "0")}:00`, energy: bucket.total / bucket.count }))
      .sort((a, b) => a.hour.localeCompare(b.hour));
  }, [history]);
  const loadAverages = useMemo(
    () =>
      [
        ["Lighting", "Lighting_Load"],
        ["HVAC", "HVAC_Load"],
        ["Computers", "Computer_Load"],
        ["Other", "Other_Load"],
      ].map(([name, key]) => ({
        name,
        load: history.length
          ? history.reduce((total, row) => total + Number(row[key]), 0) / history.length
          : 0,
      })),
    [history],
  );

  async function runScenario(event) {
    event.preventDefault();
    if (!scenario) return;
    setLoadingScenario(true);
    setError("");
    try {
      const result = await requestJson("/api/energy/simulate", {
        method: "POST",
        body: JSON.stringify(
          Object.fromEntries(SENSOR_FIELDS.map(({ key }) => [key, Number(scenario[key])])),
        ),
      });
      setScenarioResult(result);
    } catch (requestError) {
      setError(`What-if simulation: ${requestError.message}`);
    } finally {
      setLoadingScenario(false);
    }
  }

  function updateScenario(key, value) {
    setScenario((previous) => ({ ...previous, [key]: Number(value) }));
    setScenarioResult(null);
  }

  const live = building?.current;
  const modelRows = metrics?.models || [];
  const historicalChart = history.map((row) => ({
    date: row.Date.slice(5),
    energy: Number(row.Total_Energy_Consumption),
    occupancy: Number(row.Occupancy),
    temperature: Number(row.Temperature),
  }));
  const liveChart = liveReadings.map((reading) => ({
    time: new Date(reading.timestamp).toLocaleTimeString(),
    occupancy: Number(reading.Occupancy),
    temperature: Number(reading.Temperature),
    humidity: Number(reading.Humidity),
    lighting: Number(reading.Lighting_Load),
    hvac: Number(reading.HVAC_Load),
    computers: Number(reading.Computer_Load),
    other: Number(reading.Other_Load),
    prediction: Number(reading.predicted_energy_consumption),
  }));
  const lastUpdated = live?.timestamp;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="Smart Building home">
          <span className="brand-mark">T</span>
          <span>twin<span className="brand-light">space</span></span>
        </a>
        <nav aria-label="Dashboard sections">
          <a href="#overview">Overview</a>
          <a href="#analytics">Analytics</a>
          <a href="#simulation">What-if lab</a>
        </nav>
        <div className="topbar-status">
          <span className={`status-dot ${isLive ? "" : "offline"}`} />
          {isLive ? "Simulation live" : "Waiting for simulator"}
        </div>
      </header>

      <section className="hero" id="overview">
        <div>
          <p className="eyebrow">SMART BUILDING / ENERGY INTELLIGENCE</p>
          <h1>Building energy,<br /><span>in a living view.</span></h1>
          <p className="hero-copy">
            A machine-learning digital twin for understanding and exploring
            building energy performance.
          </p>
          <div className="source-note"><span className="sim-label">SIMULATED REAL-TIME DATA</span> No physical IoT sensors are connected.</div>
        </div>
        <div className="hero-graphic" aria-hidden="true">
          <div className="orb orb-one" /><div className="orb orb-two" />
          <div className="building-illustration">
            <div className="roof-line" />
            <div className="building-floor"><i /><i /><i /><b>HVAC</b></div>
            <div className="building-floor"><i /><i /><i /><b>LIGHT</b></div>
            <div className="building-floor"><i /><i /><i /><b>IT</b></div>
            <div className="building-base" />
          </div>
          <div className="graphic-caption">LIVE BUILDING MODEL <span>01 / 01</span></div>
        </div>
      </section>

      {error && <div className="error-banner" role="alert">{error}</div>}

      <section className="section-block">
        <div className="section-title">
          <div><p className="eyebrow">01 / CURRENT CONDITIONS</p><h2>Building overview</h2></div>
          <div className="live-meta">
            <span className={`live-pill ${isLive ? "" : "offline"}`}>
              <i /> {isLive ? "LIVE · SIMULATED" : "OFFLINE · SIMULATED"}
            </span>
            <time className="exact-timestamp" dateTime={lastUpdated || undefined}>
              {lastUpdated || "Waiting for first reading"}
            </time>
          </div>
        </div>
        <div className="overview-layout">
          <div className="metric-grid">
            <MetricCard label="Occupancy" value={live?.Occupancy} unit="% occupancy" accent="mint" />
            <MetricCard label="Temperature" value={live?.Temperature} unit="°C" accent="blue" />
            <MetricCard label="Humidity" value={live?.Humidity} unit="%" accent="lavender" />
            <MetricCard label="Lighting load" value={live?.Lighting_Load} unit="dataset units · simulated" />
            <MetricCard label="HVAC load" value={live?.HVAC_Load} unit="dataset units · simulated" />
            <MetricCard label="Computer load" value={live?.Computer_Load} unit="dataset units · simulated" />
            <MetricCard label="Other load" value={live?.Other_Load} unit="dataset units · simulated" />
            <MetricCard label="Predicted energy" value={building?.predicted_energy_consumption} unit="model estimate" accent="energy" />
          </div>
          <aside className="panel building-status">
            <div className="status-head"><span className="eyebrow">TWIN STATUS</span><span className="status-badge">● ACTIVE</span></div>
            <div className="status-id"><span>BUILDING ID</span><strong>{building?.building_id || "—"}</strong></div>
            <div className="status-state"><span className="state-icon">⌂</span><div><span>OPERATING STATE</span><strong>{live?.operating_state || "Awaiting readings"}</strong></div></div>
            <div className="component-list">
              <div><span className="component-icon light-icon">◉</span><span>Lighting system</span><b>{formatNumber(live?.Lighting_Load)} load units</b></div>
              <div><span className="component-icon hvac-icon">⌁</span><span>HVAC system</span><b>{formatNumber(live?.HVAC_Load)} load units</b></div>
              <div><span className="component-icon it-icon">▦</span><span>Computer systems</span><b>{formatNumber(live?.Computer_Load)} load units</b></div>
            </div>
            <p className="source-footnote">All live readings are generated by the local simulator.</p>
          </aside>
        </div>
        <div className="live-chart-grid">
          <ChartPanel title="Live load profile" subtitle="Latest simulated occupancy and equipment readings · 1-second updates">
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={liveChart}>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="time" tick={{ fill: "#82949e", fontSize: 10 }} minTickGap={30} />
                <YAxis domain={[0, 100]} tick={{ fill: "#82949e", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Legend />
                <Line isAnimationActive animationDuration={800} type="monotone" dataKey="occupancy" name="Occupancy %" stroke="#79e3bc" dot={false} strokeWidth={2} />
                <Line isAnimationActive animationDuration={800} type="monotone" dataKey="lighting" name="Lighting" stroke="#f4cf77" dot={false} />
                <Line isAnimationActive animationDuration={800} type="monotone" dataKey="hvac" name="HVAC" stroke="#8eb8ff" dot={false} />
                <Line isAnimationActive animationDuration={800} type="monotone" dataKey="computers" name="Computer" stroke="#c1a8ff" dot={false} />
                <Line isAnimationActive animationDuration={800} type="monotone" dataKey="other" name="Other" stroke="#ffad91" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Live predicted energy" subtitle="Saved ML model prediction for each simulated reading">
            <ResponsiveContainer width="100%" height={230}>
              <AreaChart data={liveChart}>
                <defs><linearGradient id="liveEnergyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#79e3bc" stopOpacity={0.28} /><stop offset="95%" stopColor="#79e3bc" stopOpacity={0} /></linearGradient></defs>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="time" tick={{ fill: "#82949e", fontSize: 10 }} minTickGap={30} />
                <YAxis tick={{ fill: "#82949e", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Area isAnimationActive animationDuration={800} type="monotone" dataKey="prediction" name="ML predicted energy" stroke="#79e3bc" fill="url(#liveEnergyFill)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Live environment" subtitle="Temperature and humidity from simulated readings">
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={liveChart}>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="time" tick={{ fill: "#82949e", fontSize: 10 }} minTickGap={30} />
                <YAxis yAxisId="humidity" domain={[25, 95]} tick={{ fill: "#82949e", fontSize: 10 }} />
                <YAxis yAxisId="temperature" orientation="right" domain={[18, 42]} tick={{ fill: "#82949e", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Legend />
                <Line isAnimationActive animationDuration={800} yAxisId="humidity" type="monotone" dataKey="humidity" name="Humidity %" stroke="#8eb8ff" dot={false} strokeWidth={2} />
                <Line isAnimationActive animationDuration={800} yAxisId="temperature" type="monotone" dataKey="temperature" name="Temperature °C" stroke="#ffad91" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
        </div>
      </section>

      <section className="section-block" id="analytics">
        <div className="section-title">
          <div><p className="eyebrow">02 / HISTORICAL CSV ANALYTICS</p><h2>Energy analytics</h2></div>
          <span className="source-chip">SOURCE · HISTORICAL CSV</span>
        </div>
        <div className="chart-grid">
          <ChartPanel title="Energy trend" subtitle="Historical target values from the provided CSV" className="wide-chart">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={historicalChart}>
                <defs><linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#79e3bc" stopOpacity={0.28} /><stop offset="95%" stopColor="#79e3bc" stopOpacity={0} /></linearGradient></defs>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "#82949e", fontSize: 11 }} minTickGap={28} />
                <YAxis tick={{ fill: "#82949e", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Area type="monotone" dataKey="energy" name="Historical energy" stroke="#79e3bc" fill="url(#energyFill)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Average energy by hour" subtitle="Time-based pattern across historical records">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={hourlyEnergy}>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="hour" tick={{ fill: "#82949e", fontSize: 10 }} interval={2} />
                <YAxis tick={{ fill: "#82949e", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Line type="monotone" dataKey="energy" name="Average energy" stroke="#8eb8ff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Occupancy & energy" subtitle="Historical occupancy compared with target consumption">
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" />
                <XAxis type="number" dataKey="occupancy" name="Occupancy" unit="%" tick={{ fill: "#82949e", fontSize: 11 }} />
                <YAxis type="number" dataKey="energy" name="Energy" tick={{ fill: "#82949e", fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Scatter name="Historical records" data={historicalChart} fill="#79e3bc" fillOpacity={0.65} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Average individual loads" subtitle="Historical load levels in dataset units">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={loadAverages} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#82949e", fontSize: 10 }} />
                <YAxis tick={{ fill: "#82949e", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Bar dataKey="load" name="Average load" fill="#c1a8ff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Model comparison" subtitle={`Held-out test results · selected by lowest MAE${metrics ? ` · Best: ${metrics.best_model}` : ""}`}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={modelRows} layout="vertical" margin={{ left: 10, right: 16 }}>
                <CartesianGrid stroke="#263641" strokeDasharray="3 5" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#82949e", fontSize: 11 }} />
                <YAxis type="category" dataKey="model" width={115} tick={{ fill: "#b8c5cb", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#14232d", border: "1px solid #314550", borderRadius: 8 }} />
                <Legend />
                <Bar dataKey="mae" name="MAE" fill="#79e3bc" radius={[0, 4, 4, 0]} />
                <Bar dataKey="rmse" name="RMSE" fill="#8eb8ff" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            {metrics && <div className="model-footnote">Test rows: {metrics.test_rows} · Split: chronological 80/20 · R² for selected model: {formatNumber(metrics.best_metrics.r2, 3)}</div>}
            {modelRows.length > 0 && (
              <div className="table-scroll">
                <table className="metrics-table">
                  <thead><tr><th>Model</th><th>MAE</th><th>MSE</th><th>RMSE</th><th>R²</th></tr></thead>
                  <tbody>
                    {modelRows.map((row) => (
                      <tr key={row.model}>
                        <td>{row.model}</td><td>{formatNumber(row.mae, 2)}</td>
                        <td>{formatNumber(row.mse, 2)}</td><td>{formatNumber(row.rmse, 2)}</td>
                        <td>{formatNumber(row.r2, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </ChartPanel>
        </div>
      </section>

      <section className="section-block" id="simulation">
        <div className="section-title">
          <div><p className="eyebrow">03 / MACHINE-LEARNING SANDBOX</p><h2>What-if simulation</h2></div>
          <span className="source-chip model-chip">SOURCE · SAVED ML MODEL</span>
        </div>
        <div className="simulation-layout">
          <form className="panel scenario-form" onSubmit={runScenario}>
            <div className="panel-heading">
              <div><h2>Adjust building conditions</h2><p>Change inputs and compare predictions from the trained model.</p></div>
              <span className="scenario-icon">↗</span>
            </div>
            <div className="slider-grid">
              {SENSOR_FIELDS.map((field) => (
                <label className="slider-control" key={field.key}>
                  <span><span>{field.label}</span><b>{formatNumber(scenario?.[field.key])} <small>{field.unit}</small></b></span>
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={field.key === "Occupancy" ? 1 : 0.1}
                    value={scenario?.[field.key] ?? field.min}
                    onChange={(event) => updateScenario(field.key, event.target.value)}
                    disabled={!scenario}
                  />
                </label>
              ))}
            </div>
            <button type="submit" className="run-button" disabled={!scenario || loadingScenario}>
              {loadingScenario ? "Running saved model…" : "Run what-if simulation"}
              <span>→</span>
            </button>
            <p className="simulation-disclaimer">Results are ML estimates from the current simulated baseline; they are not metered energy or guaranteed savings.</p>
          </form>
          <aside className="panel result-panel">
            <div className="eyebrow">PREDICTION COMPARISON</div>
            {scenarioResult ? (
              <>
                <div className="result-change">
                  <span>Predicted change</span>
                  <strong className={scenarioResult.energy_difference > 0 ? "change-up" : "change-down"}>
                    {scenarioResult.energy_difference > 0 ? "+" : ""}{formatNumber(scenarioResult.energy_difference)}
                  </strong>
                  <small>energy units · {scenarioResult.percentage_change > 0 ? "+" : ""}{formatNumber(scenarioResult.percentage_change)}%</small>
                </div>
                <div className="result-pair">
                  <div><span>Baseline prediction</span><strong>{formatNumber(scenarioResult.baseline_prediction)}</strong></div>
                  <div><span>Scenario prediction</span><strong>{formatNumber(scenarioResult.scenario_prediction)}</strong></div>
                </div>
                <div className="result-bar"><i style={{ width: `${Math.min(100, Math.abs(scenarioResult.percentage_change))}%` }} /></div>
                <p className="model-footnote">Both values were produced by {scenarioResult.model}.</p>
              </>
            ) : (
              <div className="result-empty">
                <span className="empty-spark">✳</span>
                <strong>Your scenario, modeled.</strong>
                <p>Adjust one or more building inputs, then run the saved ML model to see the predicted change.</p>
              </div>
            )}
          </aside>
        </div>
      </section>

      <footer>
        <div className="brand"><span className="brand-mark">T</span><span>twin<span className="brand-light">space</span></span></div>
        <span>Academic demonstration · Simulated telemetry · Not a building control system</span>
      </footer>
    </main>
  );
}

export default App;
