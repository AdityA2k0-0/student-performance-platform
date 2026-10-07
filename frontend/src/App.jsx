import { useEffect, useState } from "react";

const FIELDS = [
  { key: "study_hours", label: "Study hours / day", min: 0, max: 8, step: 0.5 },
  { key: "attendance", label: "Attendance %", min: 40, max: 100, step: 1 },
  { key: "previous_marks", label: "Previous marks", min: 30, max: 100, step: 1 },
  { key: "assignment_score", label: "Assignment score", min: 30, max: 100, step: 1 },
  { key: "sleep_hours", label: "Sleep hours", min: 4, max: 9, step: 0.5 },
];
const DEFAULTS = { study_hours: 3, attendance: 75, previous_marks: 65, assignment_score: 70, sleep_hours: 7 };

const API_URL = "https://student-performance-platform-8xzj.onrender.com";

const api = (path, opts) =>
  fetch(API_URL + "/api" + path, opts).then((r) => r.json());

function Bars({ data, labelKey, valueKey, signed }) {
  const max = Math.max(...data.map((d) => Math.abs(d[valueKey])), 0.01);
  return data.map((d) => (
    <div className="bar-row" key={d[labelKey]}>
      <span>{d[labelKey]}</span>
      <div className={"bar " + (signed ? (d[valueKey] < 0 ? "neg" : "pos") : "")}
           style={{ width: (Math.abs(d[valueKey]) / max) * 160 + 4 }} />
      <span style={{ width: 50 }}>{d[valueKey]}</span>
    </div>
  ));
}

function Predict() {
  const [vals, setVals] = useState(DEFAULTS);
  const [res, setRes] = useState(null);
  const [history, setHistory] = useState([]);

  // live "what-if": re-predict whenever a slider moves (not saved to history)
  useEffect(() => {
    const t = setTimeout(() => {
      api("/predict?save=false", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(vals) }).then(setRes);
    }, 200);
    return () => clearTimeout(t);
  }, [vals]);

  const save = () =>
    api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(vals) })
      .then(() => api("/history").then(setHistory));
  useEffect(() => { api("/history").then(setHistory); }, []);

  return (
    <div className="grid">
      <div className="card">
        <h2>Student details (what-if simulator)</h2>
        {FIELDS.map((f) => (
          <div className="slider" key={f.key}>
            <label><span>{f.label}</span><b>{vals[f.key]}</b></label>
            <input type="range" {...f} value={vals[f.key]} onChange={(e) => setVals({ ...vals, [f.key]: +e.target.value })} />
          </div>
        ))}
        <button onClick={save}>Save prediction to history</button>
      </div>
      {res && (
        <div className="card">
          <h2>Prediction</h2>
          <div className="big">{res.score}</div>
          <p>Grade <b>{res.grade}</b> &nbsp;
            <span className={"pill " + (res.at_risk ? "risk" : "ok")}>{res.at_risk ? "At risk" : "On track"}</span></p>
          <h2>Why? (impact on score)</h2>
          <Bars data={res.explanation} labelKey="feature" valueKey="impact" signed />
          <h2>Suggestions</h2>
          <ul>{res.suggestions.map((s) => <li key={s}>{s}</li>)}</ul>
        </div>
      )}
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h2>Prediction history</h2>
        <table>
          <thead><tr><th>Time</th><th>Study hrs</th><th>Attendance</th><th>Prev marks</th><th>Score</th><th>Grade</th></tr></thead>
          <tbody>{history.map((h, i) => (
            <tr key={i}><td>{h.time}</td><td>{h.inputs.study_hours}</td><td>{h.inputs.attendance}</td>
              <td>{h.inputs.previous_marks}</td><td>{h.score}</td><td>{h.grade}</td></tr>))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Dashboard() {
  const [d, setD] = useState(null);
  useEffect(() => { api("/students").then(setD); }, []);
  if (!d) return <p>Loading...</p>;
  const dist = Object.entries(d.distribution).map(([g, n]) => ({ grade: "Grade " + g, count: n }));
  const atRisk = d.students.filter((s) => s.at_risk);
  return (
    <div className="grid">
      <div className="card">
        <h2>Class overview</h2>
        <div className="stat">
          <div><b>{d.students.length}</b>Students</div>
          <div><b>{d.average}</b>Avg predicted score</div>
          <div><b>{atRisk.length}</b>At risk</div>
        </div>
        <h2 style={{ marginTop: 20 }}>Grade distribution</h2>
        <Bars data={dist} labelKey="grade" valueKey="count" />
      </div>
      <div className="card">
        <h2>Students</h2>
        <table>
          <thead><tr><th>Name</th><th>Attendance</th><th>Score</th><th>Grade</th><th>Status</th></tr></thead>
          <tbody>{d.students.map((s) => (
            <tr key={s.id}><td>{s.name}</td><td>{s.attendance}%</td><td>{s.score}</td><td>{s.grade}</td>
              <td><span className={"pill " + (s.at_risk ? "risk" : "ok")}>{s.at_risk ? "At risk" : "OK"}</span></td></tr>))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Model() {
  const [m, setM] = useState(null);
  useEffect(() => { api("/model-info").then(setM); }, []);
  if (!m) return <p>Loading...</p>;
  return (
    <div className="grid">
      <div className="card">
        <h2>Model performance ({m.metrics.algorithm})</h2>
        <div className="stat">
          <div><b>{m.metrics.r2}</b>R² (regression)</div>
          <div><b>{m.metrics.mae}</b>MAE (marks)</div>
          <div><b>{(m.metrics.accuracy * 100).toFixed(1)}%</b>Grade accuracy</div>
        </div>
        <p style={{ fontSize: 13, color: "#6b7280" }}>Trained on {m.metrics.train_rows} rows, tested on {m.metrics.test_rows} (synthetic data).</p>
      </div>
      <div className="card">
        <h2>Feature importance</h2>
        <Bars data={m.importance} labelKey="feature" valueKey="importance" />
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("predict");
  const tabs = { predict: ["Predict", Predict], dashboard: ["Teacher Dashboard", Dashboard], model: ["Model Info", Model] };
  const Page = tabs[tab][1];
  return (
    <>
      <header><h1>Student Performance Analytics Platform</h1></header>
      <nav>{Object.entries(tabs).map(([k, [label]]) => (
        <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>))}
      </nav>
      <main><Page /></main>
    </>
  );
}
