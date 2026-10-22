import { useEffect, useState } from 'react'
import { BarChart, Bar, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getModelComparison, getMetrics, getModelInfo, getClassReport, getConfusionMatrix } from '../api'

const pct = (v) => v != null ? `${(v * 100).toFixed(2)}%` : 'n/a'

export default function ModelPerformance() {
  const [comp, setComp] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [info, setInfo] = useState(null)
  const [report, setReport] = useState(null)
  const [cm, setCm] = useState(null)
  const [tab, setTab] = useState('comparison')

  useEffect(() => {
    getModelComparison().then(setComp).catch(() => {})
    getMetrics().then(setMetrics).catch(() => {})
    getModelInfo().then(setInfo).catch(() => {})
    getClassReport().then(setReport).catch(() => {})
    getConfusionMatrix().then(setCm).catch(() => {})
  }, [])

  if (!comp && !metrics) return <div className="loading"><div className="spinner" /> <p>Loading…</p></div>

  const best = comp?.best_per_algorithm || []
  const chartData = best.map(r => ({
    model: r.model,
    Accuracy: r.accuracy,
    'Macro F1': r.macro_f1,
    'Macro Precision': r.macro_precision,
    'Macro Recall': r.macro_recall,
  }))

  const tabs = ['comparison', 'report', 'confusion', 'info']

  return (
    <div>
      <h2>Model Performance</h2>

      {/* ---- summary stats ---- */}
      {metrics && (
        <div className="grid">
          <div className="card stat"><span className="muted">Best Model</span><b>{metrics.model}</b></div>
          <div className="card stat"><span className="muted">Test Accuracy</span><b>{pct(metrics.accuracy)}</b></div>
          <div className="card stat"><span className="muted">Test Macro F1</span><b>{pct(metrics.macro_f1)}</b></div>
          <div className="card stat"><span className="muted">Weighted F1</span><b>{pct(metrics.weighted_f1)}</b></div>
          <div className="card stat"><span className="muted">Train Time</span><b>{metrics.train_seconds}s</b></div>
          <div className="card stat"><span className="muted">Test Size</span><b>{metrics.test_size?.toLocaleString()}</b></div>
        </div>
      )}

      {/* ---- tabs ---- */}
      <div className="tabs">
        {tabs.map(t => <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t === 'comparison' ? 'Comparison' : t === 'report' ? 'Classification Report' : t === 'confusion' ? 'Confusion Matrix' : 'Model Info'}</button>)}
      </div>

      {/* ---- comparison ---- */}
      {tab === 'comparison' && comp && (
        <div className="card">
          <h3>Validation Performance by Algorithm</h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" /><YAxis domain={[0, 1]} /><Tooltip formatter={v => v.toFixed(4)} /><Legend />
              <Bar dataKey="Accuracy" fill="#14213d" /><Bar dataKey="Macro F1" fill="#3b82f6" /><Bar dataKey="Macro Precision" fill="#8b5cf6" /><Bar dataKey="Macro Recall" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
          <table style={{ marginTop: 16 }}>
            <thead><tr><th>Model</th><th>Params</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th><th>Precision</th><th>Recall</th><th>Train (s)</th><th>Best?</th></tr></thead>
            <tbody>{best.map(r => (
              <tr key={r.model} style={r.model === comp.selected?.model ? { background: '#eff6ff' } : {}}>
                <td><strong>{r.model}</strong></td>
                <td className="muted">{JSON.stringify(r.params)}</td>
                <td>{pct(r.accuracy)}</td><td>{pct(r.macro_f1)}</td><td>{pct(r.weighted_f1)}</td>
                <td>{pct(r.macro_precision)}</td><td>{pct(r.macro_recall)}</td><td>{r.train_seconds}</td>
                <td>{r.model === comp.selected?.model ? '✅' : ''}</td>
              </tr>
            ))}</tbody>
          </table>
          <p className="muted" style={{ marginTop: 12 }}>Selection: {comp.selection_rule}</p>
          <p className="muted">Macro F1 weights all classes equally. Weighted F1 accounts for class support.</p>
        </div>
      )}

      {/* ---- classification report ---- */}
      {tab === 'report' && report && (
        <div className="card">
          <h3>Per-Category Performance (test set)</h3>
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table>
              <thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(report).filter(([k]) => !['accuracy','macro avg','weighted avg'].includes(k)).map(([cat, m]) => (
                <tr key={cat}>
                  <td>{cat}</td><td>{pct(m.precision)}</td><td>{pct(m.recall)}</td><td>{pct(m['f1-score'])}</td><td>{m.support}</td>
                </tr>
              ))}
              {report['macro avg'] && <tr style={{ fontWeight: 600, borderTop: '2px solid var(--border)' }}><td>Macro Average</td><td>{pct(report['macro avg'].precision)}</td><td>{pct(report['macro avg'].recall)}</td><td>{pct(report['macro avg']['f1-score'])}</td><td>{report['macro avg'].support}</td></tr>}
              {report['weighted avg'] && <tr style={{ fontWeight: 600 }}><td>Weighted Average</td><td>{pct(report['weighted avg'].precision)}</td><td>{pct(report['weighted avg'].recall)}</td><td>{pct(report['weighted avg']['f1-score'])}</td><td>{report['weighted avg'].support}</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ---- confusion matrix ---- */}
      {tab === 'confusion' && cm && (
        <div className="card">
          <h3>Confusion Matrix (test set, row-normalised)</h3>
          <p className="muted" style={{ marginBottom: 8 }}>Rows = true category, Columns = predicted category. Colour intensity = row-normalised accuracy.</p>
          <div style={{ overflow: 'auto', maxHeight: 600 }}>
            <table className="cm-table">
              <thead><tr><th></th>{cm.labels.map(l => <th key={l} style={{ fontSize: 9, transform: 'rotate(-45deg)', whiteSpace: 'nowrap' }}>{l}</th>)}</tr></thead>
              <tbody>{cm.matrix.map((row, i) => {
                const rowSum = row.reduce((a, b) => a + b, 0)
                return (
                  <tr key={i}>
                    <td style={{ fontSize: 10, fontWeight: 600, whiteSpace: 'nowrap' }}>{cm.labels[i]}</td>
                    {row.map((v, j) => {
                      const norm = rowSum > 0 ? v / rowSum : 0
                      const bg = `rgba(59,130,246,${norm * 0.8})`
                      const color = norm > 0.5 ? '#fff' : '#14213d'
                      return <td key={j} className="cm-cell" style={{ background: bg, color, fontSize: 10 }}>{v > 0 ? v : ''}</td>
                    })}
                  </tr>
                )
              })}</tbody>
            </table>
          </div>
        </div>
      )}

      {/* ---- model info ---- */}
      {tab === 'info' && info && (
        <div className="card">
          <h3>Selected Model Details</h3>
          <table>
            <tbody>
              <tr><td><strong>Model</strong></td><td>{info.model_name}</td></tr>
              <tr><td><strong>Version</strong></td><td>{info.model_version}</td></tr>
              <tr><td><strong>Parameters</strong></td><td><code>{JSON.stringify(info.params)}</code></td></tr>
              <tr><td><strong>Score Type</strong></td><td>{info.score_type}</td></tr>
              <tr><td><strong>Input Mode</strong></td><td>{info.input_mode}</td></tr>
              <tr><td><strong>Categories</strong></td><td>{info.n_categories}</td></tr>
              <tr><td><strong>Trained At</strong></td><td>{info.trained_at_utc}</td></tr>
              {info.split && <>
                <tr><td><strong>Train Size</strong></td><td>{info.split.train_size?.toLocaleString()}</td></tr>
                <tr><td><strong>Validation Size</strong></td><td>{info.split.validation_size?.toLocaleString()}</td></tr>
                <tr><td><strong>Test Size</strong></td><td>{info.split.test_size?.toLocaleString()}</td></tr>
                <tr><td><strong>Stratified</strong></td><td>{info.split.stratified ? 'Yes' : 'No'}</td></tr>
                <tr><td><strong>Evaluated On</strong></td><td>{info.split.evaluated_on}</td></tr>
              </>}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
