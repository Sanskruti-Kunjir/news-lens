import { useEffect, useState } from 'react'
import { BarChart, Bar, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, PieChart, Pie, Cell } from 'recharts'
import { api, errorMessage } from '../api'

const pct = (v) => `${(v * 100).toFixed(1)}%`
const COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#14b8a6']

export default function Dashboard() {
  const [d, setD] = useState(null)
  const [hist, setHist] = useState(null)

  useEffect(() => {
    Promise.allSettled([
      api.get('/model-info'), api.get('/analytics/metrics'),
      api.get('/analytics/model-comparison'), api.get('/dataset/summary'),
      api.get('/analytics/prediction-history'),
    ]).then(([info, metrics, comparison, dataset, predHist]) =>
      setD({
        info: info.value?.data, metrics: metrics.value?.data,
        comparison: comparison.value?.data, dataset: dataset.value?.data,
      })
    )
    api.get('/history', { params: { page: 1, page_size: 5 } })
      .then(r => setHist(r.data)).catch(() => {})
  }, [])

  if (!d) return <div className="loading"><div className="spinner" /> <p>Loading dashboard…</p></div>
  if (!d.metrics || !d.info) {
    return (
      <div className="card">
        <h2>Setup needed</h2>
        <p>No trained model was found, or the backend is not running. Run <code>python -m ml.train</code>, then start the backend with <code>python -m uvicorn backend.app.main:app --port 8000</code>.</p>
      </div>
    )
  }

  const compare = (d.comparison?.best_per_algorithm || []).map((r) => ({ model: r.model, Accuracy: r.accuracy, 'Macro F1': r.macro_f1, 'Weighted F1': r.weighted_f1 }))
  const cats = d.dataset ? Object.entries(d.dataset.category_counts).slice(0, 12).map(([name, count]) => ({ name, count })) : []
  const pieData = cats.slice(0, 8)

  return (
    <div>
      <h2>Dashboard</h2>

      {/* ---- stat cards ---- */}
      <div className="grid">
        <div className="card stat"><span className="muted">Dataset records</span><b>{d.dataset ? d.dataset.total_records.toLocaleString() : 'n/a'}</b></div>
        <div className="card stat"><span className="muted">Categories</span><b>{d.info.n_categories}</b></div>
        <div className="card stat"><span className="muted">Selected model</span><b>{d.info.model_name}</b></div>
        <div className="card stat"><span className="muted">Test accuracy</span><b>{pct(d.metrics.accuracy)}</b></div>
        <div className="card stat"><span className="muted">Test macro F1</span><b>{pct(d.metrics.macro_f1)}</b></div>
        <div className="card stat"><span className="muted">Weighted F1</span><b>{pct(d.metrics.weighted_f1)}</b></div>
      </div>
      {d.info.sample_size && <p className="muted">Note: this model was trained on a random sample of {d.info.sample_size.toLocaleString()} articles.</p>}

      {/* ---- model comparison ---- */}
      <div className="card">
        <h3>Model Comparison (validation set)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={compare}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" /><YAxis domain={[0, 1]} /><Tooltip formatter={(v) => v.toFixed(4)} /><Legend />
            <Bar dataKey="Accuracy" fill="#14213d" /><Bar dataKey="Macro F1" fill="#3b82f6" /><Bar dataKey="Weighted F1" fill="#8b5cf6" /></BarChart>
        </ResponsiveContainer>
        <p className="muted">Final test scores belong only to the selected model, retrained on train + validation and evaluated once on the test set.</p>
      </div>

      {/* ---- category distribution ---- */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        {cats.length > 0 && (
          <div className="card">
            <h3>Top Categories</h3>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={cats} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} /><Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {pieData.length > 0 && (
          <div className="card">
            <h3>Category Share</h3>
            <ResponsiveContainer width="100%" height={340}>
              <PieChart>
                <Pie data={pieData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={120} label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false} style={{ fontSize: 10 }}>
                  {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ---- recent predictions ---- */}
      {hist && hist.total > 0 && (
        <div className="card">
          <h3>Recent Predictions</h3>
          <table>
            <thead><tr><th>Headline</th><th>Category</th><th>Sentiment</th><th>Time</th></tr></thead>
            <tbody>{hist.items.map(it => (
              <tr key={it.id}>
                <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.headline}</td>
                <td><span className="badge badge-cat">{it.predicted_category}</span></td>
                <td>{it.sentiment_label ? <SentBadge label={it.sentiment_label} /> : <span className="muted">—</span>}</td>
                <td className="muted">{it.created_at?.slice(0, 16).replace('T', ' ')}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function SentBadge({ label }) {
  const cls = label === 'Positive' ? 'badge-positive' : label === 'Negative' ? 'badge-negative' : 'badge-neutral'
  return <span className={`badge ${cls}`}>{label}</span>
}
