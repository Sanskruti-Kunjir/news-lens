import { useEffect, useState } from 'react'
import { api, errorMessage } from '../api'

export default function Predict() {
  const [headline, setHeadline] = useState('')
  const [description, setDescription] = useState('')
  const [mode, setMode] = useState('combined')
  const [modes, setModes] = useState(['combined'])
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.get('/model-info').then((r) => {
      setModes(r.data.supported_input_modes)
      setMode(r.data.input_mode)
    }).catch(() => {})
  }, [])

  async function submit(e) {
    e.preventDefault()
    setError('')
    if (!headline.trim()) return setError('Please enter a headline.')
    setLoading(true)
    try {
      const r = await api.post('/predict', { headline, short_description: description, input_mode: mode })
      setResult(r.data)
    } catch (err) {
      setResult(null)
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  function clear() {
    setHeadline(''); setDescription(''); setResult(null); setError('')
  }

  const fmt = (c) => (result.score_type === 'probability' ? `${(c.score * 100).toFixed(1)}% (probability)` : `${c.score.toFixed(3)} (raw decision score, not a probability)`)

  return (
    <div>
      <h2>News Prediction</h2>
      <form className="card" onSubmit={submit}>
        <label>Headline *
          <input value={headline} onChange={(e) => setHeadline(e.target.value)} maxLength={500} placeholder="e.g. Senate passes new budget bill after late-night vote" />
        </label>
        <label>Short description (optional)
          <textarea value={description} onChange={(e) => setDescription(e.target.value)} maxLength={2000} rows={3} />
        </label>
        <label>Input strategy
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            {modes.map((m) => <option key={m} value={m}>{m === 'combined' ? 'Headline + description' : 'Headline only'}</option>)}
          </select>
        </label>
        <div className="row">
          <button type="submit" disabled={loading}>{loading ? 'Predicting…' : 'Predict'}</button>
          <button type="button" className="secondary" onClick={clear}>Clear</button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      {result && (
        <>
          {/* ---- Category prediction ---- */}
          <div className="card">
            <p className="muted">Predicted category</p>
            <p className="big">{result.predicted_category}</p>
            <h3>Top Candidates</h3>
            <ol>{result.candidates.map((c) => <li key={c.category}><strong>{c.category}</strong> — {fmt(c)}</li>)}</ol>
            <p className="muted" style={{ marginTop: 12 }}>Model: {result.model_name} ({result.model_version}) · {result.latency_ms} ms</p>
          </div>

          {/* ---- Sentiment panel ---- */}
          {result.sentiment && (
            <div className="card">
              <h3>Sentiment Analysis</h3>
              <div className="sentiment-panel">
                <SentBadge label={result.sentiment.sentiment} />
                <span className="muted">Compound: {result.sentiment.scores.compound >= 0 ? '+' : ''}{result.sentiment.scores.compound}</span>
              </div>
              <div style={{ display: 'flex', gap: 16, margin: '12px 0' }}>
                <ScoreBar label="Positive" value={result.sentiment.scores.positive} cls="pos" />
                <ScoreBar label="Neutral" value={result.sentiment.scores.neutral} cls="neu" />
                <ScoreBar label="Negative" value={result.sentiment.scores.negative} cls="neg" />
              </div>
              <p className="muted">{result.sentiment.note}</p>
              <details style={{ marginTop: 8 }}>
                <summary className="muted" style={{ cursor: 'pointer' }}>Limitations</summary>
                <ul className="muted">{result.sentiment.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
              </details>
            </div>
          )}

          {/* ---- Explainable AI panel ---- */}
          <div className="card">
            <h3>Explainable AI — Influential Words</h3>
            {result.explanation.terms.length === 0 ? (
              <p className="muted">{result.explanation.note || 'No explanation available for this model.'}</p>
            ) : (
              <>
                <p className="muted" style={{ marginBottom: 12 }}>Method: {result.explanation.method}</p>
                {result.explanation.terms.map((t) => {
                  const maxC = Math.max(...result.explanation.terms.map(x => Math.abs(x.contribution)))
                  const pctW = maxC > 0 ? Math.abs(t.contribution) / maxC * 100 : 0
                  return (
                    <div className="contrib-bar" key={t.term}>
                      <code>{t.term}</code>
                      <div className="contrib-track">
                        <div className={`contrib-fill ${t.direction}`} style={{ width: `${pctW}%` }} />
                      </div>
                      <span style={{ minWidth: 60, fontSize: 12, color: t.direction === 'supports' ? '#10b981' : '#ef4444' }}>
                        {t.contribution > 0 ? '+' : ''}{t.contribution}
                      </span>
                    </div>
                  )
                })}
                <p className="muted" style={{ marginTop: 12 }}>{result.explanation.note}</p>
                <p className="muted" style={{ fontStyle: 'italic' }}>⚠ Influential words indicate model behavior, not factual correctness. The prediction may still be wrong.</p>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function SentBadge({ label }) {
  const cls = label === 'Positive' ? 'badge-positive' : label === 'Negative' ? 'badge-negative' : 'badge-neutral'
  return <span className={`badge ${cls}`} style={{ fontSize: 16, padding: '6px 16px' }}>{label}</span>
}

function ScoreBar({ label, value, cls }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span>{label}</span><span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="sentiment-bar">
        <div className={`sentiment-fill ${cls}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}
