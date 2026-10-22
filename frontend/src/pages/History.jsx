import { useCallback, useEffect, useState } from 'react'
import { api, errorMessage } from '../api'

const PAGE_SIZE = 10

export default function History() {
  const [data, setData] = useState(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [order, setOrder] = useState('newest')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      const r = await api.get('/history', { params: { page, page_size: PAGE_SIZE, search: search || undefined, order } })
      setData(r.data); setError('')
    } catch (err) { setError(errorMessage(err)) }
  }, [page, search, order])

  useEffect(() => { load() }, [load])

  async function remove(id) {
    try { await api.delete(`/history/${id}`); load() } catch (err) { setError(errorMessage(err)) }
  }
  async function clearAll() {
    if (!window.confirm('Delete ALL prediction history? This cannot be undone.')) return
    try { await api.delete('/history'); setPage(1); load() } catch (err) { setError(errorMessage(err)) }
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1

  return (
    <div>
      <h2>Prediction History</h2>
      <div className="card row">
        <input placeholder="Search headlines…" value={search} onChange={(e) => { setSearch(e.target.value); setPage(1) }} />
        <select value={order} onChange={(e) => setOrder(e.target.value)}>
          <option value="newest">Newest first</option><option value="oldest">Oldest first</option>
        </select>
        <button className="danger" onClick={clearAll} disabled={!data || data.total === 0}>Clear history</button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {data && data.total === 0 && <div className="card"><p>No predictions yet. Try the Predict page first.</p></div>}
      {data && data.total > 0 && (
        <div className="card">
          <table>
            <thead><tr><th>Headline</th><th>Predicted</th><th>Sentiment</th><th>Model</th><th>Time (UTC)</th><th></th></tr></thead>
            <tbody>{data.items.map((it) => (
              <tr key={it.id}>
                <td style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.headline}</td>
                <td><span className="badge badge-cat">{it.predicted_category}</span></td>
                <td>{it.sentiment_label ? <SentBadge label={it.sentiment_label} /> : <span className="muted">—</span>}</td>
                <td className="muted">{it.model_version}</td>
                <td className="muted">{it.created_at?.slice(0, 19).replace('T', ' ')}</td>
                <td><button className="secondary" onClick={() => remove(it.id)}>Delete</button></td>
              </tr>))}</tbody>
          </table>
          <div className="row">
            <button className="secondary" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button>
            <span className="muted">Page {page} of {pages} ({data.total} records)</span>
            <button className="secondary" disabled={page >= pages} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        </div>
      )}
    </div>
  )
}

function SentBadge({ label }) {
  const cls = label === 'Positive' ? 'badge-positive' : label === 'Negative' ? 'badge-negative' : 'badge-neutral'
  return <span className={`badge ${cls}`}>{label}</span>
}
