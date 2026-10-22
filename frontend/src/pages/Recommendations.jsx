import { useCallback, useEffect, useState } from 'react'
import { getRecommendations, getPreferences, putPreferences, getBookmarks, addBookmark, removeBookmark } from '../api'

export default function Recommendations() {
  const [recs, setRecs] = useState(null)
  const [prefs, setPrefs] = useState([])
  const [allCats, setAllCats] = useState([])
  const [bookmarks, setBookmarks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('recs')

  const loadAll = useCallback(async () => {
    setLoading(true)
    try {
      const [recData, prefData, bkData] = await Promise.all([
        getRecommendations(24), getPreferences(), getBookmarks(),
      ])
      setRecs(recData.items || [])
      setAllCats(recData.available_categories || [])
      setPrefs(prefData.preferred_categories || [])
      setBookmarks(bkData.items || [])
      setError('')
    } catch (err) {
      setError(err?.response?.data?.detail || 'Could not load recommendations. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  function togglePref(cat) {
    setPrefs(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])
  }

  async function savePrefs() {
    try {
      await putPreferences(prefs)
      loadAll()
    } catch (err) { setError('Failed to save preferences') }
  }

  async function resetPrefs() {
    setPrefs([])
    try { await putPreferences([]); loadAll() } catch {}
  }

  async function doBookmark(article) {
    try {
      await addBookmark({ article_id: article.article_id, headline: article.headline, category: article.category })
      loadAll()
    } catch {}
  }

  async function doUnbookmark(id) {
    try { await removeBookmark(id); loadAll() } catch {}
  }

  if (loading) return <div className="loading"><div className="spinner" /><p>Loading recommendations…</p></div>

  return (
    <div>
      <h2>Recommendations</h2>
      {error && <p className="error">{error}</p>}

      <div className="tabs">
        <button className={`tab ${tab === 'recs' ? 'active' : ''}`} onClick={() => setTab('recs')}>Recommendations</button>
        <button className={`tab ${tab === 'prefs' ? 'active' : ''}`} onClick={() => setTab('prefs')}>Preferences ({prefs.length})</button>
        <button className={`tab ${tab === 'bookmarks' ? 'active' : ''}`} onClick={() => setTab('bookmarks')}>Bookmarks ({bookmarks.length})</button>
      </div>

      {/* ---- Preferences tab ---- */}
      {tab === 'prefs' && (
        <div className="card">
          <h3>Select Your Favourite Categories</h3>
          <p className="muted" style={{ marginBottom: 12 }}>Choose categories you're interested in. Recommendations will prioritise articles from these categories.</p>
          <div className="pref-grid">
            {allCats.map(cat => (
              <div key={cat} className={`pref-chip ${prefs.includes(cat) ? 'selected' : ''}`} onClick={() => togglePref(cat)}>
                {prefs.includes(cat) ? '✓' : '+'} {cat}
              </div>
            ))}
          </div>
          <div className="row" style={{ marginTop: 16 }}>
            <button onClick={savePrefs}>Save &amp; Refresh</button>
            <button className="secondary" onClick={resetPrefs}>Reset All</button>
          </div>
        </div>
      )}

      {/* ---- Recommendations tab ---- */}
      {tab === 'recs' && (
        <>
          {prefs.length > 0 && (
            <p className="muted" style={{ marginBottom: 12 }}>Showing recommendations based on your {prefs.length} preferred categor{prefs.length === 1 ? 'y' : 'ies'}: {prefs.join(', ')}</p>
          )}
          {recs.length === 0 ? (
            <div className="card"><p>No recommendations available. Try selecting some category preferences first.</p></div>
          ) : (
            <div className="article-grid">
              {recs.map(article => (
                <ArticleCard key={article.article_id} article={article}
                  isBookmarked={bookmarks.some(b => b.article_id === article.article_id)}
                  onBookmark={() => doBookmark(article)} />
              ))}
            </div>
          )}
        </>
      )}

      {/* ---- Bookmarks tab ---- */}
      {tab === 'bookmarks' && (
        <>
          {bookmarks.length === 0 ? (
            <div className="card"><p>No bookmarks yet. Bookmark articles from the Recommendations tab.</p></div>
          ) : (
            <div className="card">
              <table>
                <thead><tr><th>Headline</th><th>Category</th><th>Saved At</th><th></th></tr></thead>
                <tbody>{bookmarks.map(b => (
                  <tr key={b.id}>
                    <td>{b.headline}</td>
                    <td><span className="badge badge-cat">{b.category}</span></td>
                    <td className="muted">{b.created_at?.slice(0, 16).replace('T', ' ')}</td>
                    <td><button className="secondary" onClick={() => doUnbookmark(b.id)}>Remove</button></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </>
      )}

      <p className="muted" style={{ marginTop: 20 }}>
        ℹ Recommendations are content-based, using TF-IDF cosine similarity and category preferences.
        Articles come from the historical HuffPost dataset and are not current news.
        This is a single-user local demonstration.
      </p>
    </div>
  )
}

function ArticleCard({ article, isBookmarked, onBookmark }) {
  return (
    <div className="card article-card">
      <h4>{article.headline}</h4>
      <div className="article-meta">
        <span className="badge badge-cat">{article.category}</span>
        {article.date && <span className="muted">{article.date.slice(0, 10)}</span>}
        <span className="muted" style={{ fontSize: 11 }}>Score: {article.relevance_score}</span>
      </div>
      <p className="muted" style={{ marginTop: 6, fontSize: 12 }}>{article.reason}</p>
      <button style={{ marginTop: 10, fontSize: 12, padding: '5px 12px' }}
        className={isBookmarked ? 'secondary' : ''} onClick={onBookmark} disabled={isBookmarked}>
        {isBookmarked ? '✓ Bookmarked' : '🔖 Bookmark'}
      </button>
    </div>
  )
}
