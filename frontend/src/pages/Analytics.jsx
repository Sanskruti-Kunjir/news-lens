import { useEffect, useState } from 'react'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { getCategories, getTrends, getKeywords, getTimeline, getPredictionHist } from '../api'

const COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#14b8a6','#f97316','#6366f1']

export default function Analytics() {
  const [tab, setTab] = useState('categories')
  const [cats, setCats] = useState(null)
  const [trends, setTrends] = useState(null)
  const [keywords, setKeywords] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [predHist, setPredHist] = useState(null)
  const [kwCat, setKwCat] = useState('')

  useEffect(() => {
    getCategories().then(setCats).catch(() => {})
    getTrends().then(setTrends).catch(() => {})
    getKeywords(null, 25).then(setKeywords).catch(() => {})
    getTimeline().then(setTimeline).catch(() => {})
    getPredictionHist().then(setPredHist).catch(() => {})
  }, [])

  function loadKw(cat) {
    setKwCat(cat)
    getKeywords(cat || null, 25).then(setKeywords).catch(() => {})
  }

  const tabs = ['categories', 'timeline', 'trending', 'keywords', 'sentiment', 'history']

  return (
    <div>
      <h2>News Analytics</h2>
      <div className="tabs">
        {tabs.map(t => <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
          {t === 'categories' ? 'Categories' : t === 'timeline' ? 'Timeline' : t === 'trending' ? 'Trending' : t === 'keywords' ? 'Keywords' : t === 'sentiment' ? 'Sentiment' : 'Pred. History'}
        </button>)}
      </div>

      {/* ======== Categories ======== */}
      {tab === 'categories' && cats && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card">
            <h3>Category Distribution ({cats.n_categories} categories, {cats.total_records?.toLocaleString()} articles)</h3>
            <ResponsiveContainer width="100%" height={Math.max(300, cats.categories.length * 22)}>
              <BarChart data={cats.categories.slice(0, 25)} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="category" width={140} tick={{ fontSize: 11 }} /><Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3>Top 8 Categories (share)</h3>
            <ResponsiveContainer width="100%" height={360}>
              <PieChart><Pie data={cats.categories.slice(0, 8)} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={130} label={({ category, percentage }) => `${category} ${percentage}%`} labelLine={false} style={{ fontSize: 10 }}>
                {cats.categories.slice(0, 8).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie><Tooltip /></PieChart>
            </ResponsiveContainer>
            <p className="muted">{cats.source}</p>
          </div>
        </div>
      )}

      {/* ======== Timeline ======== */}
      {tab === 'timeline' && timeline && (
        <div className="card">
          <h3>Articles Over Time</h3>
          {!timeline.available ? <p className="muted">Date-based timeline unavailable for this dataset.</p> : (
            <>
              <p className="muted">Date range: {timeline.date_range?.min} → {timeline.date_range?.max}</p>
              <h4 style={{ marginTop: 16 }}>By Year</h4>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={timeline.by_year}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="year" /><YAxis /><Tooltip /><Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} /></BarChart>
              </ResponsiveContainer>
              <h4 style={{ marginTop: 16 }}>Monthly Trend</h4>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={timeline.by_month.filter((_, i) => i % 3 === 0)}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="month" tick={{ fontSize: 9 }} interval={3} /><YAxis /><Tooltip /><Line type="monotone" dataKey="count" stroke="#3b82f6" dot={false} /></LineChart>
              </ResponsiveContainer>
              <p className="muted">{timeline.note}</p>
            </>
          )}
        </div>
      )}

      {/* ======== Trending ======== */}
      {tab === 'trending' && trends && (
        <div className="card">
          <h3>Trending Topics</h3>
          {!trends.available ? <p className="muted">{trends.reason || 'Trend analysis unavailable.'}</p> : (
            <>
              <p className="muted">Method: {trends.method}</p>
              <p className="muted">Comparing {trends.recent_period?.join(', ')} vs {trends.earlier_period?.join(', ')}</p>
              <table style={{ marginTop: 12 }}>
                <thead><tr><th>Topic</th><th>Direction</th><th>Change</th><th>Recent Avg</th><th>Earlier Avg</th></tr></thead>
                <tbody>{trends.trends.map(t => (
                  <tr key={t.topic}>
                    <td>{t.topic}</td>
                    <td><span className={`badge badge-${t.direction}`}>{t.direction === 'rising' ? '📈' : t.direction === 'declining' ? '📉' : '➡'} {t.direction}</span></td>
                    <td style={{ color: t.change_percent > 0 ? '#10b981' : t.change_percent < 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>
                      {t.change_percent > 0 ? '+' : ''}{t.change_percent}%
                    </td>
                    <td>{t.recent_avg}</td><td>{t.earlier_avg}</td>
                  </tr>
                ))}</tbody>
              </table>
              <p className="muted" style={{ marginTop: 12 }}>{trends.note}</p>
            </>
          )}
        </div>
      )}

      {/* ======== Keywords ======== */}
      {tab === 'keywords' && keywords && (
        <div className="card">
          <h3>Keyword Analysis</h3>
          <div className="row" style={{ marginBottom: 16 }}>
            <select value={kwCat} onChange={e => loadKw(e.target.value)} style={{ width: 200 }}>
              <option value="">All categories</option>
              {cats?.categories?.map(c => <option key={c.category} value={c.category}>{c.category}</option>)}
            </select>
          </div>
          <p className="muted">{keywords.method} · {keywords.total_documents?.toLocaleString()} documents</p>
          {keywords.keywords?.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={Math.max(300, keywords.keywords.length * 24)}>
                <BarChart data={keywords.keywords} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="term" width={120} tick={{ fontSize: 12 }} /><Tooltip />
                  <Bar dataKey="tfidf_score" fill="#8b5cf6" name="TF-IDF Score" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : <p className="muted">No keywords found for this filter.</p>}
        </div>
      )}

      {/* ======== Sentiment ======== */}
      {tab === 'sentiment' && predHist && (
        <div className="card">
          <h3>Sentiment Distribution (from predictions)</h3>
          {predHist.sentiment_distribution?.length === 0 ? (
            <p className="muted">No sentiment data yet. Make some predictions on the Predict page first.</p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart><Pie data={predHist.sentiment_distribution} dataKey="count" nameKey="sentiment" cx="50%" cy="50%" outerRadius={100} label>
                  {predHist.sentiment_distribution.map((d, i) => <Cell key={i} fill={d.sentiment === 'Positive' ? '#10b981' : d.sentiment === 'Negative' ? '#ef4444' : '#3b82f6'} />)}
                </Pie><Tooltip /><Legend /></PieChart>
              </ResponsiveContainer>
              <div>
                <table>
                  <thead><tr><th>Sentiment</th><th>Count</th></tr></thead>
                  <tbody>{predHist.sentiment_distribution.map(d => (
                    <tr key={d.sentiment}><td><span className={`badge badge-${d.sentiment?.toLowerCase()}`}>{d.sentiment}</span></td><td>{d.count}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ======== Prediction History ======== */}
      {tab === 'history' && predHist && (
        <div className="card">
          <h3>Prediction History Analytics</h3>
          <div className="grid" style={{ marginBottom: 16 }}>
            <div className="stat"><span className="muted">Total Predictions</span><b>{predHist.total_predictions}</b></div>
          </div>
          {predHist.category_distribution?.length > 0 && (
            <>
              <h4>Predicted Category Distribution</h4>
              <ResponsiveContainer width="100%" height={Math.max(200, predHist.category_distribution.length * 24)}>
                <BarChart data={predHist.category_distribution.slice(0, 15)} layout="vertical" margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="category" width={140} tick={{ fontSize: 11 }} /><Tooltip />
                  <Bar dataKey="count" fill="#14213d" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
          {predHist.by_date?.length > 0 && (
            <>
              <h4 style={{ marginTop: 16 }}>Predictions Over Time</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={predHist.by_date}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" tick={{ fontSize: 10 }} /><YAxis /><Tooltip /><Line type="monotone" dataKey="count" stroke="#3b82f6" /></LineChart>
              </ResponsiveContainer>
            </>
          )}
          <p className="muted" style={{ marginTop: 12 }}>These analytics reflect your actual prediction usage, not the dataset.</p>
        </div>
      )}
    </div>
  )
}
