import axios from 'axios'

// One place for all backend calls.
export const api = axios.create({ baseURL: '/api', timeout: 30000 })

export function errorMessage(err) {
  const detail = err?.response?.data?.detail
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ')
  if (typeof detail === 'string') return detail
  if (err?.code === 'ERR_NETWORK' || !err?.response) return 'Cannot reach the backend. Is it running on port 8000?'
  return err?.message || 'Something went wrong.'
}

// ------ convenience wrappers (all return .data)
export const getHealth          = () => api.get('/health').then(r => r.data)
export const getModelInfo       = () => api.get('/model-info').then(r => r.data)
export const postPredict        = (body) => api.post('/predict', body).then(r => r.data)
export const postExplain        = (body) => api.post('/explain', body).then(r => r.data)
export const postSentiment      = (body) => api.post('/sentiment/predict', body).then(r => r.data)
export const getMetrics         = () => api.get('/analytics/metrics').then(r => r.data)
export const getModelComparison = () => api.get('/analytics/model-comparison').then(r => r.data)
export const getClassReport     = () => api.get('/analytics/classification-report').then(r => r.data)
export const getConfusionMatrix = () => api.get('/analytics/confusion-matrix').then(r => r.data)
export const getDatasetSummary  = () => api.get('/dataset/summary').then(r => r.data)
export const getRecommendations = (limit) => api.get('/recommendations', { params: { limit } }).then(r => r.data)
export const getPreferences     = () => api.get('/preferences').then(r => r.data)
export const putPreferences     = (cats) => api.put('/preferences', { preferred_categories: cats }).then(r => r.data)
export const getBookmarks       = () => api.get('/bookmarks').then(r => r.data)
export const addBookmark        = (b) => api.post('/bookmarks', b).then(r => r.data)
export const removeBookmark     = (id) => api.delete(`/bookmarks/${id}`).then(r => r.data)
export const getTrends          = () => api.get('/analytics/trends').then(r => r.data)
export const getCategories      = () => api.get('/analytics/categories').then(r => r.data)
export const getKeywords        = (cat, limit) => api.get('/analytics/keywords', { params: { category: cat || undefined, limit } }).then(r => r.data)
export const getTimeline        = () => api.get('/analytics/timeline').then(r => r.data)
export const getSentimentDist   = () => api.get('/analytics/sentiment-distribution').then(r => r.data)
export const getPredictionHist  = () => api.get('/analytics/prediction-history').then(r => r.data)
