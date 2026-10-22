import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'

import Analytics from './pages/Analytics'
import Recommendations from './pages/Recommendations'
import History from './pages/History'

const navItems = [
  { section: 'Core' },
  { to: '/',            label: 'Dashboard',       icon: '📊', end: true },
  { to: '/predict',     label: 'Predict',         icon: '🔮' },
  
  { section: 'Explore' },
  { to: '/analytics',   label: 'Analytics',       icon: '📈' },
  { to: '/discover',    label: 'Discover',        icon: '🧭' },
  { section: 'Activity' },
  { to: '/history',     label: 'History',         icon: '🕐' },
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="sidebar-brand-icon">N</div>
            <h1>NewsLens AI<small>Intelligent News Analysis</small></h1>
          </div>
        </div>
        <nav>
          {navItems.map((item, i) =>
            item.section
              ? <div key={i} className="sidebar-section">{item.section}</div>
              : <NavLink key={item.to} to={item.to} end={item.end}>
                  <span className="nav-icon">{item.icon}</span>
                  {item.label}
                </NavLink>
          )}
        </nav>
        <div className="sidebar-footer">
          NewsLens AI v1.0 · ML-Powered
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/discover" element={<Recommendations />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  )
}
