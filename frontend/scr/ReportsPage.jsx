import { useEffect, useState } from 'react'
import { API_BASE } from './config.js'
import { useAuth } from './AuthContext.jsx'

const FILTERS = ['All', 'New', 'Reviewed']

export default function ReportsPage() {
  const { token } = useAuth()
  const [reports, setReports] = useState([])
  const [error, setError] = useState(null)
  const [busyId, setBusyId] = useState(null)
  const [filter, setFilter] = useState('All')
  const [search, setSearch] = useState('')

  function load() {
    fetch(`${API_BASE}/api/reports`, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => res.json())
      .then(setReports)
      .catch(() => setError('Could not load reports.'))
  }

  useEffect(load, [token])

  async function markReviewed(id) {
    setBusyId(id)
    try {
      const formData = new FormData()
      formData.append('status', 'Reviewed')
      await fetch(`${API_BASE}/api/reports/${id}/status`, {
        method: 'PATCH',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })
      load()
    } finally {
      setBusyId(null)
    }
  }

  const total = reports.length
  const newCount = reports.filter((r) => r.status === 'New').length
  const reviewedCount = total - newCount

  const query = search.trim().toLowerCase()
  const visible = reports.filter((r) => {
    if (filter !== 'All' && r.status !== filter) return false
    if (!query) return true
    return [r.officer_username, r.case_number, r.name, r.location, r.description]
      .filter(Boolean)
      .some((field) => field.toLowerCase().includes(query))
  })

  return (
    <div>
      <h2 style={{ marginBottom: 18 }}>Reports</h2>

      <div className="stat-grid" style={{ marginBottom: 20 }}>
        <div className="stat-card">
          <div className="stat-value">{total}</div>
          <div className="stat-label">Total Reports</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: '#E8B93D' }}>{newCount}</div>
          <div className="stat-label">Awaiting Review</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{reviewedCount}</div>
          <div className="stat-label">Reviewed</div>
        </div>
      </div>

      <div className="toolbar">
        <div className="filter-row" style={{ marginBottom: 0 }}>
          {FILTERS.map((f) => (
            <button key={f} className={filter === f ? 'active' : ''} onClick={() => setFilter(f)}>
              {f}
            </button>
          ))}
        </div>
        <input
          className="search-input"
          placeholder="Search officer, case, location, notes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {error && <p className="form-error">{error}</p>}
      {!error && total === 0 && <p className="empty">No reports submitted yet.</p>}
      {!error && total > 0 && visible.length === 0 && <p className="empty">No reports match your filters.</p>}

      <div className="case-list">
        {visible.map((r) => (
          <div className="case-card" key={r.id}>
            <div className="case-card-head">
              <div>
                <span className="case-number">{r.case_number || 'General sighting'}</span>
                <span className="case-name">{r.name || '—'}</span>
              </div>
              <span className={`badge ${r.status === 'New' ? 'badge-status-pending' : 'badge-status-verified'}`}>
                {r.status}
              </span>
            </div>

            <div className="case-card-meta">
              Submitted by {r.officer_username} &middot; {r.created_at}
              {r.location ? ` · ${r.location}` : ''}
            </div>

            {r.description && (
              <p style={{ fontSize: 12.5, color: '#E7ECF2', marginTop: 8 }}>{r.description}</p>
            )}

            {r.photo_path && (
              <img
                src={`${API_BASE}/media/${r.photo_path}`}
                alt="Report attachment"
                style={{ maxWidth: 220, borderRadius: 8, marginTop: 8, display: 'block' }}
              />
            )}

            {r.status === 'New' && (
              <div className="row-actions" style={{ marginTop: 10 }}>
                <button disabled={busyId === r.id} onClick={() => markReviewed(r.id)}>Mark Reviewed</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
