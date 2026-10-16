import { useEffect, useState } from 'react'
import { API_BASE } from './config.js'
import { useAuth } from './AuthContext.jsx'

const COLOR = {
  teal: '#2EC4B6',
  red: '#E5484D',
  amber: '#E8B93D',
  gray: '#7C8797',
  purple: '#B18CF0',
}

const RANGES = [
  { days: 7, label: 'Last 7 days' },
  { days: 14, label: 'Last 14 days' },
  { days: 30, label: 'Last 30 days' },
]

function pct(value) {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

function dayLabel(iso) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

function Panel({ title, subtitle, wide, children }) {
  return (
    <section className={`panel ${wide ? 'panel-wide' : ''}`}>
      <h3>{title}</h3>
      {subtitle && <p className="panel-sub">{subtitle}</p>}
      {children}
    </section>
  )
}

function BarChart({ series, color }) {
  const total = series.reduce((sum, s) => sum + s.count, 0)
  if (total === 0) return <p className="empty">No activity in this period.</p>

  const max = Math.max(1, ...series.map((s) => s.count))
  // Only label some of the days when there are many, so labels don't collide.
  const labelEvery = Math.ceil(series.length / 8)

  return (
    <div className="bar-chart">
      {series.map((s, i) => (
        <div className="bar-col" key={s.date} title={`${dayLabel(s.date)}: ${s.count}`}>
          <span className="bar-value">{s.count || ''}</span>
          <div className="bar-track">
            <div className="bar" style={{ height: `${(s.count / max) * 100}%`, background: color }} />
          </div>
          <span className="bar-label">{i % labelEvery === 0 ? dayLabel(s.date) : ''}</span>
        </div>
      ))}
    </div>
  )
}

function HBars({ rows }) {
  const total = rows.reduce((sum, r) => sum + r.value, 0)
  if (total === 0) return <p className="empty">No data yet.</p>

  const max = Math.max(...rows.map((r) => r.value))

  return (
    <div className="hbars">
      {rows.map((r) => (
        <div className="hbar-row" key={r.label}>
          <span className="hbar-label">{r.label}</span>
          <div className="hbar-track">
            <div className="hbar-fill" style={{ width: `${(r.value / max) * 100}%`, background: r.color }} />
          </div>
          <span className="hbar-value">
            {r.value} <em>{Math.round((r.value / total) * 100)}%</em>
          </span>
        </div>
      ))}
    </div>
  )
}

function downloadCsv(data) {
  const lines = ['date,ai_detections,reports']
  data.detections_per_day.forEach((d, i) => {
    lines.push(`${d.date},${d.count},${data.reports_per_day[i]?.count ?? 0}`)
  })
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `talaash-analytics-${data.days}d.csv`
  link.click()
  URL.revokeObjectURL(url)
}

export default function AnalyticsPage() {
  const { token } = useAuth()
  const [days, setDays] = useState(7)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setError(null)
    fetch(`${API_BASE}/api/stats/analytics?days=${days}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error('Could not load analytics.')
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [days, token])

  if (error) return <p className="form-error">{error}</p>
  if (!data) return <p className="empty">Loading analytics...</p>

  const reviewed = data.detections_by_status.Verified + data.detections_by_status.Rejected

  const kpis = [
    { label: 'Total Cases', value: data.totals.cases },
    { label: 'Resolution Rate', value: pct(data.resolution_rate), note: 'cases marked Found' },
    { label: 'AI Detections', value: data.totals.detections },
    {
      label: 'Officer-Confirmed',
      value: pct(data.confirmation_rate),
      note: reviewed ? `of ${reviewed} reviewed alerts` : 'no alerts reviewed yet',
    },
    {
      label: 'Avg. Confidence',
      value: data.avg_confidence == null ? '—' : `${(data.avg_confidence * 100).toFixed(1)}%`,
    },
    { label: 'Reports Filed', value: data.totals.reports },
  ]

  return (
    <div>
      <div className="page-head">
        <h2>Analytics</h2>
        <div className="page-head-actions">
          <div className="filter-row" style={{ marginBottom: 0 }}>
            {RANGES.map((r) => (
              <button key={r.days} className={days === r.days ? 'active' : ''} onClick={() => setDays(r.days)}>
                {r.label}
              </button>
            ))}
          </div>
          <button className="outline-btn" onClick={() => downloadCsv(data)}>Download CSV</button>
        </div>
      </div>

      <div className="stat-grid">
        {kpis.map((k) => (
          <div className="stat-card" key={k.label}>
            <div className="stat-value">{k.value}</div>
            <div className="stat-label">{k.label}</div>
            {k.note && <div className="stat-note">{k.note}</div>}
          </div>
        ))}
      </div>

      <div className="analytics-grid">
        <Panel
          wide
          title="AI detections per day"
          subtitle="Possible matches the system flagged from camera feeds"
        >
          <BarChart series={data.detections_per_day} color={COLOR.red} />
        </Panel>

        <Panel title="Case outcomes" subtitle="Where every registered case currently stands">
          <HBars
            rows={[
              { label: 'Active', value: data.cases_by_status.Active, color: COLOR.amber },
              { label: 'Found', value: data.cases_by_status.Found, color: COLOR.teal },
              { label: 'Closed', value: data.cases_by_status.Closed, color: COLOR.gray },
            ]}
          />
        </Panel>

        <Panel title="Alert review outcomes" subtitle="What officers decided about each AI alert">
          <HBars
            rows={[
              { label: 'Pending', value: data.detections_by_status.Pending, color: COLOR.amber },
              { label: 'Verified', value: data.detections_by_status.Verified, color: COLOR.teal },
              { label: 'Rejected', value: data.detections_by_status.Rejected, color: COLOR.red },
            ]}
          />
        </Panel>

        <Panel title="Match confidence" subtitle="How certain the AI was across all detections">
          <HBars
            rows={Object.entries(data.confidence_buckets).map(([label, value]) => ({
              label,
              value,
              color: COLOR.purple,
            }))}
          />
        </Panel>

        <Panel title="Busiest cameras" subtitle="Cameras with the most detections">
          <HBars rows={data.cameras.map((c) => ({ label: c.camera, value: c.count, color: COLOR.teal }))} />
        </Panel>

        <Panel title="Reports per day" subtitle="Field sightings submitted by officers">
          <BarChart series={data.reports_per_day} color={COLOR.teal} />
        </Panel>

        <Panel title="Report review status" subtitle="How many field reports still need a look">
          <HBars
            rows={[
              { label: 'New', value: data.reports_by_status.New, color: COLOR.amber },
              { label: 'Reviewed', value: data.reports_by_status.Reviewed, color: COLOR.teal },
            ]}
          />
        </Panel>
      </div>
    </div>
  )
}
