import { useEffect, useState } from 'react'
import { API_BASE } from './config.js'
import { useAuth } from './AuthContext.jsx'

export default function SubmitReportPage() {
  const { token } = useAuth()
  const [cases, setCases] = useState([])
  const [caseId, setCaseId] = useState('')
  const [location, setLocation] = useState('')
  const [description, setDescription] = useState('')
  const [photo, setPhoto] = useState(null)
  const [status, setStatus] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/api/cases?status=Active`, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => res.json())
      .then(setCases)
      .catch(() => {})
  }, [token])

  async function handleSubmit(e) {
    e.preventDefault()
    setSubmitting(true)
    setStatus(null)

    const formData = new FormData()
    if (caseId) formData.append('case_id', caseId)
    if (location) formData.append('location', location)
    if (description) formData.append('description', description)
    if (photo) formData.append('photo', photo)

    try {
      const res = await fetch(`${API_BASE}/api/reports`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Could not submit report.')
      setStatus({ type: 'success', message: 'Report submitted.' })
      setCaseId('')
      setLocation('')
      setDescription('')
      setPhoto(null)
      e.target.reset()
    } catch (err) {
      setStatus({ type: 'error', message: err.message })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <h2 style={{ marginBottom: 18 }}>Submit Report</h2>

      <form className="case-form" style={{ maxWidth: 420 }} onSubmit={handleSubmit}>
        <label>
          Case
          <select value={caseId} onChange={(e) => setCaseId(e.target.value)}>
            <option value="">-- General sighting / not case-specific --</option>
            {cases.map((c) => (
              <option key={c.id} value={c.id}>{c.case_number} — {c.name}</option>
            ))}
          </select>
        </label>

        <label>
          Location
          <input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Where was this seen?" />
        </label>

        <label>
          Description
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            placeholder="What did you observe?"
          />
        </label>

        <label>
          Photo (optional)
          <input type="file" accept="image/*" onChange={(e) => setPhoto(e.target.files[0] || null)} />
        </label>

        <button type="submit" disabled={submitting}>
          {submitting ? 'Submitting...' : 'Submit Report'}
        </button>

        {status && (
          <p className={status.type === 'error' ? 'form-error' : 'form-success'}>{status.message}</p>
        )}
      </form>
    </div>
  )
}
