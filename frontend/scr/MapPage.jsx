import { useEffect, useRef, useState } from 'react'
import { API_BASE } from './config.js'
import { useAuth } from './AuthContext.jsx'
import { loadGoogleMaps } from './googleMapsLoader.js'

export default function MapPage() {
  const { token } = useAuth()
  const mapRef = useRef(null)
  const mapInstance = useRef(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function init() {
      try {
        const maps = await loadGoogleMaps()
        if (cancelled) return

        const res = await fetch(`${API_BASE}/api/map/data`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        const data = await res.json()
        if (cancelled) return

        const camera = data.cameras[0]
        const center = camera
          ? { lat: camera.latitude, lng: camera.longitude }
          : { lat: 26.9124, lng: 75.7873 }

        const map = new maps.Map(mapRef.current, {
          center,
          zoom: 13,
          mapId: 'TALAASH_MAP',
        })
        mapInstance.current = map

        const infoWindow = new maps.InfoWindow()

        // Camera markers (📷)
        data.cameras.forEach((cam) => {
          const marker = new maps.Marker({
            position: { lat: cam.latitude, lng: cam.longitude },
            map,
            label: { text: '📷', fontSize: '18px' },
            title: cam.name,
          })
          marker.addListener('click', () => {
            infoWindow.setContent(`<div><b>${cam.name}</b><br/>CCTV Camera</div>`)
            infoWindow.open(map, marker)
          })
        })

        // Missing-person last-seen markers (📍)
        data.cases.forEach((c) => {
          const marker = new maps.Marker({
            position: { lat: c.latitude, lng: c.longitude },
            map,
            label: { text: '📍', fontSize: '18px' },
            title: c.name,
          })
          marker.addListener('click', () => {
            infoWindow.setContent(
              `<div><b>${c.name}</b><br/>Case ${c.case_number}<br/>Status: ${c.status}<br/>${c.last_known_location || ''}</div>`
            )
            infoWindow.open(map, marker)
          })
        })

        // Today's AI match markers (🚨) at the camera's location
        data.recent_detections.forEach((d) => {
          const marker = new maps.Marker({
            position: { lat: d.latitude, lng: d.longitude },
            map,
            label: { text: '🚨', fontSize: '18px' },
            title: `${d.name} — ${d.status}`,
          })
          marker.addListener('click', () => {
            infoWindow.setContent(
              `<div><b>${d.name}</b><br/>AI Match &middot; ${(d.confidence * 100).toFixed(1)}%<br/>${d.created_at}</div>`
            )
            infoWindow.open(map, marker)
          })
        })

        setLoading(false)
      } catch (err) {
        if (!cancelled) {
          setError(err.message)
          setLoading(false)
        }
      }
    }

    init()
    return () => {
      cancelled = true
    }
  }, [token])

  return (
    <div>
      <h2 style={{ marginBottom: 18 }}>Map View</h2>

      {error && (
        <div className="form-error" style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}
      {loading && !error && <p className="empty">Loading map...</p>}

      <div
        ref={mapRef}
        style={{
          width: '100%',
          height: '520px',
          borderRadius: '10px',
          border: '1px solid #232A35',
          display: error ? 'none' : 'block',
        }}
      />

      <p className="empty" style={{ marginTop: 10 }}>
        📍 Last seen &nbsp; 📷 CCTV camera &nbsp; 🚨 AI match today
      </p>
    </div>
  )
}
