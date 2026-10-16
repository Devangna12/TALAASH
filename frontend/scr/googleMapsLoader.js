let mapsPromise = null

export function loadGoogleMaps() {
  if (mapsPromise) return mapsPromise

  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  if (!apiKey) {
    return Promise.reject(
      new Error(
        'No Google Maps API key configured. Set VITE_GOOGLE_MAPS_API_KEY in frontend/.env and restart npm run dev.'
      )
    )
  }

  mapsPromise = new Promise((resolve, reject) => {
    if (window.google?.maps) {
      resolve(window.google.maps)
      return
    }

    const callbackName = '__talaashGoogleMapsLoaded'
    window[callbackName] = () => resolve(window.google.maps)

    const script = document.createElement('script')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&callback=${callbackName}`
    script.async = true
    script.onerror = () => reject(new Error('Failed to load the Google Maps script.'))
    document.head.appendChild(script)
  })

  return mapsPromise
}
