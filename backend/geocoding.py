"""
Geocoding: converts a free-text location ("MG Road junction, Jaipur")
into latitude/longitude so it can be plotted on the Map View.

Requires GOOGLE_MAPS_SERVER_KEY as an environment variable, with the
Geocoding API enabled on that key in Google Cloud Console. This is
deliberately a SEPARATE key from the one used in the frontend
(VITE_GOOGLE_MAPS_API_KEY) -- the server key should be restricted to
the Geocoding API only (no HTTP referrer restriction, since it's
called from your backend, not a browser), while the frontend key
should be restricted by HTTP referrer (since it's exposed in the
browser). Keeping them separate limits what a leaked key can do.

If no key is configured, geocoding is skipped and the case is stored
without coordinates -- it just won't show up on Map View until it's
re-geocoded (see backfill_missing_coordinates below).
"""
import os

import requests

GOOGLE_MAPS_SERVER_KEY = os.environ.get("GOOGLE_MAPS_SERVER_KEY")
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode_location(address: str):
    """Returns (lat, lng) or (None, None) if it couldn't be geocoded
    (no key configured, address not found, or a request error)."""
    if not address or not GOOGLE_MAPS_SERVER_KEY:
        return None, None

    try:
        response = requests.get(
            GEOCODE_URL,
            params={"address": address, "key": GOOGLE_MAPS_SERVER_KEY},
            timeout=5,
        )
        data = response.json()

        if data.get("status") != "OK" or not data.get("results"):
            print(f"[geocode] Could not geocode '{address}': {data.get('status')}")
            return None, None

        location = data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]

    except Exception as e:
        print(f"[geocode] ERROR geocoding '{address}': {e}")
        return None, None
