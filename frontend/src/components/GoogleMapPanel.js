import React, { useEffect, useRef, useState } from "react";
import Cookies from "js-cookie";

const MAPS_SCRIPT_ID = "google-maps-script";
const DEFAULT_CENTER = { lat: 36.7783, lng: -119.4179 }; // California
const DEFAULT_ZOOM = 6;

function loadGoogleMapsApi(apiKey) {
  return new Promise((resolve, reject) => {
    if (window.google?.maps) {
      resolve(window.google.maps);
      return;
    }

    const existingScript = document.getElementById(MAPS_SCRIPT_ID);
    if (existingScript) {
      existingScript.addEventListener("load", () => resolve(window.google?.maps));
      existingScript.addEventListener("error", () => reject(new Error("Google Maps script failed to load.")));
      return;
    }

    const script = document.createElement("script");
    script.id = MAPS_SCRIPT_ID;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}`;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve(window.google?.maps);
    script.onerror = () => reject(new Error("Google Maps script failed to load."));
    document.head.appendChild(script);
  });
}

function getCookieLocation() {
  const lat = Number.parseFloat(Cookies.get("user_lat"));
  const lng = Number.parseFloat(Cookies.get("user_lng"));
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return null;
  }
  return { lat, lng };
}

function GoogleMapPanel() {
  const mapRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

  useEffect(() => {
    if (!apiKey) {
      setStatus("missing-key");
      return;
    }

    let marker = null;
    let map = null;
    let cancelled = false;

    const init = async () => {
      try {
        const maps = await loadGoogleMapsApi(apiKey);
        if (!maps || cancelled || !mapRef.current) {
          return;
        }

        const cookieLocation = getCookieLocation();
        const center = cookieLocation || DEFAULT_CENTER;
        map = new maps.Map(mapRef.current, {
          center,
          zoom: cookieLocation ? 11 : DEFAULT_ZOOM,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true,
        });

        if (cookieLocation) {
          marker = new maps.Marker({
            position: cookieLocation,
            map,
            title: "Your saved location",
          });
        }

        setStatus("ready");
      } catch (error) {
        console.error(error);
        setStatus("error");
      }
    };

    init();

    return () => {
      cancelled = true;
      if (marker) {
        marker.setMap(null);
      }
      map = null;
    };
  }, [apiKey]);

  if (status === "missing-key") {
    return (
      <div className="map-placeholder map-message">
        Add <code>REACT_APP_GOOGLE_MAPS_API_KEY</code> to <code>frontend/.env</code>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="map-placeholder map-message">
        Google Maps failed to load. Check your API key and Google Cloud restrictions.
      </div>
    );
  }

  return (
    <div className="map-placeholder">
      {status === "loading" && <div className="map-loading">Loading map...</div>}
      <div ref={mapRef} className="map-canvas" />
    </div>
  );
}

export default GoogleMapPanel;
