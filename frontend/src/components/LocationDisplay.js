import React, { useEffect, useState } from "react";
import Cookies from "js-cookie";

/**
 * Reads lat/lng from cookies if available, otherwise requests
 * the browser Geolocation API and stores the result in cookies.
 */
function LocationDisplay() {
  const [location, setLocation] = useState({ lat: null, lng: null });
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    // 1. Try to read from cookies first
    const savedLat = Cookies.get("user_lat");
    const savedLng = Cookies.get("user_lng");

    if (savedLat && savedLng) {
      setLocation({ lat: parseFloat(savedLat), lng: parseFloat(savedLng) });
      setStatus("success");
      return;
    }

    // 2. Fall back to Geolocation API
    if (!navigator.geolocation) {
      setStatus("unsupported");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        // Store in cookies for 7 days
        Cookies.set("user_lat", lat, { expires: 7 });
        Cookies.set("user_lng", lng, { expires: 7 });
        setLocation({ lat, lng });
        setStatus("success");
      },
      (err) => {
        console.error("Geolocation error:", err.message);
        setStatus("denied");
      }
    );
  }, []);

  return (
    <section className="location-section">
      <h2>Your Location</h2>

      {status === "loading" && <p>Detecting location...</p>}

      {status === "success" && (
        <p className="coords">
          Latitude: {location.lat} &nbsp;|&nbsp; Longitude: {location.lng}
        </p>
      )}

      {status === "denied" && (
        <p className="error">
          Location access denied. Please enable location services or enter your
          address manually.
        </p>
      )}

      {status === "unsupported" && (
        <p className="error">
          Geolocation is not supported by your browser.
        </p>
      )}
    </section>
  );
}

export default LocationDisplay;
