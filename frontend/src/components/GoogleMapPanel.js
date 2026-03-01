import React, { useEffect, useRef, useState } from "react";
import Cookies from "js-cookie";

const MAPS_SCRIPT_ID = "google-maps-script";
const TILE_SIZE = 256;
const DEFAULT_CENTER = { lat: 36.7783, lng: -119.4179 };
const DEFAULT_ZOOM = 12;

function loadGoogleMapsApi(apiKey) {
  return new Promise((resolve, reject) => {
    if (window.google?.maps) {
      resolve(window.google.maps);
      return;
    }

    const existing = document.getElementById(MAPS_SCRIPT_ID);
    if (existing) {
      existing.addEventListener("load", () => resolve(window.google?.maps));
      existing.addEventListener("error", () => reject(new Error("Google Maps script failed to load.")));
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

function cellToLatLng(cellX, cellY, centerLat, centerLon, cellM, width, height) {
  const dxM = (cellX - Math.floor(width / 2)) * cellM;
  const dyM = (cellY - Math.floor(height / 2)) * cellM;
  const lat = centerLat - dyM / 111320.0;
  const lonDenom = Math.max(1e-6, 111320.0 * Math.cos((centerLat * Math.PI) / 180));
  const lng = centerLon + dxM / lonDenom;
  return { lat, lng };
}

function cellBounds(cellX, cellY, centerLat, centerLon, cellM, width, height) {
  const center = cellToLatLng(cellX, cellY, centerLat, centerLon, cellM, width, height);
  const latHalf = (cellM / 2) / 111320.0;
  const lonHalf = (cellM / 2) / Math.max(1e-6, 111320.0 * Math.cos((centerLat * Math.PI) / 180));
  return {
    north: center.lat + latHalf,
    south: center.lat - latHalf,
    west: center.lng - lonHalf,
    east: center.lng + lonHalf,
  };
}

function GoogleMapPanel({ data, loading }) {
  const mapRef = useRef(null);
  const [status, setStatus] = useState("loading");
  const [apiKey, setApiKey] = useState("");
  const [userLocation, setUserLocation] = useState(null);

  useEffect(() => {
    const cookieLat = Number.parseFloat(Cookies.get("user_lat"));
    const cookieLng = Number.parseFloat(Cookies.get("user_lng"));
    if (Number.isFinite(cookieLat) && Number.isFinite(cookieLng)) {
      setUserLocation({ lat: cookieLat, lng: cookieLng });
      return;
    }

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setUserLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        },
        () => {},
        { enableHighAccuracy: true, timeout: 5000, maximumAge: 60000 }
      );
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadKey = async () => {
      try {
        const res = await fetch("/evac/maps/config");
        if (!res.ok) throw new Error("Failed to fetch map config");
        const cfg = await res.json();
        if (cancelled) return;
        if (cfg?.google_maps_api_key) {
          setApiKey(cfg.google_maps_api_key);
          setStatus("loading");
        } else {
          setStatus("missing-key");
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setStatus("error");
      }
    };
    loadKey();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!apiKey) {
      return undefined;
    }

    let cancelled = false;
    const mapObjects = [];

    const clearObjects = () => {
      for (const obj of mapObjects) {
        if (obj && typeof obj.setMap === "function") obj.setMap(null);
      }
      mapObjects.length = 0;
    };

    const draw = async () => {
      try {
        const maps = await loadGoogleMapsApi(apiKey);
        if (!maps || cancelled || !mapRef.current) return;

        const center = {
          lat: data?.input?.lat ?? userLocation?.lat ?? DEFAULT_CENTER.lat,
          lng: data?.input?.lon ?? userLocation?.lng ?? DEFAULT_CENTER.lng,
        };

        const map = new maps.Map(mapRef.current, {
          center,
          zoom: DEFAULT_ZOOM,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: true,
        });

        mapObjects.push(map);

        if (!data?.grid || !data?.input) {
          setStatus("ready");
          return;
        }

        const width = data.grid.w;
        const height = data.grid.h;
        const cellM = data.grid.cell_m;
        const userLat = data.input.lat;
        const userLon = data.input.lon;

        const blocked = data.blocked_cells || {};
        const fireCells = blocked.fire || [];
        const airCells = blocked.air || [];

        const paintCells = (cells, color) => {
          for (const [x, y] of cells) {
            const rect = new maps.Rectangle({
              strokeOpacity: 0,
              fillColor: color,
              fillOpacity: 0.28,
              map,
              bounds: cellBounds(x, y, userLat, userLon, cellM, width, height),
            });
            mapObjects.push(rect);
          }
        };

        paintCells(fireCells, "#dc2626");
        paintCells(airCells, "#f59e0b");

        const pathLatLng = (data?.plan?.path_latlng || [])
          .map((p) => ({
            lat: Number(p?.lat),
            lng: Number(p?.lng),
          }))
          .filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lng));
        const pathCells = data?.plan?.path_cells || [];
        const chosenCenter = data?.plan?.center || null;
        let path = pathLatLng.length > 1
          ? pathLatLng
          : pathCells.map(([x, y]) => cellToLatLng(x, y, userLat, userLon, cellM, width, height));

        if (path.length < 2 && chosenCenter) {
          path = [
            center,
            { lat: Number(chosenCenter.lat), lng: Number(chosenCenter.lon) },
          ].filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lng));
        }

        if (path.length > 1) {
          const polyline = new maps.Polyline({
            path,
            geodesic: false,
            strokeColor: "#fde047",
            strokeOpacity: 1.0,
            strokeWeight: 5,
            zIndex: 9999,
            clickable: false,
            map,
          });
          mapObjects.push(polyline);
        }

        const info = new maps.InfoWindow();

        const startLatLng = path.length > 0 ? path[0] : center;

        const userMarker = new maps.Marker({
          position: startLatLng,
          map,
            title: "Your location",
          icon: {
            path: maps.SymbolPath.CIRCLE,
            scale: 7,
            fillColor: "#7e22ce",
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2,
          },
        });
        mapObjects.push(userMarker);

        const centers = data?.objects?.centers || [];
        for (const c of centers) {
          const isChosen =
            !!chosenCenter && c.place_id && chosenCenter.place_id && c.place_id === chosenCenter.place_id;
          const marker = new maps.Marker({
            position: { lat: c.lat, lng: c.lon },
            map,
            title: c.name || c.type || "Center",
            label: isChosen ? "G" : "C",
            icon: isChosen
              ? {
                  path: maps.SymbolPath.CIRCLE,
                  scale: 10,
                  fillColor: "#22c55e",
                  fillOpacity: 1,
                  strokeColor: "#14532d",
                  strokeWeight: 2,
                }
              : undefined,
          });
          marker.addListener("click", () => {
            info.setContent(`<strong>${c.name || "Emergency Center"}</strong><br/>${c.type || ""}`);
            info.open(map, marker);
          });
          mapObjects.push(marker);
        }

        const buildings = data?.objects?.buildings_nonsafe || [];
        for (const b of buildings) {
          const ring = new maps.Circle({
            center: { lat: b.lat, lng: b.lon },
            radius: 18,
            strokeOpacity: 0,
            fillColor: "#2563eb",
            fillOpacity: 0.18,
            map,
          });
          mapObjects.push(ring);

          const marker = new maps.Marker({
            position: { lat: b.lat, lng: b.lon },
            map,
            title: b.name || "Building",
            icon: {
              path: maps.SymbolPath.CIRCLE,
              scale: 4,
              fillColor: "#1d4ed8",
              fillOpacity: 0.9,
              strokeColor: "#ffffff",
              strokeWeight: 1,
            },
          });
          marker.addListener("click", () => {
            info.setContent(`<strong>${b.name || "Building"}</strong><br/>${b.type || "building"}`);
            info.open(map, marker);
          });
          mapObjects.push(marker);
        }

        if (path.length > 0) {
          const bounds = new maps.LatLngBounds();
          for (const p of path) {
            bounds.extend(p);
          }
          bounds.extend(startLatLng);
          map.fitBounds(bounds, 60);
        }

        setStatus("ready");
      } catch (e) {
        console.error(e);
        if (!cancelled) setStatus("error");
      }
    };

    draw();

    return () => {
      cancelled = true;
      clearObjects();
    };
  }, [apiKey, data, userLocation]);

  if (status === "missing-key") {
    return (
      <div className="map-placeholder map-message">
        Missing backend <code>GOOGLE_API_KEY</code>. Set it before starting <code>backend/AStarI.py</code>.
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="map-placeholder map-message">
        Google map failed to load. Verify API key and Maps JavaScript API access.
      </div>
    );
  }

  return (
    <div className="map-placeholder">
      {(loading || status === "loading") && <div className="map-loading">Loading map...</div>}
      <div ref={mapRef} className="map-canvas" />
    </div>
  );
}

export default GoogleMapPanel;
