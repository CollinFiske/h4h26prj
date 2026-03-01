# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import heapq
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timezone
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


app = Flask(__name__)
CORS(app)

# -------------------------
# Grid settings (5 miles - CHANGE)
# -------------------------
RADIUS_M = 8047
W, H = 64, 64
CELL_M = (2 * RADIUS_M) / W  # ~250m

SAME_PLACE_THRESH_M = 80  # building == center if within 80m

# -------------------------
# Helpers
# -------------------------
def now_utc():
    return datetime.now(timezone.utc).isoformat()

def empty_grid(val=0.0):
    return np.full((H, W), val, dtype=np.float32)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def get_inputs():
    data = request.get_json(silent=True) or {}
    lat = data.get("lat") or request.args.get("lat") or request.cookies.get("lat")
    lon = data.get("lon") or request.args.get("lon") or request.cookies.get("lon")
    if lat is None or lon is None:
        return None
    lang = data.get("lang") or request.args.get("lang") or request.cookies.get("lang") or "en"
    disability = data.get("disability") or request.args.get("disability") or request.cookies.get("disability") or "none"
    return float(lat), float(lon), str(lang), str(disability)

# ============================================================
# CHANGE THIS: Google / Places API calls (placeholders)
# ============================================================
import requests

PLACE_TYPES = ["hospital", "fire_station", "police"]
BUILDING_PLACE_TYPES = [
    "school",
    "supermarket",
    "shopping_mall",
    "restaurant",
    "lodging",
    "place_of_worship",
    "bank",
    "pharmacy",
]

def _nearby_search(location_lat, location_lon, place_type, radius_m):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{location_lat},{location_lon}",
        "radius": int(radius_m),
        "type": place_type,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_emergency_centers(lat, lon, max_results_per_type=5):
    all_centers = []

    for place_type in PLACE_TYPES:
        try:
            data = _nearby_search(lat, lon, place_type, RADIUS_M)
        except requests.RequestException as e:
            print(f"Google API request failed for {place_type}: {e}")
            continue

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"Google API error for {place_type}: {data.get('status')}")
            continue

        for place in data.get("results", [])[:max_results_per_type]:
            all_centers.append({
                "name": place.get("name"),
                "type": place_type,
                "lat": place["geometry"]["location"]["lat"],
                "lon": place["geometry"]["location"]["lng"],
                "place_id": place.get("place_id")
            })

    return all_centers

def get_buildings(lat, lon, max_results_per_type=20):
    buildings = []
    seen_place_ids = set()

    for place_type in BUILDING_PLACE_TYPES:
        try:
            data = _nearby_search(lat, lon, place_type, RADIUS_M)
        except requests.RequestException as e:
            print(f"Google API request failed for building type {place_type}: {e}")
            continue

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"Google API error for building type {place_type}: {data.get('status')}")
            continue

        for place in data.get("results", [])[:max_results_per_type]:
            place_id = place.get("place_id")
            if place_id and place_id in seen_place_ids:
                continue
            if place_id:
                seen_place_ids.add(place_id)

            buildings.append({
                "name": place.get("name"),
                "type": place_type,
                "lat": place["geometry"]["location"]["lat"],
                "lon": place["geometry"]["location"]["lng"],
                "place_id": place_id
            })

    return buildings

# ============================================================
# CHANGE THIS: ML model outputs (placeholders)
# ============================================================
def ml_fire_blocked(lat, lon, when_iso=None):
    """
    CHANGE THIS:
      Your fire model should output an HxW MASK of where fire is / will be.
      Return: fire_blocked (H,W) with True=blocked.
    For now: no fire.
    """
    return np.zeros((H, W), dtype=bool)

def ml_air_risk(lat, lon, when_iso=None):
    """
    CHANGE THIS:
      Air model should output HxW risk/probability grid.
      Return: air_risk (H,W) float in [0,1].
    For now: no air.
    """
    return empty_grid(0.0)

# -------------------------
# Mapping (lat/lon -> grid cell)
# -------------------------
def user_cell():
    return (W // 2, H // 2)

def latlon_to_cell(lat, lon, center_lat, center_lon):
    # CHANGE LATER: real mapping. This is a rough placeholder.
    dx = int((lon - center_lon) * 500)
    dy = int((center_lat - lat) * 500)
    x = max(0, min(W - 1, (W // 2) + dx))
    y = max(0, min(H - 1, (H // 2) + dy))
    return (x, y)

# -------------------------
# Buildings -> blocked cells
# -------------------------
def same_place(a, b):
    return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) <= SAME_PLACE_THRESH_M

def filter_non_safe_buildings(buildings, centers):
    out = []
    for b in buildings:
        if any(same_place(b, c) for c in centers):
            continue
        out.append(b)
    return out

def buildings_to_blocked(buildings, user_lat, user_lon):
    blocked = np.zeros((H, W), dtype=bool)
    for b in buildings:
        x, y = latlon_to_cell(b["lat"], b["lon"], user_lat, user_lon)
        blocked[y, x] = True
    return blocked

# -------------------------
# A* pathfinding (Pacman)
# -------------------------
def a_star(blocked, start, goal):
    sx, sy = start
    gx, gy = goal

    def inside(x, y):
        return 0 <= x < W and 0 <= y < H

    def h(x, y):
        return abs(x - gx) + abs(y - gy)

    pq = [(h(sx, sy), 0.0, sx, sy)]
    best = {(sx, sy): 0.0}
    parent = {}
    moves = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:
        _, g, x, y = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in parent:
                x, y = parent[(x, y)]
                path.append((x, y))
            path.reverse()
            return {"status": "ok", "path": path}

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not inside(nx, ny):
                continue
            if blocked[ny, nx]:
                continue

            ng = g + 1.0
            if (nx, ny) not in best or ng < best[(nx, ny)]:
                best[(nx, ny)] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))

    return {"status": "no_path", "path": []}

# -------------------------
# Endpoint: NOW
# -------------------------
@app.post("/evac/now")
def evac_now():
    got = get_inputs()
    if not got:
        return jsonify({"error": "Send lat/lon via JSON, query, or cookies."}), 400

    lat, lon, lang, disability = got

    centers = get_emergency_centers(lat, lon)
    buildings = get_buildings(lat, lon)

    # closest centers first
    centers = sorted(centers, key=lambda c: haversine_m(lat, lon, c["lat"], c["lon"]))

    # buildings except centers are blocked objects
    buildings_nonsafe = filter_non_safe_buildings(buildings, centers)
    building_blocked = buildings_to_blocked(buildings_nonsafe, lat, lon)

    # fire is ALWAYS blocked objects (from ML)
    fire_blocked = ml_fire_blocked(lat, lon)

    # air:
    # - healthy: allowed (NOT blocked)
    # - disability: blocked (ALL air cells, or only high risk if you change it later)
    air = ml_air_risk(lat, lon)
    if disability == "none":
        air_blocked = np.zeros((H, W), dtype=bool)
    else:
        air_blocked = air > 0.0  # any air -> blocked

    # final blocked map
    blocked = building_blocked | fire_blocked | air_blocked

    start = user_cell()

    # try centers until one is reachable
    chosen = None
    chosen_goal = None
    chosen_path = None

    for c in centers:
        goal = latlon_to_cell(c["lat"], c["lon"], lat, lon)
        res = a_star(blocked, start, goal)
        if res["status"] == "ok":
            chosen = c
            chosen_goal = goal
            chosen_path = res["path"]
            break

    warning = "ALERTA: Evacúa" if lang == "es" else "ALERT: Evacuate"

    return jsonify({
        "time_utc": now_utc(),
        "mode": "now",
        "input": {"lat": lat, "lon": lon, "lang": lang, "disability": disability},
        "grid": {"w": W, "h": H, "cell_m": CELL_M, "radius_m": RADIUS_M},
        "rules": {
            "fire": "always blocked",
            "air": "healthy allowed, disability blocked",
            "buildings": "non-emergency buildings blocked"
        },
        "objects": {
            "centers": centers,
            "buildings_nonsafe": buildings_nonsafe
        },
        "plan": {
            "status": "ok" if chosen else "no_path",
            "center": chosen,
            "goal_cell": list(chosen_goal) if chosen_goal else None,
            "path_cells": [list(p) for p in chosen_path] if chosen_path else []
        },
        "ui": {"warning": warning}
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "time_utc": now_utc()})

if __name__ == "__main__":
    app.run(debug=True)
