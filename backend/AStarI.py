# app.py
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import heapq
from math import radians, sin, cos, sqrt, atan2
from math import log, tan, pi, floor
from datetime import datetime, timezone
from collections import defaultdict
import time
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


app = Flask(__name__)
CORS(app)

# -------------------------
# Grid settings (5 miles - CHANGE)
# -------------------------
RADIUS_M = 1200
W, H = 240, 240
CELL_M = (2 * RADIUS_M) / W  # ~10m

SAME_PLACE_THRESH_M = 20  # treat as same place only when very close
BUILDING_BLOCK_RADIUS_M = 30
TILE_SIZE = 256
DEFAULT_TILE_Z = 16
LAST_TILESET = {"zoom": None, "tiles": {}}

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
    location_name = (
        data.get("location_name")
        or request.args.get("location_name")
        or request.cookies.get("location_name")
        or ""
    )
    lat = data.get("lat") or request.args.get("lat") or request.cookies.get("lat")
    lon = data.get("lon") or request.args.get("lon") or request.cookies.get("lon")

    # If location_name is provided, it overrides cookie/request coordinates.
    if location_name:
        geo = geocode_location_name(location_name)
        if geo:
            lat, lon = geo

    if lat is None or lon is None:
        return None
    lang = data.get("lang") or request.args.get("lang") or request.cookies.get("lang") or "en"
    disability = data.get("disability") or request.args.get("disability") or request.cookies.get("disability") or "none"
    return float(lat), float(lon), str(lang), str(disability), str(location_name)

# ============================================================
# CHANGE THIS: Google / Places API calls (placeholders)
# ============================================================
import requests

PLACE_TYPES = ["hospital", "fire_station", "police"]
BUILDING_PLACE_TYPES = [
    "university",
    "school",
    "library",
    "museum",
    "supermarket",
    "shopping_mall",
    "restaurant",
    "lodging",
    "place_of_worship",
    "bank",
    "pharmacy",
]
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

def _overpass_json(query, timeout=30):
    r = requests.post(OVERPASS_API_URL, data={"data": query}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def geocode_location_name(location_name):
    if not GOOGLE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": location_name, "key": GOOGLE_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None
        loc = data["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None

def _norm_name(s):
    return " ".join((s or "").strip().lower().split())

def _place_fingerprint(place):
    loc = place.get("geometry", {}).get("location", {})
    lat = loc.get("lat")
    lon = loc.get("lng")
    if lat is None or lon is None:
        return None
    return (
        _norm_name(place.get("name")),
        round(float(lat), 5),
        round(float(lon), 5),
    )

def _nearby_search(location_lat, location_lon, place_type, radius_m):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{location_lat},{location_lon}",
        "radius": int(radius_m),
        "type": place_type,
    }
    return _paged_search(url, params)

def _nearby_keyword_search(location_lat, location_lon, keyword, radius_m):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{location_lat},{location_lon}",
        "radius": int(radius_m),
        "keyword": keyword,
    }
    return _paged_search(url, params)

def _text_search(location_lat, location_lon, query, radius_m):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{location_lat},{location_lon}",
        "radius": int(radius_m),
        "query": query,
    }
    return _paged_search(url, params)

def _paged_search(url, params, max_pages=3):
    all_results = []
    page_params = dict(params)
    for _ in range(max_pages):
        r = requests.get(url, params=page_params, timeout=15)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            return data

        all_results.extend(data.get("results", []))
        token = data.get("next_page_token")
        if not token:
            break
        time.sleep(2.0)
        page_params = {"key": params["key"], "pagetoken": token}

    return {"status": "OK" if all_results else "ZERO_RESULTS", "results": all_results}

def get_emergency_centers(lat, lon, max_results_per_type=5, radius_m=None):
    radius = RADIUS_M if radius_m is None else radius_m
    all_centers = []

    for place_type in PLACE_TYPES:
        try:
            data = _nearby_search(lat, lon, place_type, radius)
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

def collect_emergency_centers(lat, lon):
    # Expand radius progressively so geocoded locations still get reachable centers.
    radii = [RADIUS_M, 3000, 5000, 8000]
    combined = []
    seen = set()

    def add_center(c):
        key = (
            _norm_name(c.get("name")),
            c.get("type"),
            round(float(c.get("lat", 0.0)), 5),
            round(float(c.get("lon", 0.0)), 5),
        )
        if key in seen:
            return
        seen.add(key)
        combined.append(c)

    for r in radii:
        for c in get_osm_emergency_centers(lat, lon, r, max_results=60):
            add_center(c)
        for c in get_emergency_centers(lat, lon, max_results_per_type=10, radius_m=r):
            add_center(c)
        if combined:
            break

    return combined

def get_osm_emergency_centers(lat, lon, radius_m, max_results=40):
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|fire_station|police"](around:{int(radius_m)},{lat},{lon});
      way["amenity"~"hospital|fire_station|police"](around:{int(radius_m)},{lat},{lon});
      relation["amenity"~"hospital|fire_station|police"](around:{int(radius_m)},{lat},{lon});
    );
    out tags center;
    """
    try:
        data = _overpass_json(query, timeout=30)
    except (requests.RequestException, ValueError) as e:
        print(f"Overpass emergency centers fetch failed: {e}")
        return []

    centers = []
    for el in data.get("elements", [])[:max_results]:
        tags = el.get("tags", {})
        amenity = tags.get("amenity")
        if not amenity:
            continue
        lat0 = el.get("lat") or el.get("center", {}).get("lat")
        lon0 = el.get("lon") or el.get("center", {}).get("lon")
        if lat0 is None or lon0 is None:
            continue
        centers.append({
            "name": tags.get("name") or amenity.replace("_", " ").title(),
            "type": amenity,
            "lat": float(lat0),
            "lon": float(lon0),
            "place_id": f"osm:{el.get('type')}:{el.get('id')}",
        })
    return centers

def get_buildings(lat, lon, max_results_per_type=60):
    buildings = []
    seen_place_ids = set()
    seen_fingerprints = set()

    def append_unique(place, place_type):
        place_id = place.get("place_id")
        fp = _place_fingerprint(place)
        if place_id and place_id in seen_place_ids:
            return
        if fp and fp in seen_fingerprints:
            return

        loc = place.get("geometry", {}).get("location", {})
        plat = loc.get("lat")
        plon = loc.get("lng")
        if plat is None or plon is None:
            return

        if place_id:
            seen_place_ids.add(place_id)
        if fp:
            seen_fingerprints.add(fp)

        buildings.append({
            "name": place.get("name"),
            "type": place_type,
            "lat": plat,
            "lon": plon,
            "place_id": place_id
        })

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
            append_unique(place, place_type)

    # Supplement with keyword + text queries to capture named buildings/halls.
    for keyword in (
        "hall",
        "residence hall",
        "academic building",
        "dormitory",
        "university hall",
        "student center",
    ):
        try:
            data = _nearby_keyword_search(lat, lon, keyword, RADIUS_M)
        except requests.RequestException as e:
            print(f"Google nearby keyword search failed for '{keyword}': {e}")
            continue

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"Google nearby keyword search error for '{keyword}': {data.get('status')}")
            continue

        for place in data.get("results", [])[:max_results_per_type]:
            append_unique(place, "nearby_keyword")

    for query in (
        "hall",
        "building",
        "campus building",
        "university building",
        "academic hall",
    ):
        try:
            data = _text_search(lat, lon, query, RADIUS_M)
        except requests.RequestException as e:
            print(f"Google text search failed for '{query}': {e}")
            continue

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"Google text search error for '{query}': {data.get('status')}")
            continue

        for place in data.get("results", [])[:max_results_per_type]:
            append_unique(place, "textsearch")

    return buildings

def get_osm_buildings(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:25];
    (
      way["building"](around:{int(radius_m)},{lat},{lon});
      relation["building"](around:{int(radius_m)},{lat},{lon});
    );
    out tags center geom;
    """
    try:
        data = _overpass_json(query, timeout=30)
    except (requests.RequestException, ValueError) as e:
        print(f"Overpass building fetch failed: {e}")
        return []

    out = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        center = el.get("center", {})
        clat = center.get("lat")
        clon = center.get("lon")
        if clat is None or clon is None:
            continue

        geom = el.get("geometry") or []
        polygon = []
        for p in geom:
            plat = p.get("lat")
            plon = p.get("lon")
            if plat is None or plon is None:
                continue
            polygon.append({"lat": float(plat), "lon": float(plon)})

        out.append({
            "name": tags.get("name") or tags.get("addr:housename") or "Building",
            "type": "osm_building",
            "lat": float(clat),
            "lon": float(clon),
            "place_id": f"osm:{el.get('type')}:{el.get('id')}",
            "polygon": polygon if len(polygon) >= 3 else None,
        })
    return out

def get_osm_walk_graph(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"footway|path|pedestrian|living_street|residential|service|tertiary|secondary|primary|unclassified|steps|cycleway|track"](around:{int(radius_m)},{lat},{lon});
    );
    (._;>;);
    out body;
    """
    try:
        data = _overpass_json(query, timeout=35)
    except (requests.RequestException, ValueError) as e:
        print(f"Overpass walk graph fetch failed: {e}")
        return {}, {}

    node_coords = {}
    ways = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            node_coords[el["id"]] = (float(el["lat"]), float(el["lon"]))
        elif el.get("type") == "way":
            nodes = el.get("nodes") or []
            if len(nodes) >= 2:
                ways.append(nodes)

    adj = defaultdict(dict)
    for node_list in ways:
        for i in range(len(node_list) - 1):
            a = node_list[i]
            b = node_list[i + 1]
            if a not in node_coords or b not in node_coords:
                continue
            lat1, lon1 = node_coords[a]
            lat2, lon2 = node_coords[b]
            w = haversine_m(lat1, lon1, lat2, lon2)
            if b not in adj[a] or w < adj[a][b]:
                adj[a][b] = w
                adj[b][a] = w
    return node_coords, adj

def point_in_latlon_polygon(lat, lon, poly):
    # Ray cast with x=lon, y=lat
    x, y = lon, lat
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi = poly[i]["lon"]
        yi = poly[i]["lat"]
        xj = poly[j]["lon"]
        yj = poly[j]["lat"]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside

def edge_crosses_building(a_lat, a_lon, b_lat, b_lon, building_polys):
    # Sample points along edge; practical approximation for obstacle crossing.
    samples = 8
    for t in range(samples + 1):
        k = t / samples
        lat = a_lat + (b_lat - a_lat) * k
        lon = a_lon + (b_lon - a_lon) * k
        for poly in building_polys:
            if point_in_latlon_polygon(lat, lon, poly):
                return True
    return False

def nearest_graph_node(target_lat, target_lon, node_coords, allowed_nodes=None):
    best_id = None
    best_d = float("inf")
    for nid, (nlat, nlon) in node_coords.items():
        if allowed_nodes is not None and nid not in allowed_nodes:
            continue
        d = haversine_m(target_lat, target_lon, nlat, nlon)
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id

def a_star_graph(adj, node_coords, start_id, goal_id):
    if start_id is None or goal_id is None:
        return {"status": "no_path", "path": []}
    if start_id == goal_id:
        return {"status": "ok", "path": [start_id]}

    def h(nid):
        nlat, nlon = node_coords[nid]
        glat, glon = node_coords[goal_id]
        return haversine_m(nlat, nlon, glat, glon)

    pq = [(h(start_id), 0.0, start_id)]
    best = {start_id: 0.0}
    parent = {}

    while pq:
        _, g, nid = heapq.heappop(pq)
        if nid == goal_id:
            path = [nid]
            while nid in parent:
                nid = parent[nid]
                path.append(nid)
            path.reverse()
            return {"status": "ok", "path": path}

        for nxt, w in adj.get(nid, {}).items():
            ng = g + w
            if nxt not in best or ng < best[nxt]:
                best[nxt] = ng
                parent[nxt] = nid
                heapq.heappush(pq, (ng + h(nxt), ng, nxt))

    return {"status": "no_path", "path": []}

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
    # Grid mapping in meters, consistent with cell_to_latlon.
    meters_per_deg_lon = max(1e-6, 111320.0 * cos(radians(center_lat)))
    dx_m = (lon - center_lon) * meters_per_deg_lon
    dy_m = (center_lat - lat) * 111320.0
    x = int(round((W // 2) + (dx_m / CELL_M)))
    y = int(round((H // 2) + (dy_m / CELL_M)))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return (x, y)

def cell_to_latlon(cell_x, cell_y, center_lat, center_lon):
    # Converts grid cell center to approximate geographic coordinates.
    dx_m = (cell_x - (W // 2)) * CELL_M
    dy_m = (cell_y - (H // 2)) * CELL_M
    lat = center_lat - (dy_m / 111320.0)
    lon_denom = max(1e-6, 111320.0 * cos(radians(center_lat)))
    lon = center_lon + (dx_m / lon_denom)
    return lat, lon

def latlon_to_world_px(lat, lon, z):
    n = (2 ** z) * TILE_SIZE
    x = (lon + 180.0) / 360.0 * n
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_rad = radians(lat)
    y = (1.0 - log(tan(lat_rad) + (1.0 / cos(lat_rad))) / pi) / 2.0 * n
    return x, y

def latlon_to_tile_coord(lat, lon, z):
    world_x, world_y = latlon_to_world_px(lat, lon, z)
    return int(floor(world_x / TILE_SIZE)), int(floor(world_y / TILE_SIZE))

def build_tile_index(user_lat, user_lon, building_blocked, fire_blocked, air_blocked, centers, start_cell, path_cells, zoom=DEFAULT_TILE_Z):
    tiles = {}

    def add_point(layer, cell_x, cell_y):
        lat, lon = cell_to_latlon(cell_x, cell_y, user_lat, user_lon)
        world_x, world_y = latlon_to_world_px(lat, lon, zoom)
        tx = int(floor(world_x / TILE_SIZE))
        ty = int(floor(world_y / TILE_SIZE))
        px = int(world_x - (tx * TILE_SIZE))
        py = int(world_y - (ty * TILE_SIZE))
        tkey = f"{zoom}/{tx}/{ty}"
        if tkey not in tiles:
            tiles[tkey] = {
                "z": zoom,
                "x": tx,
                "y": ty,
                "layers": {
                    "buildings": [],
                    "fire": [],
                    "air": [],
                    "centers": [],
                    "start": [],
                    "path": [],
                },
            }
        tiles[tkey]["layers"][layer].append({
            "cell": [int(cell_x), int(cell_y)],
            "pixel": [px, py],
            "lat": lat,
            "lon": lon,
        })

    by, bx = np.where(building_blocked)
    for y, x in zip(by, bx):
        add_point("buildings", int(x), int(y))

    fy, fx = np.where(fire_blocked)
    for y, x in zip(fy, fx):
        add_point("fire", int(x), int(y))

    ay, ax = np.where(air_blocked)
    for y, x in zip(ay, ax):
        add_point("air", int(x), int(y))

    for c in centers:
        cx, cy = latlon_to_cell(c["lat"], c["lon"], user_lat, user_lon)
        add_point("centers", int(cx), int(cy))

    if start_cell:
        add_point("start", int(start_cell[0]), int(start_cell[1]))

    for p in path_cells or []:
        add_point("path", int(p[0]), int(p[1]))

    return tiles

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

def dedupe_buildings(buildings):
    out = []
    seen = set()
    for b in buildings:
        key = (
            _norm_name(b.get("name")),
            round(float(b.get("lat", 0.0)), 5),
            round(float(b.get("lon", 0.0)), 5),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out

def point_in_polygon(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside

def buildings_to_blocked(buildings, user_lat, user_lon):
    blocked = np.zeros((H, W), dtype=bool)
    pad_cells = max(1, int(np.ceil(BUILDING_BLOCK_RADIUS_M / CELL_M)))
    for b in buildings:
        polygon = b.get("polygon")
        if polygon:
            poly_cells = [latlon_to_cell(p["lat"], p["lon"], user_lat, user_lon) for p in polygon]
            xs = [p[0] for p in poly_cells]
            ys = [p[1] for p in poly_cells]
            min_x = max(0, min(xs))
            max_x = min(W - 1, max(xs))
            min_y = max(0, min(ys))
            max_y = min(H - 1, max(ys))
            for ny in range(min_y, max_y + 1):
                for nx in range(min_x, max_x + 1):
                    if point_in_polygon(nx + 0.5, ny + 0.5, poly_cells):
                        blocked[ny, nx] = True
            continue

        x, y = latlon_to_cell(b["lat"], b["lon"], user_lat, user_lon)
        for ny in range(max(0, y - pad_cells), min(H, y + pad_cells + 1)):
            for nx in range(max(0, x - pad_cells), min(W, x + pad_cells + 1)):
                if (nx - x) ** 2 + (ny - y) ** 2 <= pad_cells ** 2:
                    blocked[ny, nx] = True
    return blocked

def carve_open_disk(mask, cx, cy, radius_cells=2):
    for ny in range(max(0, cy - radius_cells), min(H, cy + radius_cells + 1)):
        for nx in range(max(0, cx - radius_cells), min(W, cx + radius_cells + 1)):
            if (nx - cx) ** 2 + (ny - cy) ** 2 <= radius_cells ** 2:
                mask[ny, nx] = False

def mask_to_cells(mask):
    ys, xs = np.where(mask)
    return [[int(x), int(y)] for y, x in zip(ys, xs)]

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
        return jsonify({"error": "Send lat/lon or location_name via JSON, query, or cookies."}), 400

    lat, lon, lang, disability, location_name = got

    centers = collect_emergency_centers(lat, lon)
    buildings = dedupe_buildings(get_osm_buildings(lat, lon, RADIUS_M))

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
    carve_open_disk(blocked, start[0], start[1], radius_cells=2)
    center_cells = [list(latlon_to_cell(c["lat"], c["lon"], lat, lon)) for c in centers]

    # OSM graph routing: vertices are OSM nodes, edges are walkable ways.
    node_coords, raw_adj = get_osm_walk_graph(lat, lon, RADIUS_M)
    building_polys = [b.get("polygon") for b in buildings_nonsafe if b.get("polygon")]

    allowed_nodes = set()
    for nid, (nlat, nlon) in node_coords.items():
        if any(point_in_latlon_polygon(nlat, nlon, poly) for poly in building_polys):
            continue
        allowed_nodes.add(nid)

    adj = defaultdict(dict)
    for a, nbrs in raw_adj.items():
        if a not in allowed_nodes:
            continue
        a_lat, a_lon = node_coords[a]
        for b, w in nbrs.items():
            if b not in allowed_nodes:
                continue
            b_lat, b_lon = node_coords[b]
            if edge_crosses_building(a_lat, a_lon, b_lat, b_lon, building_polys):
                continue
            adj[a][b] = w

    # try centers until one is reachable
    chosen = None
    chosen_goal = None
    chosen_path = None
    graph_path_latlng = []

    start_node = nearest_graph_node(lat, lon, node_coords, allowed_nodes=allowed_nodes) if node_coords else None
    if start_node is not None and centers:
        for c in centers:
            goal_node = nearest_graph_node(c["lat"], c["lon"], node_coords, allowed_nodes=allowed_nodes)
            if goal_node is None:
                continue
            res = a_star_graph(adj, node_coords, start_node, goal_node)
            if res["status"] == "ok":
                chosen = c
                chosen_goal = latlon_to_cell(c["lat"], c["lon"], lat, lon)
                graph_path_latlng = [
                    {"lat": node_coords[nid][0], "lng": node_coords[nid][1]}
                    for nid in res["path"]
                ]
                chosen_path = [latlon_to_cell(p["lat"], p["lng"], lat, lon) for p in graph_path_latlng]
                break

    # Fallback: no graph path -> straight line to nearest discovered center.
    if not chosen and centers:
        chosen = centers[0]
        chosen_goal = latlon_to_cell(chosen["lat"], chosen["lon"], lat, lon)
        graph_path_latlng = [
            {"lat": lat, "lng": lon},
            {"lat": chosen["lat"], "lng": chosen["lon"]},
        ]
        chosen_path = [start, chosen_goal]

    # Final fallback: if no centers found, route to nearest walk-graph node.
    if not chosen and node_coords:
        nearest_id = nearest_graph_node(lat, lon, node_coords, allowed_nodes=allowed_nodes)
        if nearest_id is not None:
            nlat, nlon = node_coords[nearest_id]
            chosen = {
                "name": "Nearest walkable point",
                "type": "fallback",
                "lat": nlat,
                "lon": nlon,
                "place_id": f"osm:node:{nearest_id}",
            }
            chosen_goal = latlon_to_cell(nlat, nlon, lat, lon)
            graph_path_latlng = [
                {"lat": lat, "lng": lon},
                {"lat": nlat, "lng": nlon},
            ]
            chosen_path = [start, chosen_goal]

    warning = "ALERTA: Evacúa" if lang == "es" else "ALERT: Evacuate"
    path_latlng = graph_path_latlng or [
        {"lat": cell_to_latlon(p[0], p[1], lat, lon)[0], "lng": cell_to_latlon(p[0], p[1], lat, lon)[1]}
        for p in (chosen_path or [])
    ]
    tile_index = build_tile_index(
        lat,
        lon,
        building_blocked,
        fire_blocked,
        air_blocked,
        centers,
        start,
        chosen_path,
        zoom=DEFAULT_TILE_Z,
    )
    LAST_TILESET["zoom"] = DEFAULT_TILE_Z
    LAST_TILESET["tiles"] = tile_index

    return jsonify({
        "time_utc": now_utc(),
        "mode": "now",
        "input": {
            "lat": lat,
            "lon": lon,
            "lang": lang,
            "disability": disability,
            "location_name": location_name
        },
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
        "blocked_cells": {
            "buildings": mask_to_cells(building_blocked),
            "fire": mask_to_cells(fire_blocked),
            "air": mask_to_cells(air_blocked),
            "all": mask_to_cells(blocked),
        },
        "map": {
            "start_cell": list(start),
            "center_cells": center_cells,
        },
        "plan": {
            "status": "ok" if chosen else "no_path",
            "center": chosen,
            "goal_cell": list(chosen_goal) if chosen_goal else None,
            "path_cells": [list(p) for p in chosen_path] if chosen_path else [],
            "path_latlng": path_latlng
        },
        "map_tiles_2d": {
            "format": "z/x/y",
            "spec_url": "https://developers.google.com/maps/documentation/tile/2d-tiles-overview",
            "tile_url_template": "/evac/2dtiles/{z}/{x}/{y}",
            "zoom": DEFAULT_TILE_Z,
            "tile_count": len(tile_index),
            "tiles": [tile_index[k] for k in sorted(tile_index.keys())],
        },
        "ui": {"warning": warning}
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "time_utc": now_utc()})

@app.get("/evac/maps/config")
def maps_config():
    return jsonify({
        "has_google_maps_api_key": bool(GOOGLE_API_KEY),
        "google_maps_api_key": GOOGLE_API_KEY if GOOGLE_API_KEY else None,
    })

@app.get("/evac/2dtiles/<int:z>/<int:x>/<int:y>")
def evac_2d_tile(z, x, y):
    if LAST_TILESET.get("zoom") != z:
        return jsonify({"error": "No tile data for this zoom. Run /evac/now first."}), 404
    tile = LAST_TILESET.get("tiles", {}).get(f"{z}/{x}/{y}")
    if not tile:
        return jsonify({
            "z": z,
            "x": x,
            "y": y,
            "layers": {
                "buildings": [],
                "fire": [],
                "air": [],
                "centers": [],
                "start": [],
                "path": [],
            },
        })
    return jsonify(tile)

if __name__ == "__main__":
    app.run(debug=True)
