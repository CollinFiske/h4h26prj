'''
  "."  — free / passable
  "F"  — fire zone      (burn_probability >= FIRE_THRESH)   blocked
  "A"  — smoke/air zone (hazard_pred_class == 1, medium)    passable but costly
  "H"  — high hazard    (hazard_pred_class == 2, high)      blocked
  "#"  — physical obstacle (reserved; not used in geo mode)
  "S"  — start (user location)
  "G"  — safe evacuation goal

Coordinate system
The user will be in the center of the grid, each square on the grid expands to roughly 90m of terrain. The grid is 25×25 squares, so it covers about a 1.1km radius around the user.  The A* pathfinding operates on this grid, treating "F" and "H" cells as blocked and "A" cells as passable but with a higher cost.

'''

from __future__ import annotations
import heapq
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


from routes.realtime_data import (
    get_nearest_calfire_perimeter_metrics,
    vc_hourly_wind,
    slope_proxy_from_elevation,
    downwind_alignment,
    wind_components_uv_ms,
    )
# Grid and pathfinding parameters — these can be tuned based on real-world testing and user feedback.

CELL_M         = 90          # metres per grid cell (matches USGS elevation step)
GRID_RADIUS    = 12          # cells in each direction from centre → 25×25 grid
FIRE_THRESH    = 0.55        # burn_probability above this from model1 cell marked "F"
SMOKE_COST     = 6           # extra A* cost for medium-hazard cells (non-disabled person can pass but it's undesirable)
NUM_SAFE_ZONES = 3           # how many evac destinations to search for

# known safe destination types — in a real deployment these would come from
# Google Places (shelters, hospitals, community centres) via the GOOGLE_API_KEY.
_SAFE_ZONE_LABELS = ["Emergency Shelter", "Community Centre", "Hospital"]


# coordinate helpers 

def _metres_per_deg_lat(lat: float) -> float:
    return 111_320.0
def _metres_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))

def latlon_to_cell(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[int, int]:
    #convert (lat, lon) to integer grid (col, row) relative to center
    dy = (lat - center_lat) * _metres_per_deg_lat(center_lat)
    dx = (lon - center_lon) * _metres_per_deg_lon(center_lat)
    col = int(round(dx / CELL_M)) + GRID_RADIUS
    row = int(round(-dy / CELL_M)) + GRID_RADIUS   # row increases downward
    return col, row


def cell_to_latlon(col: int, row: int, center_lat: float, center_lon: float) -> tuple[float, float]:
    #Convert grid (col, row) back to (lat, lon).
    dx = (col - GRID_RADIUS) * CELL_M
    dy = -(row - GRID_RADIUS) * CELL_M
    lat = center_lat + dy / _metres_per_deg_lat(center_lat)
    lon = center_lon + dx / _metres_per_deg_lon(center_lat)
    return lat, lon

#  Grid builder and A* pathfinder on the grid.  The main public interface is the EvacuationRouter class at the bottom, which is instantiated and called from the Flask route handler in routes/predict.py.

@dataclass
class GridCell:
    col: int
    row: int
    lat: float
    lon: float
    cell_type: str = "."          # "." | "F" | "A" | "H" | "S" | "G"
    burn_prob: float = 0.0
    hazard_class: int = 0
    heat_weight: float = 0.0


def build_hazard_grid(
    center_lat: float,
    center_lon: float,
    fire_svc,                      # FireHazardService wrapper
    radius: int = GRID_RADIUS,
) -> list[list[GridCell]]:
    """
    Query the ML models for every cell in the grid and classify each cell.

    25×25 grid -> 625 ML inference calls.  They are batched into
    a single DataFrame predict so it's actually one model call per model.
    """
    size   = 2 * radius + 1
    now_iso = datetime.now(timezone.utc).isoformat()

    # Build batch of all grid points
    rows = []
    for r in range(size):
        for c in range(size):
            lat, lon = cell_to_latlon(c, r, center_lat, center_lon)
            rows.append({"col": c, "row": r, "lat": lat, "lon": lon})

    points_df = pd.DataFrame(rows)

    # We need the perimeter metrics for every point.  For speed we reuse the
    # same perimeter metrics (fetched once for the centre) for the whole grid —
    # the fire is far enough away that the same perimeter applies to all cells.
    
    import os
    from datetime import timedelta

    vc_key = os.getenv("VISUAL_CROSSING_API_KEY")
    target_time = datetime.now(timezone.utc) + timedelta(hours=1)
    # exports so tests can patch pathfinder.<fn>

    # Fetch shared context once
    pm   = get_nearest_calfire_perimeter_metrics(center_lat, center_lon)
    wind = vc_hourly_wind(center_lat, center_lon, target_time, api_key=vc_key)
    wind_speed_ms  = wind["wind_speed_ms"]
    wind_dir_deg_to = wind["wind_dir_deg_to"]
    wind_u, wind_v  = wind_components_uv_ms(wind_speed_ms, wind_dir_deg_to)

    # Build one point dict per cell (slope varies per cell, everything else shared)
    point_dicts = []
    for _, row in points_df.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        align = downwind_alignment(lat, lon, pm["center_lat"], pm["center_lon"], wind_dir_deg_to)
        slope = slope_proxy_from_elevation(lat, lon)   # LRU-cached

        point_dicts.append({
            "time_utc":           now_iso,
            "lat":                lat,
            "lon":                lon,
            "wind_speed_ms":      wind_speed_ms,
            "wind_dir_deg_to":    wind_dir_deg_to,
            "wind_u_ms":          wind_u,
            "wind_v_ms":          wind_v,
            "dist_to_center_km":  pm["dist_to_center_km"],
            "downwind_alignment": align,
            "r_boundary_km":      pm["r_boundary_km"],
            "dist_to_front_km":   pm["dist_to_front_km"],
            "slope_proxy":        slope,
        })

    # Batch inference using the existing FireHazardService
    # predict_one() operates on a single dict; we call it per cell.
    # (Future optimisation: add a predict_batch() to FireHazardService)
    grid: list[list[GridCell]] = [[None] * size for _ in range(size)]

    for i, (pt, meta_row) in enumerate(zip(point_dicts, points_df.itertuples())):
        try:
            ml = fire_svc.predict_one(pt)
        except Exception:
            logger.warning("ML inference failed for cell (%d,%d), marking safe", meta_row.col, meta_row.row)
            ml = {"burn_probability": 0.0, "hazard_pred_class": 0, "heat_weight": 0.0}

        burn_prob     = ml["burn_probability"]
        hazard_class  = ml["hazard_pred_class"]
        heat_weight   = ml["heat_weight"]

        # Classify cell
        if burn_prob >= FIRE_THRESH:
            cell_type = "F"
        elif hazard_class == 2:
            cell_type = "H"        # high hazard → treat as blocked like fire
        elif hazard_class == 1:
            cell_type = "A"        # medium hazard → passable but costly
        else:
            cell_type = "."

        grid[meta_row.row][meta_row.col] = GridCell(
            col=meta_row.col,
            row=meta_row.row,
            lat=float(meta_row.lat),
            lon=float(meta_row.lon),
            cell_type=cell_type,
            burn_prob=burn_prob,
            hazard_class=hazard_class,
            heat_weight=heat_weight,
        )

    return grid

# safe zone placement 

def _find_safe_zone_candidates(
    grid: list[list[GridCell]],
    num: int = NUM_SAFE_ZONES,
) -> list[tuple[int, int]]:
    """
    Place evacuation goals at free cells near the grid edges that have the
    lowest cumulative hazard in their neighbourhood.

    In production these would be replaced by real shelter locations from the
    Google Places API (using GOOGLE_API_KEY).
    """
    size = len(grid)
    edge_cells: list[tuple[float, int, int]] = []

    for r in range(size):
        for c in range(size):
            cell = grid[r][c]
            if cell.cell_type not in (".", "A"):
                continue
            # Prefer cells near the edges (far from centre = away from fire)
            dist_from_centre = math.hypot(c - GRID_RADIUS, r - GRID_RADIUS)
            # Score: prefer far from centre and low heat_weight
            score = -dist_from_centre + cell.heat_weight * 10
            edge_cells.append((score, c, r))

    edge_cells.sort()
    chosen: list[tuple[int, int]] = []
    min_sep = GRID_RADIUS // 2    # goals must be at least this far apart

    for _, c, r in edge_cells:
        if len(chosen) >= num:
            break
        too_close = any(
            math.hypot(c - gc, r - gr) < min_sep for gc, gr in chosen
        )
        if not too_close:
            chosen.append((c, r))

    return chosen


#  A* on grid 

def _in_bounds(col: int, row: int, size: int) -> bool:
    return 0 <= col < size and 0 <= row < size


def _is_blocked(cell: GridCell, has_disability: bool) -> bool:
    """Fire and high-hazard are always blocked.  Smoke is blocked if disabled."""
    if cell.cell_type in ("F", "H"):
        return True
    if has_disability and cell.cell_type == "A":
        return True
    return False


def _step_cost(cell: GridCell, has_disability: bool) -> float:
    if not has_disability and cell.cell_type == "A":
        return 1.0 + SMOKE_COST
    # Also penalise by heat_weight so the path naturally avoids borderline cells
    return 1.0 + cell.heat_weight * 2.0


def _heuristic(col: int, row: int, goals: set[tuple[int, int]]) -> float:
    return min(abs(col - gc) + abs(row - gr) for gc, gr in goals)

def run_astar(
    grid: list[list[GridCell]],
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    has_disability: bool,
) -> tuple[Optional[tuple[int, int]], Optional[list[tuple[int, int]]], Optional[float]]:
    """
    A* multi-goal search on the hazard grid.

    Returns (chosen_goal, path_as_col_row_list, total_cost) or (None, None, None).
    """
    size = len(grid)
    sc, sr = start
    pq      = [(float(_heuristic(sc, sr, goals)), 0.0, sc, sr)]
    best_g  = {(sc, sr): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    moves   = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while pq:
        _, g, c, r = heapq.heappop(pq)

        if (c, r) in goals:
            # Reconstruct path
            path, node = [], (c, r)
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start)
            path.reverse()
            return (c, r), path, g

        if g > best_g.get((c, r), math.inf) + 1e-9:
            continue

        for dc, dr in moves:
            nc, nr = c + dc, r + dr
            if not _in_bounds(nc, nr, size):
                continue
            neighbour = grid[nr][nc]
            if _is_blocked(neighbour, has_disability):
                continue

            ng = g + _step_cost(neighbour, has_disability)
            if ng < best_g.get((nc, nr), math.inf):
                best_g[(nc, nr)] = ng
                parent[(nc, nr)] = (c, r)
                h = _heuristic(nc, nr, goals)
                heapq.heappush(pq, (ng + h, ng, nc, nr))

    return None, None, None


# public interface 

@dataclass
class EvacuationRouter:
    """
    High-level router.  Instantiate once per prediction request.

    Parameters
    ----------
    center_lat / center_lon : user's location
    has_disability          : affects which cells are passable
    fire_svc                : FireHazardService instance (already loaded)
    """
    center_lat:     float
    center_lon:     float
    has_disability: bool  = False
    fire_svc:       object = field(default=None, repr=False)

    def run(self) -> dict:
        """
        Build the hazard grid, find safe zones, run A*, and return results.

        Returns
        -------
        {
            "path_latlon":   [(lat, lon), ...],   # waypoints for map polyline
            "safe_zone":     {"lat": ..., "lon": ..., "label": "Emergency Shelter"},
            "path_col_row":  [(col, row), ...],   # raw grid indices (debug/viz)
            "grid_stats":    {"free": int, "fire": int, "smoke": int, "high_hazard": int},
            "reachable":     bool,
            "route_cost":    float | None,
        }
        """
        logger.info("Building hazard grid centred on (%.5f, %.5f)", self.center_lat, self.center_lon)
        grid = build_hazard_grid(self.center_lat, self.center_lon, self.fire_svc)

        size  = len(grid)
        start = (GRID_RADIUS, GRID_RADIUS)    # user is at centre
        grid[start[1]][start[0]].cell_type = "S"

        # Safe zone goals
        goal_cells = _find_safe_zone_candidates(grid, num=NUM_SAFE_ZONES)
        if not goal_cells:
            return {
                "path_latlon":  [],
                "safe_zone":    None,
                "path_col_row": [],
                "grid_stats":   _grid_stats(grid),
                "reachable":    False,
                "route_cost":   None,
            }

        goals = set(goal_cells)
        for gc, gr in goals:
            grid[gr][gc].cell_type = "G"

        chosen, path_cr, cost = run_astar(grid, start, goals, self.has_disability)

        if chosen is None:
            return {
                "path_latlon":  [],
                "safe_zone":    None,
                "path_col_row": [],
                "grid_stats":   _grid_stats(grid),
                "reachable":    False,
                "route_cost":   None,
            }

        # Convert grid path → lat/lon waypoints
        path_ll = [
            cell_to_latlon(c, r, self.center_lat, self.center_lon)
            for c, r in path_cr
        ]

        goal_lat, goal_lon = cell_to_latlon(chosen[0], chosen[1], self.center_lat, self.center_lon)

        # Label the chosen goal (index into the ordered goal list)
        goal_idx   = goal_cells.index(chosen) if chosen in goal_cells else 0
        goal_label = _SAFE_ZONE_LABELS[goal_idx % len(_SAFE_ZONE_LABELS)]

        return {
            "path_latlon": [{"lat": lat, "lon": lon} for lat, lon in path_ll],
            "safe_zone":   {"lat": goal_lat, "lon": goal_lon, "label": goal_label},
            "path_col_row": [{"col": c, "row": r} for c, r in path_cr],
            "grid_stats":  _grid_stats(grid),
            "reachable":   True,
            "route_cost":  float(cost),
        }


def _grid_stats(grid: list[list[GridCell]]) -> dict:
    counts = {"free": 0, "fire": 0, "smoke": 0, "high_hazard": 0, "start": 0, "goal": 0}
    for row in grid:
        for cell in row:
            mapping = {".": "free", "F": "fire", "A": "smoke", "H": "high_hazard", "S": "start", "G": "goal"}
            counts[mapping.get(cell.cell_type, "free")] += 1
    return counts

