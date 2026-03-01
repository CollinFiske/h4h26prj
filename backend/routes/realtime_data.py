"""
    CAL FIRE ArcGIS FeatureServer  (public, no key)
    Visual Crossing Timeline API   (key via env VISUAL_CROSSING_API_KEY)
    USGS Elevation Point Query     (public, no key)
"""

import os
import math
import functools
import requests
from datetime import datetime, timedelta, timezone

from shapely.geometry import shape, Point
from shapely.ops import transform
from pyproj import Transformer


# coord transformers 

_TO_3310 = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True).transform
_TO_4326 = Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True).transform

CALFIRE_LAYER0 = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Historic_Fire_Perimeters/FeatureServer/0"
)


# CAL FIRE perimeter queries 

def query_calfire_perimeters_near_point(
    lat: float,
    lon: float,
    search_km: float = 100.0,
    max_features: int = 25,
    where: str = "YEAR_ >= 1950",
) -> dict:
    """Return GeoJSON FeatureCollection of perimeters within search_km of (lat, lon)."""
    url    = f"{CALFIRE_LAYER0}/query"
    params = {
        "where":             where,
        "geometry":          f"{lon},{lat}",
        "geometryType":      "esriGeometryPoint",
        "inSR":              "4326",
        "spatialRel":        "esriSpatialRelIntersects",
        "distance":          str(search_km),
        "units":             "esriSRUnit_Kilometer",
        "outFields":         "FIRE_NAME,YEAR_,ALARM_DATE,CONT_DATE,GIS_ACRES,IRWINID,INC_NUM",
        "returnGeometry":    "true",
        "outSR":             "4326",
        "f":                 "geojson",
        "resultRecordCount": str(max_features),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _choose_nearest_feature(feature_collection: dict, lat: float, lon: float):
    """Return (best_feature, dist_km) for the perimeter boundary closest to (lat, lon)."""
    feats = feature_collection.get("features") or []
    if not feats:
        raise ValueError("No candidate perimeters returned from CAL FIRE query.")

    pt_m = transform(_TO_3310, Point(lon, lat))
    best_feat, best_dist_km = None, None

    for feat in feats:
        geom_m  = transform(_TO_3310, shape(feat["geometry"]))
        d_km    = pt_m.distance(geom_m.boundary) / 1000.0
        if best_dist_km is None or d_km < best_dist_km:
            best_dist_km = d_km
            best_feat    = feat

    return best_feat, float(best_dist_km)


def _perimeter_metrics(feat: dict, lat: float, lon: float) -> dict:
    """Extract geometric + metadata fields from a single perimeter feature."""
    geom_m     = transform(_TO_3310, shape(feat["geometry"]))
    pt_m       = transform(_TO_3310, Point(lon, lat))
    centroid_m = geom_m.centroid

    center_lon, center_lat = transform(_TO_4326, centroid_m).coords[0]

    props = feat.get("properties") or {}
    return {
        "fire_name":         props.get("FIRE_NAME"),
        "year":              props.get("YEAR_"),
        "irwinid":           props.get("IRWINID"),
        "inc_num":           props.get("INC_NUM"),
        "center_lat":        float(center_lat),
        "center_lon":        float(center_lon),
        "r_boundary_km":     math.sqrt(geom_m.area / math.pi) / 1000.0,
        "dist_to_front_km":  pt_m.distance(geom_m.boundary) / 1000.0,
        "dist_to_center_km": pt_m.distance(centroid_m) / 1000.0,
    }


def get_nearest_calfire_perimeter_metrics(
    lat: float,
    lon: float,
    search_km: float = 100.0,
    max_features: int = 25,
    where: str = "YEAR_ >= 1950",
) -> dict:
    """
    High-level helper: query CAL FIRE and return metrics for the nearest perimeter.

    Returns a dict with fire_name, year, irwinid, inc_num, center_lat/lon,
    r_boundary_km, dist_to_front_km, dist_to_center_km.
    """
    gj       = query_calfire_perimeters_near_point(lat, lon, search_km, max_features, where)
    best, _  = _choose_nearest_feature(gj, lat, lon)
    return _perimeter_metrics(best, lat, lon)


# bearing /vector helpers 

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from (lat1, lon1) to (lat2, lon2), degrees clockwise from north."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x    = math.sin(dlon) * math.cos(phi2)
    y    = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _unit_vec(bearing: float) -> tuple[float, float]:
    """(east, north) unit vector for a bearing (0=N, 90=E)."""
    ang = math.radians(bearing)
    return math.sin(ang), math.cos(ang)


def wind_components_uv_ms(
    wind_speed_ms: float, wind_dir_deg_to: float
) -> tuple[float, float]:
    """u=eastward, v=northward wind components in m/s."""
    ux, uy = _unit_vec(wind_dir_deg_to)
    return wind_speed_ms * ux, wind_speed_ms * uy


def downwind_alignment(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    wind_dir_deg_to: float,
) -> float:
    """
    Dot product of wind direction and the center→point unit vector, in [-1, 1].

    +1 means the point is directly downwind of the fire center.
    -1 means it is directly upwind.
    """
    b        = bearing_deg(center_lat, center_lon, lat, lon)
    px, py   = _unit_vec(b)
    wx, wy   = _unit_vec(wind_dir_deg_to)
    return max(-1.0, min(1.0, wx * px + wy * py))


#  weather: Visual Crossing hourly wind 

def vc_hourly_wind(
    lat: float,
    lon: float,
    target_time_utc: datetime,
    api_key: str,
    wind_dir_is_from: bool = True,
) -> dict:
    """
    Fetch hourly wind forecast from Visual Crossing nearest to target_time_utc.

    Param:
    wind_dir_is_from : Visual Crossing reports wind direction as 'from' by default.
                       When True (default), the direction is flipped 180° so the
                       model receives the 'toward' convention it was trained on.

    Returns:
    dict with:
        time_utc       : datetime (UTC, minute/second zeroed)
        wind_speed_ms  : float — speed in **m/s** (converted from km/h)
        wind_dir_deg_to: float — direction wind blows TOWARD (degrees)

    """
    url    = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{lat},{lon}"
    )
    params = {
        "key":         api_key,
        "unitGroup":   "metric",       # windspeed → km/h
        "include":     "hours",
        "contentType": "json",
        "timezone":    "UTC",
        "forecastDays": 2,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    target_hour = target_time_utc.replace(minute=0, second=0, microsecond=0)
    best, best_dt = None, None

    for day in data.get("days", []):
        day_date = day.get("datetime")
        for hr in day.get("hours", []):
            dt = datetime.fromisoformat(
                f"{day_date}T{hr['datetime']}"
            ).replace(tzinfo=timezone.utc)
            if best is None or abs(dt - target_hour) < abs(best_dt - target_hour):
                best    = hr
                best_dt = dt

    if best is None:
        raise RuntimeError("No hourly forecast found in Visual Crossing response.")

    wind_speed_kph = float(best.get("windspeed", 0.0))
    wind_speed_ms  = wind_speed_kph / 3.6          # km/h → m/s

    wind_dir_raw   = float(best.get("winddir", 0.0))
    wind_dir_to    = (wind_dir_raw + 180.0) % 360.0 if wind_dir_is_from else wind_dir_raw

    return {
        "time_utc":        target_hour,
        "wind_speed_ms":   wind_speed_ms,
        "wind_dir_deg_to": wind_dir_to,
    }


# terrain: USGS elevation + slope proxy 

@functools.lru_cache(maxsize=1024)
def _epqs_elevation_m(lat: float, lon: float) -> float:
    """Fetch elevation from USGS EPQS (cached by lat/lon)."""
    r = requests.get(
        "https://epqs.nationalmap.gov/v1/json",
        params={"x": lon, "y": lat, "units": "Meters"},
        timeout=20,
    )
    r.raise_for_status()
    return float(r.json()["value"])


@functools.lru_cache(maxsize=512)
def slope_proxy_from_elevation(lat: float, lon: float, step_m: float = 90.0) -> float:
    """
    Estimate terrain slope at (lat, lon) by sampling elevation at 4 neighbouring
    points (N/S/E/W at step_m metres apart).

    Returns slope_proxy in [0, 1] where 30° → 1.0.

    Results are LRU-cached, so repeated calls for the same coordinate are free.
    """
    dlat = step_m / 111_320.0
    dlon = step_m / (111_320.0 * math.cos(math.radians(lat)) + 1e-9)

    zN = _epqs_elevation_m(lat + dlat, lon)
    zS = _epqs_elevation_m(lat - dlat, lon)
    zE = _epqs_elevation_m(lat, lon + dlon)
    zW = _epqs_elevation_m(lat, lon - dlon)

    dz_dy = (zN - zS) / (2.0 * step_m)
    dz_dx = (zE - zW) / (2.0 * step_m)
    slope_deg = math.degrees(math.atan(math.hypot(dz_dx, dz_dy)))

    return max(0.0, min(1.0, slope_deg / 30.0))


# main 

def build_point_next_hour(
    lat: float,
    lon: float,
    pm: dict | None = None,
    *,
    vc_api_key: str | None = None,
    incident_created_time_utc: datetime | None = None,
    search_km_if_pm_missing: float = 100.0,
) -> dict:
    """
    Build the feature dict consumed by FireHazardService.predict_one().

    param:
    lat / lon                  : location of interest
    pm                         : output of get_nearest_calfire_perimeter_metrics();
                                 if None, it is fetched automatically
    vc_api_key                 : Visual Crossing key (falls back to env var
                                 VISUAL_CROSSING_API_KEY)
    incident_created_time_utc  : if provided, adds hour_index to the point dict
    search_km_if_pm_missing    : radius used when pm is fetched automatically
    """
    if pm is None:
        pm = get_nearest_calfire_perimeter_metrics(
            lat=lat, lon=lon, search_km=search_km_if_pm_missing
        )

    vc_api_key = vc_api_key or os.getenv("VISUAL_CROSSING_API_KEY")
    if not vc_api_key:
        raise ValueError(
            "Missing Visual Crossing API key. "
            "Set env VISUAL_CROSSING_API_KEY or pass vc_api_key=..."
        )

    target_time_utc = datetime.now(timezone.utc) + timedelta(hours=1)

    wind           = vc_hourly_wind(lat, lon, target_time_utc, api_key=vc_api_key)
    wind_speed_ms  = wind["wind_speed_ms"]
    wind_dir_to    = wind["wind_dir_deg_to"]

    wind_u_ms, wind_v_ms = wind_components_uv_ms(wind_speed_ms, wind_dir_to)

    center_lat = float(pm["center_lat"])
    center_lon = float(pm["center_lon"])

    align = downwind_alignment(lat, lon, center_lat, center_lon, wind_dir_to)
    slope = slope_proxy_from_elevation(lat, lon)

    point: dict = {
        "time_utc":          target_time_utc.isoformat(),
        "lat":               float(lat),
        "lon":               float(lon),
        "wind_speed_ms":     float(wind_speed_ms),
        "wind_dir_deg_to":   float(wind_dir_to),
        "wind_u_ms":         float(wind_u_ms),
        "wind_v_ms":         float(wind_v_ms),
        "dist_to_center_km": float(pm["dist_to_center_km"]),
        "downwind_alignment": float(align),
        "r_boundary_km":     float(pm["r_boundary_km"]),
        "dist_to_front_km":  float(pm["dist_to_front_km"]),
        "slope_proxy":       float(slope),
    }

    # bearing_from_center_deg is not used by either model, so we omit it

    if incident_created_time_utc is not None:
        if incident_created_time_utc.tzinfo is None:
            incident_created_time_utc = incident_created_time_utc.replace(
                tzinfo=timezone.utc
            )
        point["hour_index"] = int(
            (target_time_utc - incident_created_time_utc).total_seconds() // 3600
        )

    if pm.get("fire_name") is not None:
        point["fire_id"] = f"{pm['fire_name']}_{pm['year']}"

    return point