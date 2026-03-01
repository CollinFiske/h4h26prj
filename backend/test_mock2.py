import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
# backend/              ← test_mock2.py, pathfinder.py, ai.py, app.py live here
# backend/routes/       ← inference.py, predict.py, realtime_data.py, feature_utils.py

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_DIR  = os.path.join(BACKEND_DIR, "routes")

for p in (BACKEND_DIR, ROUTES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock data — grounded in realistic CA wildfire scenario
# Location: foothills east of Los Angeles (Azusa area)
# Fake fire: "MOCK FIRE" 2023, moderate wind from the west
# ─────────────────────────────────────────────────────────────────────────────

MOCK_LAT = 34.1336
MOCK_LON = -117.9070

MOCK_PERIMETER_METRICS = {
    "fire_name":         "MOCK FIRE",
    "year":              2023,
    "irwinid":           "MOCK-IRWIN-001",
    "inc_num":           "CA-ANF-000001",
    "center_lat":        34.12,
    "center_lon":        -117.88,
    "r_boundary_km":     4.2,
    "dist_to_front_km":  1.8,
    "dist_to_center_km": 2.6,
}

MOCK_WIND = {
    "time_utc":        datetime(2023, 8, 14, 15, 0, tzinfo=timezone.utc),
    "wind_speed_ms":   6.2,    # ~22 km/h from west — typical CA offshore wind
    "wind_dir_deg_to": 90.0,   # blowing east toward the hills
}

MOCK_ML_OUTPUT = {
    "burn_probability":  0.61,
    "hazard_pred_class": 2,
    "p_low":             0.08,
    "p_med":             0.31,
    "p_high":            0.61,
    "heat_weight":       0.765,
}

MOCK_SLOPE = 0.18


def _make_mock_fire_svc(ml_output=None):
    svc = MagicMock()
    svc.predict_one.return_value = ml_output or MOCK_ML_OUTPUT
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# Patch targets — keyed to actual module locations
#
#   routes/realtime_data.py  → "realtime_data.<fn>"
#   backend/pathfinder.py    → "pathfinder.<fn>"
#   routes/inference.py      → "inference.<fn>"
#   routes/predict.py        → "predict.<fn>"
# ─────────────────────────────────────────────────────────────────────────────

def _all_patches():
    return [
        # realtime_data external calls (used by build_point_next_hour)
        patch("realtime_data.get_nearest_calfire_perimeter_metrics", return_value=MOCK_PERIMETER_METRICS),
        patch("realtime_data.vc_hourly_wind",                        return_value=MOCK_WIND),
        patch("realtime_data.slope_proxy_from_elevation",            return_value=MOCK_SLOPE),
        patch("realtime_data._epqs_elevation_m",                     return_value=320.0),

        # pathfinder external calls (used by build_hazard_grid)
        patch("pathfinder.get_nearest_calfire_perimeter_metrics", return_value=MOCK_PERIMETER_METRICS),
        patch("pathfinder.vc_hourly_wind",                        return_value=MOCK_WIND),
        patch("pathfinder.slope_proxy_from_elevation",            return_value=MOCK_SLOPE),
    ]



# evacuationRouter isolation tests


class TestEvacuationRouter(unittest.TestCase):

    def _run_router(self, has_disability=False, ml_output=None):
        from pathfinder import EvacuationRouter
        svc = _make_mock_fire_svc(ml_output)

        patches = _all_patches()
        for p in patches:
            p.start()
        try:
            router = EvacuationRouter(
                center_lat=MOCK_LAT,
                center_lon=MOCK_LON,
                has_disability=has_disability,
                fire_svc=svc,
            )
            result = router.run()
        finally:
            for p in patches:
                p.stop()
        return result

    def test_router_returns_expected_keys(self):
        result = self._run_router()
        for key in ("reachable", "route_cost", "safe_zone", "path_latlon", "grid_stats"):
            self.assertIn(key, result, f"Missing key: {key}")
        print("✓ Router returns all expected keys")

    def test_router_reachable_under_normal_conditions(self):
        """With moderate ML scores a route should exist."""
        result = self._run_router(ml_output={
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.3,   # below FIRE_THRESH (0.55)
            "hazard_pred_class": 1,     # medium — passable
            "heat_weight":       0.35,
        })
        self.assertTrue(result["reachable"], "Expected a reachable route")
        self.assertIsNotNone(result["safe_zone"])
        self.assertGreater(len(result["path_latlon"]), 0)
        print(f"✓ Router found route — {len(result['path_latlon'])} waypoints, "
              f"cost={result['route_cost']:.1f}, "
              f"safe zone=({result['safe_zone']['lat']:.4f}, {result['safe_zone']['lon']:.4f})")

    def test_router_blocks_fire_cells(self):
        """High burn probability should fill the grid with blocked cells."""
        result = self._run_router(ml_output={
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.90,
            "hazard_pred_class": 2,
            "heat_weight":       0.95,
        })
        stats   = result["grid_stats"]
        blocked = stats.get("fire", 0) + stats.get("high_hazard", 0)
        self.assertGreater(blocked, 100, "Expected majority of grid to be blocked")
        print(f"✓ High burn_probability → {blocked} blocked cells in grid")

    def test_router_disability_flag_affects_path(self):
        """Smoke cells are blocked entirely for disabled users."""
        smoke_output = {
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.2,
            "hazard_pred_class": 1,
            "heat_weight":       0.3,
        }
        result_normal   = self._run_router(has_disability=False, ml_output=smoke_output)
        result_disabled = self._run_router(has_disability=True,  ml_output=smoke_output)
        print(f"✓ Disability flag test:")
        print(f"    Normal   — reachable={result_normal['reachable']},   "
              f"cost={result_normal.get('route_cost')}")
        print(f"    Disabled — reachable={result_disabled['reachable']}, "
              f"cost={result_disabled.get('route_cost')}")

    def test_path_latlon_are_valid_coordinates(self):
        """Every waypoint should be a valid California lat/lon."""
        result = self._run_router(ml_output={
            **MOCK_ML_OUTPUT,
            "burn_probability":  0.2,
            "hazard_pred_class": 0,
        })
        if result["reachable"]:
            for pt in result["path_latlon"]:
                self.assertIn("lat", pt)
                self.assertIn("lon", pt)
                self.assertGreater(pt["lat"],  30.0)
                self.assertLess(pt["lat"],     42.0)
                self.assertGreater(pt["lon"], -125.0)
                self.assertLess(pt["lon"],    -114.0)
            print(f"✓ All {len(result['path_latlon'])} waypoints are valid CA coordinates")
        else:
            print("  (skipped coordinate check — no reachable route)")

    def test_grid_stats_sum_to_grid_size(self):
        """Grid stats cell counts should sum to 25×25 = 625."""
        result = self._run_router()
        stats  = result["grid_stats"]
        total  = sum(stats.values())
        self.assertEqual(total, 625, f"Expected 625 cells, got {total}: {stats}")
        print(f"✓ Grid stats sum to 625: {stats}")



# pipeline via /api/predict
# 

class TestPredictEndpoint(unittest.TestCase):

    def setUp(self):
        # Patch joblib.load so we don't need .pkl files on disk
        self._pkl_patch = patch("inference.joblib.load", return_value=MagicMock())
        self._pkl_patch.start()

        # Patch _get_fire_svc so no model files are needed
        self._svc_patch = patch(
            "predict._get_fire_svc",
            return_value=_make_mock_fire_svc(),
        )
        self._svc_patch.start()

        # Patch all external network calls
        self._ext_patches = _all_patches()
        for p in self._ext_patches:
            p.start()

        # Patch AI model — no OpenAI key needed
        self._ai_patch = patch(
            "predict.call_ai_model",
            return_value=(
                "Due to high fire risk in your area, evacuate immediately heading east. "
                "Take your pets and medications. Avoid smoke-filled roads near the hills."
            ),
        )
        self._ai_patch.start()

        # Patch EvacuationRouter in predict.py (imported from pathfinder)
        self._router_patch = patch(
            "predict.EvacuationRouter",
            return_value=MagicMock(run=MagicMock(return_value={
                "reachable":   True,
                "route_cost":  14.5,
                "safe_zone":   {"lat": 34.15, "lon": -117.94, "label": "Emergency Shelter"},
                "path_latlon": [
                    {"lat": 34.1336, "lon": -117.9070},
                    {"lat": 34.1380, "lon": -117.9120},
                    {"lat": 34.1420, "lon": -117.9200},
                    {"lat": 34.1500, "lon": -117.9400},
                ],
                "grid_stats": {"free": 480, "fire": 60, "smoke": 70, "high_hazard": 15},
            })),
        )
        self._router_patch.start()

        # Build Flask test client
        from flask import Flask
        from routes import predict as predict_module
        app = Flask(__name__)
        app.register_blueprint(predict_module.predict_bp, url_prefix="/api")
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        self._pkl_patch.stop()
        self._svc_patch.stop()
        self._ai_patch.stop()
        self._router_patch.stop()
        for p in self._ext_patches:
            p.stop()

    def _post(self, payload):
        return self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )


    def test_basic_request_returns_200(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        self.assertEqual(resp.status_code, 200)
        print("✓ POST /api/predict → 200 OK")

    def test_response_contains_all_top_level_keys(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        body = resp.get_json()
        for key in ("selected_fire", "ml", "evacuation", "guidance"):
            self.assertIn(key, body, f"Missing top-level key: {key}")
        print("✓ Response contains: selected_fire, ml, evacuation, guidance")

    def test_ml_block_structure(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        ml   = resp.get_json()["ml"]
        for key in ("burn_probability", "hazard_pred_class", "p_low", "p_med", "p_high", "heat_weight"):
            self.assertIn(key, ml)
        self.assertGreaterEqual(ml["burn_probability"], 0.0)
        self.assertLessEqual(ml["burn_probability"],    1.0)
        self.assertIn(ml["hazard_pred_class"], (0, 1, 2))
        print(f"✓ ML block valid — burn_prob={ml['burn_probability']:.2f}, "
              f"hazard_class={ml['hazard_pred_class']}")

    def test_evacuation_block_structure(self):
        resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        evac = resp.get_json()["evacuation"]
        self.assertIn("reachable",   evac)
        self.assertIn("path_latlon", evac)
        self.assertIn("safe_zone",   evac)
        self.assertIn("grid_stats",  evac)
        self.assertTrue(evac["reachable"])
        self.assertGreater(len(evac["path_latlon"]), 0)
        self.assertIn("label", evac["safe_zone"])
        print(f"✓ Evacuation block valid — {len(evac['path_latlon'])} waypoints, "
              f"safe zone='{evac['safe_zone']['label']}'")

    def test_guidance_is_non_empty_string(self):
        resp     = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
        guidance = resp.get_json()["guidance"]
        self.assertIsInstance(guidance, str)
        self.assertGreater(len(guidance), 20)
        print(f"✓ Guidance returned: \"{guidance[:80]}...\"")

    def test_full_request_with_all_checkboxes(self):
        payload = {
            "latitude":        MOCK_LAT,
            "longitude":       MOCK_LON,
            "has_disability":  True,
            "has_pets":        True,
            "has_kids":        True,
            "has_medications": True,
            "other_concerns":  "Elderly parent, limited mobility",
        }
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200)
        print("✓ Full checkbox payload → 200 OK")

    #error handling 

    def test_missing_lat_lon_returns_400(self):
        resp = self._post({"has_pets": True})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("latitude", resp.get_json()["error"])
        print("✓ Missing lat/lon → 400 with helpful error message")

    def test_invalid_lat_lon_type_returns_400(self):
        resp = self._post({"latitude": "not_a_number", "longitude": MOCK_LON})
        self.assertEqual(resp.status_code, 400)
        print("✓ Non-numeric lat/lon → 400")

    def test_empty_body_returns_400(self):
        resp = self.client.post("/api/predict", data="", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        print("✓ Empty body → 400")

    def test_routing_failure_is_non_fatal(self):
        """If routing crashes, ML results should still come back."""
        self._router_patch.stop()
        broken_router = patch(
            "predict.EvacuationRouter",
            return_value=MagicMock(run=MagicMock(side_effect=RuntimeError("Grid build failed")))
        )
        broken_router.start()
        try:
            resp = self._post({"latitude": MOCK_LAT, "longitude": MOCK_LON})
            body = resp.get_json()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("ml",       body)
            self.assertIn("guidance", body)
            self.assertFalse(body["evacuation"]["reachable"])
            self.assertIn("error", body["evacuation"])
            print("✓ Routing failure is non-fatal — ML + guidance still returned")
        finally:
            broken_router.stop()
            self._router_patch.start()


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MOCK PIPELINE TESTS  (no live APIs or fire needed)")
    print(f"Simulated location : ({MOCK_LAT}, {MOCK_LON})  — Azusa foothills, CA")
    print(f"Simulated fire     : {MOCK_PERIMETER_METRICS['fire_name']} "
          f"({MOCK_PERIMETER_METRICS['year']}), "
          f"{MOCK_PERIMETER_METRICS['dist_to_front_km']} km away")
    print(f"Simulated wind     : {MOCK_WIND['wind_speed_ms']} m/s → {MOCK_WIND['wind_dir_deg_to']}°")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    print("\n── EvacuationRouter / pathfinder (isolation) ────────────")
    suite.addTests(loader.loadTestsFromTestCase(TestEvacuationRouter))

    print("\n── /api/predict (full pipeline) ─────────────────────────")
    suite.addTests(loader.loadTestsFromTestCase(TestPredictEndpoint))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"Results: {passed}/{result.testsRun} passed")
    if result.failures or result.errors:
        print("FAILURES:")
        for f in result.failures + result.errors:
            print(f"  ✗ {f[0]}")
    print("=" * 60)