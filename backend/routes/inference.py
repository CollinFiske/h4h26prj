import numpy as np
import pandas as pd
import joblib

from feature_utils import build_fire_features, build_hazard_features, align_features


class FireHazardService:
    def __init__(
        self,
        fire_model_path: str = "model_1/fire_spread_model.pkl",
        fire_feat_path: str  = "model_1/fire_spread_features.pkl",
        haz_model_path: str  = "model_2/hazard_model.pkl",
        haz_feat_path: str   = "model_2/hazard_features.pkl",
    ):
        self.fire_model     = joblib.load(fire_model_path)
        self.fire_feat_cols = joblib.load(fire_feat_path)
        self.haz_model      = joblib.load(haz_model_path)
        self.haz_feat_cols  = joblib.load(haz_feat_path)

    def predict_one(self, point: dict) -> dict:
        """
        Run both models for a single location/time point.
        param: point : dict with keys produced by realtime_data.build_point_next_hour()

        returns:
        dict with:
            burn_probability  : float [0, 1]
            hazard_pred_class : int   {0=low, 1=med, 2=high}
            p_low / p_med / p_high : float probabilities
            heat_weight       : float heatmap score [0, 1]
        """
        df = pd.DataFrame([point])

        # model 1
        df_fire = build_fire_features(df, drop_cols=True)
        X1      = align_features(df_fire, self.fire_feat_cols)
        burn_prob = float(self.fire_model.predict_proba(X1)[:, 1][0])

        #model2
        df["burn_probability"] = burn_prob
        df_haz = build_hazard_features(df, drop_cols=True)
        X2     = align_features(df_haz, self.haz_feat_cols)
        probs  = self.haz_model.predict_proba(X2)[0]

        pred_class  = int(np.argmax(probs))
        heat_weight = float(0.5 * probs[1] + probs[2])

        return {
            "burn_probability":  burn_prob,
            "hazard_pred_class": pred_class,
            "p_low":             float(probs[0]),
            "p_med":             float(probs[1]),
            "p_high":            float(probs[2]),
            "heat_weight":       heat_weight,
        }