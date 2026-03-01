import numpy as np
import pandas as pd


# columns are identifiers/targets not model inputs 

FIRE_DROP_COLS = {
    "fire_id", "burned_label", "time_utc", "lat", "lon", "wind_dir_deg_to",
}

HAZARD_DROP_COLS = {
    "fire_id", "hazard_class_0_1_2", "hazard_level", "hazard_score",
    "time_utc", "lat", "lon", "wind_dir_deg_to",
}


def compute_smoke_proxy(
    burn_prob: np.ndarray,
    dist_to_front_km: np.ndarray,
    downwind_alignment: np.ndarray,
    wind_speed_ms: np.ndarray,
) -> np.ndarray:
    """
    estimating smoke/ember exposure at a point.

    param:
    burn_prob           : probability that the fire front reaches this cell (0-1)
    dist_to_front_km    : distance to nearest fire perimeter boundary (km)
    downwind_alignment  : dot-product of wind direction and center->point vector (-1 to 1)
    wind_speed_ms       : wind speed in m/s

    returns"
    smoke_proxy in [0, 1]
    """
    burn  = np.nan_to_num(burn_prob,          nan=0.0)
    dist  = np.nan_to_num(dist_to_front_km,   nan=1e6)
    align = np.nan_to_num(downwind_alignment,  nan=0.0)
    wind  = np.nan_to_num(wind_speed_ms,       nan=0.0)

    d        = np.maximum(dist, 0.0)
    downwind = np.clip((align + 1.0) / 2.0, 0.0, 1.0)
    L        = 10.0 + 2.8 * wind          # effective smoke plume length (km)
    decay    = np.exp(-d / L)

    smoke_proxy = burn * (0.25 + 0.75 * downwind) * decay
    return np.clip(smoke_proxy, 0.0, 1.0)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    #Parse time_utc and add hour / dayofyear / dayofweek columns
    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
        df["hour"]      = df["time_utc"].dt.hour.fillna(0).astype(int)
        df["dayofyear"] = df["time_utc"].dt.dayofyear.fillna(0).astype(int)
        df["dayofweek"] = df["time_utc"].dt.dayofweek.fillna(0).astype(int)
    else:
        # Fallback: derive hour from hour_index if present
        hour_index = df.get("hour_index", pd.Series(0, index=df.index))
        df["hour"]      = (hour_index.astype(int) % 24)
        df["dayofyear"] = 0
        df["dayofweek"] = 0
    return df


def _add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    #Add engineered wind columns; consumes wind_dir_deg_to (does not drop)
    df["wind_speed_squared"]    = df["wind_speed_ms"] ** 2
    df["wind_hour_interaction"] = df["wind_speed_ms"] * df["hour"]

    angles = np.deg2rad(df["wind_dir_deg_to"].astype(float))
    df["wind_dir_sin"] = np.sin(angles)
    df["wind_dir_cos"] = np.cos(angles)

    df["wind_x_downwind"]   = df["wind_speed_ms"] * df["downwind_alignment"]
    df["wind_x_dist_front"] = df["wind_speed_ms"] / df["dist_to_front_km"].clip(lower=0.1)
    return df


# model-specific feature builders 

def build_fire_features(df_in: pd.DataFrame, drop_cols: bool = True) -> pd.DataFrame:
    """
    Feature engineering for Model 1 (fire spread / burn probability).

    Parameters
    ----------
    df_in     : raw input DataFrame
    drop_cols : if True, remove FIRE_DROP_COLS before returning
                (set False during training if you need the target column)
    """
    df = df_in.copy()
    df = _add_time_features(df)
    df = _add_wind_features(df)

    if drop_cols:
        df = df.drop(columns=[c for c in FIRE_DROP_COLS if c in df.columns])

    return df


def build_hazard_features(df_in: pd.DataFrame, drop_cols: bool = True) -> pd.DataFrame:
    """
      param:
    df_in     : burn_probability  from DataFrame from Model1
    drop_cols : if True, remove HAZARD_DROP_COLS before returning
    """
    df = df_in.copy()

    if "burn_probability" not in df.columns:
        raise ValueError(
            "build_hazard_features() requires a 'burn_probability' column. "
            "Run Model 1 first."
        )
    df["burn_probability"] = pd.to_numeric(
        df["burn_probability"], errors="coerce"
    ).fillna(0.0)

    df["smoke_proxy"] = compute_smoke_proxy(
        df["burn_probability"].values,
        df["dist_to_front_km"].values,
        df["downwind_alignment"].values,
        df["wind_speed_ms"].values,
    )

    df = _add_time_features(df)
    df = _add_wind_features(df)

    if "slope_proxy" not in df.columns:
        df["slope_proxy"] = 0.0

    # smoke/terrain interactions
    df["smoke_x_wind"]  = df["smoke_proxy"] * df["wind_speed_ms"]
    df["smoke_x_slope"] = df["smoke_proxy"] * df["slope_proxy"]
    df["smoke_x_front"] = df["smoke_proxy"] / df["dist_to_front_km"].clip(lower=0.1)

    if drop_cols:
        df = df.drop(columns=[c for c in HAZARD_DROP_COLS if c in df.columns])

    return df


def align_features(df_feat: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    ensure df_feat has exactly feat_cols, in that order.
    missing columns are filled with 0.0; extra columns are dropped.
    """
    X = df_feat.copy()
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0.0
    return X[feat_cols]