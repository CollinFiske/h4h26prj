import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import joblib

"""
Model 2 — Hazard/Air Pollution Predicter 
Dataset: FINAL_hazard_ml_30fires_48h.csv
Target : hazard_class_0_1_2 -> (0=low, 1=med, 2=high)
"""

#removed for training because either they are the target or identifiers or raw coords would confuse 
DROP_COLS = [
    "fire_id",
    "hazard_class_0_1_2",
    "hazard_level",
    "hazard_score",
    "time_utc",
    "lat", "lon",
    "wind_dir_deg_to",
]

#takes raw input data and creates smarter features for the model such we convert time into readable data for the model
#
#
df2 = pd.read_csv("model_1/fire_spread_with_probs.csv")[["lat", "lon", "time_utc", "burn_probability"]]
df2["time_utc"] = pd.to_datetime(df2["time_utc"], utc=True)

def compute_smoke_proxy(burn_prob, dist_to_front_km, downwind_alignment, wind_speed_ms):
    d = np.maximum(dist_to_front_km, 0)
    downwind = np.clip((downwind_alignment + 1) / 2, 0, 1)
    L = 10 + 2.8 * wind_speed_ms
    decay = np.exp(-d / L)
    smoke_proxy = burn_prob * (0.25 + 0.75 * downwind) * decay
    return np.clip(smoke_proxy, 0, 1)
    
def build_hazard_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    
    # burn_probability now acts as smoke_proxy
    df = df.merge(df2, on=["lat", "lon", "time_utc"], how="left")
    df["burn_probability"] = df["burn_probability"].fillna(0.0)
    
    df["smoke_proxy"] = compute_smoke_proxy(
    df["burn_probability"].values,
    df["dist_to_front_km"].values,
    df["downwind_alignment"].values,
    df["wind_speed_ms"].values
    )
    #time features 
    if "time_utc" in df.columns:
        # handle both string and datetime
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
        df["hour"] = df["time_utc"].dt.hour.fillna(0).astype(int) #int 14
        df["dayofyear"] = df["time_utc"].dt.dayofyear.fillna(0).astype(int) #228
        df["dayofweek"] = df["time_utc"].dt.dayofweek.fillna(0).astype(int) #2(wednesday)
    else:
        # fallback if you only have hour_index
        hour_index = df.get("hour_index", 0)
        df["hour"] = (hour_index.astype(int) % 24) if hasattr(hour_index, "astype") else (int(hour_index) % 24)
        df["dayofyear"] = 0
        df["dayofweek"] = 0

    #wind features 
    df["wind_speed_squared"] = df["wind_speed_ms"] ** 2 #squaring captures acceration rate bc 20mph doesnt spread fire 2x than 10mph. (Rothermel model)
    df["wind_hour_interaction"] = df["wind_speed_ms"] * df["hour"] #wind hour varies during the day so it takes that into consideration 

    # wind direction sin/cos
    angles = np.deg2rad(df["wind_dir_deg_to"].astype(float)) 
    df["wind_dir_sin"] = np.sin(angles) #bc direction is circular, these will tell the dir is different bc 1 deg versus 359 is nearly same but actaully far
    df["wind_dir_cos"] = np.cos(angles)

    # interactions
    df["wind_x_downwind"] = df["wind_speed_ms"] * df["downwind_alignment"] #tells you how much wind energy is directed at point of interest. 
    df["wind_x_dist_front"] = df["wind_speed_ms"] / (df["dist_to_front_km"].clip(lower=0.1)) #wind/dist to fire front bc strong wind that is close to the fire front is more dangerous than far away.

    # smoke/terrain interactions
    df["smoke_x_wind"] = df["smoke_proxy"] * df["wind_speed_ms"] #is able to spot potential fire bc strong wind can carry embers(traveling w smoke)
    df["smoke_x_slope"] = df["smoke_proxy"] * df["slope_proxy"] #fire spreading can depend on slope too 
    df["smoke_x_front"] = df["smoke_proxy"] / (df["dist_to_front_km"].clip(lower=0.1)) #same as fire front, smoke near fire front is more dangerous than away so it calculates that

    # Drop columns that aren't features
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def predict_hazard( 
    df_points: pd.DataFrame,
    model_path: str = "hazard_model.pkl",
    feature_list_path: str = "hazard_features.pkl",
    return_probabilities: bool = True
) -> pd.DataFrame:
   
    model = joblib.load(model_path) #load XGBoost model from .pkl file to memory
    feat_cols = joblib.load(feature_list_path) #loads saved list of col names

    X_feat = build_hazard_features(df_points) 

    # ensure all expected feature columns exist, if there is data missing, fill in the spot with 0.0 to prevent crashing
    for col in feat_cols:
        if col not in X_feat.columns:
            X_feat[col] = 0.0
    #reorder the columns to mach what the model was trained on 
    X_feat = X_feat[feat_cols]

    # predict
    if hasattr(model, "predict_proba") and return_probabilities:
        probs = model.predict_proba(X_feat)  # shape (N,3), returns a 2D array, calc prob for low/med/high
        pred_class = probs.argmax(axis=1) #picks class w high prob for each row(axis 1)

        out = df_points.copy()
        out["hazard_pred_class"] = pred_class.astype(int) 
        out["p_low"] = probs[:, 0]
        out["p_med"] = probs[:, 1]
        out["p_high"] = probs[:, 2]
        # smooth heatmap weight: 0..1
        out["heat_weight"] = 0.5 * out["p_med"] + 1.0 * out["p_high"] #computing heatmap score based on intensity
        return out

    # fallback: if we cannot get a probablity score, it predicts a class label and return
    pred_class = model.predict(X_feat)
    out = df_points.copy()
    out["hazard_pred_class"] = pred_class.astype(int)
    return out


if __name__ == "__main__":
   
    df = pd.read_csv("datasets_train/FINAL_hazard_ml_30fires_48h.csv", parse_dates=["time_utc"])

    #feature matrix(modified inputs)
    X = build_hazard_features(df)
    y = df["hazard_class_0_1_2"].astype(int) #target

    print("Features used:", list(X.columns))
    print("Class counts:", y.value_counts().sort_index().to_dict())

    #train and test split group 80% train, 20% test and it is gruped like fire_id
    groups = df["fire_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #randomly generated to cross validate data
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")
    print(f"Test fires : {df['fire_id'].iloc[test_idx].unique()}")

    #model training
    def train_model(X_train, X_test, y_train, y_test, algo):
        if algo == "rf":
            model = RandomForestClassifier(
                n_estimators=600, #600 trees
                max_depth=18, 
                min_samples_split=10, #needs at least 10 to splite further
                min_samples_leaf=4, #needs at 4 samples to be a leaf node
                max_features="sqrt", #each split consides root of n features to reduce overfitting 
                class_weight="balanced", #upweights minority classes to balance it out
                random_state=42,
                n_jobs=-1
            )

        elif algo == "xgb":
            model = xgb.XGBClassifier(
                n_estimators=600, 
                max_depth=7, #less depth need bc builds on errors
                learning_rate=0.06, #slower but accurate
                subsample=0.85, #each tree sees 85% of rows to add randomness and better generalization to testing data
                colsample_bytree=0.85, #each tree sees 85% of features
                reg_lambda=1.0, #L2 reg to penalize large weights
                reg_alpha=0.1, #L1 reg to encourage spare weights
                objective="multi:softprob", #ouputs prob for 3 class 0,1,2
                num_class=3,
                eval_metric="mlogloss", 
                random_state=42,
                verbosity=0
            )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred) 
        f1m = f1_score(y_test, pred, average="macro")
        cm  = confusion_matrix(y_test, pred)

        print(f"\n{'-'*40}")
        print(f"Model   : {algo.upper()} (multi-class)")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro F1: {f1m:.3f}")
        print("Confusion matrix:\n", cm)
        print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

        return model
    
    rf_model  = train_model(X_train, X_test, y_train, y_test, algo="rf") 
    xgb_model = train_model(X_train, X_test, y_train, y_test, algo="xgb")

    # overfit check
    print("\n" + "-"*40)
    print("Overfit check (train vs test accuracy):")
    print(f"RF   train: {rf_model.score(X_train,  y_train):.3f}   test: {rf_model.score(X_test,  y_test):.3f}")
    print(f"XGB  train: {xgb_model.score(X_train, y_train):.3f}   test: {xgb_model.score(X_test, y_test):.3f}")

    #extracts features XGBoost model most relied to see if there is bias or outliers
    feat_imp = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 15 features (XGB):")
    print(feat_imp.head(15).to_string())

    # save both models, will load same colum and order as training
    joblib.dump(xgb_model, "model_2/hazard_model.pkl")
    joblib.dump(rf_model, "model_2/hazard_rf_model.pkl")
    joblib.dump(list(X.columns), "model_2/hazard_features.pkl")

    print("\nSaved as hazard_model.pkl (XGB) \nSaved as hazard_rf_model.pkl (RF) \nSaved as hazard_features.pkl (feature list)")
    
    df = df.merge(df2, on=["lat", "lon", "time_utc"], how="left")
    print(f"smoke_proxy null rate(debugging): {df['burn_probability'].isna().mean():.1%}")
    df["smoke_proxy"] = df["burn_probability"].fillna(0.0)

