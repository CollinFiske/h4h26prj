import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import joblib

if __name__ == "__main__":
    df = pd.read_csv("datasets_train/FINAL_fire_spread_ml_30fires_48h.csv", parse_dates=["time_utc"])


    # Temporal features from time_utc
    df['hour']       = df['time_utc'].dt.hour
    df['dayofyear']  = df['time_utc'].dt.dayofyear
    df['dayofweek']  = df['time_utc'].dt.dayofweek

    # Wind features  (using correct column names: wind_speed_ms, wind_dir_deg_to)
    df['wind_speed_squared']     = df['wind_speed_ms'] ** 2
    df['wind_hour_interaction']  = df['wind_speed_ms'] * df['hour']

    # Wind direction → sin/cos encoding (already have u/v components too, both are useful)
    angles = np.deg2rad(df['wind_dir_deg_to'].astype(float))
    df['wind_dir_sin'] = np.sin(angles)
    df['wind_dir_cos'] = np.cos(angles)

    # Interaction: how much is wind pushing toward the fire front
    df['wind_x_downwind']    = df['wind_speed_ms'] * df['downwind_alignment']
    df['wind_x_dist_front']  = df['wind_speed_ms'] / (df['dist_to_front_km'].clip(lower=0.1))



    DROP_COLS = [
        'fire_id',          # group identifier, not a feature
        'burned_label',     # target
        'time_utc',         # replaced by hour/dayofyear
        'lat',              # raw coords not useful without spatial model
        'lon',
        'wind_dir_deg_to',  # replaced by sin/cos encoding
    ]

    X = df.drop(columns=DROP_COLS)
    y = df['burned_label'].astype(int)

    print("Features used:", list(X.columns))
    print(f"Class balance → 0: {(y==0).sum()}  1: {(y==1).sum()}  ratio: {(y==0).sum()/(y==1).sum():.1f}:1")

    groups   = df['fire_id']
    #splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #train_idx, test_idx = next(splitter.split(X, y, groups))
    #TEST_FIRES = ['fire_01', 'fire_07', 'fire_12', 'fire_21', 'fire_22', 'fire_26']
    #test_mask  = df['fire_id'].isin(TEST_FIRES)
    #train_idx  = df.index[~test_mask]
    #test_idx   = df.index[test_mask]


    #X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    #y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    X_train, y_train = X, y

    #print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")
    #print(f"Test fires : {df['fire_id'].iloc[test_idx].unique()}")


    # Class imbalance weight — critical since most cells don't burn
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\npos_weight (for XGB scale_pos_weight): {pos_weight:.2f}")


    #def train_model(X_train, X_test, y_train, y_test, algo, threshold=0.5):
    def train_model(X_train, y_train, algo, threshold=0.5):
        if algo == 'rf':
            model = RandomForestClassifier(
                n_estimators=500,          # 1000 is overkill for this size
                max_depth=15,              # unconstrained trees overfit badly
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',   # handles class imbalance
                random_state=42,
                n_jobs=-1
            )

        elif algo == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,                   # min loss reduction for split
                reg_lambda=1,
                reg_alpha=0.1,
                scale_pos_weight=pos_weight,   # handles class imbalance
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )

        model.fit(X_train, y_train)
        print(f"Trained: {algo.upper()}")
        return model
    '''
        #probs  = model.predict_proba(X_test)[:, 1]
        #y_pred = (probs > threshold).astype(int)

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, probs)
        cm    = confusion_matrix(y_test, y_pred)

        print(f"\n{'-'*40}")
        print(f"Model   : {algo.upper()}  (threshold={threshold})")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1       : {f1:.3f}")
        print(f"AUC-ROC  : {auc:.3f}")
        print(f"Confusion matrix:\n{cm}")

        return model, probs

    '''

    # Train both
    #rf_model,  rf_probs  = train_model(X_train, X_test, y_train, y_test, algo='rf',  threshold=0.4)
    #xgb_model, xgb_probs = train_model(X_train, X_test, y_train, y_test, algo='xgb', threshold=0.4)
    rf_model  = train_model(X_train, y_train, algo='rf',  threshold=0.4)
    xgb_model = train_model(X_train, y_train, algo='xgb', threshold=0.4)

    #print("\n" + "-"*40)
    #print("Overfit check (train vs test accuracy):")
    #print(f"RF   train: {rf_model.score(X_train,  y_train):.3f}   test: {rf_model.score(X_test,  y_test):.3f}")
    #print(f"XGB  train: {xgb_model.score(X_train, y_train):.3f}   test: {xgb_model.score(X_test, y_test):.3f}")

    feat_imp = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 15 features (XGB):")
    print(feat_imp.head(15).to_string())

    joblib.dump(xgb_model,       "model_1/fire_spread_model.pkl")
    joblib.dump(rf_model,        "model_1/fire_spread_rf_model.pkl")
    joblib.dump(list(X.columns), "model_1/fire_spread_features.pkl")

    #df["burn_probability"] = np.nan
    #df.loc[test_idx, "burn_probability"] = xgb_probs
    #df.loc[test_idx].to_csv("model_1/fire_spread_with_probs.csv", index=False)
    all_probs = xgb_model.predict_proba(X)[:, 1]
    df["burn_probability"] = all_probs
    #df.to_csv("model_1/fire_spread_with_probs.csv", index=False)

    df[["lat", "lon", "time_utc", "burn_probability"]].to_csv(
        "model_1/fire_spread_with_probs.csv", index=False
    )


    print("\nSaved → fire_spread_with_probs.csv for model2")
    #print("\nSaved → fire_spread_model.pkl (XGB)")
    #print("Saved → fire_spread_rf_model.pkl (RF)")
    #print("Saved → fire_spread_features.pkl (feature list)")
