"""
TRAIN_EVAP_MODELS.PY — EMERALD LANKA
======================================
Trains 4 XGBoost evaporation models — one per fuel type.
Uses real weather data + sales volume + date features.

This is the genuine ML approach:
  evap = f(sales_volume, temperature, humidity, wind, ET0, solar, month, season)

Run:
  python scripts/train_evap_models.py

Requires:
  data/sales_data_clean.csv     (from preprocess.py)
  data/fuel_sales_evaporation.csv
  data/weather_data.csv         (from fetch_weather.py)
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

FUEL_TYPES = [
    "petrol",
    "super_petrol",
    "diesel",
    "super_diesel",
]

# Map fuel names to CSV columns
EVAP_COL = {
    "petrol":       "petrol_evap_L",
    "super_petrol": "super_petrol_evap_L",
    "diesel":       "diesel_evap_L",
    "super_diesel": "super_diesel_evap_L",
}
SALES_COL = {
    "petrol":       "petrol_sales_L",
    "super_petrol": "super_petrol_sales_L",
    "diesel":       "diesel_sales_L",
    "super_diesel": "super_diesel_sales_L",
}

MODELS_DIR = "models/evaporation"

XGBOOST_PARAMS = {
    "n_estimators":         500,
    "max_depth":            4,        # shallower than sales model — less data to train
    "learning_rate":        0.05,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "random_state":         42,
    "n_jobs":               -1,
    "early_stopping_rounds": 40,
}


# ─── LOAD AND MERGE DATA ─────────────────────────────────────────────────────

def load_merged_data() -> pd.DataFrame:
    """
    Merge evaporation data with weather data.
    Returns a single DataFrame with evaporation + sales + weather per day.
    """
    # Evaporation data
    evap_path = "data/fuel_sales_evaporation.csv"
    if not os.path.exists(evap_path):
        raise FileNotFoundError(
            f"Missing: {evap_path}\n"
            "Place your evaporation CSV in the data/ folder."
        )
    evap_df = pd.read_csv(evap_path)
    evap_df["date"] = pd.to_datetime(evap_df["date"])

    # Weather data
    wx_path = "data/weather_data.csv"
    if not os.path.exists(wx_path):
        raise FileNotFoundError(
            f"Missing: {wx_path}\n"
            "Run: python scripts/fetch_weather.py first."
        )
    wx_df = pd.read_csv(wx_path)
    wx_df["date"] = pd.to_datetime(wx_df["date"])

    # Merge on date
    df = pd.merge(evap_df, wx_df, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Evaporation rows : {len(evap_df)}")
    print(f"  Weather rows     : {len(wx_df)}")
    print(f"  After merge      : {len(df)} rows")
    print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")

    return df


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, fuel: str) -> tuple:
    """
    Build the feature matrix X and target vector y for one fuel type.

    Features:
      - Sales volume (most important — more sales = more evaporation)
      - All weather variables (temp, humidity, wind, ET0, solar, VPD)
      - Derived weather features (is_hot_day, evap_risk, etc.)
      - Date features (month, season, day_of_week)
      - Lag features (yesterday's evaporation — tanks stay warm overnight)
    """
    d = df.copy()
    target_col = EVAP_COL[fuel]
    sales_col  = SALES_COL[fuel]

    # ── Drop OOS days ─────────────────────────────────────────────────────────
    # Out-of-stock days have zero sales AND zero evaporation
    # These corrupt the model — remove them
    d = d[d[sales_col] > 0].copy().reset_index(drop=True)
    print(f"  After removing OOS days: {len(d)} rows "
          f"({len(df)-len(d)} OOS days removed)")

    # ── Date features ─────────────────────────────────────────────────────────
    d["month"]        = d["date"].dt.month
    d["day_of_week"]  = d["date"].dt.dayofweek
    d["day_of_year"]  = d["date"].dt.dayofyear
    d["quarter"]      = d["date"].dt.quarter

    # Cyclical month encoding
    d["month_sin"]    = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"]    = np.cos(2 * np.pi * d["month"] / 12)

    # Season flags
    d["is_monsoon"]   = d["month"].isin([5,6,7,8,9]).astype(int)
    d["is_dry_season"]= d["month"].isin([12,1,2,3,4]).astype(int)

    # ── Sales features ────────────────────────────────────────────────────────
    # Sales volume is the primary driver of evaporation
    d["sales_L"]      = d[sales_col]
    d["sales_lag1"]   = d[sales_col].shift(1)   # yesterday's sales
    d["sales_roll7"]  = d[sales_col].shift(1).rolling(7, min_periods=1).mean()

    # ── Evaporation lag features ───────────────────────────────────────────────
    # Yesterday's evaporation (tanks stay warm, correlation is high)
    d["evap_lag1"]    = d[target_col].shift(1)
    d["evap_lag2"]    = d[target_col].shift(2)
    d["evap_roll7"]   = d[target_col].shift(1).rolling(7, min_periods=1).mean()

    # ── Weather lag features ──────────────────────────────────────────────────
    # Yesterday's temperature still affects today (thermal mass of tanks)
    d["temp_max_lag1"] = d["temp_max_c"].shift(1)
    d["et0_lag1"]      = d["et0_mm"].shift(1)

    # ── Drop NaN rows (from lag features) ─────────────────────────────────────
    d = d.dropna().reset_index(drop=True)

    # ── Feature columns ────────────────────────────────────────────────────────
    weather_cols = [
        "temp_max_c", "temp_min_c", "temp_mean_c", "temp_range_c",
        "precip_mm", "humidity_max", "humidity_min", "humidity_mean",
        "wind_max_kmh", "wind_mean_kmh",
        "et0_mm", "solar_mj", "vpd_kpa",
        "is_hot_day", "is_very_hot", "is_dry_day", "is_humid_day",
        "is_rainy_day", "is_heavy_rain", "no_rain", "is_windy",
        "evap_risk",
        "temp_max_lag1", "et0_lag1",
    ]
    date_cols = [
        "month", "day_of_week", "day_of_year", "quarter",
        "month_sin", "month_cos", "is_monsoon", "is_dry_season",
    ]
    sales_cols = ["sales_L", "sales_lag1", "sales_roll7"]
    evap_lag_cols = ["evap_lag1", "evap_lag2", "evap_roll7"]

    feature_cols = weather_cols + date_cols + sales_cols + evap_lag_cols

    # Keep only columns that exist in the data
    feature_cols = [c for c in feature_cols if c in d.columns]

    X = d[feature_cols]
    y = d[target_col]

    return X, y, feature_cols, d


# ─── TRAIN ONE MODEL ─────────────────────────────────────────────────────────

def train_one_model(df: pd.DataFrame, fuel: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training evaporation model: {fuel}")
    print(f"{'='*60}")

    X, y, feature_cols, data = build_features(df, fuel)
    print(f"  Dataset: {len(X)} rows, {len(feature_cols)} features")

    # ── Cross-validation ──────────────────────────────────────────────────────
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_maes  = []
    cv_mapes = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        es_split = int(len(X_tr) * 0.9)
        fold_model = XGBRegressor(**XGBOOST_PARAMS)
        fold_model.fit(
            X_tr.iloc[:es_split], y_tr.iloc[:es_split],
            eval_set=[(X_tr.iloc[es_split:], y_tr.iloc[es_split:])],
            verbose=False,
        )
        pred = np.maximum(0, fold_model.predict(X_val))
        mae  = mean_absolute_error(y_val, pred)
        mask = y_val > 0
        mape = np.mean(np.abs((y_val[mask] - pred[mask]) / y_val[mask])) * 100
        cv_maes.append(mae)
        cv_mapes.append(mape)

    cv_mae  = np.mean(cv_maes)
    cv_mape = np.mean(cv_mapes)
    print(f"\n  Cross-validation (5-fold):")
    print(f"    CV MAE  : {cv_mae:.5f} L  (±{np.std(cv_maes):.5f})")
    print(f"    CV MAPE : {cv_mape:.2f}%  (±{np.std(cv_mapes):.2f}%)")

    # ── Holdout test ──────────────────────────────────────────────────────────
    test_size = 20
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    es_split  = int(len(X_train) * 0.9)
    hm        = XGBRegressor(**XGBOOST_PARAMS)
    hm.fit(
        X_train.iloc[:es_split], y_train.iloc[:es_split],
        eval_set=[(X_train.iloc[es_split:], y_train.iloc[es_split:])],
        verbose=False,
    )
    h_pred = np.maximum(0, hm.predict(X_test))
    h_mae  = mean_absolute_error(y_test, h_pred)
    h_rmse = np.sqrt(mean_squared_error(y_test, h_pred))
    mask   = y_test > 0
    h_mape = np.mean(np.abs((y_test[mask] - h_pred[mask]) / y_test[mask])) * 100
    print(f"\n  Holdout test (last 20 days):")
    print(f"    MAE  : {h_mae:.5f} L")
    print(f"    RMSE : {h_rmse:.5f} L")
    print(f"    MAPE : {h_mape:.2f}%")

    # ── Final model on ALL data ───────────────────────────────────────────────
    best_n = hm.best_iteration + 1
    print(f"\n  Training final model on ALL {len(X)} rows (n_estimators={best_n})...")
    final_params = {k: v for k, v in XGBOOST_PARAMS.items()
                    if k not in ("early_stopping_rounds", "n_estimators")}
    final_params["n_estimators"] = best_n

    final_model = XGBRegressor(**final_params)
    final_model.fit(X, y)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model,  f"{MODELS_DIR}/{fuel}_evap_model.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/{fuel}_evap_features.pkl")
    joblib.dump({
        "cv_mae":      cv_mae,
        "cv_mape":     cv_mape,
        "holdout_mae": h_mae,
        "holdout_mape": h_mape,
        "mae":  cv_mae,
        "mape": cv_mape,
    }, f"{MODELS_DIR}/{fuel}_evap_metrics.pkl")

    print(f"  Saved: {MODELS_DIR}/{fuel}_evap_model.pkl")

    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    print(f"\n  TOP 10 FEATURES:")
    for feat, val in imp.head(10).items():
        print(f"    {feat:<35} {val:.4f}")

    return {"cv_mape": cv_mape, "holdout_mape": h_mape,
            "mae": cv_mae, "mape": cv_mape}


# ─── TRAIN ALL 4 ─────────────────────────────────────────────────────────────

def train_all_evap_models(
    evap_path    : str = "data/fuel_sales_evaporation.csv",
    weather_path : str = "data/weather_data.csv",
    models_dir   : str = "models/evaporation",
) -> dict:
    global MODELS_DIR
    MODELS_DIR = models_dir

    print("\nLoading and merging datasets...")
    df = load_merged_data()

    metrics = {}
    for fuel in FUEL_TYPES:
        metrics[fuel] = train_one_model(df, fuel)

    print("\n" + "="*60)
    print("  EVAPORATION MODELS — TRAINING COMPLETE")
    print("="*60)
    print(f"\n{'Fuel':<20} {'CV MAPE':>10} {'Holdout':>10}")
    print("-"*42)
    for fuel, m in metrics.items():
        print(f"{fuel:<20} {m['cv_mape']:>9.2f}% {m['holdout_mape']:>9.2f}%")

    print(f"\nAll 4 models saved to {models_dir}/")
    return metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_all_evap_models()