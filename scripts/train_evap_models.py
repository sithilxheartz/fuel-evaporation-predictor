"""
TRAIN_EVAP_MODELS.PY — EMERALD LANKA
======================================
Trains 4 XGBoost evaporation models.

KEY FIX: Training now uses BOTH:
  1. fuel_sales_evaporation.csv  (Apr 2025 → Apr 2026) — has real evap values
  2. Firebase fuelSaleHistory    (Feb 2026 → today)    — has recent real sales

For dates in Firebase that are AFTER the CSV ends (Apr 20 → today):
  - Sales come from Firebase
  - Evaporation is calculated using the formula rates from CSV patterns
  - This gives the model recent data to learn from

This means:
  - Model trains on Apr 2025 → today (not just Apr 2025 → Apr 2026)
  - LATEST_DATE in API becomes today's date automatically
  - UI shows correct predictions from latest date

Run:
  python scripts/train_evap_models.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

FUEL_TYPES  = ["petrol", "super_petrol", "diesel", "super_diesel"]
MODELS_DIR  = "models/evaporation"

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

XGBOOST_PARAMS = {
    "n_estimators":          500,
    "max_depth":             4,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "random_state":          42,
    "n_jobs":                -1,
    "early_stopping_rounds": 40,
}


# ─── FIREBASE HELPERS ────────────────────────────────────────────────────────

def _init_firebase():
    import firebase_admin
    from firebase_admin import credentials
    if firebase_admin._apps:
        return
    env = os.environ.get("FIREBASE_CREDENTIALS")
    if env:
        cred = credentials.Certificate(json.loads(env))
    else:
        key = os.path.join("firebase", "serviceAccountKey.json")
        if not os.path.exists(key):
            print("  ⚠️  Firebase key not found — training on CSV only")
            return False
        cred = credentials.Certificate(key)
    firebase_admin.initialize_app(cred)
    return True


def fetch_firebase_sales() -> pd.DataFrame:
    """
    Fetch all sales from Firebase fuelSaleHistory.
    Returns DataFrame with same column format as CSV.
    """
    try:
        if not _init_firebase():
            return pd.DataFrame()

        from firebase_admin import firestore
        db   = firestore.client()
        docs = db.collection("fuelSaleHistory").stream()

        records = []
        for doc in docs:
            data = doc.to_dict()
            try:
                d = pd.to_datetime(doc.id)
            except Exception:
                continue

            def safe(key):
                v = data.get(key)
                try:
                    f = float(v) if v is not None else 0.0
                    return f if f > 0 else 0.0
                except Exception:
                    return 0.0

            records.append({
                "date":                  d,
                "petrol_sales_L":        safe("92PetrolSale"),
                "super_petrol_sales_L":  safe("95PetrolSale"),
                "diesel_sales_L":        safe("dieselSale"),
                "super_diesel_sales_L":  safe("superDieselSale"),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        print(f"  Firebase sales: {len(df)} records "
              f"({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    except Exception as e:
        print(f"  ⚠️  Firebase fetch failed: {e}")
        return pd.DataFrame()


# ─── LOAD AND MERGE DATA ─────────────────────────────────────────────────────

def load_merged_data() -> pd.DataFrame:
    """
    Load evaporation CSV and merge with Firebase sales data.

    Strategy:
      Phase 1 (CSV dates): use CSV evap values directly (real measured data)
      Phase 2 (Firebase dates after CSV): use Firebase sales + estimate evap
             using monthly evaporation rates learned from CSV

    This gives the model recent data including the latest sales patterns.
    """
    # ── Load CSV ──────────────────────────────────────────────────────────────
    evap_path = "data/fuel_sales_evaporation.csv"
    if not os.path.exists(evap_path):
        raise FileNotFoundError(f"Missing: {evap_path}")

    csv_df = pd.read_csv(evap_path)
    csv_df["date"] = pd.to_datetime(csv_df["date"])
    csv_end = csv_df["date"].max()
    print(f"  CSV data:     {len(csv_df)} rows "
          f"({csv_df['date'].min().date()} → {csv_end.date()})")

    # ── Load weather ──────────────────────────────────────────────────────────
    wx_path = "data/weather_data.csv"
    if not os.path.exists(wx_path):
        raise FileNotFoundError(
            f"Missing: {wx_path}\nRun: python scripts/fetch_weather.py first.")

    wx_df = pd.read_csv(wx_path)
    wx_df["date"] = pd.to_datetime(wx_df["date"])
    print(f"  Weather data: {len(wx_df)} rows "
          f"({wx_df['date'].min().date()} → {wx_df['date'].max().date()})")

    # ── Merge CSV + weather ───────────────────────────────────────────────────
    merged_csv = pd.merge(csv_df, wx_df, on="date", how="inner")
    merged_csv = merged_csv.sort_values("date").reset_index(drop=True)
    print(f"  CSV + weather merge: {len(merged_csv)} rows")

    # ── Fetch Firebase sales ──────────────────────────────────────────────────
    print("  Fetching Firebase fuelSaleHistory...")
    firebase_df = fetch_firebase_sales()

    if firebase_df.empty:
        print("  Using CSV data only for training")
        return merged_csv

    # ── Get Firebase dates AFTER CSV ends ─────────────────────────────────────
    new_dates_df = firebase_df[firebase_df["date"] > csv_end].copy()

    if new_dates_df.empty:
        print("  No new dates in Firebase after CSV — training on CSV only")
        return merged_csv

    print(f"  New Firebase dates: {len(new_dates_df)} rows "
          f"({new_dates_df['date'].min().date()} → "
          f"{new_dates_df['date'].max().date()})")

    # ── Calculate evaporation rates from CSV ──────────────────────────────────
    # Learn monthly evaporation rates from the CSV data
    monthly_rates = {}
    for fuel in FUEL_TYPES:
        monthly_rates[fuel] = {}
        for month in range(1, 13):
            mask = (
                (merged_csv["date"].dt.month == month) &
                (merged_csv[SALES_COL[fuel]] > 0)
            )
            if mask.sum() > 0:
                rates = (merged_csv[mask][EVAP_COL[fuel]] /
                         merged_csv[mask][SALES_COL[fuel]])
                monthly_rates[fuel][month] = rates.mean()
            else:
                monthly_rates[fuel][month] = 0.0009  # default fallback

    # ── Build new rows for Firebase dates ─────────────────────────────────────
    # Get weather for new dates from weather_deploy.csv or weather_data.csv
    deploy_wx = pd.DataFrame()
    for wx_file in ["data/weather_deploy.csv", "data/weather_full.csv"]:
        if os.path.exists(wx_file):
            try:
                deploy_wx = pd.read_csv(wx_file)
                deploy_wx["date"] = pd.to_datetime(deploy_wx["date"])
                break
            except Exception:
                pass

    # Merge new Firebase dates with weather
    if not deploy_wx.empty:
        new_with_wx = pd.merge(new_dates_df, deploy_wx, on="date", how="left")
    else:
        new_with_wx = new_dates_df.copy()

    # Add evaporation estimates using monthly rates
    for fuel in FUEL_TYPES:
        evap_col   = EVAP_COL[fuel]
        sales_col  = SALES_COL[fuel]
        new_with_wx[evap_col] = new_with_wx.apply(
            lambda row: (
                row[sales_col] *
                monthly_rates[fuel].get(row["date"].month, 0.0009)
                if row[sales_col] > 0 else 0.0
            ), axis=1
        )

    # Add total columns if missing
    if "total_sales_L" not in new_with_wx.columns:
        new_with_wx["total_sales_L"] = sum(
            new_with_wx[SALES_COL[f]] for f in FUEL_TYPES)
    if "total_evap_L" not in new_with_wx.columns:
        new_with_wx["total_evap_L"] = sum(
            new_with_wx[EVAP_COL[f]] for f in FUEL_TYPES)

    # ── Combine CSV + new Firebase rows ───────────────────────────────────────
    # Only keep columns that exist in both
    common_cols = list(set(merged_csv.columns) & set(new_with_wx.columns))
    combined = pd.concat(
        [merged_csv[common_cols], new_with_wx[common_cols]],
        ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    # Fill missing weather columns with seasonal averages
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    print(f"\n  ✅ Combined training dataset:")
    print(f"     Total rows:  {len(combined)}")
    print(f"     Date range:  {combined['date'].min().date()} → "
          f"{combined['date'].max().date()}")
    print(f"     CSV rows:    {len(merged_csv)}")
    print(f"     Firebase rows added: {len(new_with_wx)}")

    return combined


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, fuel: str) -> tuple:
    """Build feature matrix X and target y for one fuel."""
    d = df.copy()
    target_col = EVAP_COL[fuel]
    sales_col  = SALES_COL[fuel]

    # Remove OOS days
    d = d[d[sales_col] > 0].copy().reset_index(drop=True)
    print(f"  After removing OOS: {len(d)} rows")

    # Date features
    d["month"]         = d["date"].dt.month
    d["day_of_week"]   = d["date"].dt.dayofweek
    d["day_of_year"]   = d["date"].dt.dayofyear
    d["quarter"]       = d["date"].dt.quarter
    d["month_sin"]     = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"]     = np.cos(2 * np.pi * d["month"] / 12)
    d["is_monsoon"]    = d["month"].isin([5,6,7,8,9]).astype(int)
    d["is_dry_season"] = d["month"].isin([12,1,2,3,4]).astype(int)

    # Sales features
    d["sales_L"]     = d[sales_col]
    d["sales_lag1"]  = d[sales_col].shift(1)
    d["sales_roll7"] = d[sales_col].shift(1).rolling(7, min_periods=1).mean()

    # Evaporation lag features
    d["evap_lag1"]  = d[target_col].shift(1)
    d["evap_lag2"]  = d[target_col].shift(2)
    d["evap_roll7"] = d[target_col].shift(1).rolling(7, min_periods=1).mean()

    # Weather lag features
    d["temp_max_lag1"] = d["temp_max_c"].shift(1) if "temp_max_c" in d.columns else 30.0
    d["et0_lag1"]      = d["et0_mm"].shift(1)     if "et0_mm"     in d.columns else 5.0

    # Drop NaN rows from lag features
    d = d.dropna().reset_index(drop=True)

    # Feature columns
    weather_cols = [c for c in [
        "temp_max_c","temp_min_c","temp_mean_c","temp_range_c",
        "precip_mm","humidity_max","humidity_min","humidity_mean",
        "wind_max_kmh","wind_mean_kmh",
        "et0_mm","solar_mj","vpd_kpa",
        "is_hot_day","is_very_hot","is_dry_day","is_humid_day",
        "is_rainy_day","is_heavy_rain","no_rain","is_windy",
        "evap_risk","temp_max_lag1","et0_lag1",
    ] if c in d.columns]

    date_cols = [
        "month","day_of_week","day_of_year","quarter",
        "month_sin","month_cos","is_monsoon","is_dry_season",
    ]
    sales_cols    = ["sales_L","sales_lag1","sales_roll7"]
    evap_lag_cols = ["evap_lag1","evap_lag2","evap_roll7"]

    feature_cols = weather_cols + date_cols + sales_cols + evap_lag_cols
    feature_cols = [c for c in feature_cols if c in d.columns]

    return d[feature_cols], d[target_col], feature_cols, d


# ─── TRAIN ONE MODEL ─────────────────────────────────────────────────────────

def train_one_model(df: pd.DataFrame, fuel: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {fuel}")
    print(f"{'='*60}")

    X, y, feature_cols, data = build_features(df, fuel)
    print(f"  Dataset: {len(X)} rows, {len(feature_cols)} features")

    # Cross-validation
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_mapes = [], []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        es_split    = int(len(X_tr) * 0.9)

        fold_model = XGBRegressor(**XGBOOST_PARAMS)
        fold_model.fit(
            X_tr.iloc[:es_split], y_tr.iloc[:es_split],
            eval_set=[(X_tr.iloc[es_split:], y_tr.iloc[es_split:])],
            verbose=False,
        )
        pred = np.maximum(0, fold_model.predict(X_val))
        mae  = mean_absolute_error(y_val, pred)
        mask = y_val > 0
        mape = np.mean(np.abs((y_val[mask] - pred[mask]) /
                               y_val[mask])) * 100
        cv_maes.append(mae)
        cv_mapes.append(mape)

    cv_mae  = np.mean(cv_maes)
    cv_mape = np.mean(cv_mapes)
    print(f"\n  Cross-validation:")
    print(f"    CV MAE  : {cv_mae:.5f} L")
    print(f"    CV MAPE : {cv_mape:.2f}%")

    # Holdout test
    test_size       = 20
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    es_split        = int(len(X_train) * 0.9)

    hm = XGBRegressor(**XGBOOST_PARAMS)
    hm.fit(
        X_train.iloc[:es_split], y_train.iloc[:es_split],
        eval_set=[(X_train.iloc[es_split:], y_train.iloc[es_split:])],
        verbose=False,
    )
    h_pred = np.maximum(0, hm.predict(X_test))
    h_mae  = mean_absolute_error(y_test, h_pred)
    h_rmse = np.sqrt(mean_squared_error(y_test, h_pred))
    mask   = y_test > 0
    h_mape = np.mean(np.abs((y_test[mask] - h_pred[mask]) /
                             y_test[mask])) * 100

    print(f"\n  Holdout (last 20 days):")
    print(f"    MAE  : {h_mae:.5f} L")
    print(f"    RMSE : {h_rmse:.5f} L")
    print(f"    MAPE : {h_mape:.2f}%")

    # Final model on all data
    best_n      = hm.best_iteration + 1
    final_params = {k: v for k, v in XGBOOST_PARAMS.items()
                    if k not in ("early_stopping_rounds","n_estimators")}
    final_params["n_estimators"] = best_n

    final_model = XGBRegressor(**final_params)
    final_model.fit(X, y)

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model,  f"{MODELS_DIR}/{fuel}_evap_model.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/{fuel}_evap_features.pkl")
    joblib.dump({
        "cv_mae":       cv_mae,
        "cv_mape":      cv_mape,
        "holdout_mae":  h_mae,
        "holdout_mape": h_mape,
        "mae":          cv_mae,
        "mape":         cv_mape,
    }, f"{MODELS_DIR}/{fuel}_evap_metrics.pkl")

    print(f"\n  Saved: {MODELS_DIR}/{fuel}_evap_model.pkl")

    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    print(f"\n  TOP 10 FEATURES:")
    for feat, val in imp.head(10).items():
        print(f"    {feat:<35} {val:.4f}")

    return {
        "cv_mape":      cv_mape,
        "holdout_mape": h_mape,
        "mae":          cv_mae,
        "mape":         cv_mape,
    }


# ─── TRAIN ALL 4 ─────────────────────────────────────────────────────────────

def train_all_evap_models(
    evap_path:   str = "data/fuel_sales_evaporation.csv",
    weather_path:str = "data/weather_data.csv",
    models_dir:  str = "models/evaporation",
) -> dict:
    global MODELS_DIR
    MODELS_DIR = models_dir

    print("\nLoading and merging training datasets...")
    print("(CSV + Firebase fuelSaleHistory)")
    df = load_merged_data()

    metrics = {}
    for fuel in FUEL_TYPES:
        metrics[fuel] = train_one_model(df, fuel)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"\n{'Fuel':<20} {'CV MAPE':>10} {'Holdout':>10}")
    print("-"*42)
    for fuel, m in metrics.items():
        print(f"{fuel:<20} {m['cv_mape']:>9.2f}% "
              f"{m['holdout_mape']:>9.2f}%")

    print(f"\nAll 4 models saved to {models_dir}/")
    return metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_all_evap_models()