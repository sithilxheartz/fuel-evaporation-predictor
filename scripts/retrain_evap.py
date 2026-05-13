"""
RETRAIN_EVAP.PY — EMERALD LANKA (Fixed: missing fuel fields)
=============================================================
Key fix:
  Only calculates evaporation for fuels that ACTUALLY have sales
  data in Firebase fuelSaleHistory for that date.

  If 2026-05-13 only has dieselSale and superDieselSale:
    → only diesel and super_diesel evaporation is calculated
    → petrol and super_petrol are stored as 0 (not fake averages)
    → when petrol sales are added later → retrain again → updates correctly

Uses TWO separate weather files:
  weather_data.csv    → training (Apr 2025 → Apr 2026, 384 rows)
  weather_deploy.csv  → deployment (Firebase sales date range)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── FIREBASE ────────────────────────────────────────────────────────────────

def _init_firebase():
    import firebase_admin
    from firebase_admin import credentials
    if firebase_admin._apps:
        return
    env_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if env_creds:
        cred = credentials.Certificate(json.loads(env_creds))
    else:
        key_path = os.path.join("firebase", "serviceAccountKey.json")
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Firebase key not found: {key_path}\n"
                "Add FIREBASE_CREDENTIALS env var on Railway."
            )
        cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")


def _fetch_all_sales() -> pd.DataFrame:
    """
    Fetch all sales from Firebase fuelSaleHistory.
    Returns None for missing fuel fields — not fake averages.
    """
    _init_firebase()
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
            """Returns actual value or None if field missing/null."""
            v = data.get(key)
            if v is None:
                return None          # ← field not entered yet
            try:
                f = float(v)
                return f if f > 0 else None   # ← 0 = OOS, treat as None
            except Exception:
                return None

        records.append({
            "date":         d,
            "petrol":       safe("92PetrolSale"),
            "super_petrol": safe("95PetrolSale"),
            "diesel":       safe("dieselSale"),
            "super_diesel": safe("superDieselSale"),
        })

    if not records:
        raise ValueError("No records found in Firebase fuelSaleHistory")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  Fetched {len(df)} sales records")
    print(f"  Range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Show how many records have each fuel
    for fuel, col in [("petrol","petrol"), ("super_petrol","super_petrol"),
                      ("diesel","diesel"), ("super_diesel","super_diesel")]:
        count = df[col].notna().sum()
        print(f"    {fuel:<20}: {count}/{len(df)} days have sales data")

    return df


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def full_evap_retrain_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION RETRAIN PIPELINE")
    print("="*60)
    started = datetime.now()

    from scripts.fetch_weather import (
        fetch_historical, _add_derived_features, fetch_forecast
    )

    # ── Step 1: Sales from Firebase ──────────────────────────────────────────
    print("\n[1/6] Fetching sales from Firebase fuelSaleHistory...")
    _init_firebase()
    sales_df   = _fetch_all_sales()
    first_date = sales_df["date"].min().date()
    last_date  = sales_df["date"].max().date()
    print(f"  Will calculate: {first_date} → {last_date}")

    # ── Step 2: Training weather (full evap CSV range → 384 rows) ────────────
    print("\n[2/6] Fetching TRAINING weather (full evap CSV range)...")
    try:
        evap_csv = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_csv["date"] = pd.to_datetime(evap_csv["date"])
        t_start  = evap_csv["date"].min().strftime("%Y-%m-%d")
        t_end    = evap_csv["date"].max().strftime("%Y-%m-%d")
    except Exception:
        t_start  = "2025-04-01"
        t_end    = "2026-04-19"

    try:
        train_wx = fetch_historical(t_start, t_end)
        train_wx.to_csv("data/weather_data.csv", index=False)
        print(f"  Training weather: {len(train_wx)} days "
              f"({t_start} → {t_end})")
    except Exception as e:
        print(f"  ⚠️  Training weather failed: {e} — using existing")

    # ── Step 3: Deployment weather (Firebase sales date range) ───────────────
    print(f"\n[3/6] Fetching DEPLOYMENT weather "
          f"({first_date} → {last_date})...")
    deploy_wx_df = pd.DataFrame()
    try:
        deploy_wx = fetch_historical(str(first_date), str(last_date))
        deploy_wx.to_csv("data/weather_deploy.csv", index=False)
        deploy_wx_df = deploy_wx
        print(f"  Deployment weather: {len(deploy_wx)} days")

        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Forecast: {len(fcast)} days")
    except Exception as e:
        print(f"  ⚠️  Deployment weather failed: {e}")
        try:
            deploy_wx_df = pd.read_csv("data/weather_deploy.csv")
            deploy_wx_df["date"] = pd.to_datetime(deploy_wx_df["date"])
            print(f"  Using cached: {len(deploy_wx_df)} days")
        except Exception:
            print("  No deployment weather — will use formula fallback")

    # ── Step 4: Train on FULL 384-row dataset ────────────────────────────────
    print("\n[4/6] Training models on full 384-row dataset...")
    from scripts.train_evap_models import train_all_evap_models
    metrics = train_all_evap_models()

    # ── Step 5: Load fresh models ─────────────────────────────────────────────
    print("\n[5/6] Loading fresh models...")
    from scripts.predict_evap import (
        load_evap_models,
        predict_evaporation_ml,
        FUEL_PRICES_LKR,
        FALLBACK_MONTHLY_RATES,
    )
    evap_models = load_evap_models("models/evaporation")
    print(f"  Loaded {len(evap_models)} models")

    if deploy_wx_df.empty:
        try:
            deploy_wx_df = pd.read_csv("data/weather_deploy.csv")
            deploy_wx_df["date"] = pd.to_datetime(deploy_wx_df["date"])
        except Exception:
            pass

    weather_lookup = {}
    if not deploy_wx_df.empty:
        weather_lookup = {
            row["date"].strftime("%Y-%m-%d"): row
            for _, row in deploy_wx_df.iterrows()
        }
    print(f"  Weather lookup: {len(weather_lookup)} dates")

    # ── Step 6: Calculate + store evaporation ────────────────────────────────
    print(f"\n[6/6] Calculating + storing {len(sales_df)} records...")
    print("  Only calculates evaporation for fuels with ACTUAL sales data")
    print("  Missing fuels stored as 0 (not fake averages)")

    FUEL_TYPES = ["petrol", "super_petrol", "diesel", "super_diesel"]

    # Firebase field name mapping
    FIREBASE_KEYS = {
        "petrol":       "petrol",
        "super_petrol": "superPetrol",
        "diesel":       "diesel",
        "super_diesel": "superDiesel",
    }

    evap_history = []
    try:
        hist_df      = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_history = hist_df["petrol_evap_L"].tail(30).tolist()
    except Exception:
        pass

    from firebase_admin import firestore
    db          = firestore.client()
    batch       = db.batch()
    batch_count = 0
    MAX_BATCH   = 400
    success     = 0
    failed      = 0

    for _, sale_row in sales_df.iterrows():
        target_date = sale_row["date"].date()
        date_str    = str(target_date)
        wx_row      = weather_lookup.get(date_str, pd.Series(dtype=float))

        try:
            doc_data = {
                "date":        date_str,
                "generatedAt": datetime.now().isoformat(),
                "modelType":   "ml_xgboost_weather",
            }

            total_L   = 0.0
            total_lkr = 0.0

            for fuel in FUEL_TYPES:
                fk      = FIREBASE_KEYS[fuel]
                sales_L = sale_row[fuel]   # None if field missing in Firebase

                # ── ONLY calculate if this fuel has actual sales data ──────
                if sales_L is None or pd.isna(sales_L):
                    # Field not entered yet — store 0, not a fake average
                    doc_data[f"{fk}EvapL"]   = 0.0
                    doc_data[f"{fk}EvapLkr"] = 0.0
                    doc_data[f"{fk}SalesL"]  = 0.0
                    continue

                sales_L = float(sales_L)

                # Calculate evaporation using ML model + weather
                if fuel in evap_models and not wx_row.empty:
                    try:
                        evap_L = predict_evaporation_ml(
                            target_date       = target_date,
                            fuel              = fuel,
                            sales_litres      = sales_L,
                            weather_row       = wx_row,
                            model_dict        = evap_models[fuel],
                            recent_evap_vals  = evap_history,
                            recent_sales_vals = [],
                        )
                    except Exception:
                        evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                else:
                    evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]

                evap_L   = round(float(max(0, evap_L)), 5)
                evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

                doc_data[f"{fk}EvapL"]   = evap_L
                doc_data[f"{fk}EvapLkr"] = evap_lkr
                doc_data[f"{fk}SalesL"]  = round(sales_L, 2)

                total_L   += evap_L
                total_lkr += evap_lkr

                if fuel == "petrol":
                    evap_history.append(evap_L)

            doc_data["totalEvapL"]   = round(total_L, 5)
            doc_data["totalEvapLkr"] = round(total_lkr, 2)

            # Weather context
            if not wx_row.empty:
                doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",   0) or 0)
                doc_data["precipMm"]    = float(wx_row.get("precip_mm",    0) or 0)
                doc_data["humidityMax"] = float(wx_row.get("humidity_max", 0) or 0)
                doc_data["et0Mm"]       = float(wx_row.get("et0_mm",       0) or 0)

            ref = db.collection("fuelEvaporation").document(date_str)
            batch.set(ref, doc_data)
            batch_count += 1
            success     += 1

            if batch_count >= MAX_BATCH:
                batch.commit()
                print(f"  Committed batch of {batch_count}...")
                batch       = db.batch()
                batch_count = 0

        except Exception as e:
            print(f"  ❌ {date_str}: {e}")
            failed += 1

    if batch_count > 0:
        batch.commit()

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETE in {elapsed}s")
    print(f"  Stored:  {success} records  📅 {first_date} → {last_date}")
    if failed:
        print(f"  Failed:  {failed}")
    print()
    print("  Model accuracy (full 384-row dataset):")
    for fuel, m in metrics.items():
        print(f"    {fuel:<20} CV: {m['cv_mape']:.2f}%  "
              f"Holdout: {m['holdout_mape']:.2f}%")
    return metrics


if __name__ == "__main__":
    full_evap_retrain_pipeline()