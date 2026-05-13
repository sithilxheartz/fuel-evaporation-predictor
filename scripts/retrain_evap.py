"""
RETRAIN_EVAP.PY — EMERALD LANKA
=================================
Full evaporation retrain pipeline.
After retraining, automatically calculates and stores ALL evaporation
data to Firebase fuelEvaporation collection.

Logic:
  1. Fetch latest weather from Open-Meteo
  2. Train all 4 XGBoost evaporation models
  3. After training: fetch all sales from Firebase fuelSaleHistory
  4. Calculate evaporation for every day (using fresh models)
  5. Store/overwrite to Firebase fuelEvaporation
     (if same date exists → overwrite with new data)

Run manually:
  python scripts/retrain_evap.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def full_evap_retrain_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION RETRAIN PIPELINE")
    print("="*60)
    started = datetime.now()

    # ── Step 1: Fetch latest weather ─────────────────────────────────────────
    print("\n[1/4] Fetching latest weather from Open-Meteo...")
    try:
        from scripts.fetch_weather import fetch_historical, _add_derived_features

        # Get date range from evap CSV
        clean_path = "data/fuel_sales_evaporation.csv"
        if os.path.exists(clean_path):
            evap_df    = pd.read_csv(clean_path)
            evap_df["date"] = pd.to_datetime(evap_df["date"])
            start_date = evap_df["date"].min().strftime("%Y-%m-%d")
        else:
            start_date = "2025-04-01"

        # Fetch up to today
        end_date = date.today().strftime("%Y-%m-%d")

        wx_df = fetch_historical(start_date, end_date)
        wx_df.to_csv("data/weather_data.csv", index=False)
        print(f"  Weather saved: {len(wx_df)} days ({start_date} → {end_date})")

        # Also fetch 8-day forecast for predictions
        from scripts.fetch_weather import fetch_forecast
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Forecast saved: {len(fcast)} days")

    except Exception as e:
        print(f"  ⚠️  Weather fetch failed: {e}")
        print("  Continuing with existing weather data...")

    # ── Step 2: Train all 4 evaporation models ────────────────────────────────
    print("\n[2/4] Training evaporation models...")
    from scripts.train_evap_models import train_all_evap_models
    metrics = train_all_evap_models()

    # ── Step 3: Store evaporation to Firebase ────────────────────────────────
    print("\n[3/4] Storing evaporation data to Firebase...")
    try:
        _store_all_to_firebase()
    except Exception as e:
        print(f"  ⚠️  Firebase storage failed: {e}")
        print("  Models were trained successfully. Firebase storage can be retried.")

    # ── Step 4: Update forecast ───────────────────────────────────────────────
    print("\n[4/4] Updating weather forecast...")
    try:
        from scripts.fetch_weather import fetch_forecast
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print("  Forecast updated")
    except Exception as e:
        print(f"  ⚠️  Forecast update failed: {e}")

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE in {elapsed}s")
    print(f"{'='*60}")
    for fuel, m in metrics.items():
        print(f"  {fuel:<20} CV MAPE: {m['cv_mape']:.2f}%  "
              f"Holdout: {m['holdout_mape']:.2f}%")

    return metrics


def _store_all_to_firebase():
    """
    After retraining, calculate evaporation for ALL dates in Firebase
    fuelSaleHistory and store/overwrite to fuelEvaporation.

    Key behaviour:
      - Fetches ALL sales records from Firebase
      - Calculates evaporation using the NEWLY TRAINED models
      - If a date already exists in fuelEvaporation → OVERWRITES it
      - This ensures data always reflects the latest model
    """
    print("  Loading freshly trained models...")
    from scripts.predict_evap import (
        load_evap_models,
        predict_evaporation_ml,
        FUEL_PRICES_LKR,
        FALLBACK_MONTHLY_RATES,
    )
    from scripts.fetch_weather import _add_derived_features

    evap_models = load_evap_models("models/evaporation")
    if not evap_models:
        raise Exception("No models found after training")

    print(f"  Loaded {len(evap_models)} fresh models")

    # Connect to Firebase
    import json
    _init_firebase()
    from firebase_admin import firestore
    db = firestore.client()

    # Fetch all sales from Firebase
    print("  Fetching sales from Firebase fuelSaleHistory...")
    docs     = db.collection("fuelSaleHistory").stream()
    sales_records = []
    for doc in docs:
        data = doc.to_dict()
        try:
            d = pd.to_datetime(doc.id)
        except Exception:
            continue

        def safe(key):
            v = data.get(key)
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        sales_records.append({
            "date":         d,
            "petrol":       safe("92PetrolSale"),
            "super_petrol": safe("95PetrolSale"),
            "diesel":       safe("dieselSale"),
            "super_diesel": safe("superDieselSale"),
        })

    if not sales_records:
        raise Exception("No sales records found in Firebase")

    sales_df   = pd.DataFrame(sales_records).sort_values("date").reset_index(drop=True)
    first_date = sales_df["date"].min().date()
    last_date  = sales_df["date"].max().date()
    print(f"  Found {len(sales_df)} sales records: {first_date} → {last_date}")

    # Load weather data
    print("  Loading weather data...")
    wx_df = pd.read_csv("data/weather_data.csv")
    wx_df["date"] = pd.to_datetime(wx_df["date"])

    # Build lookups
    sales_lookup = {
        row["date"].strftime("%Y-%m-%d"): {
            "petrol":       float(row["petrol"])       if row["petrol"] > 0       else 2500.0,
            "super_petrol": float(row["super_petrol"]) if row["super_petrol"] > 0 else 80.0,
            "diesel":       float(row["diesel"])       if row["diesel"] > 0       else 2800.0,
            "super_diesel": float(row["super_diesel"]) if row["super_diesel"] > 0 else 120.0,
        }
        for _, row in sales_df.iterrows()
    }
    weather_lookup = {
        row["date"].strftime("%Y-%m-%d"): row
        for _, row in wx_df.iterrows()
    }

    # Load evap history for lag features
    evap_history = []
    try:
        hist_df = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_history = hist_df["petrol_evap_L"].tail(30).tolist()
    except Exception:
        pass

    # Process every date
    all_dates = [
        first_date + timedelta(days=i)
        for i in range((last_date - first_date).days + 1)
    ]

    FUEL_TYPES = ["petrol", "super_petrol", "diesel", "super_diesel"]
    success = 0
    failed  = 0

    print(f"  Calculating + storing {len(all_dates)} days "
          f"(overwriting existing records)...")

    # Use batch writes for speed (max 400 per batch)
    batch     = db.batch()
    batch_count = 0
    MAX_BATCH   = 400

    for target_date in all_dates:
        date_str = str(target_date)
        sales    = sales_lookup.get(date_str, {
            "petrol": 2500.0, "super_petrol": 80.0,
            "diesel": 2800.0, "super_diesel": 120.0,
        })
        wx_row = weather_lookup.get(date_str, pd.Series(dtype=float))

        try:
            doc_data = {
                "date":       date_str,
                "generatedAt": datetime.now().isoformat(),
                "modelType":  "ml_xgboost_weather",
            }

            total_L   = 0.0
            total_lkr = 0.0

            for fuel in FUEL_TYPES:
                sales_L = float(sales.get(fuel, 2000.0))

                # Use ML model with real weather
                if fuel in evap_models and not wx_row.empty:
                    try:
                        evap_L = predict_evaporation_ml(
                            target_date    = target_date,
                            fuel           = fuel,
                            sales_litres   = sales_L,
                            weather_row    = wx_row,
                            model_dict     = evap_models[fuel],
                            recent_evap_vals  = evap_history,
                            recent_sales_vals = [],
                        )
                        method = "ml"
                    except Exception:
                        evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                        method = "formula_fallback"
                else:
                    evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                    method = "formula"

                evap_L   = round(float(max(0, evap_L)), 5)
                evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

                # Firebase field names
                fuel_key = {
                    "petrol":       "petrol",
                    "super_petrol": "superPetrol",
                    "diesel":       "diesel",
                    "super_diesel": "superDiesel",
                }[fuel]

                doc_data[f"{fuel_key}EvapL"]    = evap_L
                doc_data[f"{fuel_key}EvapLkr"]  = evap_lkr
                doc_data[f"{fuel_key}SalesL"]   = round(sales_L, 2)
                doc_data[f"method_{fuel_key}"]  = method

                total_L   += evap_L
                total_lkr += evap_lkr

                # Update lag history
                if fuel == "petrol":
                    evap_history.append(evap_L)

            # Summary fields
            doc_data["totalEvapL"]   = round(total_L, 5)
            doc_data["totalEvapLkr"] = round(total_lkr, 2)

            # Weather context
            if not wx_row.empty:
                doc_data["tempMaxC"]   = float(wx_row.get("temp_max_c",  0) or 0)
                doc_data["precipMm"]   = float(wx_row.get("precip_mm",   0) or 0)
                doc_data["humidityMax"]= float(wx_row.get("humidity_max",0) or 0)
                doc_data["et0Mm"]      = float(wx_row.get("et0_mm",      0) or 0)

            # Add to batch (SET with merge=True → creates or overwrites)
            ref = db.collection("fuelEvaporation").document(date_str)
            batch.set(ref, doc_data)
            batch_count += 1
            success     += 1

            # Commit batch when full
            if batch_count >= MAX_BATCH:
                batch.commit()
                print(f"  Committed batch of {batch_count} records...")
                batch       = db.batch()
                batch_count = 0

        except Exception as e:
            print(f"  ❌ {date_str}: {e}")
            failed += 1

    # Commit remaining
    if batch_count > 0:
        batch.commit()

    print(f"\n  ✅ Stored {success} records to Firebase fuelEvaporation")
    if failed > 0:
        print(f"  ❌ Failed: {failed} records")


def _init_firebase():
    """Initialize Firebase connection."""
    import json
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
                f"\n❌ Firebase key not found: {key_path}\n"
                "Add FIREBASE_CREDENTIALS environment variable on Railway."
            )
        cred = credentials.Certificate(key_path)

    firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")


if __name__ == "__main__":
    full_evap_retrain_pipeline()