"""
RETRAIN_EVAP.PY — EMERALD LANKA (Fixed)
=========================================
After retraining, syncs fuelEvaporation to always match fuelSaleHistory.

Key fix:
  - Date range comes from Firebase fuelSaleHistory (NOT local CSV)
  - fuelEvaporation always matches fuelSaleHistory exactly
  - Every date in fuelSaleHistory gets evaporation calculated
  - If fuelSaleHistory has new dates → fuelEvaporation gets them too
  - Second retrain → overwrites all records with fresh model results

Flow:
  1. Fetch latest weather from Open-Meteo
  2. Retrain all 4 XGBoost models
  3. Fetch ALL sales from Firebase fuelSaleHistory
  4. Calculate evaporation for every date that has a sales record
  5. Store/overwrite ALL to Firebase fuelEvaporation

Run manually:
  python scripts/retrain_evap.py
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
                "Add FIREBASE_CREDENTIALS environment variable on Railway."
            )
        cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")


def _fetch_all_sales_from_firebase() -> pd.DataFrame:
    """
    Fetch ALL sales records from Firebase fuelSaleHistory.
    This is the source of truth for which dates need evaporation calculated.
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
            v = data.get(key)
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

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
    print(f"  Fetched {len(df)} sales records from fuelSaleHistory")
    print(f"  Range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def full_evap_retrain_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION RETRAIN PIPELINE")
    print("="*60)
    started = datetime.now()

    # ── Step 1: Fetch ALL sales from Firebase first ───────────────────────────
    # We do this BEFORE fetching weather so we know the exact date range needed
    print("\n[1/5] Fetching sales from Firebase fuelSaleHistory...")
    _init_firebase()
    sales_df   = _fetch_all_sales_from_firebase()
    first_date = sales_df["date"].min().date()
    last_date  = sales_df["date"].max().date()

    print(f"  Will calculate evaporation for: {first_date} → {last_date}")
    print(f"  Total days: {(last_date - first_date).days + 1}")

    # ── Step 2: Fetch weather for exact date range ────────────────────────────
    print(f"\n[2/5] Fetching weather: {first_date} → {last_date}...")
    try:
        from scripts.fetch_weather import fetch_historical, _add_derived_features, fetch_forecast

        wx_df = fetch_historical(
            str(first_date),
            str(last_date),
        )
        wx_df.to_csv("data/weather_data.csv", index=False)
        print(f"  Weather saved: {len(wx_df)} days")

        # Also fetch 8-day forecast for predictions
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Forecast saved: {len(fcast)} days")

    except Exception as e:
        print(f"  ⚠️  Weather fetch failed: {e}")
        print("  Continuing with existing weather data...")
        wx_df = None

    # ── Step 3: Retrain all 4 evaporation models ──────────────────────────────
    print("\n[3/5] Training evaporation models...")
    from scripts.train_evap_models import train_all_evap_models
    metrics = train_all_evap_models()

    # ── Step 4: Load fresh models + weather ──────────────────────────────────
    print("\n[4/5] Loading fresh models and weather data...")
    from scripts.predict_evap import (
        load_evap_models,
        predict_evaporation_ml,
        FUEL_PRICES_LKR,
        FALLBACK_MONTHLY_RATES,
    )

    evap_models = load_evap_models("models/evaporation")
    print(f"  Loaded {len(evap_models)} fresh models")

    # Load weather
    try:
        wx_df = pd.read_csv("data/weather_data.csv")
        wx_df["date"] = pd.to_datetime(wx_df["date"])
    except Exception as e:
        print(f"  ⚠️  Weather data not available: {e}")
        wx_df = pd.DataFrame()

    # ── Step 5: Calculate + store evaporation for ALL sales dates ─────────────
    print(f"\n[5/5] Calculating + storing evaporation for ALL {len(sales_df)} dates...")
    print("  (overwrites existing records with fresh model results)")

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

    weather_lookup = {}
    if not wx_df.empty:
        weather_lookup = {
            row["date"].strftime("%Y-%m-%d"): row
            for _, row in wx_df.iterrows()
        }

    # Load evap history for lag features
    evap_history = []
    try:
        hist_df      = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_history = hist_df["petrol_evap_L"].tail(30).tolist()
    except Exception:
        pass

    FUEL_TYPES = ["petrol", "super_petrol", "diesel", "super_diesel"]

    from firebase_admin import firestore
    db = firestore.client()

    success     = 0
    failed      = 0
    batch       = db.batch()
    batch_count = 0
    MAX_BATCH   = 400

    # Process every date that has a sales record in Firebase
    for _, sale_row in sales_df.iterrows():
        target_date = sale_row["date"].date()
        date_str    = str(target_date)

        sales  = sales_lookup.get(date_str, {
            "petrol": 2500.0, "super_petrol": 80.0,
            "diesel": 2800.0, "super_diesel": 120.0,
        })
        wx_row = weather_lookup.get(date_str, pd.Series(dtype=float))

        try:
            doc_data = {
                "date":        date_str,
                "generatedAt": datetime.now().isoformat(),
                "modelType":   "ml_xgboost_weather",
            }

            total_L   = 0.0
            total_lkr = 0.0

            for fuel in FUEL_TYPES:
                sales_L = float(sales.get(fuel, 2000.0))

                # Use ML model + real weather if available
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
                        method = "ml"
                    except Exception:
                        evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                        method = "formula_fallback"
                else:
                    evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                    method = "formula"

                evap_L   = round(float(max(0, evap_L)), 5)
                evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

                # Map to Firebase field names
                fuel_key = {
                    "petrol":       "petrol",
                    "super_petrol": "superPetrol",
                    "diesel":       "diesel",
                    "super_diesel": "superDiesel",
                }[fuel]

                doc_data[f"{fuel_key}EvapL"]   = evap_L
                doc_data[f"{fuel_key}EvapLkr"] = evap_lkr
                doc_data[f"{fuel_key}SalesL"]  = round(sales_L, 2)

                total_L   += evap_L
                total_lkr += evap_lkr

                if fuel == "petrol":
                    evap_history.append(evap_L)

            # Summary
            doc_data["totalEvapL"]   = round(total_L, 5)
            doc_data["totalEvapLkr"] = round(total_lkr, 2)

            # Weather context
            if not wx_row.empty:
                doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",  0) or 0)
                doc_data["precipMm"]    = float(wx_row.get("precip_mm",   0) or 0)
                doc_data["humidityMax"] = float(wx_row.get("humidity_max",0) or 0)
                doc_data["et0Mm"]       = float(wx_row.get("et0_mm",      0) or 0)

            # Batch write — set() overwrites if exists, creates if not
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

    # Commit remaining records
    if batch_count > 0:
        batch.commit()

    elapsed = (datetime.now() - started).seconds

    print(f"\n{'='*60}")
    print(f"  RETRAIN + FIREBASE SYNC COMPLETE in {elapsed}s")
    print(f"{'='*60}")
    print(f"  ✅ Stored {success} records to Firebase fuelEvaporation")
    print(f"  📅 Range: {first_date} → {last_date}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}")
    print()
    print("  Model accuracy:")
    for fuel, m in metrics.items():
        print(f"    {fuel:<20} CV MAPE: {m['cv_mape']:.2f}%  "
              f"Holdout: {m['holdout_mape']:.2f}%")

    return metrics


if __name__ == "__main__":
    full_evap_retrain_pipeline()