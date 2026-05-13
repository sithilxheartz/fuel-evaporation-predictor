"""
BACKFILL & DAILY EVAPORATION — EMERALD LANKA (Fixed)
======================================================
Reads real sales from Firebase fuelSaleHistory,
uses REAL weather from Open-Meteo archive for each day,
runs ML evaporation model,
stores results to Firebase fuelEvaporation.

Fix: directly passes weather row to ML model instead of
     going through the forecast file lookup.

Run (backfill all missing days):
  python scripts/backfill_and_daily_evaporation.py --mode=backfill

Run (daily cron - yesterday only):
  python scripts/backfill_and_daily_evaporation.py --mode=daily
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_evap import (
    load_evap_models,
    predict_evaporation_ml,
    FUEL_PRICES_LKR,
    FALLBACK_MONTHLY_RATES,
)
from scripts.fetch_weather import fetch_historical, _add_derived_features

FUEL_TYPES = ["petrol", "super_petrol", "diesel", "super_diesel"]

# ─── FIREBASE ────────────────────────────────────────────────────────────────

_firebase_app = None

def init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return
    import firebase_admin
    from firebase_admin import credentials
    if firebase_admin._apps:
        _firebase_app = firebase_admin.get_app()
        return
    env_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if env_creds:
        cred = credentials.Certificate(json.loads(env_creds))
    else:
        key_path = os.path.join('firebase', 'serviceAccountKey.json')
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"\n❌ Firebase key not found: {key_path}\n"
                "Download from Firebase Console → Project Settings → Service Accounts"
            )
        cred = credentials.Certificate(key_path)
    import firebase_admin
    _firebase_app = firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")


def fetch_sales_from_firebase() -> pd.DataFrame:
    """Fetch all sales records from fuelSaleHistory."""
    init_firebase()
    from firebase_admin import firestore
    db   = firestore.client()
    docs = db.collection('fuelSaleHistory').stream()
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
                return float(v) if v is not None else np.nan
            except Exception:
                return np.nan
        records.append({
            'date':               d,
            'petrol':             safe('92PetrolSale'),
            'super_petrol':       safe('95PetrolSale'),
            'diesel':             safe('dieselSale'),
            'super_diesel':       safe('superDieselSale'),
        })
    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    print(f"  Fetched {len(df)} sales records")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def fetch_existing_dates() -> set:
    """Get dates already stored in fuelEvaporation."""
    init_firebase()
    from firebase_admin import firestore
    db   = firestore.client()
    docs = db.collection('fuelEvaporation').stream()
    existing = {doc.id for doc in docs}
    print(f"  Already in Firebase: {len(existing)} dates")
    return existing


def store_to_firebase(date_str: str, result: dict):
    """Store one day's evaporation to fuelEvaporation/{date}."""
    init_firebase()
    from firebase_admin import firestore
    db  = firestore.client()
    doc = {
        'date':              date_str,
        'generatedAt':       datetime.now().isoformat(),
        'modelType':         result.get('model_type', 'ml_xgboost_weather'),

        # Per fuel
        'petrolEvapL':        float(result['petrol']['evap_litres']),
        'petrolEvapLkr':      float(result['petrol']['evap_lkr']),
        'petrolSalesL':       float(result['petrol']['sales_L']),

        'superPetrolEvapL':   float(result['super_petrol']['evap_litres']),
        'superPetrolEvapLkr': float(result['super_petrol']['evap_lkr']),
        'superPetrolSalesL':  float(result['super_petrol']['sales_L']),

        'dieselEvapL':        float(result['diesel']['evap_litres']),
        'dieselEvapLkr':      float(result['diesel']['evap_lkr']),
        'dieselSalesL':       float(result['diesel']['sales_L']),

        'superDieselEvapL':   float(result['super_diesel']['evap_litres']),
        'superDieselEvapLkr': float(result['super_diesel']['evap_lkr']),
        'superDieselSalesL':  float(result['super_diesel']['sales_L']),

        # Summary
        'totalEvapL':         float(result['total_evap_L']),
        'totalEvapLkr':       float(result['total_evap_lkr']),

        # Weather
        'tempMaxC':           float(result.get('temp_max_c', 0)),
        'precipMm':           float(result.get('precip_mm', 0)),
        'humidityMax':        float(result.get('humidity_max', 0)),
        'et0Mm':              float(result.get('et0_mm', 0)),
    }
    db.collection('fuelEvaporation').document(date_str).set(doc)


# ─── CALCULATE ONE DAY ───────────────────────────────────────────────────────

def calculate_one_day(
    target_date:    date,
    sales:          dict,
    wx_row:         pd.Series,
    evap_models:    dict,
    evap_history:   list,  # list of recent evap values for lag features
    sales_history:  list,  # list of recent sales values
) -> dict:
    """
    Calculate evaporation for one day using ML model + real weather.
    Returns a clean dict ready to store to Firebase.
    """
    FUEL_MAP = {
        'petrol':       'petrol_sales_L',
        'super_petrol': 'super_petrol_sales_L',
        'diesel':       'diesel_sales_L',
        'super_diesel': 'super_diesel_sales_L',
    }

    result = {'model_type': 'ml_xgboost_weather'}
    total_L   = 0.0
    total_lkr = 0.0

    for fuel in FUEL_TYPES:
        sales_L = float(sales.get(fuel, 0.0))
        if sales_L <= 0:
            # OOS day — use monthly average
            sales_L = {'petrol': 2500.0, 'super_petrol': 80.0,
                       'diesel': 2800.0, 'super_diesel': 120.0}[fuel]

        # ML prediction with real weather
        if fuel in evap_models and not wx_row.empty:
            try:
                evap_L = predict_evaporation_ml(
                    target_date    = target_date,
                    fuel           = fuel,
                    sales_litres   = sales_L,
                    weather_row    = wx_row,
                    model_dict     = evap_models[fuel],
                    recent_evap_vals  = evap_history,
                    recent_sales_vals = sales_history,
                )
                method = 'ml'
            except Exception as e:
                evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                method = 'formula_fallback'
        else:
            evap_L = sales_L * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
            method = 'formula'

        evap_L   = round(float(evap_L), 5)
        evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

        result[fuel] = {
            'evap_litres': evap_L,
            'evap_lkr':    evap_lkr,
            'sales_L':     round(sales_L, 3),
            'method':      method,
        }
        total_L   += evap_L
        total_lkr += evap_lkr

    result['total_evap_L']   = round(total_L, 5)
    result['total_evap_lkr'] = round(total_lkr, 2)

    # Weather context
    if not wx_row.empty:
        result['temp_max_c']   = float(wx_row.get('temp_max_c', 0))
        result['precip_mm']    = float(wx_row.get('precip_mm', 0))
        result['humidity_max'] = float(wx_row.get('humidity_max', 0))
        result['et0_mm']       = float(wx_row.get('et0_mm', 0))

    return result


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_pipeline(mode: str = 'backfill'):
    print("\n" + "="*60)
    print(f"  EVAPORATION PIPELINE — {mode.upper()}")
    print("="*60)

    # 1. Load models
    print("\n[1/5] Loading evaporation ML models...")
    evap_models = load_evap_models("models/evaporation")
    print(f"  Loaded: {list(evap_models.keys())}")

    # 2. Fetch sales from Firebase
    print("\n[2/5] Fetching sales from Firebase fuelSaleHistory...")
    sales_df    = fetch_sales_from_firebase()
    last_date   = sales_df['date'].max().date()
    first_date  = sales_df['date'].min().date()

    if mode == 'daily':
        yesterday  = date.today() - timedelta(days=1)
        # Cap to last available sales date
        process_date = min(yesterday, last_date)
        date_range   = [process_date]
        print(f"  Daily mode: processing {process_date}")
    else:
        date_range = [
            (first_date + timedelta(days=i))
            for i in range((last_date - first_date).days + 1)
        ]
        print(f"  Backfill: {first_date} → {last_date} ({len(date_range)} days)")

    # 3. Find already calculated dates
    print("\n[3/5] Checking existing fuelEvaporation records...")
    existing = fetch_existing_dates()
    to_process = [d for d in date_range if str(d) not in existing]
    print(f"  To calculate: {len(to_process)} days")

    if not to_process:
        print("\n  ✅ All dates already calculated!")
        return

    # 4. Fetch real weather for date range
    print(f"\n[4/5] Fetching real weather from Open-Meteo archive...")
    start_str  = str(min(to_process))
    end_str    = str(max(to_process))
    weather_df = fetch_historical(start_str, end_str)
    weather_df = _add_derived_features(weather_df)

    # Build lookups
    sales_lookup = {
        row['date'].strftime('%Y-%m-%d'): {
            'petrol':       float(row['petrol'])       if not pd.isna(row['petrol'])       else 0.0,
            'super_petrol': float(row['super_petrol']) if not pd.isna(row['super_petrol']) else 0.0,
            'diesel':       float(row['diesel'])       if not pd.isna(row['diesel'])       else 0.0,
            'super_diesel': float(row['super_diesel']) if not pd.isna(row['super_diesel']) else 0.0,
        }
        for _, row in sales_df.iterrows()
    }
    weather_lookup = {
        row['date'].strftime('%Y-%m-%d'): row
        for _, row in weather_df.iterrows()
    }

    # Load evap history for lag features
    evap_history_list  = []
    sales_history_list = []
    try:
        hist_df = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_history_list  = hist_df['petrol_evap_L'].tail(30).tolist()
        sales_history_list = hist_df['petrol_sales_L'].tail(30).tolist()
    except Exception:
        pass

    # 5. Calculate and store
    print(f"\n[5/5] Calculating evaporation with REAL weather data...")
    success = 0
    failed  = 0

    for target_date in to_process:
        date_str = str(target_date)
        sales    = sales_lookup.get(date_str, {})
        wx_row   = weather_lookup.get(date_str, pd.Series(dtype=float))

        weather_source = "real weather" if not wx_row.empty else "seasonal avg"

        try:
            result = calculate_one_day(
                target_date   = target_date,
                sales         = sales,
                wx_row        = wx_row,
                evap_models   = evap_models,
                evap_history  = evap_history_list,
                sales_history = sales_history_list,
            )

            store_to_firebase(date_str, result)

            temp = result.get('temp_max_c', '?')
            rain = result.get('precip_mm', '?')
            print(f"  ✅ {date_str}  "
                  f"total={result['total_evap_L']:.3f}L  "
                  f"LKR={result['total_evap_lkr']:.0f}  "
                  f"temp={temp}°C  rain={rain}mm  "
                  f"[{weather_source}]")

            # Update history for next iteration
            evap_history_list.append(result['petrol']['evap_litres'])
            sales_history_list.append(sales.get('petrol', 2500.0))
            success += 1

        except Exception as e:
            print(f"  ❌ {date_str}  Error: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  DONE — {success} stored ✅  {failed} failed ❌")
    print(f"  Firebase: fuelEvaporation collection updated")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='backfill',
                        choices=['backfill', 'daily'],
                        help='backfill=all missing, daily=yesterday only')
    args = parser.parse_args()
    run_pipeline(mode=args.mode)