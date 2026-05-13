"""
EVAPORATION PIPELINE — EMERALD LANKA
======================================
Complete pipeline that:

1. Reads fuel prices from Firebase fuelTanks collection
2. Reads actual evaporation from fuel_sales_evaporation.csv
   (Apr 2025 → Apr 2026) → stores as type="actual"
3. Reads sales from Firebase fuelSaleHistory
   (Feb 2026 → today) → predicts evaporation using ML model
   for dates AFTER the CSV ends → stores as type="predicted"
4. Stores monthly summaries to fuelEvaporationSummary collection
   (one document per month: 2026-02, 2026-03, etc.)
5. Stores overall summary to fuelEvaporation/--summary--
   (last 7 days, last 30 days totals)

Run:
  python scripts/evaporation_pipeline.py

Called automatically after retrain completes.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FUEL_TYPES   = ["petrol", "super_petrol", "diesel", "super_diesel"]
FIREBASE_KEYS = {
    "petrol":       "petrol",
    "super_petrol": "superPetrol",
    "diesel":       "diesel",
    "super_diesel": "superDiesel",
}
# Fallback prices if Firebase fetch fails
DEFAULT_PRICES = {
    "petrol":       398.0,
    "super_petrol": 455.0,
    "diesel":       382.0,
    "super_diesel": 420.0,
}


# ─── FIREBASE ────────────────────────────────────────────────────────────────

_firebase_initialized = False

def init_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return
    import firebase_admin
    from firebase_admin import credentials
    if firebase_admin._apps:
        _firebase_initialized = True
        return
    env = os.environ.get("FIREBASE_CREDENTIALS")
    if env:
        cred = credentials.Certificate(json.loads(env))
    else:
        key = os.path.join("firebase", "serviceAccountKey.json")
        if not os.path.exists(key):
            raise FileNotFoundError(f"Firebase key not found: {key}")
        cred = credentials.Certificate(key)
    firebase_admin.initialize_app(cred)
    _firebase_initialized = True
    print("  ✅ Firebase connected")


def get_fuel_prices() -> dict:
    """
    Read current fuel prices from Firebase fuelTanks collection.
    Fields: fuelType, fuelPrice
    Fuel types: "92 Petrol", "95 Petrol", "Auto Diesel", "Super Diesel"
    """
    init_firebase()
    from firebase_admin import firestore
    db    = firestore.client()
    docs  = db.collection("fuelTanks").stream()
    prices = dict(DEFAULT_PRICES)  # start with defaults

    type_map = {
        "92 Petrol":    "petrol",
        "95 Petrol":    "super_petrol",
        "Auto Diesel":  "diesel",
        "Super Diesel": "super_diesel",
    }

    for doc in docs:
        data = doc.to_dict()
        ft   = data.get("fuelType", "")
        fp   = data.get("fuelPrice")
        if ft in type_map and fp:
            try:
                prices[type_map[ft]] = float(fp)
            except Exception:
                pass

    print(f"  Fuel prices (from Firebase fuelTanks):")
    for f, p in prices.items():
        print(f"    {f:<20}: LKR {p}")
    return prices


def fetch_sales_from_firebase() -> pd.DataFrame:
    """Fetch all sales from Firebase fuelSaleHistory."""
    init_firebase()
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
                f = float(v) if v is not None else None
                return f if f and f > 0 else None
            except Exception:
                return None

        records.append({
            "date":         d,
            "petrol":       safe("92PetrolSale"),
            "super_petrol": safe("95PetrolSale"),
            "diesel":       safe("dieselSale"),
            "super_diesel": safe("superDieselSale"),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  Firebase fuelSaleHistory: {len(df)} records "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def fetch_weather_for_range(start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch weather from Open-Meteo for a date range."""
    try:
        from scripts.fetch_weather import fetch_historical, _add_derived_features
        wx = fetch_historical(start_str, end_str)
        return wx
    except Exception as e:
        print(f"  ⚠️  Weather fetch failed: {e}")
        return pd.DataFrame()


# ─── STEP 1: BUILD ACTUAL EVAPORATION (from CSV) ─────────────────────────────

def build_actual_evap_records(
    evap_csv_path: str,
    fuel_prices: dict,
    wx_lookup: dict,
    evap_models: dict,
    fallback_rates: dict,
) -> list:
    """
    Read evaporation directly from fuel_sales_evaporation.csv.
    These are "actual" records — real measured/calculated values.

    For each date in the CSV that overlaps with Firebase sales,
    we use the ML model with real weather for consistency.
    For dates only in CSV (before Firebase), we use CSV evap values directly.

    Returns list of doc dicts ready to store to Firebase.
    """
    print(f"\n  Reading actual evaporation from CSV...")
    evap_df = pd.read_csv(evap_csv_path)
    evap_df["date"] = pd.to_datetime(evap_df["date"])
    print(f"  CSV rows: {len(evap_df)} "
          f"({evap_df['date'].min().date()} → {evap_df['date'].max().date()})")

    EVAP_COLS = {
        "petrol":       "petrol_evap_L",
        "super_petrol": "super_petrol_evap_L",
        "diesel":       "diesel_evap_L",
        "super_diesel": "super_diesel_evap_L",
    }
    SALES_COLS = {
        "petrol":       "petrol_sales_L",
        "super_petrol": "super_petrol_sales_L",
        "diesel":       "diesel_sales_L",
        "super_diesel": "super_diesel_sales_L",
    }

    records = []
    for _, row in evap_df.iterrows():
        target_date = row["date"].date()
        date_str    = str(target_date)
        wx_row      = wx_lookup.get(date_str, pd.Series(dtype=float))

        doc_data = {
            "date":        date_str,
            "generatedAt": datetime.now().isoformat(),
            "type":        "actual",   # ← from real CSV data
            "modelType":   "ml_xgboost_weather",
        }

        total_L   = 0.0
        total_lkr = 0.0

        for fuel in FUEL_TYPES:
            fk        = FIREBASE_KEYS[fuel]
            sales_L   = row.get(SALES_COLS[fuel], 0)
            csv_evap  = row.get(EVAP_COLS[fuel], 0)

            if pd.isna(sales_L) or sales_L <= 0:
                doc_data[f"{fk}EvapL"]   = 0.0
                doc_data[f"{fk}EvapLkr"] = 0.0
                doc_data[f"{fk}SalesL"]  = 0.0
                continue

            sales_L  = float(sales_L)
            csv_evap = float(csv_evap) if not pd.isna(csv_evap) else 0.0

            # Try ML model with weather first, fall back to CSV value
            if fuel in evap_models and not wx_row.empty:
                try:
                    from scripts.predict_evap import predict_evaporation_ml
                    evap_L = predict_evaporation_ml(
                        target_date       = target_date,
                        fuel              = fuel,
                        sales_litres      = sales_L,
                        weather_row       = wx_row,
                        model_dict        = evap_models[fuel],
                        recent_evap_vals  = [],
                        recent_sales_vals = [],
                    )
                except Exception:
                    evap_L = csv_evap if csv_evap > 0 else (
                        sales_L * fallback_rates[fuel][target_date.month])
            else:
                # Use CSV value directly — this IS the actual measured value
                evap_L = csv_evap if csv_evap > 0 else (
                    sales_L * fallback_rates[fuel][target_date.month])

            evap_L   = round(float(max(0, evap_L)), 5)
            price    = fuel_prices.get(fuel, DEFAULT_PRICES[fuel])
            evap_lkr = round(evap_L * price, 2)

            doc_data[f"{fk}EvapL"]   = evap_L
            doc_data[f"{fk}EvapLkr"] = evap_lkr
            doc_data[f"{fk}SalesL"]  = round(sales_L, 3)

            total_L   += evap_L
            total_lkr += evap_lkr

        doc_data["totalEvapL"]   = round(total_L, 5)
        doc_data["totalEvapLkr"] = round(total_lkr, 2)

        # Weather context
        if not wx_row.empty:
            doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",   0) or 0)
            doc_data["precipMm"]    = float(wx_row.get("precip_mm",    0) or 0)
            doc_data["humidityMax"] = float(wx_row.get("humidity_max", 0) or 0)
            doc_data["et0Mm"]       = float(wx_row.get("et0_mm",       0) or 0)

        records.append(doc_data)

    print(f"  Built {len(records)} actual records")
    return records


# ─── STEP 2: BUILD PREDICTED EVAPORATION (ML model) ──────────────────────────

def build_predicted_evap_records(
    sales_df: pd.DataFrame,
    csv_end_date: date,
    fuel_prices: dict,
    wx_lookup: dict,
    evap_models: dict,
    fallback_rates: dict,
) -> list:
    """
    For dates AFTER the CSV ends (predicted period):
    Use Firebase sales + ML model + weather to predict evaporation.

    Only processes dates where Firebase has sales data AND
    the date is AFTER csv_end_date.
    """
    # Filter to only dates after CSV ends
    predicted_df = sales_df[sales_df["date"].dt.date > csv_end_date].copy()

    if predicted_df.empty:
        print(f"  No predicted records needed (Firebase ends at or before CSV)")
        return []

    print(f"\n  Building predicted evaporation: "
          f"{predicted_df['date'].min().date()} → "
          f"{predicted_df['date'].max().date()} "
          f"({len(predicted_df)} days)")

    records = []
    for _, row in predicted_df.iterrows():
        target_date = row["date"].date()
        date_str    = str(target_date)
        wx_row      = wx_lookup.get(date_str, pd.Series(dtype=float))

        doc_data = {
            "date":        date_str,
            "generatedAt": datetime.now().isoformat(),
            "type":        "predicted",   # ← ML model prediction
            "modelType":   "ml_xgboost_weather",
        }

        total_L   = 0.0
        total_lkr = 0.0

        for fuel in FUEL_TYPES:
            fk      = FIREBASE_KEYS[fuel]
            sales_L = row.get(fuel)

            if sales_L is None or pd.isna(sales_L) or sales_L <= 0:
                doc_data[f"{fk}EvapL"]   = 0.0
                doc_data[f"{fk}EvapLkr"] = 0.0
                doc_data[f"{fk}SalesL"]  = 0.0
                continue

            sales_L = float(sales_L)

            if fuel in evap_models and not wx_row.empty:
                try:
                    from scripts.predict_evap import predict_evaporation_ml
                    evap_L = predict_evaporation_ml(
                        target_date       = target_date,
                        fuel              = fuel,
                        sales_litres      = sales_L,
                        weather_row       = wx_row,
                        model_dict        = evap_models[fuel],
                        recent_evap_vals  = [],
                        recent_sales_vals = [],
                    )
                except Exception:
                    evap_L = sales_L * fallback_rates[fuel][target_date.month]
            else:
                evap_L = sales_L * fallback_rates[fuel][target_date.month]

            evap_L   = round(float(max(0, evap_L)), 5)
            price    = fuel_prices.get(fuel, DEFAULT_PRICES[fuel])
            evap_lkr = round(evap_L * price, 2)

            doc_data[f"{fk}EvapL"]   = evap_L
            doc_data[f"{fk}EvapLkr"] = evap_lkr
            doc_data[f"{fk}SalesL"]  = round(sales_L, 3)

            total_L   += evap_L
            total_lkr += evap_lkr

        doc_data["totalEvapL"]   = round(total_L, 5)
        doc_data["totalEvapLkr"] = round(total_lkr, 2)

        if not wx_row.empty:
            doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",   0) or 0)
            doc_data["precipMm"]    = float(wx_row.get("precip_mm",    0) or 0)
            doc_data["humidityMax"] = float(wx_row.get("humidity_max", 0) or 0)
            doc_data["et0Mm"]       = float(wx_row.get("et0_mm",       0) or 0)

        records.append(doc_data)
        print(f"    {date_str}  total={total_L:.3f}L  "
              f"LKR={total_lkr:.0f}  [predicted]")

    print(f"  Built {len(records)} predicted records")
    return records


# ─── STEP 3: BUILD MONTHLY SUMMARIES ─────────────────────────────────────────

def build_monthly_summaries(all_records: list, fuel_prices: dict) -> dict:
    """
    Group all records by month and calculate totals.
    Returns dict: {"2026-02": {...}, "2026-03": {...}, ...}
    """
    monthly = defaultdict(lambda: {
        "totalEvapL":       0.0,
        "totalEvapLkr":     0.0,
        "petrolEvapL":      0.0,
        "petrolEvapLkr":    0.0,
        "superPetrolEvapL": 0.0,
        "superPetrolEvapLkr": 0.0,
        "dieselEvapL":      0.0,
        "dieselEvapLkr":    0.0,
        "superDieselEvapL": 0.0,
        "superDieselEvapLkr": 0.0,
        "days":             0,
        "actualDays":       0,
        "predictedDays":    0,
    })

    for rec in all_records:
        try:
            d         = date.fromisoformat(rec["date"])
            month_key = d.strftime("%Y-%m")
            m         = monthly[month_key]

            m["totalEvapL"]         += rec.get("totalEvapL",       0)
            m["totalEvapLkr"]       += rec.get("totalEvapLkr",     0)
            m["petrolEvapL"]        += rec.get("petrolEvapL",       0)
            m["petrolEvapLkr"]      += rec.get("petrolEvapLkr",     0)
            m["superPetrolEvapL"]   += rec.get("superPetrolEvapL",  0)
            m["superPetrolEvapLkr"] += rec.get("superPetrolEvapLkr",0)
            m["dieselEvapL"]        += rec.get("dieselEvapL",       0)
            m["dieselEvapLkr"]      += rec.get("dieselEvapLkr",     0)
            m["superDieselEvapL"]   += rec.get("superDieselEvapL",  0)
            m["superDieselEvapLkr"] += rec.get("superDieselEvapLkr",0)
            m["days"]               += 1

            if rec.get("type") == "actual":
                m["actualDays"] += 1
            else:
                m["predictedDays"] += 1

        except Exception:
            continue

    # Round all values
    result = {}
    for month_key, data in monthly.items():
        result[month_key] = {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in data.items()}
        result[month_key]["month"] = month_key
        result[month_key]["generatedAt"] = datetime.now().isoformat()

    return result


# ─── STEP 4: BUILD OVERALL SUMMARY ───────────────────────────────────────────

def build_overall_summary(all_records: list) -> dict:
    """
    Build last 7 days and last 30 days summary.
    Stored at fuelEvaporation/--summary--
    """
    if not all_records:
        return {}

    # Sort by date
    sorted_recs = sorted(all_records, key=lambda x: x["date"])
    last_date   = date.fromisoformat(sorted_recs[-1]["date"])

    cutoff_7d   = last_date - timedelta(days=6)   # last 7 days inclusive
    cutoff_30d  = last_date - timedelta(days=29)  # last 30 days inclusive

    def sum_range(cutoff):
        total_L   = 0.0
        total_lkr = 0.0
        per_fuel  = {f: {"L": 0.0, "lkr": 0.0} for f in FUEL_TYPES}
        days      = 0

        for rec in sorted_recs:
            try:
                d = date.fromisoformat(rec["date"])
                if d < cutoff:
                    continue
                total_L   += rec.get("totalEvapL",   0)
                total_lkr += rec.get("totalEvapLkr", 0)
                for fuel, fk in FIREBASE_KEYS.items():
                    per_fuel[fuel]["L"]   += rec.get(f"{fk}EvapL",   0)
                    per_fuel[fuel]["lkr"] += rec.get(f"{fk}EvapLkr", 0)
                days += 1
            except Exception:
                continue

        return {
            "totalEvapL":         round(total_L,   3),
            "totalEvapLkr":       round(total_lkr, 2),
            "petrolEvapL":        round(per_fuel["petrol"]["L"],       3),
            "petrolEvapLkr":      round(per_fuel["petrol"]["lkr"],     2),
            "superPetrolEvapL":   round(per_fuel["super_petrol"]["L"], 3),
            "superPetrolEvapLkr": round(per_fuel["super_petrol"]["lkr"],2),
            "dieselEvapL":        round(per_fuel["diesel"]["L"],       3),
            "dieselEvapLkr":      round(per_fuel["diesel"]["lkr"],     2),
            "superDieselEvapL":   round(per_fuel["super_diesel"]["L"], 3),
            "superDieselEvapLkr": round(per_fuel["super_diesel"]["lkr"],2),
            "days": days,
        }

    summary_7d  = sum_range(cutoff_7d)
    summary_30d = sum_range(cutoff_30d)

    return {
        "generatedAt":       datetime.now().isoformat(),
        "dataAsOf":          str(last_date),
        "last7Days":         summary_7d,
        "last30Days":        summary_30d,
        "totalRecords":      len(sorted_recs),
        "earliestDate":      sorted_recs[0]["date"],
        "latestDate":        sorted_recs[-1]["date"],
    }


# ─── STORE TO FIREBASE ────────────────────────────────────────────────────────

def store_all_to_firebase(
    daily_records:    list,
    monthly_summaries: dict,
    overall_summary:   dict,
):
    """
    Store everything to Firebase in batches.
    """
    init_firebase()
    from firebase_admin import firestore
    db = firestore.client()

    # ── Daily records ─────────────────────────────────────────────────────────
    print(f"\n  Storing {len(daily_records)} daily records...")
    batch       = db.batch()
    batch_count = 0
    MAX_BATCH   = 400
    stored      = 0

    for rec in daily_records:
        ref = db.collection("fuelEvaporation").document(rec["date"])
        batch.set(ref, rec)
        batch_count += 1
        stored      += 1

        if batch_count >= MAX_BATCH:
            batch.commit()
            print(f"    Committed batch of {batch_count}...")
            batch       = db.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()

    print(f"  ✅ Stored {stored} daily records to fuelEvaporation")

    # ── Monthly summaries ─────────────────────────────────────────────────────
    print(f"\n  Storing {len(monthly_summaries)} monthly summaries...")
    for month_key, data in monthly_summaries.items():
        db.collection("fuelEvaporationSummary").document(month_key).set(data)
        print(f"    {month_key}: {data['days']} days  "
              f"total={data['totalEvapL']:.2f}L  "
              f"LKR={data['totalEvapLkr']:.0f}")

    print(f"  ✅ Stored {len(monthly_summaries)} monthly summaries "
          f"to fuelEvaporationSummary")

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n  Storing overall summary...")
    db.collection("fuelEvaporation").document("--summary--").set(overall_summary)

    s7  = overall_summary["last7Days"]
    s30 = overall_summary["last30Days"]
    print(f"  Last 7 days:  {s7['totalEvapL']:.3f}L  LKR {s7['totalEvapLkr']:.0f}")
    print(f"  Last 30 days: {s30['totalEvapL']:.3f}L  LKR {s30['totalEvapLkr']:.0f}")
    print(f"  ✅ Overall summary stored to fuelEvaporation/--summary--")


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_evaporation_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION PIPELINE — EMERALD LANKA")
    print("="*60)
    started = datetime.now()

    from scripts.predict_evap import load_evap_models, FALLBACK_MONTHLY_RATES
    from scripts.fetch_weather import fetch_historical, _add_derived_features

    # ── 1. Get fuel prices from Firebase ──────────────────────────────────────
    print("\n[1/7] Reading fuel prices from Firebase fuelTanks...")
    fuel_prices = get_fuel_prices()

    # ── 2. Load trained ML models ─────────────────────────────────────────────
    print("\n[2/7] Loading evaporation ML models...")
    evap_models = load_evap_models("models/evaporation")
    print(f"  Loaded {len(evap_models)} models")

    # ── 3. Load evaporation CSV (actual data) ─────────────────────────────────
    print("\n[3/7] Loading fuel_sales_evaporation.csv...")
    evap_csv_path = "data/fuel_sales_evaporation.csv"
    evap_df       = pd.read_csv(evap_csv_path)
    evap_df["date"] = pd.to_datetime(evap_df["date"])
    csv_start     = evap_df["date"].min().date()
    csv_end       = evap_df["date"].max().date()
    print(f"  CSV range: {csv_start} → {csv_end} ({len(evap_df)} rows)")

    # ── 4. Fetch sales from Firebase ──────────────────────────────────────────
    print("\n[4/7] Fetching sales from Firebase fuelSaleHistory...")
    sales_df    = fetch_sales_from_firebase()
    sales_end   = sales_df["date"].max().date()

    # Gap = dates after CSV ends where Firebase has sales
    gap_days = (sales_end - csv_end).days
    print(f"  CSV ends:      {csv_end}")
    print(f"  Firebase ends: {sales_end}")
    print(f"  Gap to predict: {gap_days} days "
          f"({csv_end + timedelta(days=1)} → {sales_end})")

    # ── 5. Fetch all weather needed ───────────────────────────────────────────
    print(f"\n[5/7] Fetching weather ({csv_start} → {sales_end})...")
    try:
        wx_df = fetch_historical(str(csv_start), str(sales_end))
        wx_df.to_csv("data/weather_full.csv", index=False)
        print(f"  Got {len(wx_df)} days of weather")
    except Exception as e:
        print(f"  ⚠️  Weather fetch failed: {e}")
        try:
            wx_df = pd.read_csv("data/weather_data.csv")
            wx_df["date"] = pd.to_datetime(wx_df["date"])
            print(f"  Using cached: {len(wx_df)} days")
        except Exception:
            wx_df = pd.DataFrame()

    weather_lookup = {}
    if not wx_df.empty:
        weather_lookup = {
            row["date"].strftime("%Y-%m-%d"): row
            for _, row in wx_df.iterrows()
        }
    print(f"  Weather lookup: {len(weather_lookup)} dates")

    # ── 6. Build all records ──────────────────────────────────────────────────
    print("\n[6/7] Building evaporation records...")

    # Phase 1: Actual records from CSV
    print("\n  Phase 1: ACTUAL records (from fuel_sales_evaporation.csv)")
    actual_records = build_actual_evap_records(
        evap_csv_path  = evap_csv_path,
        fuel_prices    = fuel_prices,
        wx_lookup      = weather_lookup,
        evap_models    = evap_models,
        fallback_rates = FALLBACK_MONTHLY_RATES,
    )

    # Phase 2: Predicted records for gap period
    print("\n  Phase 2: PREDICTED records (ML model + Firebase sales)")
    predicted_records = build_predicted_evap_records(
        sales_df       = sales_df,
        csv_end_date   = csv_end,
        fuel_prices    = fuel_prices,
        wx_lookup      = weather_lookup,
        evap_models    = evap_models,
        fallback_rates = FALLBACK_MONTHLY_RATES,
    )

    # Combine all
    all_records = actual_records + predicted_records
    print(f"\n  Total records: {len(all_records)} "
          f"({len(actual_records)} actual + {len(predicted_records)} predicted)")

    # Phase 3: Monthly summaries
    print("\n  Phase 3: Building monthly summaries...")
    monthly_summaries = build_monthly_summaries(all_records, fuel_prices)
    for m, data in sorted(monthly_summaries.items()):
        print(f"    {m}: {data['days']} days  "
              f"{data['totalEvapL']:.2f}L  LKR {data['totalEvapLkr']:.0f}  "
              f"({data['actualDays']} actual + {data['predictedDays']} predicted)")

    # Phase 4: Overall summary
    print("\n  Phase 4: Building overall summary...")
    overall_summary = build_overall_summary(all_records)

    # ── 7. Store everything to Firebase ──────────────────────────────────────
    print("\n[7/7] Storing to Firebase...")
    store_all_to_firebase(
        daily_records     = all_records,
        monthly_summaries = monthly_summaries,
        overall_summary   = overall_summary,
    )

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed}s")
    print(f"{'='*60}")
    print(f"\n  Firebase collections updated:")
    print(f"    fuelEvaporation          → {len(all_records)} daily records")
    print(f"    fuelEvaporation/--summary-- → last 7d + 30d totals")
    print(f"    fuelEvaporationSummary   → {len(monthly_summaries)} monthly summaries")
    print(f"\n  Date coverage:")
    print(f"    Actual (from CSV):    {csv_start} → {csv_end}")
    if predicted_records:
        pred_start = date.fromisoformat(predicted_records[0]["date"])
        pred_end   = date.fromisoformat(predicted_records[-1]["date"])
        print(f"    Predicted (ML model): {pred_start} → {pred_end}")


if __name__ == "__main__":
    run_evaporation_pipeline()