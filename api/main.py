"""
FASTAPI BACKEND — EMERALD LANKA EVAPORATION PREDICTOR v4.0
============================================================
Now includes real-time evaporation update after every sale.

Run: uvicorn api.main:app --host 0.0.0.0 --port $PORT

Endpoints:
  GET  /                          → welcome
  GET  /health                    → server status
  GET  /predict/evaporation       → tomorrow's prediction
  GET  /predict/evaporation/7days → 7-day forecast
  GET  /predict/summary           → tomorrow + 7days combined
  GET  /evaporation/history       → history from Firebase
  POST /evaporation/update        → ★ called after every sale (real-time)
  POST /retrain                   → full retrain + Firebase sync
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.predict_evap import (
    load_evap_models,
    predict_evaporation_tomorrow,
    predict_evaporation_full,
    predict_evaporation_ml,
    FUEL_PRICES_LKR,
    FALLBACK_MONTHLY_RATES,
)


# ─── JSON HELPER ─────────────────────────────────────────────────────────────

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emerald Lanka Evaporation Predictor v4.0",
    description="Real-time evaporation update after every sale",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── STARTUP ──────────────────────────────────────────────────────────────────

print("\n🚀 Starting Evaporation Predictor API v4.0...")

print("   Loading evaporation models...")
EVAP_MODELS = load_evap_models("models/evaporation")

print("   Loading evaporation history...")
EVAP_HISTORY_DF = None
LATEST_DATE     = date.today() - timedelta(days=1)

try:
    EVAP_HISTORY_DF = pd.read_csv("data/fuel_sales_evaporation.csv")
    EVAP_HISTORY_DF["date"] = pd.to_datetime(EVAP_HISTORY_DF["date"])
    LATEST_DATE = EVAP_HISTORY_DF["date"].max().date()
    print(f"   Evap history: {len(EVAP_HISTORY_DF)} rows, latest: {LATEST_DATE}")
except Exception as e:
    print(f"   ⚠️  Evap history not found: {e}")

RETRAIN_STATUS = {"running": False, "last_run": None, "last_result": None}

print(f"   ✅ API ready! {len(EVAP_MODELS)} models loaded\n")

EVAP_ACCURACY = {
    "petrol":       {"holdout_mape": 3.27,  "cv_mape": 9.82,  "reliability": "high"},
    "super_petrol": {"holdout_mape": 13.17, "cv_mape": 20.47, "reliability": "medium"},
    "diesel":       {"holdout_mape": 4.83,  "cv_mape": 10.66, "reliability": "high"},
    "super_diesel": {"holdout_mape": 30.86, "cv_mape": 19.63, "reliability": "medium"},
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_avg_sales() -> dict:
    if EVAP_HISTORY_DF is None:
        return {"petrol": 2500.0, "super_petrol": 80.0,
                "diesel": 2800.0, "super_diesel": 120.0}
    df = EVAP_HISTORY_DF
    return {
        "petrol":       float(df["petrol_sales_L"][df["petrol_sales_L"] > 0].mean()),
        "super_petrol": float(df["super_petrol_sales_L"][df["super_petrol_sales_L"] > 0].mean()),
        "diesel":       float(df["diesel_sales_L"][df["diesel_sales_L"] > 0].mean()),
        "super_diesel": float(df["super_diesel_sales_L"][df["super_diesel_sales_L"] > 0].mean()),
    }


def build_sales_dict(n_days: int = 7) -> dict:
    avg      = get_avg_sales()
    tomorrow = LATEST_DATE + timedelta(days=1)
    result   = {}
    for fuel in ["petrol", "super_petrol", "diesel", "super_diesel"]:
        result[f"{fuel}_sales"] = [
            {"date":             str(tomorrow + timedelta(days=i)),
             "predicted_litres": round(float(avg[fuel]), 2)}
            for i in range(n_days)
        ]
    return result


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
        cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)


# ─── REQUEST SCHEMA ───────────────────────────────────────────────────────────

class EvaporationUpdateRequest(BaseModel):
    """
    Called by Flutter SalesService after every sale.
    Only include fuels that have actual sales data.
    Missing fuels (None) → stored as 0 in Firebase.
    """
    date:         str              # "2026-05-13"
    petrol:       Optional[float]  # 92PetrolSale total for the day, None if not recorded
    super_petrol: Optional[float]  # 95PetrolSale total
    diesel:       Optional[float]  # dieselSale total
    super_diesel: Optional[float]  # superDieselSale total


# ─── PREDICTION ENDPOINTS ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "Emerald Lanka Evaporation Predictor v4.0",
        "station": "Emerald Lanka Filling Station, Hettipola",
        "model":   "XGBoost + Open-Meteo weather",
        "docs":    "/docs",
        "tip":     "POST /evaporation/update after every sale for real-time sync",
    }


@app.get("/health")
def health():
    return clean_json({
        "status":           "healthy",
        "evap_models":      list(EVAP_MODELS.keys()),
        "data_latest_date": str(LATEST_DATE),
        "retrain_running":  RETRAIN_STATUS["running"],
        "last_retrain":     RETRAIN_STATUS["last_run"],
        "timestamp":        datetime.now().isoformat(),
    })


@app.get("/predict/evaporation")
def predict_tomorrow():
    try:
        tomorrow    = LATEST_DATE + timedelta(days=1)
        sales_preds = build_sales_dict(n_days=1)
        evap = predict_evaporation_tomorrow(
            sales_predictions  = sales_preds,
            tomorrow_date      = tomorrow,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )
        return clean_json({
            "generated_at":    datetime.now().isoformat(),
            "prediction_for":  str(tomorrow),
            "model_type":      "ml_xgboost_weather",
            "petrol":          evap.get("petrol",       {}),
            "super_petrol":    evap.get("super_petrol", {}),
            "diesel":          evap.get("diesel",       {}),
            "super_diesel":    evap.get("super_diesel", {}),
            "summary":         evap.get("summary",      {}),
            "fuel_prices_lkr": FUEL_PRICES_LKR,
            "accuracy":        EVAP_ACCURACY,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/evaporation/7days")
def predict_7_days():
    try:
        start_date  = LATEST_DATE + timedelta(days=1)
        sales_preds = build_sales_dict(n_days=7)
        evap_7day   = predict_evaporation_full(
            sales_predictions  = sales_preds,
            start_date         = start_date,
            n_days             = 7,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )
        return clean_json({
            "generated_at":           datetime.now().isoformat(),
            "model_type":             "ml_xgboost_weather",
            "days":                   evap_7day,
            "total_7day_evap_litres": round(
                sum(d["summary"]["total_evap_litres"] for d in evap_7day), 3),
            "total_7day_evap_lkr":    round(
                sum(d["summary"]["total_evap_lkr"] for d in evap_7day), 2),
            "accuracy":               EVAP_ACCURACY,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/summary")
def predict_summary():
    try:
        tomorrow    = LATEST_DATE + timedelta(days=1)
        sales_preds = build_sales_dict(n_days=7)
        evap_tmrw   = predict_evaporation_tomorrow(
            sales_predictions  = sales_preds,
            tomorrow_date      = tomorrow,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )
        evap_7day   = predict_evaporation_full(
            sales_predictions  = sales_preds,
            start_date         = tomorrow,
            n_days             = 7,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )
        return clean_json({
            "generated_at":      datetime.now().isoformat(),
            "data_as_of":        str(LATEST_DATE),
            "model_type":        "ml_xgboost_weather",
            "tomorrow":          evap_tmrw,
            "seven_days":        evap_7day,
            "total_7day_litres": round(
                sum(d["summary"]["total_evap_litres"] for d in evap_7day), 3),
            "total_7day_lkr":    round(
                sum(d["summary"]["total_evap_lkr"] for d in evap_7day), 2),
            "fuel_prices_lkr":   FUEL_PRICES_LKR,
            "accuracy":          EVAP_ACCURACY,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── REAL-TIME UPDATE ENDPOINT ────────────────────────────────────────────────

@app.post("/evaporation/update")
def update_evaporation_for_date(req: EvaporationUpdateRequest):
    """
    ★ Called automatically by Flutter after every sale ★

    Recalculates evaporation for one specific date using
    the latest sales totals and stores to Firebase fuelEvaporation.

    - Only calculates fuels that have actual sales (not None/0)
    - Stores 0 for fuels with no sales yet
    - Overwrites existing Firebase record for that date
    - Returns in 2-3 seconds

    Example body:
    {
      "date": "2026-05-13",
      "petrol": null,
      "super_petrol": null,
      "diesel": 2049.0,
      "super_diesel": 235.0
    }
    """
    try:
        target_date = date.fromisoformat(req.date)

        # Get real weather for this date
        wx_row = _get_weather_for_date(req.date)

        # Build sales dict
        sales = {
            "petrol":       req.petrol,
            "super_petrol": req.super_petrol,
            "diesel":       req.diesel,
            "super_diesel": req.super_diesel,
        }

        FUEL_TYPES  = ["petrol", "super_petrol", "diesel", "super_diesel"]
        FIREBASE_KEYS = {
            "petrol":       "petrol",
            "super_petrol": "superPetrol",
            "diesel":       "diesel",
            "super_diesel": "superDiesel",
        }

        doc_data = {
            "date":        req.date,
            "generatedAt": datetime.now().isoformat(),
            "modelType":   "ml_xgboost_weather",
        }

        total_L   = 0.0
        total_lkr = 0.0

        for fuel in FUEL_TYPES:
            fk      = FIREBASE_KEYS[fuel]
            sales_L = sales.get(fuel)

            # Only calculate if this fuel has actual sales data
            if sales_L is None or sales_L <= 0:
                doc_data[f"{fk}EvapL"]   = 0.0
                doc_data[f"{fk}EvapLkr"] = 0.0
                doc_data[f"{fk}SalesL"]  = 0.0
                continue

            # Calculate evaporation
            if fuel in EVAP_MODELS and wx_row is not None:
                try:
                    evap_L = predict_evaporation_ml(
                        target_date       = target_date,
                        fuel              = fuel,
                        sales_litres      = float(sales_L),
                        weather_row       = wx_row,
                        model_dict        = EVAP_MODELS[fuel],
                        recent_evap_vals  = [],
                        recent_sales_vals = [],
                    )
                except Exception:
                    evap_L = float(sales_L) * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
            else:
                evap_L = float(sales_L) * FALLBACK_MONTHLY_RATES[fuel][target_date.month]

            evap_L   = round(float(max(0, evap_L)), 5)
            evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

            doc_data[f"{fk}EvapL"]   = evap_L
            doc_data[f"{fk}EvapLkr"] = evap_lkr
            doc_data[f"{fk}SalesL"]  = round(float(sales_L), 2)

            total_L   += evap_L
            total_lkr += evap_lkr

        doc_data["totalEvapL"]   = round(total_L, 5)
        doc_data["totalEvapLkr"] = round(total_lkr, 2)

        # Weather context
        if wx_row is not None:
            doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",   0) or 0)
            doc_data["precipMm"]    = float(wx_row.get("precip_mm",    0) or 0)
            doc_data["humidityMax"] = float(wx_row.get("humidity_max", 0) or 0)
            doc_data["et0Mm"]       = float(wx_row.get("et0_mm",       0) or 0)

        # Store to Firebase
        _init_firebase()
        from firebase_admin import firestore as fs
        db = fs.client()
        db.collection("fuelEvaporation").document(req.date).set(doc_data)

        return clean_json({
            "status":        "updated",
            "date":          req.date,
            "totalEvapL":    doc_data["totalEvapL"],
            "totalEvapLkr":  doc_data["totalEvapLkr"],
            "fuels_updated": [
                f for f in FUEL_TYPES
                if doc_data.get(f"{FIREBASE_KEYS[f]}EvapL", 0) > 0
            ],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_weather_for_date(date_str: str):
    """
    Get weather for a specific date.
    Checks weather_deploy.csv first, then weather_data.csv,
    then tries live fetch, then returns None (formula fallback).
    """
    import pandas as pd

    # Try deployment weather cache first
    for wx_file in ["data/weather_deploy.csv", "data/weather_data.csv"]:
        try:
            wx_df = pd.read_csv(wx_file)
            wx_df["date"] = pd.to_datetime(wx_df["date"])
            match = wx_df[wx_df["date"].dt.strftime("%Y-%m-%d") == date_str]
            if not match.empty:
                return match.iloc[0]
        except Exception:
            continue

    # Try live fetch for today's date
    try:
        from scripts.fetch_weather import fetch_historical
        wx_df = fetch_historical(date_str, date_str)
        if not wx_df.empty:
            # Cache it
            try:
                existing = pd.read_csv("data/weather_deploy.csv")
                existing["date"] = pd.to_datetime(existing["date"])
                combined = pd.concat([existing, wx_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined.to_csv("data/weather_deploy.csv", index=False)
            except Exception:
                wx_df.to_csv("data/weather_deploy.csv", index=False)
            return wx_df.iloc[0]
    except Exception:
        pass

    return None  # Will use formula fallback


# ─── HISTORY FROM FIREBASE ───────────────────────────────────────────────────

@app.get("/evaporation/history")
def get_history(days: int = 30):
    try:
        _init_firebase()
        from firebase_admin import firestore as fs

        days   = min(days, 365)
        cutoff = date.today() - timedelta(days=days)
        db     = fs.client()
        docs   = db.collection("fuelEvaporation").stream()

        records = []
        for doc in docs:
            data     = doc.to_dict()
            date_str = doc.id
            try:
                d = date.fromisoformat(date_str)
                if d >= cutoff:
                    records.append({
                        "date":             date_str,
                        "petrolEvapL":      float(data.get("petrolEvapL",      0) or 0),
                        "superPetrolEvapL": float(data.get("superPetrolEvapL", 0) or 0),
                        "dieselEvapL":      float(data.get("dieselEvapL",      0) or 0),
                        "superDieselEvapL": float(data.get("superDieselEvapL", 0) or 0),
                        "totalEvapL":       float(data.get("totalEvapL",       0) or 0),
                        "totalEvapLkr":     float(data.get("totalEvapLkr",     0) or 0),
                        "petrolSalesL":     float(data.get("petrolSalesL",     0) or 0),
                        "dieselSalesL":     float(data.get("dieselSalesL",     0) or 0),
                        "tempMaxC":         float(data.get("tempMaxC",         0) or 0),
                        "precipMm":         float(data.get("precipMm",         0) or 0),
                        "modelType":        data.get("modelType", "ml"),
                    })
            except Exception:
                continue

        records.sort(key=lambda x: x["date"])

        total_l   = sum(r["totalEvapL"]   for r in records)
        total_lkr = sum(r["totalEvapLkr"] for r in records)

        return clean_json({
            "generated_at":     datetime.now().isoformat(),
            "days_requested":   days,
            "records_found":    len(records),
            "records":          records,
            "period_total_L":   round(total_l, 3),
            "period_total_lkr": round(total_lkr, 2),
            "annual_est_lkr":   round(total_lkr / len(records) * 365, 0)
                                if records else 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── RETRAIN ─────────────────────────────────────────────────────────────────

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Full retrain + Firebase sync.
    Use this monthly or when new sales data has been added in bulk.
    """
    if RETRAIN_STATUS["running"]:
        return {"message": "Already retraining.", "started_at": RETRAIN_STATUS["last_run"]}
    RETRAIN_STATUS["running"]  = True
    RETRAIN_STATUS["last_run"] = datetime.now().isoformat()
    background_tasks.add_task(_run_retrain)
    return {
        "message":    "✅ Retrain started. Models will be retrained and all "
                      "Firebase evaporation data updated. Check /health in 3-5 minutes.",
        "started_at": RETRAIN_STATUS["last_run"],
    }


def _run_retrain():
    global EVAP_MODELS
    try:
        print("\n🔄 Retrain started...")
        from scripts.retrain_evap import full_evap_retrain_pipeline
        full_evap_retrain_pipeline()
        EVAP_MODELS = load_evap_models("models/evaporation")
        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = "success"
        print("✅ Retrain complete!\n")
    except Exception as e:
        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = f"failed: {e}"
        print(f"❌ Retrain failed: {e}\n")