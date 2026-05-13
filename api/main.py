"""
FASTAPI BACKEND — EMERALD LANKA EVAPORATION PREDICTOR v3.0
============================================================
Every retrain automatically:
  1. Fetches latest weather from Open-Meteo
  2. Retrains 4 XGBoost models
  3. Calculates evaporation for ALL sales dates in Firebase
  4. Stores/overwrites Firebase fuelEvaporation collection

Run: uvicorn api.main:app --host 0.0.0.0 --port $PORT

Endpoints:
  GET  /                          → welcome
  GET  /health                    → server status
  GET  /predict/evaporation       → tomorrow's prediction
  GET  /predict/evaporation/7days → 7-day forecast
  GET  /predict/summary           → tomorrow + 7days combined
  GET  /evaporation/history       → history from Firebase
  POST /retrain                   → retrain + auto-store to Firebase
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from scripts.predict_evap import (
    load_evap_models,
    predict_evaporation_tomorrow,
    predict_evaporation_full,
    FUEL_PRICES_LKR,
)


# ─── JSON HELPER ─────────────────────────────────────────────────────────────

def clean_json(obj):
    """Convert numpy types to plain Python for JSON serialization."""
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
    title="Emerald Lanka Evaporation Predictor v3.0",
    description=(
        "ML evaporation forecasting. "
        "Every retrain auto-stores all historical data to Firebase."
    ),
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── STARTUP ──────────────────────────────────────────────────────────────────

print("\n🚀 Starting Evaporation Predictor API v3.0...")

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

RETRAIN_STATUS = {
    "running":      False,
    "last_run":     None,
    "last_result":  None,
    "last_stored":  None,
}

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


# ─── PREDICTION ENDPOINTS ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "Emerald Lanka Evaporation Predictor v3.0",
        "station": "Emerald Lanka Filling Station, Hettipola",
        "model":   "XGBoost + Open-Meteo weather",
        "docs":    "/docs",
        "note":    "POST /retrain to retrain models and auto-update Firebase",
    }


@app.get("/health")
def health():
    return clean_json({
        "status":             "healthy",
        "evap_models":        list(EVAP_MODELS.keys()),
        "data_latest_date":   str(LATEST_DATE),
        "retrain_running":    RETRAIN_STATUS["running"],
        "last_retrain":       RETRAIN_STATUS["last_run"],
        "last_result":        RETRAIN_STATUS["last_result"],
        "last_firebase_store": RETRAIN_STATUS["last_stored"],
        "timestamp":          datetime.now().isoformat(),
    })


@app.get("/predict/evaporation")
def predict_tomorrow():
    """Tomorrow's evaporation using ML model + real weather forecast."""
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
    """7-day evaporation forecast."""
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
    """★ FLUTTER — tomorrow + 7-day in one call."""
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


# ─── HISTORY FROM FIREBASE ───────────────────────────────────────────────────

@app.get("/evaporation/history")
def get_history(days: int = 30):
    """
    Returns evaporation history from Firebase fuelEvaporation.
    Used by Flutter HISTORY tab.
    Query: ?days=30 (default 30, max 365)
    """
    try:
        from scripts.retrain_evap import _init_firebase
        _init_firebase()
        from firebase_admin import firestore

        days   = min(days, 365)
        cutoff = date.today() - timedelta(days=days)
        db     = firestore.client()
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
                        "humidityMax":      float(data.get("humidityMax",      0) or 0),
                        "et0Mm":            float(data.get("et0Mm",            0) or 0),
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


# ─── RETRAIN — AUTO STORES TO FIREBASE ───────────────────────────────────────

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Retrain evaporation models + automatically store all data to Firebase.

    What happens:
      1. Fetch latest weather from Open-Meteo
      2. Retrain all 4 XGBoost models
      3. Fetch all sales from Firebase fuelSaleHistory
      4. Calculate evaporation for every date with the NEW models
      5. Store/overwrite to Firebase fuelEvaporation
         (retrain twice → second result overwrites first)

    Takes 3-5 minutes. Check /health for status.
    """
    if RETRAIN_STATUS["running"]:
        return {
            "message":    "Retrain already running. Check /health.",
            "started_at": RETRAIN_STATUS["last_run"],
        }
    RETRAIN_STATUS["running"]  = True
    RETRAIN_STATUS["last_run"] = datetime.now().isoformat()
    background_tasks.add_task(_run_retrain)
    return {
        "message":    (
            "✅ Retrain started. Models will be retrained and all "
            "evaporation data will be stored to Firebase automatically. "
            "Check /health in 3-5 minutes."
        ),
        "started_at": RETRAIN_STATUS["last_run"],
    }


def _run_retrain():
    global EVAP_MODELS
    try:
        print("\n🔄 Retrain + Firebase store started...")
        from scripts.retrain_evap import full_evap_retrain_pipeline
        metrics = full_evap_retrain_pipeline()

        # Reload fresh models into memory
        EVAP_MODELS = load_evap_models("models/evaporation")

        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = "success"
        RETRAIN_STATUS["last_stored"] = datetime.now().isoformat()
        print("✅ Retrain + Firebase store complete!\n")

    except Exception as e:
        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = f"failed: {str(e)}"
        print(f"❌ Retrain failed: {e}\n")