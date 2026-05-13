"""
FASTAPI BACKEND — EMERALD LANKA EVAPORATION PREDICTOR
======================================================
Standalone evaporation predictor — no dependency on sales project.

Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                          → welcome
  GET  /health                    → server status
  GET  /predict/evaporation       → tomorrow evaporation (ML + weather)
  GET  /predict/evaporation/7days → 7-day evaporation forecast
  GET  /predict/summary           → tomorrow + 7days combined (use in Flutter)
  POST /retrain                   → retrain models + refresh weather
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
# FastAPI cannot serialize numpy types (numpy.int64, numpy.float64) directly.
# This function converts everything to plain Python types before returning.

def clean_json(obj):
    """Recursively convert numpy types to plain Python for JSON serialization."""
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
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return obj


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emerald Lanka Evaporation Predictor",
    description="ML-based fuel evaporation forecasting with real weather data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── STARTUP ──────────────────────────────────────────────────────────────────

print("\n🚀 Starting Evaporation Predictor API...")

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
    print(f"   ⚠️  Could not load evaporation history: {e}")

RETRAIN_STATUS = {"running": False, "last_run": None}

print(f"   ✅ API ready! {len(EVAP_MODELS)} evap models loaded\n")

EVAP_ACCURACY = {
    "petrol":       {"holdout_mape": 3.27,  "cv_mape": 9.82,  "reliability": "high"},
    "super_petrol": {"holdout_mape": 13.17, "cv_mape": 20.47, "reliability": "medium"},
    "diesel":       {"holdout_mape": 4.83,  "cv_mape": 10.66, "reliability": "high"},
    "super_diesel": {"holdout_mape": 30.86, "cv_mape": 19.63, "reliability": "medium"},
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_average_sales() -> dict:
    """Average daily sales per fuel from evaporation CSV history."""
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
    """
    Build simple sales estimates using recent rolling average.
    The evaporation model needs sales as an input feature.
    We use the average from the last 30 non-zero days as a stable estimate.
    """
    avg      = get_average_sales()
    tomorrow = LATEST_DATE + timedelta(days=1)

    result = {}
    for fuel in ["petrol", "super_petrol", "diesel", "super_diesel"]:
        result[f"{fuel}_sales"] = [
            {
                "date":             str(tomorrow + timedelta(days=i)),
                "predicted_litres": round(float(avg[fuel]), 2)
            }
            for i in range(n_days)
        ]
    return result


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":      "Emerald Lanka Evaporation Predictor v1.0",
        "station":  "Emerald Lanka Filling Station, Hettipola",
        "model":    "XGBoost + Open-Meteo weather features",
        "docs":     "/docs",
        "endpoints": [
            "GET  /health",
            "GET  /predict/evaporation",
            "GET  /predict/evaporation/7days",
            "GET  /predict/summary",
            "POST /retrain",
        ]
    }


@app.get("/health")
def health():
    return clean_json({
        "status":           "healthy",
        "evap_models":      list(EVAP_MODELS.keys()),
        "model_count":      len(EVAP_MODELS),
        "data_latest_date": str(LATEST_DATE),
        "retrain_running":  RETRAIN_STATUS["running"],
        "last_retrain":     RETRAIN_STATUS["last_run"],
        "timestamp":        datetime.now().isoformat(),
    })


@app.get("/predict/evaporation")
def predict_tomorrow():
    """
    Tomorrow's fuel evaporation prediction using ML model + weather forecast.

    Model accuracy (holdout test on last 20 days):
      Petrol       : 3.27%  MAPE ← very accurate
      Diesel       : 4.83%  MAPE ← very accurate
      Super Petrol : 13.17% MAPE ← good
      Super Diesel : 30.86% MAPE ← fair (small volume, more variance)

    Top features learned by model:
      is_dry_season, is_monsoon, et0_mm, vpd_kpa, sales_L
    """
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
            "model_type":      "ml_xgboost_weather" if EVAP_MODELS else "formula_fallback",
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
    """7-day evaporation forecast using ML model + real weather forecast per day."""
    try:
        start_date  = LATEST_DATE + timedelta(days=1)
        sales_preds = build_sales_dict(n_days=7)

        evap_7day = predict_evaporation_full(
            sales_predictions  = sales_preds,
            start_date         = start_date,
            n_days             = 7,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )

        return clean_json({
            "generated_at":           datetime.now().isoformat(),
            "model_type":             "ml_xgboost_weather" if EVAP_MODELS else "formula_fallback",
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
    """
    ★ RECOMMENDED FOR FLUTTER ★
    Tomorrow + 7-day evaporation in one API call.
    """
    try:
        tomorrow    = LATEST_DATE + timedelta(days=1)
        sales_preds = build_sales_dict(n_days=7)

        evap_tmrw = predict_evaporation_tomorrow(
            sales_predictions  = sales_preds,
            tomorrow_date      = tomorrow,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )
        evap_7day = predict_evaporation_full(
            sales_predictions  = sales_preds,
            start_date         = tomorrow,
            n_days             = 7,
            evap_models        = EVAP_MODELS,
            historical_evap_df = EVAP_HISTORY_DF,
        )

        return clean_json({
            "generated_at":      datetime.now().isoformat(),
            "data_as_of":        str(LATEST_DATE),
            "model_type":        "ml_xgboost_weather" if EVAP_MODELS else "formula_fallback",
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


# ─── RETRAIN ─────────────────────────────────────────────────────────────────

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Retrain evaporation models + refresh weather data from Open-Meteo."""
    if RETRAIN_STATUS["running"]:
        return {"message": "Retrain already running.", "started_at": RETRAIN_STATUS["last_run"]}
    RETRAIN_STATUS["running"]  = True
    RETRAIN_STATUS["last_run"] = datetime.now().isoformat()
    background_tasks.add_task(_run_retrain)
    return {"message": "✅ Retrain started. Check /health in 2-3 minutes.",
            "started_at": RETRAIN_STATUS["last_run"]}


def _run_retrain():
    global EVAP_MODELS
    try:
        print("\n🔄 Evaporation retrain started...")
        from scripts.retrain_evap import full_evap_retrain_pipeline
        full_evap_retrain_pipeline()
        EVAP_MODELS = load_evap_models("models/evaporation")
        RETRAIN_STATUS["running"] = False
        print("✅ Evaporation retrain complete!\n")
    except Exception as e:
        RETRAIN_STATUS["running"] = False
        print(f"❌ Evaporation retrain failed: {e}\n")