"""
FASTAPI BACKEND — EMERALD LANKA EVAPORATION PREDICTOR v5.0
============================================================
Key fix: LATEST_DATE now comes from Firebase fuelSaleHistory
         so the UI always shows predictions from the most recent
         sales date, not from the old CSV end date.

Run: uvicorn api.main:app --host 0.0.0.0 --port $PORT
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
    elif isinstance(obj, (np.integer,)):  return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, (np.bool_,)):    return bool(obj)
    elif isinstance(obj, (np.ndarray,)):  return obj.tolist()
    else: return obj


# ─── APP ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emerald Lanka Evaporation Predictor v5.0",
    description="Real-time evaporation with Firebase-based latest date",
    version="5.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── FIREBASE INIT ────────────────────────────────────────────────────────────

_firebase_ready = False

def init_firebase():
    global _firebase_ready
    if _firebase_ready:
        return True
    try:
        import firebase_admin
        from firebase_admin import credentials
        if firebase_admin._apps:
            _firebase_ready = True
            return True
        env = os.environ.get("FIREBASE_CREDENTIALS")
        if env:
            cred = credentials.Certificate(json.loads(env))
        else:
            key = os.path.join("firebase", "serviceAccountKey.json")
            if not os.path.exists(key):
                return False
            cred = credentials.Certificate(key)
        firebase_admin.initialize_app(cred)
        _firebase_ready = True
        return True
    except Exception as e:
        print(f"   ⚠️  Firebase init failed: {e}")
        return False


def get_latest_date_from_firebase() -> date:
    """
    Read the latest date from Firebase fuelSaleHistory.
    This is the true source of truth for LATEST_DATE.
    Falls back to fuelEvaporation, then local CSV.
    """
    try:
        if not init_firebase():
            raise Exception("Firebase not available")

        from firebase_admin import firestore
        db = firestore.client()

        # Check fuelSaleHistory for latest sales date
        docs  = db.collection("fuelSaleHistory").stream()
        dates = []
        for doc in docs:
            try:
                dates.append(date.fromisoformat(doc.id))
            except Exception:
                pass

        if dates:
            latest = max(dates)
            print(f"   Latest date from Firebase fuelSaleHistory: {latest}")
            return latest

    except Exception as e:
        print(f"   ⚠️  Firebase date read failed: {e}")

    # Fallback to local CSV
    try:
        evap_df = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_df["date"] = pd.to_datetime(evap_df["date"])
        latest = evap_df["date"].max().date()
        print(f"   Using CSV latest date: {latest}")
        return latest
    except Exception:
        pass

    # Last resort: yesterday
    return date.today() - timedelta(days=1)


# ─── STARTUP ─────────────────────────────────────────────────────────────────

print("\n🚀 Starting Evaporation Predictor API v5.0...")

print("   Loading evaporation models...")
EVAP_MODELS = load_evap_models("models/evaporation")

print("   Loading evaporation history (for lag features)...")
EVAP_HISTORY_DF = None
try:
    EVAP_HISTORY_DF = pd.read_csv("data/fuel_sales_evaporation.csv")
    EVAP_HISTORY_DF["date"] = pd.to_datetime(EVAP_HISTORY_DF["date"])
    print(f"   Evap CSV: {len(EVAP_HISTORY_DF)} rows")
except Exception as e:
    print(f"   ⚠️  Evap history not found: {e}")

# ── KEY FIX: Read LATEST_DATE from Firebase ───────────────────────────────────
print("   Reading latest date from Firebase fuelSaleHistory...")
LATEST_DATE = get_latest_date_from_firebase()
print(f"   ✅ LATEST_DATE = {LATEST_DATE}")

RETRAIN_STATUS = {"running": False, "last_run": None, "last_result": None}

print(f"   ✅ API ready! {len(EVAP_MODELS)} models | "
      f"Predictions from: {LATEST_DATE + timedelta(days=1)}\n")

EVAP_ACCURACY = {
    "petrol":       {"holdout_mape": 3.27,  "cv_mape": 9.82,  "reliability": "high"},
    "super_petrol": {"holdout_mape": 13.17, "cv_mape": 20.47, "reliability": "medium"},
    "diesel":       {"holdout_mape": 4.83,  "cv_mape": 10.66, "reliability": "high"},
    "super_diesel": {"holdout_mape": 30.86, "cv_mape": 19.63, "reliability": "medium"},
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_avg_sales() -> dict:
    """Average sales from CSV history for use in predictions."""
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


def get_weather_for_date(date_str: str):
    """Get weather row for a specific date from cached files."""
    for wx_file in ["data/weather_deploy.csv",
                    "data/weather_full.csv",
                    "data/weather_data.csv"]:
        try:
            wx_df = pd.read_csv(wx_file)
            wx_df["date"] = pd.to_datetime(wx_df["date"])
            match = wx_df[wx_df["date"].dt.strftime("%Y-%m-%d") == date_str]
            if not match.empty:
                return match.iloc[0]
        except Exception:
            continue

    # Try live fetch
    try:
        from scripts.fetch_weather import fetch_historical
        wx_df = fetch_historical(date_str, date_str)
        if not wx_df.empty:
            try:
                existing = pd.read_csv("data/weather_deploy.csv")
                existing["date"] = pd.to_datetime(existing["date"])
                combined = pd.concat([existing, wx_df]).drop_duplicates(
                    subset=["date"], keep="last")
                combined.to_csv("data/weather_deploy.csv", index=False)
            except Exception:
                wx_df.to_csv("data/weather_deploy.csv", index=False)
            return wx_df.iloc[0]
    except Exception:
        pass

    return None


# ─── REQUEST SCHEMA ───────────────────────────────────────────────────────────

class EvaporationUpdateRequest(BaseModel):
    date:         str
    petrol:       Optional[float] = None
    super_petrol: Optional[float] = None
    diesel:       Optional[float] = None
    super_diesel: Optional[float] = None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":         "Emerald Lanka Evaporation Predictor v5.0",
        "station":     "Emerald Lanka Filling Station, Hettipola",
        "model":       "XGBoost + Open-Meteo weather",
        "latest_date": str(LATEST_DATE),
        "predicting_from": str(LATEST_DATE + timedelta(days=1)),
        "docs":        "/docs",
    }


@app.get("/health")
def health():
    return clean_json({
        "status":           "healthy",
        "evap_models":      list(EVAP_MODELS.keys()),
        "latest_date":      str(LATEST_DATE),
        "predicting_from":  str(LATEST_DATE + timedelta(days=1)),
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
            "data_as_of":      str(LATEST_DATE),
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
            "data_as_of":             str(LATEST_DATE),
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


# ─── REAL-TIME UPDATE ─────────────────────────────────────────────────────────

@app.post("/evaporation/update")
def update_evaporation(req: EvaporationUpdateRequest):
    """Called by Flutter after every sale — updates fuelEvaporation for that date."""
    try:
        target_date = date.fromisoformat(req.date)
        wx_row      = get_weather_for_date(req.date)

        FUEL_TYPES_MAP = ["petrol","super_petrol","diesel","super_diesel"]
        FIREBASE_KEYS  = {"petrol":"petrol","super_petrol":"superPetrol",
                          "diesel":"diesel","super_diesel":"superDiesel"}

        sales = {
            "petrol":       req.petrol,
            "super_petrol": req.super_petrol,
            "diesel":       req.diesel,
            "super_diesel": req.super_diesel,
        }

        doc_data = {
            "date":        req.date,
            "generatedAt": datetime.now().isoformat(),
            "modelType":   "ml_xgboost_weather",
            "type":        "predicted",
        }
        total_L = 0.0; total_lkr = 0.0

        for fuel in FUEL_TYPES_MAP:
            fk      = FIREBASE_KEYS[fuel]
            sales_L = sales.get(fuel)

            if sales_L is None or sales_L <= 0:
                doc_data[f"{fk}EvapL"]   = 0.0
                doc_data[f"{fk}EvapLkr"] = 0.0
                doc_data[f"{fk}SalesL"]  = 0.0
                continue

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

        if wx_row is not None:
            doc_data["tempMaxC"]    = float(wx_row.get("temp_max_c",   0) or 0)
            doc_data["precipMm"]    = float(wx_row.get("precip_mm",    0) or 0)
            doc_data["humidityMax"] = float(wx_row.get("humidity_max", 0) or 0)
            doc_data["et0Mm"]       = float(wx_row.get("et0_mm",       0) or 0)

        init_firebase()
        from firebase_admin import firestore as fs
        fs.client().collection("fuelEvaporation").document(req.date).set(doc_data)

        return clean_json({
            "status":       "updated",
            "date":         req.date,
            "totalEvapL":   doc_data["totalEvapL"],
            "totalEvapLkr": doc_data["totalEvapLkr"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── HISTORY ─────────────────────────────────────────────────────────────────

@app.get("/evaporation/history")
def get_history(days: int = 30):
    try:
        init_firebase()
        from firebase_admin import firestore as fs

        days   = min(days, 365)
        cutoff = date.today() - timedelta(days=days)
        db     = fs.client()
        docs   = db.collection("fuelEvaporation").stream()

        records = []
        for doc in docs:
            if doc.id == "--summary--":
                continue
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
                        "type":             data.get("type", "actual"),
                    })
            except Exception:
                continue

        records.sort(key=lambda x: x["date"])
        total_l   = sum(r["totalEvapL"]   for r in records)
        total_lkr = sum(r["totalEvapLkr"] for r in records)

        return clean_json({
            "generated_at":     datetime.now().isoformat(),
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
    Full retrain:
      1. Merges CSV + Firebase sales for training (bigger dataset)
      2. Retrains all 4 models
      3. Runs evaporation pipeline (stores all data to Firebase)
      4. Updates LATEST_DATE from Firebase
    """
    if RETRAIN_STATUS["running"]:
        return {"message": "Already retraining.", "started_at": RETRAIN_STATUS["last_run"]}
    RETRAIN_STATUS["running"]  = True
    RETRAIN_STATUS["last_run"] = datetime.now().isoformat()
    background_tasks.add_task(_run_retrain)
    return {
        "message":    "✅ Retrain started. Models will train on CSV + Firebase data. "
                      "Check /health in 3-5 minutes.",
        "started_at": RETRAIN_STATUS["last_run"],
    }


def _run_retrain():
    global EVAP_MODELS, LATEST_DATE
    try:
        print("\n🔄 Retrain started (CSV + Firebase)...")
        from scripts.retrain_evap import full_evap_retrain_pipeline
        full_evap_retrain_pipeline()

        # Reload models
        EVAP_MODELS = load_evap_models("models/evaporation")

        # Update latest date from Firebase
        LATEST_DATE = get_latest_date_from_firebase()
        print(f"✅ LATEST_DATE updated to: {LATEST_DATE}")

        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = "success"
        print("✅ Retrain complete!\n")
    except Exception as e:
        RETRAIN_STATUS["running"]     = False
        RETRAIN_STATUS["last_result"] = f"failed: {e}"
        print(f"❌ Retrain failed: {e}\n")