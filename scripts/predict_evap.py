"""
PREDICT_EVAP.PY — EMERALD LANKA
=================================
Generates evaporation predictions using the trained ML models
and real weather forecast data from Open-Meteo.

How it works:
  1. Load trained evaporation model for each fuel
  2. Get weather forecast for tomorrow / next 7 days
  3. Get sales predictions for same dates
  4. Build feature row: sales + weather + date features
  5. model.predict() → evaporation in litres
  6. Multiply by fuel price → LKR cost

This is better than the formula predictor because:
  - A 38°C day has ~40% more evaporation than a 26°C day
  - Rain cools tanks and reduces evaporation
  - Wind carries vapour away faster
  The formula treats all days in a month identically — the ML model does not.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FUEL_TYPES = ["petrol", "super_petrol", "diesel", "super_diesel"]

FUEL_PRICES_LKR = {
    "petrol":       317.0,
    "super_petrol": 377.0,
    "diesel":       303.0,
    "super_diesel": 365.0,
}

MODELS_DIR = "models/evaporation"

# Fallback monthly rates if weather API is unavailable
FALLBACK_MONTHLY_RATES = {
    "petrol":       {1:0.00101336,2:0.00103530,3:0.00096423,4:0.00085518,
                     5:0.00077217,6:0.00068872,7:0.00066873,8:0.00066608,
                     9:0.00074189,10:0.00077849,11:0.00085482,12:0.00092837},
    "super_petrol": {1:0.00115806,2:0.00118337,3:0.00110205,4:0.00097722,
                     5:0.00088283,6:0.00078707,7:0.00076451,8:0.00076126,
                     9:0.00084814,10:0.00088903,11:0.00097695,12:0.00105800},
    "diesel":       {1:0.00009048,2:0.00009243,3:0.00008609,4:0.00007636,
                     5:0.00006894,6:0.00006149,7:0.00005971,8:0.00005947,
                     9:0.00006624,10:0.00006951,11:0.00007632,12:0.00008289},
    "super_diesel": {1:0.00007963,2:0.00008138,3:0.00007580,4:0.00006702,
                     5:0.00006105,6:0.00005405,7:0.00005275,8:0.00005223,
                     9:0.00005833,10:0.00006127,11:0.00006720,12:0.00007288},
}


# ─── LOAD EVAP MODELS ────────────────────────────────────────────────────────

def load_evap_models(models_dir: str = MODELS_DIR) -> dict:
    """Load all 4 trained evaporation models."""
    models = {}
    for fuel in FUEL_TYPES:
        model_path = f"{models_dir}/{fuel}_evap_model.pkl"
        feat_path  = f"{models_dir}/{fuel}_evap_features.pkl"
        if os.path.exists(model_path) and os.path.exists(feat_path):
            models[fuel] = {
                "model":    joblib.load(model_path),
                "features": joblib.load(feat_path),
            }
            print(f"  Loaded evap model: {fuel}")
        else:
            print(f"  Evap model not found: {fuel} — will use formula fallback")
    return models


# ─── LOAD WEATHER FORECAST ───────────────────────────────────────────────────

def get_weather_for_dates(target_dates: list) -> pd.DataFrame:
    """
    Get weather data for the prediction dates.
    First tries the forecast CSV, then fetches live, then uses historical average.
    """
    # Try forecast file first (fastest, no API call)
    forecast_path = "data/weather_forecast.csv"
    if os.path.exists(forecast_path):
        fcast = pd.read_csv(forecast_path)
        fcast["date"] = pd.to_datetime(fcast["date"])
        available = set(fcast["date"].dt.strftime("%Y-%m-%d"))
        needed    = {str(d) for d in target_dates}
        if needed.issubset(available):
            mask = fcast["date"].isin(pd.to_datetime(target_dates))
            return fcast[mask].reset_index(drop=True)

    # Try live forecast fetch
    try:
        from scripts.fetch_weather import fetch_forecast, _add_derived_features
        print("  Fetching live weather forecast...")
        fcast = fetch_forecast(days_ahead=8)
        fcast = _add_derived_features(fcast)
        fcast.to_csv(forecast_path, index=False)
        mask = fcast["date"].isin(pd.to_datetime(target_dates))
        if mask.any():
            return fcast[mask].reset_index(drop=True)
    except Exception as e:
        print(f"  Live forecast failed: {e}")

    # Fallback: use historical seasonal averages
    print("  Using seasonal weather averages as fallback")
    return _build_seasonal_fallback(target_dates)


def _build_seasonal_fallback(target_dates: list) -> pd.DataFrame:
    """
    If weather API is unreachable, use monthly average weather for Hettipola.
    Based on Sri Lanka climate data.
    """
    # Monthly averages for Hettipola / Kurunegala District
    monthly_avg = {
        1:  {"temp_max_c":31.2,"temp_min_c":22.1,"temp_mean_c":26.5,"precip_mm":2.1,"humidity_max":85,"humidity_min":62,"wind_max_kmh":14,"wind_mean_kmh":8,"et0_mm":5.2,"solar_mj":20.1,"vpd_kpa":1.8},
        2:  {"temp_max_c":32.8,"temp_min_c":23.0,"temp_mean_c":27.6,"precip_mm":1.4,"humidity_max":82,"humidity_min":58,"wind_max_kmh":15,"wind_mean_kmh":9,"et0_mm":5.8,"solar_mj":22.3,"vpd_kpa":2.1},
        3:  {"temp_max_c":34.1,"temp_min_c":24.2,"temp_mean_c":28.8,"precip_mm":3.2,"humidity_max":84,"humidity_min":60,"wind_max_kmh":16,"wind_mean_kmh":9,"et0_mm":6.1,"solar_mj":23.4,"vpd_kpa":2.3},
        4:  {"temp_max_c":34.5,"temp_min_c":25.1,"temp_mean_c":29.4,"precip_mm":8.5,"humidity_max":88,"humidity_min":65,"wind_max_kmh":17,"wind_mean_kmh":10,"et0_mm":6.0,"solar_mj":22.8,"vpd_kpa":2.2},
        5:  {"temp_max_c":32.4,"temp_min_c":24.8,"temp_mean_c":28.2,"precip_mm":18.3,"humidity_max":90,"humidity_min":72,"wind_max_kmh":18,"wind_mean_kmh":11,"et0_mm":5.4,"solar_mj":19.8,"vpd_kpa":1.7},
        6:  {"temp_max_c":30.2,"temp_min_c":24.1,"temp_mean_c":26.8,"precip_mm":22.1,"humidity_max":92,"humidity_min":75,"wind_max_kmh":19,"wind_mean_kmh":12,"et0_mm":4.8,"solar_mj":17.2,"vpd_kpa":1.4},
        7:  {"temp_max_c":29.8,"temp_min_c":23.6,"temp_mean_c":26.4,"precip_mm":20.4,"humidity_max":93,"humidity_min":76,"wind_max_kmh":20,"wind_mean_kmh":12,"et0_mm":4.6,"solar_mj":16.8,"vpd_kpa":1.3},
        8:  {"temp_max_c":30.1,"temp_min_c":23.8,"temp_mean_c":26.7,"precip_mm":16.2,"humidity_max":91,"humidity_min":74,"wind_max_kmh":18,"wind_mean_kmh":11,"et0_mm":4.9,"solar_mj":17.5,"vpd_kpa":1.4},
        9:  {"temp_max_c":30.8,"temp_min_c":23.9,"temp_mean_c":27.1,"precip_mm":14.8,"humidity_max":89,"humidity_min":71,"wind_max_kmh":16,"wind_mean_kmh":10,"et0_mm":5.1,"solar_mj":18.4,"vpd_kpa":1.6},
        10: {"temp_max_c":30.5,"temp_min_c":23.5,"temp_mean_c":26.8,"precip_mm":22.6,"humidity_max":91,"humidity_min":73,"wind_max_kmh":15,"wind_mean_kmh":9,"et0_mm":4.9,"solar_mj":17.8,"vpd_kpa":1.5},
        11: {"temp_max_c":29.8,"temp_min_c":22.8,"temp_mean_c":26.1,"precip_mm":28.4,"humidity_max":90,"humidity_min":70,"wind_max_kmh":14,"wind_mean_kmh":8,"et0_mm":4.7,"solar_mj":17.0,"vpd_kpa":1.4},
        12: {"temp_max_c":30.2,"temp_min_c":22.4,"temp_mean_c":25.8,"precip_mm":12.3,"humidity_max":87,"humidity_min":66,"wind_max_kmh":13,"wind_mean_kmh":8,"et0_mm":4.8,"solar_mj":18.2,"vpd_kpa":1.5},
    }

    rows = []
    for d in target_dates:
        month = d.month if isinstance(d, date) else pd.Timestamp(d).month
        row = {"date": pd.Timestamp(d), **monthly_avg[month]}
        row["temp_range_c"]   = row["temp_max_c"] - row["temp_min_c"]
        row["humidity_mean"]  = (row["humidity_max"] + row["humidity_min"]) / 2
        row["humidity_range"] = row["humidity_max"] - row["humidity_min"]
        row["is_hot_day"]     = int(row["temp_max_c"] > 33)
        row["is_very_hot"]    = int(row["temp_max_c"] > 36)
        row["is_dry_day"]     = int(row["humidity_min"] < 40)
        row["is_humid_day"]   = int(row["humidity_max"] > 85)
        row["is_rainy_day"]   = int(row["precip_mm"] > 5)
        row["is_heavy_rain"]  = int(row["precip_mm"] > 20)
        row["no_rain"]        = int(row["precip_mm"] == 0)
        row["is_windy"]       = int(row["wind_max_kmh"] > 20)
        row["evap_risk"]      = (row["temp_mean_c"]*0.4 +
                                  (100-row["humidity_mean"])*0.35 +
                                  row["wind_mean_kmh"]*0.25)
        rows.append(row)

    return pd.DataFrame(rows)


# ─── CORE PREDICTION ─────────────────────────────────────────────────────────

def predict_evaporation_ml(
    target_date: date,
    fuel: str,
    sales_litres: float,
    weather_row: pd.Series,
    model_dict: dict,
    recent_evap_vals: list,
    recent_sales_vals: list,
) -> float:
    """
    Predict evaporation for one fuel on one date using the ML model.

    Args:
        target_date:       Date to predict for
        fuel:              e.g. "petrol"
        sales_litres:      Predicted sales for that day
        weather_row:       Weather data Series for that date
        model_dict:        {"model": XGBRegressor, "features": [...]}
        recent_evap_vals:  Last 7 days of evaporation (most recent last)
        recent_sales_vals: Last 7 days of sales

    Returns:
        Predicted evaporation in litres
    """
    model    = model_dict["model"]
    features = model_dict["features"]

    td = pd.Timestamp(target_date)

    # Build feature row as a dict
    row = {
        # Weather
        "temp_max_c":      weather_row.get("temp_max_c", 31.0),
        "temp_min_c":      weather_row.get("temp_min_c", 23.0),
        "temp_mean_c":     weather_row.get("temp_mean_c", 27.0),
        "temp_range_c":    weather_row.get("temp_range_c", 8.0),
        "precip_mm":       weather_row.get("precip_mm", 0.0),
        "humidity_max":    weather_row.get("humidity_max", 85.0),
        "humidity_min":    weather_row.get("humidity_min", 65.0),
        "humidity_mean":   weather_row.get("humidity_mean", 75.0),
        "humidity_range":  weather_row.get("humidity_range", 20.0),
        "wind_max_kmh":    weather_row.get("wind_max_kmh", 15.0),
        "wind_mean_kmh":   weather_row.get("wind_mean_kmh", 9.0),
        "et0_mm":          weather_row.get("et0_mm", 5.0),
        "solar_mj":        weather_row.get("solar_mj", 19.0),
        "vpd_kpa":         weather_row.get("vpd_kpa", 1.6),
        "is_hot_day":      weather_row.get("is_hot_day", 0),
        "is_very_hot":     weather_row.get("is_very_hot", 0),
        "is_dry_day":      weather_row.get("is_dry_day", 0),
        "is_humid_day":    weather_row.get("is_humid_day", 0),
        "is_rainy_day":    weather_row.get("is_rainy_day", 0),
        "is_heavy_rain":   weather_row.get("is_heavy_rain", 0),
        "no_rain":         weather_row.get("no_rain", 1),
        "is_windy":        weather_row.get("is_windy", 0),
        "evap_risk":       weather_row.get("evap_risk", 20.0),
        "temp_max_lag1":   weather_row.get("temp_max_c", 31.0),  # same day proxy
        "et0_lag1":        weather_row.get("et0_mm", 5.0),

        # Date
        "month":       td.month,
        "day_of_week": td.dayofweek,
        "day_of_year": td.dayofyear,
        "quarter":     td.quarter,
        "month_sin":   np.sin(2 * np.pi * td.month / 12),
        "month_cos":   np.cos(2 * np.pi * td.month / 12),
        "is_monsoon":  int(td.month in [5,6,7,8,9]),
        "is_dry_season": int(td.month in [12,1,2,3,4]),

        # Sales
        "sales_L":     sales_litres,
        "sales_lag1":  recent_sales_vals[-1] if recent_sales_vals else sales_litres,
        "sales_roll7": np.mean(recent_sales_vals[-7:]) if recent_sales_vals else sales_litres,

        # Evap lags (from recent history)
        "evap_lag1":   recent_evap_vals[-1] if recent_evap_vals else sales_litres * 0.0009,
        "evap_lag2":   recent_evap_vals[-2] if len(recent_evap_vals) >= 2 else sales_litres * 0.0009,
        "evap_roll7":  np.mean(recent_evap_vals[-7:]) if recent_evap_vals else sales_litres * 0.0009,
    }

    # Build DataFrame with only the features the model expects
    X = pd.DataFrame([{f: row.get(f, 0.0) for f in features}])
    pred = float(model.predict(X)[0])
    return max(0.0, pred)


# ─── FULL PREDICTION PIPELINE ────────────────────────────────────────────────

def predict_evaporation_full(
    sales_predictions: dict,
    start_date: date,
    n_days: int = 7,
    evap_models: dict = None,
    historical_evap_df: pd.DataFrame = None,
) -> list:
    """
    Predict evaporation for n_days using ML models + weather forecast.

    Falls back to formula predictor if ML models are not available
    or weather API is unreachable.

    Args:
        sales_predictions:  dict from get_7day_prediction()
        start_date:         first prediction date
        n_days:             number of days to predict
        evap_models:        loaded evaporation models (from load_evap_models())
        historical_evap_df: historical evaporation CSV DataFrame

    Returns:
        List of n_days dicts, each with per-fuel evaporation and summary
    """
    target_dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Get weather for all prediction dates
    weather_df = get_weather_for_dates(target_dates)

    # Get recent historical evaporation for lag features
    recent_evap = {}
    if historical_evap_df is not None:
        for fuel in FUEL_TYPES:
            col = f"{fuel}_evap_L"
            if col in historical_evap_df.columns:
                recent_evap[fuel] = historical_evap_df[col].tail(30).tolist()
            else:
                recent_evap[fuel] = []
    else:
        for fuel in FUEL_TYPES:
            recent_evap[fuel] = []

    results = []
    pred_evap_history = {fuel: list(recent_evap.get(fuel, [])) for fuel in FUEL_TYPES}

    for i, target_date in enumerate(target_dates):
        day_result = {"date": str(target_date)}

        # Get weather row for this date
        wx_match = weather_df[weather_df["date"] == pd.Timestamp(target_date)]
        if wx_match.empty:
            wx_row = pd.Series(dtype=float)
        else:
            wx_row = wx_match.iloc[0]

        total_evap_L   = 0.0
        total_evap_lkr = 0.0

        for fuel in FUEL_TYPES:
            sales_key = f"{fuel}_sales"
            if fuel in ["petrol", "diesel"]:
                sales_key = f"{fuel}_sales"
            sales_list = sales_predictions.get(
                f"{fuel}_sales",
                sales_predictions.get(fuel, [])
            )
            if isinstance(sales_list, list) and len(sales_list) > i:
                s = sales_list[i]
                sales_litres = s["predicted_litres"] if isinstance(s, dict) else float(s)
            else:
                sales_litres = 2000.0

            # ML prediction
            if evap_models and fuel in evap_models:
                try:
                    evap_L = predict_evaporation_ml(
                        target_date, fuel, sales_litres, wx_row,
                        evap_models[fuel],
                        pred_evap_history[fuel],
                        [],
                    )
                    method = "ml"
                except Exception as e:
                    evap_L = sales_litres * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                    method = "formula_fallback"
            else:
                evap_L = sales_litres * FALLBACK_MONTHLY_RATES[fuel][target_date.month]
                method = "formula"

            evap_L   = round(evap_L, 5)
            evap_lkr = round(evap_L * FUEL_PRICES_LKR[fuel], 2)

            day_result[fuel] = {
                "date":             str(target_date),
                "predicted_sales_L": round(sales_litres, 2),
                "evap_litres":       evap_L,
                "evap_lkr":          evap_lkr,
                "evap_pct_of_sales": round(evap_L / sales_litres * 100, 5)
                                     if sales_litres > 0 else 0.0,
                "method":            method,
                "weather": {
                    "temp_max_c":   wx_row.get("temp_max_c", None),
                    "precip_mm":    wx_row.get("precip_mm", None),
                    "humidity_max": wx_row.get("humidity_max", None),
                    "et0_mm":       wx_row.get("et0_mm", None),
                },
            }

            total_evap_L   += evap_L
            total_evap_lkr += evap_lkr
            pred_evap_history[fuel].append(evap_L)

        day_result["summary"] = {
            "date":             str(target_date),
            "total_evap_litres": round(total_evap_L, 4),
            "total_evap_lkr":    round(total_evap_lkr, 2),
            "annual_est_litres": round(total_evap_L * 365, 1),
            "annual_est_lkr":    round(total_evap_lkr * 365, 0),
        }

        results.append(day_result)

    return results


def predict_evaporation_tomorrow(
    sales_predictions: dict,
    tomorrow_date: date,
    evap_models: dict = None,
    historical_evap_df: pd.DataFrame = None,
) -> dict:
    """Convenience wrapper — returns only tomorrow's prediction."""
    results = predict_evaporation_full(
        sales_predictions, tomorrow_date, n_days=1,
        evap_models=evap_models,
        historical_evap_df=historical_evap_df,
    )
    return results[0]