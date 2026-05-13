"""
FETCH_WEATHER.PY — EMERALD LANKA
==================================
Downloads real weather data for Hettipola from Open-Meteo (free, no account).

TWO MODES:
  1. Historical  → downloads past weather to train the evaporation ML model
  2. Forecast    → downloads next 7 days for making predictions

Run on your PC:
  python scripts/fetch_weather.py

Called automatically by retrain_evap.py on Railway during retraining.
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

# ─── HETTIPOLA LOCATION ───────────────────────────────────────────────────────
LAT      = 7.6033
LON      = 80.0752
TIMEZONE = "Asia/Colombo"

# ─── VARIABLES TO FETCH ──────────────────────────────────────────────────────
# Why each one matters for evaporation:
#   temp_max        → hot days evaporate more fuel (physical law)
#   temp_mean       → average thermal energy of the day
#   precip_mm       → rain cools tanks and reduces evaporation
#   humidity_max    → low humidity = faster evaporation (vapour pressure gradient)
#   wind_max        → wind carries away vapour from tank vents = more evap
#   et0             → reference evapotranspiration = best single evaporation proxy
#                     (combines temp + humidity + wind + solar radiation)
#   solar_mj        → solar radiation heats the tank surface directly
#   vpd_kpa         → vapour pressure deficit = the direct physical driver

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "vapour_pressure_deficit_max",
]

# Clean column names mapping
COL_MAP = {
    "temperature_2m_max":          "temp_max_c",
    "temperature_2m_min":          "temp_min_c",
    "temperature_2m_mean":         "temp_mean_c",
    "precipitation_sum":           "precip_mm",
    "relative_humidity_2m_max":    "humidity_max",
    "relative_humidity_2m_min":    "humidity_min",
    "wind_speed_10m_max":          "wind_max_kmh",
    "wind_speed_10m_mean":         "wind_mean_kmh",
    "et0_fao_evapotranspiration":  "et0_mm",
    "shortwave_radiation_sum":     "solar_mj",
    "vapour_pressure_deficit_max": "vpd_kpa",
}


# ─── FETCH HISTORICAL ────────────────────────────────────────────────────────

def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical daily weather from Open-Meteo Archive API.
    Free, unlimited historical data, no API key required.
    """
    print(f"  Fetching weather: {start_date} to {end_date} ...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   TIMEZONE,
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise Exception(f"API error {r.status_code}: {r.text[:300]}")

    daily = r.json()["daily"]
    df    = _build_df(daily)
    print(f"  Got {len(df)} days  "
          f"(temp: {df['temp_max_c'].min():.0f}–{df['temp_max_c'].max():.0f}°C  "
          f"rainy days: {(df['precip_mm']>5).sum()})")
    return df


# ─── FETCH FORECAST ──────────────────────────────────────────────────────────

def fetch_forecast(days_ahead: int = 8) -> pd.DataFrame:
    """
    Fetch weather forecast from Open-Meteo Forecast API.
    Used at prediction time so the evaporation model knows
    how hot/humid/rainy tomorrow will be.
    """
    print(f"  Fetching {days_ahead}-day weather forecast ...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      LAT,
        "longitude":     LON,
        "daily":         ",".join(DAILY_VARS),
        "timezone":      TIMEZONE,
        "forecast_days": days_ahead,
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Forecast API error {r.status_code}: {r.text[:300]}")

    daily = r.json()["daily"]
    df    = _build_df(daily)
    print(f"  Tomorrow forecast: "
          f"temp={df.iloc[0]['temp_max_c']}°C  "
          f"humidity={df.iloc[0]['humidity_max']}%  "
          f"rain={df.iloc[0]['precip_mm']}mm  "
          f"ET0={df.iloc[0]['et0_mm']}")
    return df


# ─── BUILD DATAFRAME ─────────────────────────────────────────────────────────

def _build_df(daily: dict) -> pd.DataFrame:
    """Parse API response dict into clean DataFrame."""
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for raw, clean in COL_MAP.items():
        df[clean] = daily.get(raw, np.nan)
    df = _add_derived_features(df)
    df = _fill_missing(df)
    return df


# ─── DERIVED WEATHER FEATURES ────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from raw weather variables.
    These help the ML model capture non-linear physical relationships.
    """
    df = df.copy()

    # Temperature
    df["temp_range_c"]    = df["temp_max_c"]   - df["temp_min_c"]
    df["is_hot_day"]      = (df["temp_max_c"]  > 33).astype(int)  # >33°C very hot
    df["is_very_hot"]     = (df["temp_max_c"]  > 36).astype(int)  # >36°C extreme

    # Humidity
    df["humidity_mean"]   = (df["humidity_max"] + df["humidity_min"]) / 2
    df["humidity_range"]  = df["humidity_max"]  - df["humidity_min"]
    df["is_dry_day"]      = (df["humidity_min"] < 40).astype(int)  # very dry
    df["is_humid_day"]    = (df["humidity_max"] > 85).astype(int)  # very humid

    # Rain
    df["is_rainy_day"]    = (df["precip_mm"]   > 5).astype(int)   # significant rain
    df["is_heavy_rain"]   = (df["precip_mm"]   > 20).astype(int)  # heavy rain
    df["no_rain"]         = (df["precip_mm"]   == 0).astype(int)  # dry weather

    # Wind
    df["is_windy"]        = (df["wind_max_kmh"] > 20).astype(int) # noticeable wind

    # Evaporation risk score (0-100 scale proxy)
    # High temp + low humidity + wind = maximum evaporation conditions
    df["evap_risk"]       = (
        df["temp_mean_c"]    * 0.40 +
        (100 - df["humidity_mean"]) * 0.35 +
        df["wind_mean_kmh"]  * 0.25
    ).round(3)

    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any NaN values using column median (robust to outliers)."""
    for col in df.columns:
        if col == "date":
            continue
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ─── CACHE AWARE LOADER ──────────────────────────────────────────────────────

def load_or_fetch_historical(
    start_date: str,
    end_date:   str,
    cache_path: str = "data/weather_data.csv"
) -> pd.DataFrame:
    """
    Load from local cache if it covers the date range,
    otherwise fetch from API and save to cache.

    On your PC  → uses cache to avoid repeated API calls
    On Railway  → always fetches fresh (no persistent storage between deploys)
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached["date"] = pd.to_datetime(cached["date"])
        c_start = cached["date"].min().strftime("%Y-%m-%d")
        c_end   = cached["date"].max().strftime("%Y-%m-%d")

        if c_start <= start_date and c_end >= end_date:
            print(f"  Using cached weather ({c_start} to {c_end})")
            mask = (cached["date"] >= start_date) & (cached["date"] <= end_date)
            return cached[mask].reset_index(drop=True)
        else:
            print(f"  Cache too small — fetching full range")

    df = fetch_historical(start_date, end_date)
    df.to_csv(cache_path, index=False)
    print(f"  Saved to {cache_path}")
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  WEATHER DATA FETCH — HETTIPOLA, SRI LANKA")
    print("="*60)
    print(f"  Location: lat={LAT}, lon={LON}")
    print(f"  API: Open-Meteo (free, no account)")

    # 1. Historical weather (for training)
    print("\n[1/2] Historical weather (training data)...")
    hist = load_or_fetch_historical(
        start_date = "2025-04-01",
        end_date   = "2026-04-19",
        cache_path = "data/weather_data.csv",
    )

    print(f"\n  Summary:")
    print(f"    Rows          : {len(hist)}")
    print(f"    Date range    : {hist['date'].min().date()} → {hist['date'].max().date()}")
    print(f"    Temp range    : {hist['temp_min_c'].min():.1f}°C – {hist['temp_max_c'].max():.1f}°C")
    print(f"    Avg max temp  : {hist['temp_max_c'].mean():.1f}°C")
    print(f"    Rainy days    : {hist['is_rainy_day'].sum()} / {len(hist)}")
    print(f"    Hot days >33°C: {hist['is_hot_day'].sum()} / {len(hist)}")
    print(f"    ET0 range     : {hist['et0_mm'].min():.2f} – {hist['et0_mm'].max():.2f} mm")
    print(f"    VPD range     : {hist['vpd_kpa'].min():.2f} – {hist['vpd_kpa'].max():.2f} kPa")

    # 2. Weather forecast (for predictions)
    print("\n[2/2] Weather forecast (next 8 days)...")
    try:
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Saved to data/weather_forecast.csv")
    except Exception as e:
        print(f"  Forecast failed: {e}")
        print(f"  Predictions will use historical averages as fallback")

    print("\n" + "="*60)
    print("  Files created:")
    print("    data/weather_data.csv     ← training data (384 days)")
    print("    data/weather_forecast.csv ← 7-day forecast")
    print("\n  Next step:")
    print("    python scripts/train_evap_models.py")
    print("="*60 + "\n")