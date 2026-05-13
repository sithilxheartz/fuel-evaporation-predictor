"""
RETRAIN_EVAP.PY — EMERALD LANKA
=================================
Full evaporation model retraining pipeline.
Called automatically on Railway when /retrain_evap endpoint is triggered.

Steps:
  1. Fetch latest weather data from Open-Meteo
  2. Fetch latest sales + evaporation data from Firebase
  3. Merge all data
  4. Train all 4 evaporation XGBoost models
  5. Save updated models

Run manually:
  python scripts/retrain_evap.py
"""

import os
import sys
import pandas as pd
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def full_evap_retrain_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION RETRAIN PIPELINE")
    print("="*60)
    started = datetime.now()

    # Step 1: Fetch latest weather (covers full date range)
    print("\n[1/4] Fetching latest weather data from Open-Meteo...")
    try:
        from scripts.fetch_weather import fetch_historical, _add_derived_features
        # Get clean data range
        clean_path = "data/sales_data_clean.csv"
        if os.path.exists(clean_path):
            clean_df = pd.read_csv(clean_path)
            clean_df["date"] = pd.to_datetime(clean_df["date"])
            start_date = clean_df["date"].min().strftime("%Y-%m-%d")
            end_date   = clean_df["date"].max().strftime("%Y-%m-%d")
        else:
            start_date = "2025-04-01"
            end_date   = date.today().strftime("%Y-%m-%d")

        wx_df = fetch_historical(start_date, end_date)
        wx_df.to_csv("data/weather_data.csv", index=False)
        print(f"  Weather data saved: {len(wx_df)} days")

        # Also fetch forecast for predictions
        from scripts.fetch_weather import fetch_forecast
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Forecast saved: {len(fcast)} days")

    except Exception as e:
        print(f"  Weather fetch failed: {e}")
        print("  Will use cached or seasonal fallback data")

    # Step 2: Check evaporation CSV exists
    print("\n[2/4] Checking evaporation data...")
    evap_path = "data/fuel_sales_evaporation.csv"
    if not os.path.exists(evap_path):
        print(f"  ❌ Missing: {evap_path}")
        print("  Cannot retrain evaporation models without evaporation data.")
        return None
    else:
        evap_df = pd.read_csv(evap_path)
        print(f"  Found: {len(evap_df)} rows of evaporation data")

    # Step 3: Train all 4 models
    print("\n[3/4] Training evaporation models...")
    from scripts.train_evap_models import train_all_evap_models
    metrics = train_all_evap_models()

    # Step 4: Fetch updated weather forecast for predictions
    print("\n[4/4] Updating weather forecast...")
    try:
        from scripts.fetch_weather import fetch_forecast
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print("  Forecast updated")
    except Exception as e:
        print(f"  Forecast update failed: {e} (will use cached)")

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'='*60}")
    print(f"  EVAPORATION RETRAIN COMPLETE in {elapsed}s")
    print(f"{'='*60}")
    for fuel, m in metrics.items():
        print(f"  {fuel:<20} CV MAPE: {m['cv_mape']:.2f}%")

    return metrics


if __name__ == "__main__":
    full_evap_retrain_pipeline()