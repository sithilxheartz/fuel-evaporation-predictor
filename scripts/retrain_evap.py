"""
RETRAIN_EVAP.PY — EMERALD LANKA
=================================
Retrains evaporation models then runs the full pipeline:
  1. Fetch training weather (full CSV range → 384 rows)
  2. Train all 4 XGBoost evaporation models
  3. Run evaporation_pipeline.py:
       - Reads actual evap from CSV (type="actual")
       - Predicts gap period using ML + Firebase sales (type="predicted")
       - Stores monthly summaries to fuelEvaporationSummary
       - Stores overall summary to fuelEvaporation/--summary--

Run manually:
  python scripts/retrain_evap.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def full_evap_retrain_pipeline():
    print("\n" + "="*60)
    print("  EVAPORATION RETRAIN PIPELINE")
    print("="*60)
    started = datetime.now()

    from scripts.fetch_weather import fetch_historical, fetch_forecast

    # ── Step 1: Fetch training weather (full CSV range) ───────────────────────
    print("\n[1/3] Fetching training weather (Apr 2025 → Apr 2026)...")
    try:
        import pandas as pd
        evap_csv = pd.read_csv("data/fuel_sales_evaporation.csv")
        evap_csv["date"] = pd.to_datetime(evap_csv["date"])
        t_start  = evap_csv["date"].min().strftime("%Y-%m-%d")
        t_end    = evap_csv["date"].max().strftime("%Y-%m-%d")
    except Exception:
        t_start = "2025-04-01"
        t_end   = "2026-04-19"

    try:
        train_wx = fetch_historical(t_start, t_end)
        train_wx.to_csv("data/weather_data.csv", index=False)
        print(f"  Training weather: {len(train_wx)} days ({t_start} → {t_end})")

        # Also fetch forecast for API predictions
        fcast = fetch_forecast(days_ahead=8)
        fcast.to_csv("data/weather_forecast.csv", index=False)
        print(f"  Forecast: {len(fcast)} days")
    except Exception as e:
        print(f"  ⚠️  Weather fetch failed: {e}")

    # ── Step 2: Train all 4 models ────────────────────────────────────────────
    print("\n[2/3] Training evaporation models on full 384-row dataset...")
    from scripts.train_evap_models import train_all_evap_models
    metrics = train_all_evap_models()

    # ── Step 3: Run full evaporation pipeline ─────────────────────────────────
    print("\n[3/3] Running full evaporation pipeline...")
    from scripts.evaporation_pipeline import run_evaporation_pipeline
    run_evaporation_pipeline()

    elapsed = (datetime.now() - started).seconds
    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE in {elapsed}s")
    print(f"{'='*60}")
    print("\n  Model accuracy:")
    for fuel, m in metrics.items():
        print(f"    {fuel:<20} CV: {m['cv_mape']:.2f}%  "
              f"Holdout: {m['holdout_mape']:.2f}%")

    return metrics


if __name__ == "__main__":
    full_evap_retrain_pipeline()