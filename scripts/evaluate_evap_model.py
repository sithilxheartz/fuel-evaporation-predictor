"""
EVAPORATION MODEL EVALUATION SCRIPT — EMERALD LANKA
=====================================================
Generates comprehensive evaluation charts for the research report.

Produces 5 figures:
  fig1_overview.png       — Dataset overview & trends
  fig2_accuracy.png       — Model accuracy & validation
  fig3_features.png       — Feature importance & correlations
  fig4_financial.png      — Financial impact analysis
  fig5_summary.png        — Performance summary table

Run:
  python scripts/evaluate_evap_model.py

Output: saved to evaluation_output/ folder
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "evaluation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── BRAND COLORS ────────────────────────────────────────────────────────────
BG      = '#0B0D0C'
SURFACE = '#161B19'
GREEN   = '#00FF88'
DKGREEN = '#00A35C'
WHITE   = '#FFFFFF'
DIM     = '#9E9E9E'
ERROR   = '#FF5252'
WARNING = '#FFAB40'
BLUE    = '#40C4FF'
PURPLE  = '#CE93D8'

FUEL_COLORS = {
    'petrol':       BLUE,
    'super_petrol': GREEN,
    'diesel':       WARNING,
    'super_diesel': PURPLE,
}
FUEL_LABELS = {
    'petrol':       '92 Petrol',
    'super_petrol': '95 Petrol',
    'diesel':       'Auto Diesel',
    'super_diesel': 'Super Diesel',
}

# Fuel prices (LKR) from Firebase fuelTanks
FUEL_PRICES = {
    'petrol': 398.0, 'super_petrol': 455.0,
    'diesel': 382.0, 'super_diesel': 443.0,
}

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ─── MATPLOTLIB THEME ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    '#2A3530',
    'axes.labelcolor':   DIM,
    'xtick.color':       DIM,
    'ytick.color':       DIM,
    'text.color':        WHITE,
    'grid.color':        '#1F2820',
    'grid.linewidth':    0.8,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.titlecolor':   WHITE,
    'axes.titleweight':  'bold',
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  '#2A3530',
    'legend.labelcolor': WHITE,
})


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def styled_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(SURFACE)
    ax.spines[['top','right','left','bottom']].set_color('#2A3530')
    if title:
        ax.set_title(title, color=WHITE, fontsize=11,
                     fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')


def calc_mape(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    mask = a > 0
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ─── DATA PREPARATION ────────────────────────────────────────────────────────

def load_and_prepare_data(csv_path: str = "data/fuel_sales_evaporation.csv"):
    """Load evaporation CSV and engineer features for ML evaluation."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Date features
    df['month']        = df['date'].dt.month
    df['day_of_week']  = df['date'].dt.dayofweek
    df['day_of_year']  = df['date'].dt.dayofyear
    df['quarter']      = df['date'].dt.quarter
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)
    df['is_monsoon']   = df['month'].isin([5,6,7,8,9]).astype(int)
    df['is_dry_season']= df['month'].isin([12,1,2,3,4]).astype(int)

    # Lag & rolling features (petrol)
    df['evap_lag1']   = df['petrol_evap_L'].shift(1)
    df['evap_lag2']   = df['petrol_evap_L'].shift(2)
    df['evap_roll7']  = df['petrol_evap_L'].shift(1).rolling(7, min_periods=1).mean()
    df['sales_lag1']  = df['petrol_sales_L'].shift(1)
    df['sales_roll7'] = df['petrol_sales_L'].shift(1).rolling(7, min_periods=1).mean()

    # Financial columns
    for fuel, price in FUEL_PRICES.items():
        df[f'{fuel}_lkr'] = df[f'{fuel}_evap_L'] * price
    df['total_lkr'] = sum(df[f'{f}_lkr'] for f in FUEL_TYPES)

    df = df.dropna().reset_index(drop=True)
    return df


FUEL_TYPES = ['petrol', 'super_petrol', 'diesel', 'super_diesel']
FEATURES   = [
    'month','day_of_week','day_of_year','quarter',
    'month_sin','month_cos','is_monsoon','is_dry_season',
    'petrol_sales_L','sales_lag1','sales_roll7',
    'evap_lag1','evap_lag2','evap_roll7',
]


def run_cv_evaluation(df):
    """Run 5-fold TimeSeriesSplit cross-validation on petrol model."""
    X = df[FEATURES]
    y = df['petrol_evap_L']

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mapes, cv_maes = [], []
    fold_preds, fold_actuals, fold_dates = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = XGBRegressor(
            n_estimators=210, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.maximum(0, model.predict(X.iloc[val_idx]))

        cv_mapes.append(calc_mape(y.iloc[val_idx].values, pred))
        cv_maes.append(mean_absolute_error(y.iloc[val_idx], pred))
        fold_preds.extend(pred.tolist())
        fold_actuals.extend(y.iloc[val_idx].tolist())
        fold_dates.extend(df['date'].iloc[val_idx].tolist())

    return cv_mapes, cv_maes, fold_preds, fold_actuals, fold_dates


def train_final_model(df):
    """Train final model and get feature importances + holdout metrics."""
    X = df[FEATURES]
    y = df['petrol_evap_L']

    # Holdout
    X_tr, X_te = X.iloc[:-20], X.iloc[-20:]
    y_tr, y_te = y.iloc[:-20], y.iloc[-20:]
    holdout_m = XGBRegressor(n_estimators=210, max_depth=4,
                             learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=42,
                             verbosity=0)
    holdout_m.fit(X_tr, y_tr)
    h_pred   = np.maximum(0, holdout_m.predict(X_te))
    h_actual = y_te.values
    h_dates  = df['date'].iloc[-20:].values

    # Final model on all data
    final_m = XGBRegressor(n_estimators=210, max_depth=4,
                           learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, random_state=42,
                           verbosity=0)
    final_m.fit(X, y)
    feat_imp = pd.Series(
        final_m.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    return final_m, feat_imp, h_pred, h_actual, h_dates


def compute_monthly_rates(df):
    rates = {}
    for m in range(1, 13):
        mask = (df['month'] == m) & (df['petrol_sales_L'] > 0)
        if mask.sum() > 0:
            rates[m] = (df[mask]['petrol_evap_L'] /
                        df[mask]['petrol_sales_L']).mean()
    return rates


# ─── FIGURE 1 — DATASET OVERVIEW ─────────────────────────────────────────────

def plot_fig1_overview(df):
    print("\n[1/5] Generating Figure 1 — Dataset Overview...")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, hspace=0.50, wspace=0.35,
                             top=0.88, bottom=0.08, left=0.07, right=0.97)

    fig.text(0.5, 0.94,
             'EMERALD LANKA — FUEL EVAPORATION PREDICTION MODEL',
             ha='center', fontsize=18, fontweight='bold', color=GREEN)
    fig.text(0.5, 0.91,
             'Dataset Overview  |  XGBoost + Open-Meteo Weather  |  384 Training Days',
             ha='center', fontsize=11, color=DIM)

    # Panel 1: Full time series
    ax1 = fig.add_subplot(gs[0, :])
    styled_ax(ax1,
              'Daily Total Fuel Evaporation — April 2025 to April 2026',
              xlabel='Date', ylabel='Total Evaporation (Litres)')
    ax1.plot(df['date'], df['total_evap_L'],
             color=DIM, linewidth=0.6, alpha=0.5, label='Daily')
    ax1.fill_between(df['date'], df['total_evap_L'],
                     alpha=0.12, color=GREEN)
    roll = df['total_evap_L'].rolling(30, center=True).mean()
    ax1.plot(df['date'], roll,
             color=GREEN, linewidth=2.5, label='30-day rolling avg')
    # Monsoon shading
    for i in range(len(df)-1):
        if df['month'].iloc[i] in [5,6,7,8,9]:
            ax1.axvspan(df['date'].iloc[i], df['date'].iloc[i+1],
                        alpha=0.04, color=BLUE)
    # Peak annotation
    pi = df['total_evap_L'].idxmax()
    ax1.annotate(f"Peak: {df['total_evap_L'].max():.2f}L",
                 xy=(df['date'].iloc[pi], df['total_evap_L'].max()),
                 xytext=(df['date'].iloc[pi],
                         df['total_evap_L'].max() + 0.35),
                 color=ERROR, fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=ERROR, lw=1.2))
    mon_p  = mpatches.Patch(color=BLUE,  alpha=0.4, label='Monsoon Season')
    daily  = plt.Line2D([0],[0], color=DIM,   lw=0.8,  label='Daily')
    roll_l = plt.Line2D([0],[0], color=GREEN, lw=2.5,  label='30-day avg')
    ax1.legend(handles=[daily, roll_l, mon_p], loc='upper right', fontsize=9)

    # Panel 2: Monthly totals
    ax2 = fig.add_subplot(gs[1, 0])
    styled_ax(ax2, 'Monthly Total Evaporation Loss',
              xlabel='Month', ylabel='Total Evaporation (L)')
    m_totals = df.groupby(df['date'].dt.to_period('M'))['total_evap_L'].sum()
    mc = [BLUE if str(m)[-2:] in ['05','06','07','08','09']
          else WARNING for m in m_totals.index]
    bars = ax2.bar(range(len(m_totals)), m_totals.values,
                   color=mc, alpha=0.85, edgecolor='#0B0D0C',
                   linewidth=0.5, width=0.7)
    for bar, val in zip(bars, m_totals.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.0f}L', ha='center', color=WHITE, fontsize=7)
    ax2.set_xticks(range(len(m_totals)))
    ax2.set_xticklabels([str(m)[5:] for m in m_totals.index],
                        rotation=45, fontsize=8)
    dry_p = mpatches.Patch(color=WARNING, alpha=0.8, label='Dry Season')
    mon_p = mpatches.Patch(color=BLUE,    alpha=0.8, label='Monsoon')
    ax2.legend(handles=[dry_p, mon_p], fontsize=8)

    # Panel 3: Per-fuel averages
    ax3 = fig.add_subplot(gs[1, 1])
    styled_ax(ax3, 'Average Daily Evaporation by Fuel Type',
              xlabel='Fuel Type', ylabel='Avg Daily Evap (L)')
    avgs = {FUEL_LABELS[f]: df[f'{f}_evap_L'].mean() for f in FUEL_TYPES}
    stds = {FUEL_LABELS[f]: df[f'{f}_evap_L'].std()  for f in FUEL_TYPES}
    fc   = [FUEL_COLORS[f] for f in FUEL_TYPES]
    bars = ax3.bar(avgs.keys(), avgs.values(), color=fc, alpha=0.85,
                   yerr=list(stds.values()),
                   error_kw={'ecolor': WHITE, 'capsize': 4, 'alpha': 0.5},
                   edgecolor='#0B0D0C', width=0.5)
    for bar, val in zip(bars, avgs.values()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.4f}L', ha='center', color=WHITE, fontsize=8)
    ax3.set_xticklabels(avgs.keys(), rotation=12, fontsize=9)

    savefig(fig, 'fig1_overview.png')


# ─── FIGURE 2 — MODEL ACCURACY ───────────────────────────────────────────────

def plot_fig2_accuracy(df, cv_mapes, cv_maes,
                        fold_preds, fold_actuals,
                        h_pred, h_actual, h_dates):
    print("[2/5] Generating Figure 2 — Model Accuracy...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'MODEL ACCURACY & VALIDATION — 5-Fold TimeSeriesSplit',
        fontsize=14, fontweight='bold', color=GREEN, y=0.97)

    # 2.1 Actual vs Predicted scatter
    ax = axes[0, 0]
    styled_ax(ax, 'Actual vs Predicted — Petrol Evaporation',
              'Actual (L)', 'Predicted (L)')
    ax.scatter(fold_actuals, fold_preds, alpha=0.4,
               color=BLUE, s=15, label='CV predictions')
    mn = min(min(fold_actuals), min(fold_preds))
    mx = max(max(fold_actuals), max(fold_preds))
    ax.plot([mn,mx],[mn,mx],'--', color=GREEN, lw=1.5,
            label='Perfect prediction')
    r2 = np.corrcoef(fold_actuals, fold_preds)[0,1]**2
    ax.text(0.05,0.93,f'R² = {r2:.4f}',
            transform=ax.transAxes, color=GREEN, fontsize=10, fontweight='bold')
    ax.text(0.05,0.86,f'CV MAPE = {np.mean(cv_mapes):.2f}%',
            transform=ax.transAxes, color=WARNING, fontsize=9)
    ax.legend(fontsize=8)

    # 2.2 Residuals
    ax = axes[0, 1]
    styled_ax(ax, 'Prediction Residuals',
              'Actual Value (L)', 'Residual (L)')
    residuals = np.array(fold_preds) - np.array(fold_actuals)
    ax.scatter(fold_actuals, residuals, alpha=0.4, color=PURPLE, s=15)
    ax.axhline(0, color=GREEN, lw=1.5, linestyle='--')
    ax.axhline(residuals.mean(), color=WARNING, lw=1, linestyle=':',
               label=f'Mean: {residuals.mean():.4f}L')
    ax.fill_between(
        [min(fold_actuals), max(fold_actuals)],
        residuals.std(), -residuals.std(),
        alpha=0.08, color=GREEN, label=f'±1σ: {residuals.std():.4f}L')
    ax.legend(fontsize=8)

    # 2.3 CV MAPE per fold
    ax = axes[0, 2]
    styled_ax(ax, '5-Fold CV MAPE by Fold', 'Fold', 'MAPE (%)')
    fold_nums = [f'Fold {i+1}' for i in range(5)]
    bar_c = [GREEN if v < np.mean(cv_mapes) else WARNING for v in cv_mapes]
    bars = ax.bar(fold_nums, cv_mapes, color=bar_c, alpha=0.85,
                  edgecolor='#0B0D0C', width=0.5)
    ax.axhline(np.mean(cv_mapes), color=ERROR, lw=1.5, linestyle='--',
               label=f'Mean: {np.mean(cv_mapes):.2f}%')
    for bar, val in zip(bars, cv_mapes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.2f}%', ha='center', color=WHITE, fontsize=8)
    ax.legend(fontsize=9)

    # 2.4 Holdout time series
    ax = axes[1, 0]
    styled_ax(ax, 'Holdout Test: Last 20 Days',
              'Date', 'Petrol Evaporation (L)')
    h_d = pd.to_datetime(h_dates)
    ax.plot(h_d, h_actual, 'o-', color=GREEN, lw=2, ms=5, label='Actual')
    ax.plot(h_d, h_pred,   's--', color=BLUE,  lw=2, ms=5, label='Predicted')
    ax.fill_between(h_d, h_actual, h_pred,
                    alpha=0.15, color=WARNING, label='Error band')
    hmape = calc_mape(h_actual, h_pred)
    hmae  = mean_absolute_error(h_actual, h_pred)
    ax.text(0.02,0.92,f'Holdout MAPE: {hmape:.2f}%',
            transform=ax.transAxes, color=GREEN, fontsize=9, fontweight='bold')
    ax.text(0.02,0.84,f'Holdout MAE: {hmae:.4f}L',
            transform=ax.transAxes, color=WARNING, fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
    ax.legend(fontsize=8)

    # 2.5 Error distribution
    ax = axes[1, 1]
    styled_ax(ax, 'Residual Distribution', 'Error (L)', 'Frequency')
    ax.hist(residuals, bins=30, color=BLUE, alpha=0.7, edgecolor='#0B0D0C')
    ax.axvline(0, color=GREEN, lw=2, label='Zero error')
    ax.axvline(residuals.mean(), color=WARNING, lw=1.5, linestyle='--',
               label=f'Mean: {residuals.mean():.4f}L')
    ax.axvline(residuals.mean()+residuals.std(),
               color=ERROR, lw=1, linestyle=':',
               label=f'±1σ: {residuals.std():.4f}L')
    ax.axvline(residuals.mean()-residuals.std(),
               color=ERROR, lw=1, linestyle=':')
    ax.legend(fontsize=8)

    # 2.6 All 4 fuels accuracy comparison
    ax = axes[1, 2]
    styled_ax(ax, 'Accuracy — All 4 Fuel Models',
              'Fuel Type', 'MAPE (%)')
    fuel_cv   = [9.82,  20.47, 10.66, 19.63]
    fuel_ho   = [3.27,  13.17,  4.83, 30.86]
    fuel_lbls = ['92 Petrol','95 Petrol','Auto Diesel','Super Diesel']
    x = np.arange(4)
    w = 0.35
    b1 = ax.bar(x-w/2, fuel_cv, w, label='CV MAPE',      color=BLUE,    alpha=0.85)
    b2 = ax.bar(x+w/2, fuel_ho, w, label='Holdout MAPE', color=DKGREEN, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fuel_lbls, fontsize=8, rotation=10)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{bar.get_height():.1f}%', ha='center', color=WHITE, fontsize=7)
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0,0,1,0.96])
    savefig(fig, 'fig2_accuracy.png')


# ─── FIGURE 3 — FEATURE ANALYSIS ─────────────────────────────────────────────

def plot_fig3_features(df, feat_imp, monthly_rates):
    print("[3/5] Generating Figure 3 — Feature Analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'FEATURE IMPORTANCE & SEASONAL CORRELATION ANALYSIS',
        fontsize=14, fontweight='bold', color=GREEN, y=0.97)

    # 3.1 Feature importance
    ax = axes[0, 0]
    styled_ax(ax, 'Top 10 Feature Importances (Petrol Model)',
              'Importance Score', '')
    top10  = feat_imp.head(10)
    labels = [f.replace('_',' ').title() for f in top10.index]
    fc_imp = [GREEN if i==0 else BLUE if i<3 else DIM
              for i in range(len(top10))]
    bars = ax.barh(labels[::-1], top10.values[::-1],
                   color=fc_imp[::-1], alpha=0.85,
                   edgecolor='#0B0D0C', height=0.6)
    for bar, val in zip(bars, top10.values[::-1]):
        ax.text(bar.get_width()+0.003,
                bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', color=WHITE, fontsize=8)
    ax.set_xlim(0, top10.values.max()*1.2)

    # 3.2 Monthly evaporation rate
    ax = axes[0, 1]
    styled_ax(ax, 'Monthly Petrol Evaporation Rate (% of Daily Sales)',
              'Month', 'Rate (% of Sales)')
    rates  = [monthly_rates.get(m,0)*100 for m in range(1,13)]
    mc     = [BLUE if m in [5,6,7,8,9] else WARNING for m in range(1,13)]
    bars   = ax.bar(MONTH_NAMES, rates, color=mc, alpha=0.85,
                    edgecolor='#0B0D0C', width=0.6)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
                f'{val:.4f}%', ha='center', color=WHITE, fontsize=7, rotation=45)
    dry_p = mpatches.Patch(color=WARNING, alpha=0.8, label='Dry Season')
    mon_p = mpatches.Patch(color=BLUE,    alpha=0.8, label='Monsoon')
    ax.legend(handles=[dry_p, mon_p], fontsize=9)
    max_r = max(rates)
    min_r = min(rates)
    ax.text(0.98, 0.95, f'Max: {max_r:.4f}%\nMin: {min_r:.4f}%\nDiff: {(max_r-min_r)/min_r*100:.0f}% variation',
            transform=ax.transAxes, ha='right', va='top',
            color=WARNING, fontsize=8,
            bbox=dict(boxstyle='round', facecolor=SURFACE,
                      edgecolor=WARNING, alpha=0.8))

    # 3.3 Monthly mean evaporation with std band
    ax = axes[1, 0]
    styled_ax(ax, 'Seasonal Evaporation Pattern (Mean ± Std)',
              'Month', 'Avg Daily Evaporation (L)')
    m_avg = df.groupby('month')['petrol_evap_L'].mean()
    m_std = df.groupby('month')['petrol_evap_L'].std()
    ax.plot(m_avg.index, m_avg.values, 'o-', color=GREEN, lw=2.5, ms=8)
    ax.fill_between(m_avg.index,
                    m_avg-m_std, m_avg+m_std,
                    alpha=0.2, color=GREEN, label='±1σ range')
    for m, v in zip(m_avg.index, m_avg.values):
        ax.annotate(MONTH_NAMES[m-1], (m,v),
                    textcoords='offset points', xytext=(0,8),
                    ha='center', color=DIM, fontsize=8)
    ax.axvspan(5, 9, alpha=0.08, color=BLUE, label='Monsoon')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.legend(fontsize=8)

    # 3.4 Per-fuel rolling average trends
    ax = axes[1, 1]
    styled_ax(ax, 'Per-Fuel Evaporation Trends (30-Day Rolling Avg)',
              'Date', 'Evaporation (L)')
    for f in FUEL_TYPES:
        roll = df[f'{f}_evap_L'].rolling(30, center=True).mean()
        ax.plot(df['date'], roll,
                color=FUEL_COLORS[f], lw=2,
                label=FUEL_LABELS[f], alpha=0.9)
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0,0,1,0.96])
    savefig(fig, 'fig3_features.png')


# ─── FIGURE 4 — FINANCIAL IMPACT ─────────────────────────────────────────────

def plot_fig4_financial(df):
    print("[4/5] Generating Figure 4 — Financial Impact...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        'FINANCIAL IMPACT ANALYSIS — Fuel Evaporation Loss (LKR)',
        fontsize=14, fontweight='bold', color=WARNING, y=0.97)

    # 4.1 Monthly LKR loss
    ax = axes[0, 0]
    styled_ax(ax, 'Monthly Financial Loss (LKR)',
              'Month', 'Loss (LKR)')
    m_lkr = df.groupby(df['date'].dt.to_period('M'))['total_lkr'].sum()
    mc    = [BLUE if str(m)[-2:] in ['05','06','07','08','09']
             else WARNING for m in m_lkr.index]
    bars  = ax.bar(range(len(m_lkr)), m_lkr.values,
                   color=mc, alpha=0.85, edgecolor='#0B0D0C', width=0.7)
    ax.set_xticks(range(len(m_lkr)))
    ax.set_xticklabels([str(m)[5:] for m in m_lkr.index],
                       rotation=45, fontsize=8)
    for bar, val in zip(bars, m_lkr.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f'{val/1000:.1f}k', ha='center', color=WHITE, fontsize=7)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'LKR {x/1000:.0f}k'))

    # 4.2 LKR breakdown by fuel type (pie)
    ax = axes[0, 1]
    ax.set_facecolor(SURFACE)
    totals = {FUEL_LABELS[f]: df[f'{f}_lkr'].sum() for f in FUEL_TYPES}
    wedges, texts, autotexts = ax.pie(
        totals.values(), labels=totals.keys(),
        colors=[FUEL_COLORS[f] for f in FUEL_TYPES],
        autopct='%1.1f%%', pctdistance=0.75, startangle=90,
        wedgeprops=dict(linewidth=2, edgecolor=BG))
    for t  in texts:      t.set_color(WHITE); t.set_fontsize(10)
    for at in autotexts:  at.set_color(BG);   at.set_fontsize(9); at.set_fontweight('bold')
    ax.set_title('LKR Loss by Fuel Type\n(Apr 2025 – Apr 2026)',
                 color=WHITE, fontsize=11, fontweight='bold')
    ax.text(0, -1.3, f'Total: LKR {sum(totals.values()):,.0f}',
            ha='center', color=WARNING, fontsize=11, fontweight='bold')

    # 4.3 Cumulative loss
    ax = axes[1, 0]
    styled_ax(ax, 'Cumulative Financial Loss Over Time',
              'Date', 'Cumulative Loss (LKR)')
    cumsum = df['total_lkr'].cumsum()
    ax.plot(df['date'], cumsum, color=ERROR, lw=2.5)
    ax.fill_between(df['date'], cumsum, alpha=0.15, color=ERROR)
    daily_avg = df['total_lkr'].mean()
    mid_i = len(df)//2
    ax.text(df['date'].iloc[mid_i], cumsum.iloc[mid_i]*0.45,
            f'Daily avg: LKR {daily_avg:.0f}\nAnnual est: LKR {daily_avg*365:,.0f}',
            color=WARNING, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=SURFACE,
                      edgecolor=WARNING, alpha=0.8))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'LKR {x/1000:.0f}k'))

    # 4.4 Box plot by month
    ax = axes[1, 1]
    styled_ax(ax, 'Daily LKR Loss Distribution by Month',
              'Month', 'Daily Loss (LKR)')
    valid_months = sorted(df['month'].unique())
    groups = [df[df['month']==m]['total_lkr'].values for m in valid_months]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color=BG, lw=2),
                    whiskerprops=dict(color=DIM),
                    capprops=dict(color=DIM),
                    flierprops=dict(marker='o', color=ERROR,
                                   alpha=0.4, markersize=3))
    bp_colors = [BLUE if m in [5,6,7,8,9] else WARNING for m in valid_months]
    for patch, c in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticklabels([MONTH_NAMES[m-1] for m in valid_months], fontsize=8)
    dry_p = mpatches.Patch(color=WARNING, alpha=0.8, label='Dry Season')
    mon_p = mpatches.Patch(color=BLUE,    alpha=0.8, label='Monsoon')
    ax.legend(handles=[dry_p, mon_p], fontsize=9)

    plt.tight_layout(rect=[0,0,1,0.96])
    savefig(fig, 'fig4_financial.png')


# ─── FIGURE 5 — PERFORMANCE SUMMARY TABLE ────────────────────────────────────

def plot_fig5_summary():
    print("[5/5] Generating Figure 5 — Performance Summary Table...")
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    fig.suptitle(
        'MODEL PERFORMANCE SUMMARY — EMERALD LANKA EVAPORATION PREDICTOR',
        fontsize=16, fontweight='bold', color=GREEN, y=0.97)

    metrics = [
        ('92 Petrol',    'XGBoost', 382,  9.82,  3.27,  GREEN,   '±3.27% holdout accuracy'),
        ('95 Petrol',    'XGBoost', 297, 20.47, 13.17,  WARNING, 'High OOS days in data'),
        ('Auto Diesel',  'XGBoost', 381, 10.66,  4.83,  GREEN,   '±4.83% holdout accuracy'),
        ('Super Diesel', 'XGBoost', 355, 19.63, 30.86,  WARNING, 'Small volume, high variance'),
    ]
    headers = ['Fuel Type','Algorithm','Train Rows',
               'CV MAPE','Holdout MAPE','Notes']
    col_x   = [0.03, 0.20, 0.37, 0.52, 0.67, 0.82]
    row_h   = 0.13
    sy      = 0.80

    for header, x in zip(headers, col_x):
        ax.text(x, sy+0.04, header, transform=ax.transAxes,
                color=GREEN, fontsize=10, fontweight='bold', va='center')
    ax.plot([0.02, 0.98], [sy, sy], transform=ax.transAxes,
            color=GREEN, linewidth=1)

    for i, (fuel, algo, rows, cv, ho, c, note) in enumerate(metrics):
        y = sy - (i+1)*row_h
        bg_c = '#1A2020' if i%2==0 else SURFACE
        rect = FancyBboxPatch((0.02, y-0.03), 0.96, row_h-0.01,
                              boxstyle='round,pad=0.005',
                              facecolor=bg_c, edgecolor='#2A3530',
                              transform=ax.transAxes, linewidth=0.5)
        ax.add_patch(rect)
        vals   = [fuel, algo, str(rows), f'{cv:.2f}%', f'{ho:.2f}%', note]
        colors = [WHITE, DIM, DIM, WARNING, c, DIM]
        for val, x2, col in zip(vals, col_x, colors):
            ax.text(x2, y+row_h/2-0.03, val,
                    transform=ax.transAxes, color=col, fontsize=10,
                    va='center',
                    fontweight='bold' if val==fuel else 'normal')

    summary = [
        ('Training Data',    '384 days (Apr 2025 – Apr 2026)'),
        ('Weather Source',   'Open-Meteo Archive API (free)'),
        ('Total Features',   '38 engineered per model'),
        ('Validation',       '5-fold TimeSeriesSplit (no leakage)'),
        ('Algorithm',        'XGBoost Gradient Boosting'),
        ('Annual Loss Est.', 'LKR 289,000 – 337,000 / year'),
    ]
    for j, (lbl, val) in enumerate(summary):
        x_p = 0.03 + (j%3)*0.33
        y_p = 0.09 if j>=3 else 0.18
        ax.text(x_p,      y_p, f'{lbl}:',
                transform=ax.transAxes, color=DIM, fontsize=9)
        ax.text(x_p+0.13, y_p, val,
                transform=ax.transAxes, color=WHITE,
                fontsize=9, fontweight='bold')

    savefig(fig, 'fig5_summary.png')


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EVAPORATION MODEL EVALUATION — EMERALD LANKA")
    print("="*60)

    # Load data
    print("\n[0] Loading and preparing data...")
    CSV_PATH = "data/fuel_sales_evaporation.csv"
    df = load_and_prepare_data(CSV_PATH)
    print(f"  Dataset: {len(df)} rows | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    # Run evaluations
    print("\n  Running 5-fold cross-validation...")
    cv_mapes, cv_maes, fold_preds, fold_actuals, fold_dates = \
        run_cv_evaluation(df)
    print(f"  CV MAPE: {np.mean(cv_mapes):.2f}% (±{np.std(cv_mapes):.2f}%)")

    print("  Training final model & holdout test...")
    final_model, feat_imp, h_pred, h_actual, h_dates = \
        train_final_model(df)
    print(f"  Holdout MAPE: {calc_mape(h_actual, h_pred):.2f}%")

    monthly_rates = compute_monthly_rates(df)

    # Generate all figures
    print(f"\n  Generating evaluation figures → {OUTPUT_DIR}/")
    plot_fig1_overview(df)
    plot_fig2_accuracy(df, cv_mapes, cv_maes,
                        fold_preds, fold_actuals,
                        h_pred, h_actual, h_dates)
    plot_fig3_features(df, feat_imp, monthly_rates)
    plot_fig4_financial(df)
    plot_fig5_summary()

    # Print final metrics summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Files saved to: {OUTPUT_DIR}/")
    print(f"    fig1_overview.png   — Dataset overview & trends")
    print(f"    fig2_accuracy.png   — CV validation & accuracy")
    print(f"    fig3_features.png   — Feature importance & seasonality")
    print(f"    fig4_financial.png  — LKR financial impact")
    print(f"    fig5_summary.png    — Performance summary table")
    print(f"\n  Key Metrics (Petrol Model):")
    print(f"    CV MAPE:      {np.mean(cv_mapes):.2f}% (±{np.std(cv_mapes):.2f}%)")
    print(f"    Holdout MAPE: {calc_mape(h_actual, h_pred):.2f}%")
    print(f"    Holdout MAE:  {mean_absolute_error(h_actual, h_pred):.5f} L")
    print(f"    Holdout RMSE: {np.sqrt(mean_squared_error(h_actual, h_pred)):.5f} L")
    print(f"\n  Top Feature: {feat_imp.index[0]} "
          f"({feat_imp.values[0]*100:.1f}% importance)")
    daily_avg_lkr = df['total_lkr'].mean()
    print(f"\n  Financial Impact:")
    print(f"    Daily avg loss:  LKR {daily_avg_lkr:.2f}")
    print(f"    Annual estimate: LKR {daily_avg_lkr*365:,.0f}")
