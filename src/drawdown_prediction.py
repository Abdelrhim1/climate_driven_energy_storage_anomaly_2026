import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
import json
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from pathlib import Path

# Dynamic Path Discovery for Reorganized Structure
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Windows DLL loading fix for Conda environments (still useful for eccodes)
if sys.platform == 'win32':
    env_path = r'C:\Users\abdel\miniforge3\envs\ds_env'
    bin_path = os.path.join(env_path, 'Library', 'bin')
    if os.path.exists(bin_path):
        os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory') and os.path.exists(bin_path):
        os.add_dll_directory(bin_path)

# Define Industrial Clusters (NRW, Bavaria, Baden-Württemberg)
CLUSTERS = {
    'Nordrhein-Westfalen': {'lat': 51.2, 'lon': 6.8, 'weight': 0.45},
    'Bayern': {'lat': 48.1, 'lon': 11.5, 'weight': 0.30},
    'Baden-Wuerttemberg': {'lat': 48.7, 'lon': 9.0, 'weight': 0.25}
}

def calculate_weighted_hdd(ds: xr.Dataset) -> pd.Series:
    """Calculate HDD weighted by industrial cluster density."""
    t2m = ds.t2m - 273.15 # Kelvin to Celsius
    weighted_hdds = []
    
    for date in ds.time:
        t_snapshot = t2m.sel(time=date)
        daily_hdd = 0
        for name, info in CLUSTERS.items():
            t_cluster = t_snapshot.sel(latitude=info['lat'], longitude=info['lon'], method='nearest').mean().values.item()
            hdd = float(max(0, 15.5 - t_cluster))
            daily_hdd += hdd * info['weight']
        weighted_hdds.append(daily_hdd)
        
    return pd.Series(weighted_hdds, index=pd.to_datetime(ds.time.values), name='Weighted_HDD')

def get_climate_anomalies(ds_t: xr.Dataset, ds_p: xr.Dataset, ds_clime: xr.Dataset) -> pd.DataFrame:
    """Calculate anomalies for Temp and Precip."""
    germany_box = {'latitude': slice(55, 47), 'longitude': slice(6, 15)}
    
    # Ensure dimensions are reduced (collapsed step/lat/lon)
    t_target = ds_t.t2m.sel(**germany_box).mean(dim=['latitude', 'longitude'])
    if 'step' in t_target.dims:
        t_target = t_target.mean(dim='step')
        
    p_target = ds_p.tp.sel(**germany_box).mean(dim=['latitude', 'longitude'])
    if 'step' in p_target.dims:
        p_target = p_target.mean(dim='step')
    
    t_target_daily = t_target.resample(time='1D').mean()
    p_target_daily = p_target.resample(time='1D').sum()
    
    # Simplified anomaly relative to slice mean
    t_anomaly = t_target_daily - t_target_daily.mean()
    p_anomaly = p_target_daily - p_target_daily.mean()
    
    return pd.DataFrame({
        'Temp_Anomaly': t_anomaly.values,
        'Precip_Anomaly': p_anomaly.values
    }, index=pd.to_datetime(t_target_daily.time.values))

def build_and_predict(df: pd.DataFrame):
    """High-accuracy XGBoost Model with Temporal Lagging."""
    features = ['Weighted_HDD', 'Temp_Anomaly', 'Precip_Anomaly', 'Gas_Price_EUR_MWh']
    target = 'Storage_Fill_Percent'
    
    # Feature Engineering: Lags to capture sequential drawdown trends
    for f in features:
        df[f'{f}_lag1'] = df[f].shift(1)
        df[f'{f}_lag2'] = df[f].shift(2)
    
    df = df.dropna()
    extended_features = [c for c in df.columns if c != target]
    
    X = df[extended_features].values
    y = df[target].values
    
    print(f"Training XGBoost Predictor on {len(df)} days...")
    model = XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=5)
    model.fit(X, y)
    
    preds = model.predict(X)
    importance = dict(zip(extended_features, [float(x) for x in model.feature_importances_]))
    residuals = y - preds
    
    return preds, y, df.index, importance, residuals

if __name__ == "__main__":
    print("Initializing Robust Modeling Workflow...")
    print(f"DEBUG: SCRIPT_DIR = {SCRIPT_DIR}")
    print(f"DEBUG: ROOT_DIR = {ROOT_DIR}")
    print(f"DEBUG: DATA_DIR = {DATA_DIR}")

    # Path Alignment
    GAS_DATA_PATH = DATA_DIR / 'german_gas_q1_2026.csv'
    GRIB_DATA_PATH = DATA_DIR / 'era5_land_2026_jan_feb_REAL.grib'
    CLIME_DATA_PATH = DATA_DIR / 'era5_climatology_1991_2020_REAL.nc'
    
    if not GAS_DATA_PATH.exists():
        print(f"Error: Energy data CSV not found at {GAS_DATA_PATH}")
        sys.exit(1)
        
    gas_data = pd.read_csv(GAS_DATA_PATH, parse_dates=['Date']).set_index('Date')
    
    try:
        print("Loading Climate Data...")
        temp_dir = os.environ.get('TEMP', os.path.expanduser('~'))
        idx_t = os.path.join(temp_dir, 'era5_final_t2m.idx')
        idx_p = os.path.join(temp_dir, 'era5_final_tp.idx')
        
        germany_box = {'latitude': slice(55, 47), 'longitude': slice(6, 15)}
        
        ds_t = xr.open_dataset(GRIB_DATA_PATH, engine='cfgrib', 
                              chunks={'time': 12},
                              backend_kwargs={'filter_by_keys': {'shortName': '2t'}, 'indexpath': idx_t})
        
        ds_p = xr.open_dataset(GRIB_DATA_PATH, engine='cfgrib', 
                              chunks={'time': 12},
                              backend_kwargs={'filter_by_keys': {'shortName': 'tp'}, 'indexpath': idx_p})
        
        ds_t = ds_t.sel(**germany_box)
        ds_p = ds_p.sel(**germany_box)
        ds_clime = xr.open_dataset(CLIME_DATA_PATH)
        
        print("Calculating Features...")
        hdd = calculate_weighted_hdd(ds_t)
        
        # Pass both datasets separately to avoid merge conflicts
        anoms = get_climate_anomalies(ds_t, ds_p, ds_clime)
        
        df = gas_data.join(hdd).join(anoms).dropna()
        
        preds, actuals, dates, importance, residuals = build_and_predict(df)
        
        results = pd.DataFrame({
            'Actual': actuals, 
            'Predicted': preds, 
            'Residual': residuals
        }, index=dates)
        results.to_csv(DATA_DIR / 'model_results_q1_2026.csv')
        
        with open(MODELS_DIR / 'feature_importance.json', 'w') as f:
            json.dump(importance, f)
            
        feb_results = results.loc['2026-02-01':'2026-02-28']
        crisis_delta = feb_results['Residual'].mean()
        
        print(f"\n--- Prediction Results ---")
        print(f"February 2026 'Crisis Delta' (Mean Residual): {crisis_delta:.4f} % points")
        print(f"Model RMSE: {np.sqrt(mean_squared_error(actuals, preds)):.4f}")
        print("Results saved to model_results_q1_2026.csv and feature_importance.json")
        
    except Exception as e:
        print(f"Error in modelling workflow: {e}")
        import traceback
        traceback.print_exc()
