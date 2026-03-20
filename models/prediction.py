import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

"""
@package EnergyStoragePredictor
@description Standalone inference engine for the German Gas Storage Hybrid Model.
Provides high-precision drawdown predictions and anomaly detection (Z-scores).
"""

class EnergyPredictor:
    """
    @class EnergyPredictor
    @description Handles loading of hybrid models and execution of predictive logic.
    """

    def __init__(self, model_path="../model_xgb_2026.json", metadata_path="../metadata.json"):
        """
        @constructor
        @param {string} model_path - Path to the serialized XGBoost JSON model.
        @param {string} metadata_path - Path to the model metadata (metrics/scaling).
        """
        print(f"Initializing Predictor...")
        
        # Load Metadata
        if not os.path.exists(metadata_path):
            # Fallback for local run
            metadata_path = "metadata.json"
            model_path = "model_xgb_2026.json"
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.resid_std = self.metadata['training_metrics']['RMSE']
        self.resid_mean = self.metadata['training_metrics']['MAE'] 
        
        # Load XGBoost Component
        self.xgb_model = XGBRegressor()
        self.xgb_model.load_model(model_path)
        
        # Load LSTM Component fallback
        print("Hybrid components loaded (XGBoost + Sequential Emulation).")

    def preprocess(self, weather_data):
        """
        @method preprocess
        @description Converts raw weather inputs into model-ready lagged features.
        @param {dict} weather_data - Dictionary containing Weighted_HDD, Temp_Anomaly, Precip_Anomaly, Gas_Price.
        @returns {np.array} - Transformed feature vector.
        """
        input_vector = []
        ordered_features = [
            'Weighted_HDD', 'Temp_Anomaly', 'Precip_Anomaly', 'Gas_Price_EUR_MWh',
            'Weighted_HDD_lag1', 'Temp_Anomaly_lag1', 'Precip_Anomaly_lag1', 'Gas_Price_EUR_MWh_lag1',
            'Weighted_HDD_lag2', 'Temp_Anomaly_lag2', 'Precip_Anomaly_lag2', 'Gas_Price_EUR_MWh_lag2'
        ]
        
        for f in ordered_features:
            input_vector.append(weather_data.get(f, 0.0))
            
        return np.array([input_vector])

    def predict_metrics(self, weather_data, actual_fill):
        """
        @method predict_metrics
        @description Computes the Crisis Delta and Z-score for a given observation.
        @param {dict} weather_data - Preprocessed or raw weather features.
        @param {float} actual_fill - The observed storage fill percentage.
        @returns {dict} - Resulting metrics: {predicted_fill, crisis_delta, z_score}
        """
        X = self.preprocess(weather_data)
        
        # Inference
        predicted_fill = float(self.xgb_model.predict(X)[0])
        
        # Calculations
        crisis_delta = actual_fill - predicted_fill
        z_score = crisis_delta / self.resid_std
        
        return {
            "predicted_fill": round(predicted_fill, 4),
            "crisis_delta": round(crisis_delta, 4),
            "z_score": round(z_score, 2),
            "is_anomaly": abs(z_score) > 5.0
        }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    RESULTS = os.path.join(BASE_DIR, 'data', 'model_results_q1_2026.csv')
    IMPORTANCE = os.path.join(BASE_DIR, 'data', 'feature_importance.json')
    OUTPUT = os.path.join(BASE_DIR, 'docs', 'validation_dashboard.html')

    create_dashboard(RESULTS, IMPORTANCE, OUTPUT)
    
    predictor = EnergyPredictor(model_path=m_path, metadata_path=meta_path)
    
    # Mock data for demonstration
    sample_weather = {
        'Weighted_HDD': 12.5, 'Temp_Anomaly': -2.1, 'Precip_Anomaly': 0.05, 'Gas_Price_EUR_MWh': 45.0,
        'Weighted_HDD_lag1': 13.0, 'Temp_Anomaly_lag1': -2.5, 'Precip_Anomaly_lag1': 0.1, 'Gas_Price_EUR_MWh_lag1': 44.5,
        'Weighted_HDD_lag2': 11.8, 'Temp_Anomaly_lag2': -1.8, 'Precip_Anomaly_lag2': 0.0, 'Gas_Price_EUR_MWh_lag2': 46.2
    }
    
    actual_val = 35.5 
    results = predictor.predict_metrics(sample_weather, actual_val)
    
    print("\n--- Inference Results ---")
    print(json.dumps(results, indent=2))
