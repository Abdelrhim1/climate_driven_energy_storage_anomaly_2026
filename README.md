# German Gas Storage Forecasting and Anomaly Detection

## Overview

This project models German natural gas storage drawdown for Q1 2026 using a hybrid machine-learning approach that integrates climatic indicators and energy market variables.

The system predicts expected storage levels and identifies abnormal deviations through residual analysis.

The project demonstrates how weather-driven demand signals and market dynamics influence storage behavior.

---

## Key Features

* Climate-aware gas storage forecasting
* Integration of ERA5-Land climate data
* Industrial Heating Degree Day (HDD) weighting
* Lagged feature modeling
* XGBoost regression model
* Residual-based anomaly detection
* Interactive Plotly dashboard
* Automated LaTeX reporting

---

## Data Sources

### Climate Data

* ERA5-Land reanalysis dataset
* 2m temperature
* precipitation anomalies

### Energy Market Data

* Natural gas prices (EUR/MWh)
* historical storage levels

---

## Feature Engineering

The model uses several engineered features to capture energy demand dynamics:

* Temperature anomaly
* Heating Degree Days (HDD)
* Industrial HDD weighting
* Precipitation anomaly
* Gas price signals
* Lagged temporal indicators

These variables allow the model to capture both weather-driven consumption and market-driven storage strategies.

---

## Model

Due to TensorFlow environment limitations, the project pivoted from an LSTM architecture to a robust lagged feature XGBoost regression model.

The final model achieved:

RMSE ≈ 0.0045

This approach maintains strong predictive performance while simplifying deployment.

---

## Anomaly Detection

Anomalies are detected using statistical residual monitoring.

Steps:

1. Predict expected storage levels
2. Compute prediction residuals
3. Apply a 5σ statistical threshold
4. Flag abnormal deviations

This allows early detection of unusual storage behavior.

---

## Dashboard

The interactive dashboard includes:

* Actual vs predicted storage levels
* Residual monitoring
* Anomaly detection markers
* Feature importance visualization

---

## Project Structure

src/ – modeling and dashboard scripts
models/ – trained XGBoost model
data/ – climate and storage datasets
reports/ – LaTeX report and outputs

---

## Technologies

Python
XGBoost
Pandas
Plotly
Scikit-learn
ERA5 climate data
LaTeX reporting

---

## Applications

Energy storage monitoring

Energy market analytics

Climate-driven demand forecasting

## Documentation

A full technical description of the methodology, validation results, and feature importance analysis is available in the project report:

report.pdf
