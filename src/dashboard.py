import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path

# Dynamic Path Discovery for Reorganized Structure
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

def create_dashboard(results_file: str, importance_file: str, output_file: str):
    """
    Generate an interactive validation dashboard with 5σ anomaly detection.
    """
    if not os.path.exists(results_file) or not os.path.exists(importance_file):
        print("Data files not found. Dashboard creation skipped.")
        return

    # Load Results
    df = pd.read_csv(results_file, parse_dates=['Date'])
    
    # Load Importance
    with open(importance_file, 'r') as f:
        importance = json.load(f)
    
    # Calculate 5σ Anomaly
    resid_mean = df['Residual'].mean()
    resid_std = df['Residual'].std()
    threshold = 5 * resid_std
    
    df['Anomaly_5s'] = (df['Residual'].abs() > threshold)
    anomalies = df[df['Anomaly_5s']]
    
    # Create Dashboard
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Actual vs Predicted Storage Level", "Residuals and 5σ Anomalies", "Feature Importance"),
        vertical_spacing=0.1,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "bar"}]]
    )
    
    # 1. Actual vs Predicted
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], name='Actual', line=dict(color='#00d1b2', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], name='Predicted', line=dict(color='#ff3860', width=3, dash='dot')), row=1, col=1)
    
    # 2. Residuals and 5σ
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Residual'], name='Residuals', line=dict(color='#3273dc', width=2)), row=2, col=1)
    # 5s Bands
    fig.add_hline(y=threshold, line_dash="dash", line_color="black", annotation_text="5σ Upper", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="black", annotation_text="5σ Lower", row=2, col=1)
    
    # Mark Anomalies
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Date'], y=anomalies['Residual'],
            mode='markers', name='5σ Anomaly',
            marker=dict(color='orange', size=15, symbol='star', line=dict(color='black', width=1.5))
        ), row=2, col=1)
        
    # 3. Feature Importance
    sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    fig.add_trace(go.Bar(
        x=list(sorted_importance.keys()), 
        y=list(sorted_importance.values()),
        marker_color='#485fc7',
        name='Importance'
    ), row=3, col=1)
    
    # Premium Light Styling
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#FFFFFF",
        height=1400,
        title={
            'text': "Energy Storage Validation Dashboard - Q1 2026",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 32, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        font=dict(family="Verdana, sans-serif", size=14, color="#2C3E50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#BDC3C7",
            borderwidth=1
        ),
        margin=dict(t=150, b=100, l=80, r=80),
    )
    
    # Explicitly update all axes
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor='#7F8C8D',
        gridcolor='#ECF0F1', tickfont=dict(color='#2C3E50')
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='#7F8C8D',
        gridcolor='#ECF0F1', tickfont=dict(color='#2C3E50')
    )
    
    # Force subplot titles to be dark and visible
    fig.update_annotations(font=dict(color='#2C3E50', size=22, family='Arial Black'))

    fig.write_html(output_file, include_plotlyjs='cdn', config={'responsive': True})
    print(f"Premium light-mode dashboard generated: {output_file}")

if __name__ == "__main__":
    RESULTS = DATA_DIR / 'model_results_q1_2026.csv'
    IMPORTANCE = MODELS_DIR / 'feature_importance.json'
    OUTPUT = REPORTS_DIR / 'validation_dashboard.html'
    
    create_dashboard(str(RESULTS), str(IMPORTANCE), str(OUTPUT))
