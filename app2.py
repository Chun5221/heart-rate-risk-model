# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 09:07:48 2025

@author: chun5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="❤️ Heart Rate Risk Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00b894;
    }
</style>
""", unsafe_allow_html=True)

# Risk calculation data based on meta-analysis
RISK_DATA = {
    'Coronary Heart Disease': {'rr': 1.07, 'ci_lower': 1.05, 'ci_upper': 1.10},
    'Sudden Cardiac Death': {'rr': 1.09, 'ci_lower': 1.00, 'ci_upper': 1.18},
    'Heart Failure': {'rr': 1.18, 'ci_lower': 1.10, 'ci_upper': 1.27},
    'Total Stroke': {'rr': 1.06, 'ci_lower': 1.02, 'ci_upper': 1.10},
    'Cardiovascular Disease': {'rr': 1.15, 'ci_lower': 1.11, 'ci_upper': 1.18},
    'Total Cancer': {'rr': 1.14, 'ci_lower': 1.06, 'ci_upper': 1.23},
    'All-cause Mortality': {'rr': 1.17, 'ci_lower': 1.14, 'ci_upper': 1.19}
}

def calculate_relative_risk(baseline_hr, current_hr, risk_type):
    """Calculate relative risk based on heart rate difference"""
    hr_difference = current_hr - baseline_hr
    hr_increase_per_10 = hr_difference / 10
    
    rr_data = RISK_DATA[risk_type]
    relative_risk = rr_data['rr'] ** hr_increase_per_10
    
    return relative_risk

def get_risk_color(relative_risk):
    """Get color based on risk level"""
    if relative_risk < 1.1:
        return "#27ae60"  # Green
    elif relative_risk < 1.3:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(relative_risk):
    """Get risk level description"""
    if relative_risk < 1.1:
        return "Low Risk"
    elif relative_risk < 1.3:
        return "Moderate Risk"
    else:
        return "High Risk"

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assess your cardiovascular disease risk based on resting heart rate</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Input Parameters")
        
        # Age and gender for context
        age = st.slider("Age", 20, 90, 50, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # Heart rate inputs
        st.markdown("### 💓 Heart Rate Information")
        current_hr = st.slider(
            "Current Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your current resting heart rate in beats per minute"
        )
        
        baseline_hr = st.slider(
            "Baseline/Reference Heart Rate (bpm)", 
            40, 120, 60, 
            help="Reference heart rate (typically 60 bpm for healthy adults)"
        )
        
        # Additional health context
        st.markdown("### 🏥 Health Context")
        has_conditions = st.multiselect(
            "Pre-existing conditions (optional)",
            ["Hypertension", "Diabetes", "Smoking", "Obesity", "Family History of Heart Disease"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Risk Assessment Results")
        
        # Calculate risks for all conditions
        risks = {}
        for condition in RISK_DATA.keys():
            risks[condition] = calculate_relative_risk(baseline_hr, current_hr, condition)
        
        # Create risk visualization
        conditions = list(risks.keys())
        risk_values = list(risks.values())
        colors = [get_risk_color(risk) for risk in risk_values]
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=risk_values,
                marker_color=colors,
                text=[f"{risk:.2f}x" for risk in risk_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Relative Risk: %{y:.2f}x<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Relative Risk by Condition",
            xaxis_title="Health Conditions",
            yaxis_title="Relative Risk",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Baseline Risk (1.0)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level indicators
        st.markdown("### 🎯 Risk Level Summary")
        cols = st.columns(3)
        
        high_risk = sum(1 for risk in risk_values if risk >= 1.3)
        moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.3)
        low_risk = sum(1 for risk in risk_values if risk < 1.1)
        
        with cols[0]:
            st.metric("🔴 High Risk Conditions", high_risk)
        with cols[1]:
            st.metric("🟡 Moderate Risk Conditions", moderate_risk)
        with cols[2]:
            st.metric("🟢 Low Risk Conditions", low_risk)
    
    with col2:
        st.markdown("### 💡 Heart Rate Status")
        
        # Heart rate status card
        hr_diff = current_hr - baseline_hr
        if hr_diff > 0:
            status = f"⬆️ {hr_diff} bpm above baseline"
            status_color = "#e74c3c"
        elif hr_diff < 0:
            status = f"⬇️ {abs(hr_diff)} bpm below baseline"
            status_color = "#27ae60"
        else:
            status = "✅ At baseline"
            status_color = "#27ae60"
        
        st.markdown(f"""
        <div class="risk-card" style="background: {status_color};">
            <h3>Current Status</h3>
            <p style="font-size: 1.2rem; margin: 0;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Normal heart rate ranges
        st.markdown("### 📏 Normal Ranges")
        st.markdown("""
        <div class="info-box">
            <strong>Normal Resting Heart Rate:</strong><br>
            • Adults: 60-100 bpm<br>
            • Athletes: 40-60 bpm<br>
            • Elderly: 60-100 bpm
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed risk breakdown
    st.markdown("### 📋 Detailed Risk Analysis")
    
    # Create detailed table
    risk_df = pd.DataFrame({
        'Condition': conditions,
        'Relative Risk': [f"{risk:.2f}x" for risk in risk_values],
        'Risk Level': [get_risk_level(risk) for risk in risk_values],
        'Confidence Interval': [f"{RISK_DATA[cond]['ci_lower']:.2f} - {RISK_DATA[cond]['ci_upper']:.2f}" 
                               for cond in conditions]
    })
    
    # Color code the table
    def style_risk_level(val):
        if val == "High Risk":
            return 'background-color: #ffebee; color: #c62828'
        elif val == "Moderate Risk":
            return 'background-color: #fff3e0; color: #ef6c00'
        else:
            return 'background-color: #e8f5e8; color: #2e7d32'
    
    styled_df = risk_df.style.applymap(style_risk_level, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Trend analysis
    st.markdown("### 📊 Heart Rate vs Risk Trend")
    
    # Create trend chart
    hr_range = np.arange(40, 121, 5)
    
    fig_trend = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cardiovascular Disease', 'Heart Failure', 'All-cause Mortality', 'Total Cancer'),
        vertical_spacing=0.12
    )
    
    selected_conditions = ['Cardiovascular Disease', 'Heart Failure', 'All-cause Mortality', 'Total Cancer']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, condition in enumerate(selected_conditions):
        row, col = positions[i]
        trend_risks = [calculate_relative_risk(baseline_hr, hr, condition) for hr in hr_range]
        
        fig_trend.add_trace(
            go.Scatter(
                x=hr_range, 
                y=trend_risks,
                mode='lines+markers',
                name=condition,
                line=dict(width=3),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Add current point
        current_risk = calculate_relative_risk(baseline_hr, current_hr, condition)
        fig_trend.add_trace(
            go.Scatter(
                x=[current_hr], 
                y=[current_risk],
                mode='markers',
                name=f"Your {condition} Risk",
                marker=dict(size=12, color='red', symbol='star'),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig_trend.update_layout(
        height=600,
        title_text="Risk Trends Across Heart Rate Range",
        showlegend=False
    )
    
    fig_trend.update_xaxes(title_text="Heart Rate (bpm)")
    fig_trend.update_yaxes(title_text="Relative Risk")
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💊 Recommendations")
    
    max_risk = max(risk_values)
    if max_risk >= 1.3:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ High Risk Detected</strong><br>
            Consider consulting with a healthcare provider about your elevated heart rate. 
            Lifestyle modifications and medical evaluation may be beneficial.
        </div>
        """, unsafe_allow_html=True)
    elif max_risk >= 1.1:
        st.markdown("""
        <div class="warning-box">
            <strong>⚡ Moderate Risk</strong><br>
            Your heart rate is slightly elevated. Consider lifestyle improvements such as 
            regular exercise, stress management, and maintaining a healthy weight.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>✅ Good Heart Rate Range</strong><br>
            Your heart rate appears to be in a healthy range. Continue maintaining 
            a healthy lifestyle with regular exercise and proper nutrition.
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator is for educational purposes only. 
        The results are based on population-level meta-analysis data and should not replace 
        professional medical advice. Always consult with a healthcare provider for personal 
        health assessments and treatment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Risk calculations based on meta-analysis of resting heart rate and cardiovascular disease outcomes</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
