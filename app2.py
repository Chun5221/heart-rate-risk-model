# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 09:07:48 2025
Updateed on Tue Jul 15 09:46 2025
Updateed on Thu Jul 17 13:51 2025

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
    # From paper 1 (baseline 65 bpm)
    'All-cause Mortality': {'rr': 1.4, 'baseline': 65, 'category': 'Mortality'},
    'Cardiovascular Mortality': {'rr': 1.42, 'baseline': 65, 'category': 'Mortality'},
    'Heart Failure Mortality': {'rr': 1.67, 'baseline': 65, 'category': 'Mortality'},
    'Coronary Heart Disease': {'rr': 1.18, 'baseline': 65, 'category': 'Cardiovascular'},
    'Total Stroke': {'rr': 1.32, 'baseline': 65, 'category': 'Cardiovascular'},
    'Hemorrhagic Stroke': {'rr': 1.29, 'baseline': 65, 'category': 'Cardiovascular'},
    'Ischemic Stroke': {'rr': 1.28, 'baseline': 65, 'category': 'Cardiovascular'},
    
    # From paper 2 (baseline 60 bpm)
    'ESRD (End-Stage Renal Disease)': {'rr': 1.14, 'baseline': 60, 'category': 'Kidney'},
    'Diabetes Mortality': {'rr': 1.26, 'baseline': 60, 'category': 'Mortality'},
    'Kidney Disease Mortality': {'rr': 1.24, 'baseline': 60, 'category': 'Mortality'}
}

def calculate_relative_risk(current_hr, risk_type):
    """Calculate relative risk based on heart rate difference from study baseline"""
    rr_data = RISK_DATA[risk_type]
    study_baseline = rr_data['baseline']
    
    # Calculate difference from study baseline
    hr_difference = current_hr - study_baseline
    hr_increase_per_10 = hr_difference / 10
    
    # Calculate relative risk
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
        
        # Show study baselines for reference
        st.markdown("### 📚 Study Reference Points")
        st.info("📊 Study baselines: 65 bpm (cardiovascular outcomes) and 60 bpm (kidney/diabetes outcomes)")
        
        # Risk category filter
        st.markdown("### 🔍 Filter by Category")
        selected_categories = st.multiselect(
            "Select risk categories to display:",
            ["Mortality", "Cardiovascular", "Kidney"],
            default=["Mortality", "Cardiovascular", "Kidney"]
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
        filtered_conditions = [cond for cond, data in RISK_DATA.items() 
                              if data['category'] in selected_categories]
        
        for condition in filtered_conditions:
            risks[condition] = calculate_relative_risk(current_hr, condition)
        
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
        cardiovascular_baseline = 65
        kidney_baseline = 60
        
        cv_diff = current_hr - cardiovascular_baseline
        kidney_diff = current_hr - kidney_baseline
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Status vs Study Baselines</h4>
            <p><strong>Cardiovascular studies:</strong> {cv_diff:+d} bpm from 65 bpm baseline</p>
            <p><strong>Kidney/Diabetes studies:</strong> {kidney_diff:+d} bpm from 60 bpm baseline</p>
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
    
    # Create detailed table with categories
    conditions = list(risks.keys())
    risk_df = pd.DataFrame({
        'Condition': conditions,
        'Category': [RISK_DATA[cond]['category'] for cond in conditions],
        'Study Baseline': [f"{RISK_DATA[cond]['baseline']} bpm" for cond in conditions],
        'Risk per 10 bpm': [f"{RISK_DATA[cond]['rr']:.2f}x" for cond in conditions],
        'Your Relative Risk': [f"{risks[cond]:.2f}x" for cond in conditions],
        'Risk Level': [get_risk_level(risks[cond]) for cond in conditions]
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
    
    # Create trend chart for all conditions in selected categories
    hr_range = np.arange(40, 121, 5)
    
    # Get all conditions for selected categories
    trend_conditions = [cond for cond, data in RISK_DATA.items() 
                       if data['category'] in selected_categories]
    
    if len(trend_conditions) > 0:
        # Calculate number of rows and columns needed
        n_conditions = len(trend_conditions)
        n_cols = min(3, n_conditions)  # Maximum 3 columns
        n_rows = (n_conditions + n_cols - 1) // n_cols  # Ceiling division
        
        fig_trend = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=trend_conditions,
            vertical_spacing=0.25,  # Increased from 0.12
            horizontal_spacing=0.15,  # Increased from 0.08
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        for i, condition in enumerate(trend_conditions):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            trend_risks = [calculate_relative_risk(hr, condition) for hr in hr_range]
            
            # Add baseline reference line
            baseline_hr = RISK_DATA[condition]['baseline']
            fig_trend.add_hline(
                y=1.0, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                row=row, col=col
            )
            
            # Add vertical line for study baseline
            fig_trend.add_vline(
                x=baseline_hr,
                line_dash="dot",
                line_color="blue",
                opacity=0.5,
                row=row, col=col
            )
            
            # Main trend line
            fig_trend.add_trace(
                go.Scatter(
                    x=hr_range, 
                    y=trend_risks,
                    mode='lines+markers',
                    name=condition,
                    line=dict(width=3),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add current point
            current_risk = calculate_relative_risk(current_hr, condition)
            fig_trend.add_trace(
                go.Scatter(
                    x=[current_hr], 
                    y=[current_risk],
                    mode='markers',
                    name=f"Your Risk",
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=False,
                    hovertemplate=f'<b>{condition}</b><br>HR: {current_hr} bpm<br>Risk: {current_risk:.2f}x<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Calculate appropriate height with more generous spacing
        chart_height = max(500, n_rows * 350)  # Increased from 250
        
        fig_trend.update_layout(
            height=chart_height,
            title_text=f"Risk Trends for {', '.join(selected_categories)} Categories",
            showlegend=False,
            font=dict(size=10),
            margin=dict(l=50, r=50, t=80, b=50)  # Added margins
        )
        
        # Update axes for each subplot with more spacing
        for i in range(n_conditions):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            fig_trend.update_xaxes(
                title_text="Heart Rate (bpm)", 
                row=row, col=col,
                title_standoff=20,  # Add space between axis and title
                tickfont=dict(size=9)
            )
            fig_trend.update_yaxes(
                title_text="Relative Risk", 
                row=row, col=col,
                title_standoff=20,  # Add space between axis and title
                tickfont=dict(size=9)
            )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Add legend explanation
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>📖 Chart Legend:</strong><br>
            • <span style="color: gray;">Gray dashed line</span>: Baseline risk (1.0x)<br>
            • <span style="color: blue;">Blue dotted line</span>: Study baseline heart rate<br>
            • <span style="color: red;">Red star</span>: Your current risk level
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("Please select at least one category to view trend analysis.")
    
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
        <em>Risk calculations based on meta-analysis studies:<br>
        • Cardiovascular outcomes: baseline 65 bpm<br>
        • Kidney and diabetes outcomes: baseline 60 bpm</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
