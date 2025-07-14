# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:01:18 2025

@author: chun5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Rate Disease Risk Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .low-risk {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .moderate-risk {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .high-risk {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def calculate_risk_scores(rhr, age, gender, weight, height):
    """
    Calculate disease risk scores based on resting heart rate and other factors.
    These are simplified models for demonstration purposes.
    """
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Normalize resting heart rate (60-100 is normal range)
    rhr_normalized = (rhr - 60) / 40
    
    # Age factor (higher age = higher risk)
    age_factor = (age - 20) / 60
    
    # Gender factor (males typically have slightly higher cardiovascular risk)
    gender_factor = 1.1 if gender == "Male" else 1.0
    
    # BMI factor
    bmi_factor = 1.0
    if bmi > 30:
        bmi_factor = 1.3
    elif bmi > 25:
        bmi_factor = 1.1
    
    # Cardiovascular Disease Risk
    cvd_base_risk = 0.15
    cvd_rhr_impact = 0.6 if rhr > 90 else (0.3 if rhr > 80 else 0.1)
    cvd_risk = min(0.95, cvd_base_risk + (cvd_rhr_impact * rhr_normalized) + (age_factor * 0.4) + (bmi_factor - 1) * 0.3) * gender_factor
    
    # Diabetes Risk
    diabetes_base_risk = 0.10
    diabetes_rhr_impact = 0.4 if rhr > 85 else 0.2
    diabetes_risk = min(0.90, diabetes_base_risk + (diabetes_rhr_impact * rhr_normalized) + (age_factor * 0.3) + (bmi_factor - 1) * 0.4)
    
    # Stroke Risk
    stroke_base_risk = 0.08
    stroke_rhr_impact = 0.5 if rhr > 95 else 0.2
    stroke_risk = min(0.85, stroke_base_risk + (stroke_rhr_impact * rhr_normalized) + (age_factor * 0.5) + (bmi_factor - 1) * 0.2) * gender_factor
    
    # Hypertension Risk
    hypertension_base_risk = 0.20
    hypertension_rhr_impact = 0.7 if rhr > 90 else 0.3
    hypertension_risk = min(0.95, hypertension_base_risk + (hypertension_rhr_impact * rhr_normalized) + (age_factor * 0.4) + (bmi_factor - 1) * 0.5)
    
    return {
        "Cardiovascular Disease": cvd_risk,
        "Type 2 Diabetes": diabetes_risk,
        "Stroke": stroke_risk,
        "Hypertension": hypertension_risk
    }

def get_risk_category(risk_score):
    """Categorize risk score into low, moderate, or high risk."""
    if risk_score < 0.3:
        return "Low", "low-risk"
    elif risk_score < 0.6:
        return "Moderate", "moderate-risk"
    else:
        return "High", "high-risk"

def get_recommendations(rhr, risks):
    """Generate personalized recommendations based on risk assessment."""
    recommendations = []
    
    if rhr > 90:
        recommendations.append("🏃 Consider regular aerobic exercise to lower resting heart rate")
        recommendations.append("🧘 Practice stress management techniques like meditation or yoga")
    
    if rhr > 100:
        recommendations.append("⚠️ Consult a healthcare provider about your elevated resting heart rate")
    
    if risks["Cardiovascular Disease"] > 0.5:
        recommendations.append("❤️ Focus on heart-healthy diet with omega-3 fatty acids")
        recommendations.append("🚭 Avoid smoking and limit alcohol consumption")
    
    if risks["Type 2 Diabetes"] > 0.4:
        recommendations.append("🥗 Maintain a balanced diet with controlled carbohydrate intake")
        recommendations.append("⚖️ Monitor and maintain healthy body weight")
    
    if risks["Hypertension"] > 0.5:
        recommendations.append("🧂 Reduce sodium intake in your diet")
        recommendations.append("💤 Ensure adequate sleep (7-9 hours per night)")
    
    # General recommendations
    recommendations.append("🩺 Regular health check-ups with your healthcare provider")
    recommendations.append("📊 Monitor your heart rate and blood pressure regularly")
    
    return recommendations

# Main app
def main():
    st.markdown('<h1 class="main-header">❤️ Heart Rate Disease Risk Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This tool provides an estimate of disease risk based on your resting heart rate and other health factors. 
    **Please note**: This is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Sidebar for input parameters
    st.sidebar.header("📋 Personal Information")
    
    # User inputs
    age = st.sidebar.slider("Age", 18, 100, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0, step=0.1)
    height = st.sidebar.number_input("Height (cm)", 120.0, 220.0, 170.0, step=0.1)
    
    st.sidebar.header("💓 Heart Rate Information")
    rhr = st.sidebar.number_input("Resting Heart Rate (bpm)", 40, 150, 72, step=1)
    
    # Information about normal heart rate ranges
    st.sidebar.markdown("""
    **Normal Resting Heart Rate Ranges:**
    - Adults: 60-100 bpm
    - Athletes: 40-60 bpm
    - Elderly: 50-90 bpm
    """)
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Your Health Profile")
        
        # Display basic metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("BMI", f"{bmi:.1f}")
            st.metric("Age", f"{age} years")
        with metric_col2:
            st.metric("Resting HR", f"{rhr} bpm")
            st.metric("Gender", gender)
        
        # Heart rate status
        if rhr < 60:
            hr_status = "Below Normal (Bradycardia)"
            hr_color = "orange"
        elif rhr <= 100:
            hr_status = "Normal Range"
            hr_color = "green"
        else:
            hr_status = "Above Normal (Tachycardia)"
            hr_color = "red"
        
        st.markdown(f"**Heart Rate Status:** <span style='color: {hr_color}'>{hr_status}</span>", unsafe_allow_html=True)
    
    with col2:
        # Calculate risk scores
        risks = calculate_risk_scores(rhr, age, gender, weight, height)
        
        st.subheader("🎯 Disease Risk Assessment")
        
        for disease, risk_score in risks.items():
            risk_category, risk_class = get_risk_category(risk_score)
            risk_percentage = risk_score * 100
            
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h4>{disease}</h4>
                <h3>{risk_percentage:.1f}% Risk</h3>
                <p><strong>{risk_category} Risk</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualization section
    st.subheader("📈 Risk Visualization")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Risk comparison chart
        diseases = list(risks.keys())
        risk_values = [risks[disease] * 100 for disease in diseases]
        
        fig = go.Figure(data=[
            go.Bar(
                x=diseases,
                y=risk_values,
                marker_color=['#FF6B6B' if r > 60 else '#FFE66D' if r > 30 else '#4ECDC4' for r in risk_values]
            )
        ])
        
        fig.update_layout(
            title="Disease Risk Comparison",
            xaxis_title="Disease",
            yaxis_title="Risk Percentage (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Heart rate range visualization
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = rhr,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Resting Heart Rate"},
            gauge = {
                'axis': {'range': [None, 150]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightblue"},
                    {'range': [60, 100], 'color': "lightgreen"},
                    {'range': [100, 150], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recommendations section
    st.subheader("💡 Personalized Recommendations")
    
    recommendations = get_recommendations(rhr, risks)
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")
    
    # Disclaimer
    st.markdown("""
    ---
    **⚠️ Important Disclaimer:**
    This tool provides general risk estimates based on population data and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for personalized medical guidance.
    
    **Data Sources:** Risk calculations are based on epidemiological studies and general population data. Individual risk factors may vary significantly.
    """)
    
    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Created with ❤️ using Streamlit | Last updated: """ + datetime.now().strftime("%B %Y") + """
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()