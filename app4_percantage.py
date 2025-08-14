# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:52:58 2025
Updated on Thu Aug 14 16:20 2025

@author: chun5

Heart Rate Risk Percentile Calculator
Shows user's risk percentile within their age/gender group
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# Page configuration
st.set_page_config(
    page_title="❤️ Heart Rate Risk Percentile Calculator",
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
    
    .percentile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .high-risk-card {
        background: linear-gradient(135deg, #ff7675 0%, #e84393 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .moderate-risk-card {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .low-risk-card {
        background: linear-gradient(135deg, #00b894 0%, #55a3ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00b894;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
    }
</style>
""", unsafe_allow_html=True)

# Sample diseases (you can modify this list)
DISEASES = [
    'Hypertension', 'Type 2 Diabetes', 'Atrial Fibrillation', 
    'Heart Failure', 'Ischemic Heart Disease', 'Depression',
    'Anxiety', 'Chronic Kidney Disease', 'Asthma'
]

def get_age_group(age):
    """Categorize age into groups"""
    if age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70+"

def calculate_linear_predictor(age, gender, hr, bmi, smoking_status, drinking_status, disease):
    """
    Mock function to calculate linear predictor
    In your real implementation, this would use actual model coefficients
    """
    # Mock linear predictor calculation
    # Replace this with your actual Cox regression model
    
    base_lp = 0.0
    
    # Age effect (older = higher risk for most diseases)
    base_lp += age * 0.02
    
    # Gender effect (varies by disease)
    if gender == "Female":
        if disease in ["Atrial Fibrillation", "Myocardial Infarction"]:
            base_lp -= 0.5  # Lower risk for women
        else:
            base_lp += 0.1  # Slightly higher risk for some conditions
    
    # Heart rate effect
    if hr < 60:
        base_lp += 0.1
    elif 60 <= hr < 70:
        base_lp += 0.0  # Reference
    elif 70 <= hr < 80:
        base_lp += 0.15
    elif 80 <= hr < 90:
        base_lp += 0.3
    else:  # >= 90
        base_lp += 0.5
    
    # BMI effect
    base_lp += (bmi - 22) * 0.05
    
    # Smoking effect
    if smoking_status == "Former Smoker":
        base_lp += 0.2
    elif smoking_status == "Current Smoker":
        base_lp += 0.4
    
    # Drinking effect
    if drinking_status == "Former Drinker":
        base_lp += 0.1
    elif drinking_status == "Current Drinker":
        base_lp += 0.05
    
    # Add some disease-specific variations
    disease_multipliers = {
        'Hypertension': 1.2,
        'Type 2 Diabetes': 1.1,
        'Atrial Fibrillation': 0.8,
        'Heart Failure': 0.9,
        'Depression': 0.7,
        'Anxiety': 0.6
    }
    
    base_lp *= disease_multipliers.get(disease, 1.0)
    
    # Add some random variation to simulate real data
    np.random.seed(hash(str(age) + gender + str(hr) + disease) % 2**32)
    base_lp += np.random.normal(0, 0.1)
    
    return base_lp

def get_percentile_in_group(user_lp, age, gender, disease):
    """
    Mock function to calculate percentile within age/gender group
    In your real implementation, this would lookup from your LP distribution data
    """
    age_group = get_age_group(age)
    
    # Mock: Generate sample distribution for this age/gender/disease group
    # This simulates the LP distribution you'll provide later
    np.random.seed(hash(age_group + gender + disease) % 2**32)
    
    # Create a mock distribution of linear predictors for this group
    n_samples = 10000
    if gender == "Male":
        group_lps = np.random.normal(-0.2, 0.8, n_samples)
    else:
        group_lps = np.random.normal(-0.1, 0.75, n_samples)
    
    # Age adjustment for the group distribution
    age_mid = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70+": 75}[age_group]
    group_lps += (age_mid - 45) * 0.01
    
    # Calculate percentile
    percentile = (np.sum(group_lps < user_lp) / len(group_lps)) * 100
    
    return min(99.9, max(0.1, percentile))

def get_risk_interpretation(percentile):
    """Get risk level and color based on percentile"""
    if percentile >= 90:
        return "Very High Risk", "#e74c3c", "⚠️"
    elif percentile >= 75:
        return "High Risk", "#e67e22", "🔶"
    elif percentile >= 50:
        return "Moderate Risk", "#f39c12", "🟡"
    elif percentile >= 25:
        return "Low-Moderate Risk", "#27ae60", "🟢"
    else:
        return "Low Risk", "#2ecc71", "✅"

def categorize_diseases(disease_name):
    """Categorize diseases for filtering"""
    cardiovascular = ['Hypertension', 'Atrial Fibrillation', 'Heart Failure', 'Ischemic Heart Disease']
    metabolic = ['Type 2 Diabetes', 'Chronic Kidney Disease']
    mental_health = ['Depression', 'Anxiety']
    respiratory = ['Asthma']
    
    if disease_name in cardiovascular:
        return 'Cardiovascular'
    elif disease_name in metabolic:
        return 'Metabolic'
    elif disease_name in mental_health:
        return 'Mental Health'
    elif disease_name in respiratory:
        return 'Respiratory'
    else:
        return 'Other'

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Percentile Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your risk percentile within your age and gender group</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Your Information")
        
        # Personal information
        age = st.slider("Age", 20, 80, 45, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        bmi = st.slider("BMI", 15.0, 40.0, 24.0, step=0.1, help="Body Mass Index")
        
        # Heart rate
        st.markdown("### 💓 Heart Rate")
        current_hr = st.slider(
            "Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your resting heart rate in beats per minute"
        )
        
        # Lifestyle factors
        st.markdown("### 🚬 Lifestyle")
        smoking_status = st.selectbox(
            "Smoking Status", 
            ["Never Smoker", "Former Smoker", "Current Smoker"]
        )
        
        drinking_status = st.selectbox(
            "Drinking Status",
            ["Never Drinker", "Former Drinker", "Current Drinker"]
        )
        
        # Disease selection
        st.markdown("### 🎯 Focus Diseases")
        selected_diseases = st.multiselect(
            "Select diseases to analyze:",
            DISEASES,
            default=['Hypertension', 'Type 2 Diabetes', 'Heart Failure', 'Depression'],
            help="Choose which diseases to assess your risk for"
        )
    
    # Main content area
    if not selected_diseases:
        st.warning("Please select at least one disease to analyze.")
        return
    
    # Calculate percentiles for selected diseases
    results = []
    for disease in selected_diseases:
        user_lp = calculate_linear_predictor(
            age, gender, current_hr, bmi, smoking_status, drinking_status, disease
        )
        percentile = get_percentile_in_group(user_lp, age, gender, disease)
        risk_level, color, emoji = get_risk_interpretation(percentile)
        
        results.append({
            'disease': disease,
            'percentile': percentile,
            'risk_level': risk_level,
            'color': color,
            'emoji': emoji,
            'category': categorize_diseases(disease)
        })
    
    # Sort results by percentile (highest first)
    results.sort(key=lambda x: x['percentile'], reverse=True)
    
    # Display results
    age_group = get_age_group(age)
    st.markdown(f"### 📈 Your Risk Profile Among {gender}s Aged {age_group}")
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    high_risk_count = sum(1 for r in results if r['percentile'] >= 75)
    moderate_risk_count = sum(1 for r in results if 50 <= r['percentile'] < 75)
    low_risk_count = sum(1 for r in results if r['percentile'] < 50)
    
    with col1:
        st.markdown(f"""
        <div class="high-risk-card">
            <h3>🔴 High Risk</h3>
            <h2>{high_risk_count}</h2>
            <p>conditions ≥75th percentile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="moderate-risk-card">
            <h3>🟡 Moderate Risk</h3>
            <h2>{moderate_risk_count}</h2>
            <p>conditions 50-74th percentile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="low-risk-card">
            <h3>🟢 Lower Risk</h3>
            <h2>{low_risk_count}</h2>
            <p>conditions <50th percentile</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    st.markdown("### 📋 Detailed Risk Assessment")
    
    # Create visualization
    diseases_list = [r['disease'] for r in results]
    percentiles = [r['percentile'] for r in results]
    colors = [r['color'] for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            y=diseases_list,
            x=percentiles,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentiles],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Percentile: %{x:.1f}%<br>Among %{customdata}<extra></extra>',
            customdata=[f"{gender}s aged {age_group}"] * len(diseases_list)
        )
    ])
    
    fig.update_layout(
        title=f"Your Risk Percentiles Among {gender}s Aged {age_group}",
        xaxis_title="Percentile Within Your Age/Gender Group",
        yaxis_title="Health Conditions",
        height=max(400, len(diseases_list) * 40),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add reference lines
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Average (50th percentile)")
    fig.add_vline(x=75, line_dash="dash", line_color="orange", annotation_text="High Risk (75th percentile)")
    fig.add_vline(x=90, line_dash="dash", line_color="red", annotation_text="Very High Risk (90th percentile)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual disease cards
    st.markdown("### 📊 Individual Risk Analysis")
    
    for i, result in enumerate(results):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            percentile = result['percentile']
            if percentile >= 90:
                card_class = "high-risk-card"
                interpretation = f"You are in the **top {100-percentile:.1f}%** of {gender.lower()}s aged {age_group} for {result['disease']} risk."
                action = "⚠️ **Strongly consider** consulting with a healthcare provider about prevention strategies."
            elif percentile >= 75:
                card_class = "moderate-risk-card"
                interpretation = f"You are in the **top {100-percentile:.1f}%** of {gender.lower()}s aged {age_group} for {result['disease']} risk."
                action = "🔶 **Consider** discussing this with your healthcare provider during routine visits."
            elif percentile >= 50:
                card_class = "percentile-card"
                interpretation = f"Your {result['disease']} risk is **above average** (top {100-percentile:.1f}%) among {gender.lower()}s aged {age_group}."
                action = "🟡 **Monitor** this condition and maintain healthy lifestyle habits."
            else:
                card_class = "low-risk-card"
                interpretation = f"Your {result['disease']} risk is **below average** ({percentile:.1f}th percentile) among {gender.lower()}s aged {age_group}."
                action = "✅ **Continue** your current healthy practices."
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4>{result['emoji']} {result['disease']}</h4>
                <h3>{percentile:.1f}th Percentile</h3>
                <p>{interpretation}</p>
                <p><em>{action}</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mini gauge chart for each disease
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentile,
                title={'text': f"{result['disease']}<br>Percentile"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': result['color']},
                    'steps': [
                        {'range': [0, 25], 'color': "#d5f4e6"},
                        {'range': [25, 50], 'color': "#ffeeb3"},
                        {'range': [50, 75], 'color': "#ffccb3"},
                        {'range': [75, 100], 'color': "#ffb3b3"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Personalized recommendations
    st.markdown("### 💡 Personalized Recommendations")
    
    highest_risk = results[0] if results else None
    
    if highest_risk and highest_risk['percentile'] >= 90:
        st.markdown(f"""
        <div class="warning-box">
            <strong>🚨 Priority Alert:</strong> Your highest risk is for <strong>{highest_risk['disease']}</strong> 
            (top {100-highest_risk['percentile']:.1f}% among {gender.lower()}s aged {age_group}).<br><br>
            <strong>Immediate Actions:</strong><br>
            • Schedule an appointment with your healthcare provider<br>
            • Discuss targeted screening and prevention strategies<br>
            • Focus on lifestyle modifications specific to this condition
        </div>
        """, unsafe_allow_html=True)
    elif highest_risk and highest_risk['percentile'] >= 75:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚡ Health Focus Area:</strong> Your highest risk is for <strong>{highest_risk['disease']}</strong> 
            (top {100-highest_risk['percentile']:.1f}% among {gender.lower()}s aged {age_group}).<br><br>
            <strong>Recommended Actions:</strong><br>
            • Bring this up during your next healthcare visit<br>
            • Consider targeted lifestyle improvements<br>
            • Monitor relevant health markers regularly
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>✅ Overall Good Profile:</strong> Your risk profile is generally favorable compared to peers in your age/gender group.<br><br>
            <strong>Maintain Your Health:</strong><br>
            • Continue your current healthy lifestyle<br>
            • Keep up with routine preventive care<br>
            • Stay informed about age-appropriate health screenings
        </div>
        """, unsafe_allow_html=True)
    
    # Heart rate specific advice
    if current_hr >= 85:
        st.markdown("""
        <div class="warning-box">
            <strong>💓 Heart Rate Focus:</strong> Your resting heart rate is elevated. This may be contributing to higher risk percentiles.<br>
            <strong>Consider:</strong> Regular cardio exercise, stress management, adequate sleep, and medical evaluation if persistently high.
        </div>
        """, unsafe_allow_html=True)
    
    # Model explanation
    with st.expander("📚 Understanding Your Results"):
        st.markdown(f"""
        ### How Percentiles Work
        
        **Your Comparison Group:** {gender}s aged {age_group}
        
        **What Percentiles Mean:**
        - **50th percentile** = Average risk in your group
        - **75th percentile** = Higher risk than 75% of your peers
        - **90th percentile** = Higher risk than 90% of your peers (top 10%)
        - **25th percentile** = Lower risk than 75% of your peers
        
        **Risk Categories:**
        - 🔴 **Very High (≥90th percentile):** Top 10% - Consider immediate medical consultation
        - 🟠 **High (75-89th percentile):** Top 11-25% - Discuss with healthcare provider
        - 🟡 **Moderate (50-74th percentile):** Above average - Monitor and maintain healthy habits
        - 🟢 **Low-Moderate (25-49th percentile):** Below average - Continue current practices
        - ✅ **Low (<25th percentile):** Bottom 25% - Excellent relative risk profile
        
        **Factors Considered:**
        - Age, gender, heart rate, BMI, smoking status, drinking status
        
        **Important Notes:**
        - This shows your risk relative to peers, not absolute risk
        - Multiple factors contribute to disease risk beyond those measured
        - Results are for educational purposes and don't replace medical advice
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator shows your risk percentile within your age/gender group 
        for educational purposes only. Results are based on population models and should not replace professional 
        medical advice. Individual risk factors not included in the model may significantly affect your actual risk. 
        Always consult healthcare providers for personal health assessments.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
