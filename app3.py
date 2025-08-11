# -*- coding: utf-8 -*-
"""
Updated Heart Rate Risk Calculator using Complete TWB Model Data
Created on Tue Jul 15 09:07:48 2025
Updated on Thu Jul 17 13:51 2025
Updated on Thu Aug 11 10:55 2025 with Complete TWB Model Data (HR + Demographics + Lifestyle)

@author: chun5
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
    page_title="❤️ Comprehensive Health Risk Calculator - TWB Model",
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
    
    .factor-box {
        background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #8b5cf6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_twb_data():
    """Load and process complete TWB model data"""
    twb_data = {
        'Atrial Fibrillation': {
            # Heart Rate Categories
            'HR_cat<60': 1.278584, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.007198,
            'HR_cat80-89': 1.329859, 'HR_cat>=90': 1.870907,
            # Demographics
            'AGE': 1.097335, 'MALE': 1.0, 'FEMALE': 0.43779, 'BMI': 1.054307,
            # Lifestyle
            'Never_smoke': 1.0, 'Ever_smoke': 1.041755, 'Now_smoke': 0.796538,
            'Never_drink': 1.0, 'Ever_drink': 1.120964, 'Now_drink': 1.082538,
            'category': 'Cardiovascular'
        },
        'Anxiety': {
            'HR_cat<60': 0.9853143, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.0337353,
            'HR_cat80-89': 1.0830412, 'HR_cat>=90': 1.0332187,
            'AGE': 1.0224173, 'MALE': 1.0, 'FEMALE': 1.7524471, 'BMI': 0.9789276,
            'Never_smoke': 1.0, 'Ever_smoke': 1.1469517, 'Now_smoke': 1.2107771,
            'Never_drink': 1.0, 'Ever_drink': 1.2639205, 'Now_drink': 1.1566027,
            'category': 'Mental Health'
        },
        'Chronic Kidney Disease': {
            'HR_cat<60': 0.962877, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.120832,
            'HR_cat80-89': 1.344067, 'HR_cat>=90': 2.055936,
            'AGE': 1.070072, 'MALE': 1.0, 'FEMALE': 0.714564, 'BMI': 1.085128,
            'Never_smoke': 1.0, 'Ever_smoke': 1.035606, 'Now_smoke': 1.035984,
            'Never_drink': 1.0, 'Ever_drink': 1.327176, 'Now_drink': 0.906842,
            'category': 'Kidney'
        },
        'GERD': {
            'HR_cat<60': 1.0175125, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1123568,
            'HR_cat80-89': 1.1324623, 'HR_cat>=90': 1.2009102,
            'AGE': 1.0111128, 'MALE': 1.0, 'FEMALE': 1.1882805, 'BMI': 1.0031746,
            'Never_smoke': 1.0, 'Ever_smoke': 1.1259326, 'Now_smoke': 0.9475566,
            'Never_drink': 1.0, 'Ever_drink': 1.0541823, 'Now_drink': 1.0703914,
            'category': 'Digestive'
        },
        'Heart Failure': {
            'HR_cat<60': 1.099216, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.047627,
            'HR_cat80-89': 1.324929, 'HR_cat>=90': 1.384662,
            'AGE': 1.080752, 'MALE': 1.0, 'FEMALE': 0.948374, 'BMI': 1.113224,
            'Never_smoke': 1.0, 'Ever_smoke': 1.165321, 'Now_smoke': 1.289472,
            'Never_drink': 1.0, 'Ever_drink': 1.136773, 'Now_drink': 0.980753,
            'category': 'Cardiovascular'
        },
        'Myocardial Infarction': {
            'HR_cat<60': 1.305421, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.097465,
            'HR_cat80-89': 1.15584, 'HR_cat>=90': 1.156146,
            'AGE': 1.082279, 'MALE': 1.0, 'FEMALE': 0.355145, 'BMI': 1.092264,
            'Never_smoke': 1.0, 'Ever_smoke': 1.557379, 'Now_smoke': 2.347051,
            'Never_drink': 1.0, 'Ever_drink': 1.266607, 'Now_drink': 0.690009,
            'category': 'Cardiovascular'
        },
        'Type 2 Diabetes': {
            'HR_cat<60': 0.891278, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.268673,
            'HR_cat80-89': 1.65573, 'HR_cat>=90': 2.01645,
            'AGE': 1.059158, 'MALE': 1.0, 'FEMALE': 1.098451, 'BMI': 1.118891,
            'Never_smoke': 1.0, 'Ever_smoke': 1.145229, 'Now_smoke': 1.203484,
            'Never_drink': 1.0, 'Ever_drink': 1.183832, 'Now_drink': 1.130815,
            'category': 'Metabolic'
        },
        'Anemia': {
            'HR_cat<60': 0.9614963, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1027278,
            'HR_cat80-89': 1.2191359, 'HR_cat>=90': 1.0937225,
            'AGE': 0.9938358, 'MALE': 1.0, 'FEMALE': 3.0926062, 'BMI': 0.9927205,
            'Never_smoke': 1.0, 'Ever_smoke': 1.2353906, 'Now_smoke': 0.9478152,
            'Never_drink': 1.0, 'Ever_drink': 1.1823343, 'Now_drink': 0.8406682,
            'category': 'Blood'
        },
        'Angina Pectoris': {
            'HR_cat<60': 1.154145, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.930245,
            'HR_cat80-89': 1.032199, 'HR_cat>=90': 0.907764,
            'AGE': 1.058539, 'MALE': 1.0, 'FEMALE': 0.831779, 'BMI': 1.057477,
            'Never_smoke': 1.0, 'Ever_smoke': 1.227097, 'Now_smoke': 1.229498,
            'Never_drink': 1.0, 'Ever_drink': 1.212417, 'Now_drink': 0.994733,
            'category': 'Cardiovascular'
        },
        'Asthma': {
            'HR_cat<60': 0.898631, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.022859,
            'HR_cat80-89': 1.055431, 'HR_cat>=90': 1.280111,
            'AGE': 1.018948, 'MALE': 1.0, 'FEMALE': 1.417606, 'BMI': 1.052391,
            'Never_smoke': 1.0, 'Ever_smoke': 1.198245, 'Now_smoke': 1.171488,
            'Never_drink': 1.0, 'Ever_drink': 1.05014, 'Now_drink': 0.927569,
            'category': 'Respiratory'
        },
        'Atherosclerosis': {
            'HR_cat<60': 1.154199, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.954146,
            'HR_cat80-89': 0.989229, 'HR_cat>=90': 0.824199,
            'AGE': 1.079968, 'MALE': 1.0, 'FEMALE': 0.622169, 'BMI': 1.076141,
            'Never_smoke': 1.0, 'Ever_smoke': 1.244718, 'Now_smoke': 1.220403,
            'Never_drink': 1.0, 'Ever_drink': 1.284505, 'Now_drink': 1.050576,
            'category': 'Cardiovascular'
        },
        'Cardiac Arrhythmia': {
            'HR_cat<60': 1.198335, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.069528,
            'HR_cat80-89': 1.25314, 'HR_cat>=90': 1.516303,
            'AGE': 1.034404, 'MALE': 1.0, 'FEMALE': 1.221919, 'BMI': 1.003799,
            'Never_smoke': 1.0, 'Ever_smoke': 1.063968, 'Now_smoke': 0.942808,
            'Never_drink': 1.0, 'Ever_drink': 1.164741, 'Now_drink': 1.056686,
            'category': 'Cardiovascular'
        },
        'Depression': {
            'HR_cat<60': 1.1269417, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1207691,
            'HR_cat80-89': 1.4019456, 'HR_cat>=90': 1.7691724,
            'AGE': 1.0124351, 'MALE': 1.0, 'FEMALE': 1.8196446, 'BMI': 0.9903191,
            'Never_smoke': 1.0, 'Ever_smoke': 1.3618666, 'Now_smoke': 1.749551,
            'Never_drink': 1.0, 'Ever_drink': 1.4718567, 'Now_drink': 0.9993047,
            'category': 'Mental Health'
        },
        'Hypertension': {
            'HR_cat<60': 0.943739, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.21525,
            'HR_cat80-89': 1.478731, 'HR_cat>=90': 1.908702,
            'AGE': 1.060353, 'MALE': 1.0, 'FEMALE': 0.89628, 'BMI': 1.122546,
            'Never_smoke': 1.0, 'Ever_smoke': 1.047533, 'Now_smoke': 1.10408,
            'Never_drink': 1.0, 'Ever_drink': 1.203899, 'Now_drink': 1.324933,
            'category': 'Cardiovascular'
        },
        'Ischemic Heart Disease': {
            'HR_cat<60': 1.192775, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.024668,
            'HR_cat80-89': 0.983446, 'HR_cat>=90': 0.984206,
            'AGE': 1.079249, 'MALE': 1.0, 'FEMALE': 0.817353, 'BMI': 1.074744,
            'Never_smoke': 1.0, 'Ever_smoke': 1.116484, 'Now_smoke': 1.238498,
            'Never_drink': 1.0, 'Ever_drink': 1.249432, 'Now_drink': 0.890876,
            'category': 'Cardiovascular'
        },
        'Ischemic Stroke': {
            'HR_cat<60': 1.101362, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.315702,
            'HR_cat80-89': 1.543814, 'HR_cat>=90': 1.654512,
            'AGE': 1.08316, 'MALE': 1.0, 'FEMALE': 0.833408, 'BMI': 1.053648,
            'Never_smoke': 1.0, 'Ever_smoke': 1.0919, 'Now_smoke': 1.504703,
            'Never_drink': 1.0, 'Ever_drink': 1.299209, 'Now_drink': 1.251103,
            'category': 'Cardiovascular'
        },
        'Migraine': {
            'HR_cat<60': 0.952232, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.994199,
            'HR_cat80-89': 1.171267, 'HR_cat>=90': 1.25287,
            'AGE': 0.986428, 'MALE': 1.0, 'FEMALE': 2.58151, 'BMI': 1.015953,
            'Never_smoke': 1.0, 'Ever_smoke': 1.182575, 'Now_smoke': 1.386739,
            'Never_drink': 1.0, 'Ever_drink': 0.763077, 'Now_drink': 0.776311,
            'category': 'Neurological'
        },
        'Parkinson\'s Disease': {
            'HR_cat<60': 1.241971, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.440464,
            'HR_cat80-89': 1.296338, 'HR_cat>=90': 1.759019,
            'AGE': 1.14334, 'MALE': 1.0, 'FEMALE': 0.525192, 'BMI': 1.031024,
            'Never_smoke': 1.0, 'Ever_smoke': 0.629942, 'Now_smoke': 0.828633,
            'Never_drink': 1.0, 'Ever_drink': 1.002674, 'Now_drink': 0.502161,
            'category': 'Neurological'
        }
    }
    return twb_data

def get_hr_category(heart_rate):
    """Determine heart rate category based on bpm"""
    if heart_rate < 60:
        return 'HR_cat<60'
    elif heart_rate <= 69:
        return 'HR_cat60-69'
    elif heart_rate <= 79:
        return 'HR_cat70-79'
    elif heart_rate <= 89:
        return 'HR_cat80-89'
    else:
        return 'HR_cat>=90'

def calculate_comprehensive_risk(disease_data, age, gender, bmi, heart_rate, smoking_status, alcohol_status):
    """Calculate comprehensive risk using all TWB model factors"""
    # Start with baseline risk
    risk = 1.0
    
    # Heart rate category
    hr_category = get_hr_category(heart_rate)
    risk *= disease_data.get(hr_category, 1.0)
    
    # Age (continuous variable - risk per year)
    risk *= disease_data.get('AGE', 1.0) ** age
    
    # Gender
    if gender == 'Female':
        risk *= disease_data.get('FEMALE', 1.0)
    # Male is baseline (1.0), so no multiplication needed
    
    # BMI (continuous variable - risk per BMI unit)
    risk *= disease_data.get('BMI', 1.0) ** bmi
    
    # Smoking status
    if smoking_status == 'Current Smoker':
        risk *= disease_data.get('Now_smoke', 1.0)
    elif smoking_status == 'Former Smoker':
        risk *= disease_data.get('Ever_smoke', 1.0)
    # Never smoker is baseline (1.0)
    
    # Alcohol status
    if alcohol_status == 'Current Drinker':
        risk *= disease_data.get('Now_drink', 1.0)
    elif alcohol_status == 'Former Drinker':
        risk *= disease_data.get('Ever_drink', 1.0)
    # Never drinker is baseline (1.0)
    
    return risk

def get_risk_color(risk_ratio):
    """Get color based on risk level"""
    if risk_ratio < 1.1:
        return "#27ae60"  # Green
    elif risk_ratio < 1.5:
        return "#f39c12"  # Orange
    elif risk_ratio < 2.0:
        return "#e67e22"  # Dark Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(risk_ratio):
    """Get risk level description"""
    if risk_ratio < 1.1:
        return "Low Risk"
    elif risk_ratio < 1.5:
        return "Moderate Risk"
    elif risk_ratio < 2.0:
        return "High Risk"
    else:
        return "Very High Risk"

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight and height"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def main():
    # Load TWB data
    twb_data = load_twb_data()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Comprehensive Health Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personalized disease risk assessment using Taiwan Biobank model</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 👤 Personal Information")
        
        # Basic demographics
        age = st.slider("Age", 20, 90, 50, help="Your current age in years")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # BMI calculation
        st.markdown("#### 📏 Body Measurements")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        with col2:
            height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
        
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        st.info(f"📊 BMI: {bmi:.1f} ({bmi_category})")
        
        # Heart rate
        st.markdown("### 💓 Heart Rate Information")
        heart_rate = st.slider(
            "Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your resting heart rate in beats per minute"
        )
        
        hr_category = get_hr_category(heart_rate)
        category_display = {
            'HR_cat<60': '<60 bpm (Bradycardia)',
            'HR_cat60-69': '60-69 bpm (Normal)',
            'HR_cat70-79': '70-79 bpm (Normal)',
            'HR_cat80-89': '80-89 bpm (Upper Normal)',
            'HR_cat>=90': '≥90 bpm (Tachycardia)'
        }
        st.info(f"📊 HR Category: {category_display[hr_category]}")
        
        # Lifestyle factors
        st.markdown("### 🚬 Smoking Status")
        smoking_status = st.selectbox(
            "Smoking History",
            ["Never Smoker", "Former Smoker", "Current Smoker"],
            help="Your smoking history"
        )
        
        st.markdown("### 🍷 Alcohol Consumption")
        alcohol_status = st.selectbox(
            "Alcohol History",
            ["Never Drinker", "Former Drinker", "Current Drinker"],
            help="Your alcohol consumption history"
        )
        
        # Risk category filter
        st.markdown("### 🔍 Disease Categories")
        available_categories = list(set([data['category'] for data in twb_data.values()]))
        selected_categories = st.multiselect(
            "Select categories to display:",
            available_categories,
            default=available_categories
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Comprehensive Risk Assessment")
        
        # Calculate comprehensive risks
        risks = {}
        filtered_diseases = [disease for disease, data in twb_data.items() 
                           if data['category'] in selected_categories]
        
        for disease in filtered_diseases:
            disease_data = twb_data[disease]
            risk = calculate_comprehensive_risk(
                disease_data, age, gender, bmi, heart_rate, 
                smoking_status, alcohol_status
            )
            risks[disease] = risk
        
        if risks:
            # Create risk visualization
            diseases = list(risks.keys())
            risk_values = list(risks.values())
            colors = [get_risk_color(risk) for risk in risk_values]
            
            # Sort by risk level for better visualization
            sorted_data = sorted(zip(diseases, risk_values, colors), key=lambda x: x[1], reverse=True)
            diseases, risk_values, colors = zip(*sorted_data)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=diseases,
                    x=risk_values,
                    marker_color=colors,
                    text=[f"{risk:.2f}x" for risk in risk_values],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Risk Ratio: %{x:.2f}x<extra></extra>',
                    orientation='h'
                )
            ])
            
            fig.update_layout(
                title="Personalized Disease Risk Assessment",
                xaxis_title="Risk Ratio (vs. Baseline Population)",
                yaxis_title="Disease",
                height=max(500, len(diseases) * 25),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10),
                margin=dict(l=200, r=50, t=80, b=50)
            )
            
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                         annotation_text="Population Baseline (1.0)", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level indicators
            st.markdown("### 🎯 Risk Level Summary")
            cols = st.columns(4)
            
            very_high_risk = sum(1 for risk in risk_values if risk >= 2.0)
            high_risk = sum(1 for risk in risk_values if 1.5 <= risk < 2.0)
            moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.5)
            low_risk = sum(1 for risk in risk_values if risk < 1.1)
            
            with cols[0]:
                st.metric("🔴 Very High Risk", very_high_risk)
            with cols[1]:
                st.metric("🟠 High Risk", high_risk)
            with cols[2]:
                st.metric("🟡 Moderate Risk", moderate_risk)
            with cols[3]:
                st.metric("🟢 Low Risk", low_risk)
        else:
            st.warning("Please select at least one category to view results.")
    
    with col2:
        st.markdown("### 💡 Personal Profile")
        
        # Personal summary
        st.markdown(f"""
        <div class="info-box">
            <h4>👤 Your Profile</h4>
            <p><strong>Age:</strong> {age} years</p>
            <p><strong>Gender:</strong> {gender}</p>
            <p><strong>BMI:</strong> {bmi:.1f} ({bmi_category})</p>
            <p><strong>Heart Rate:</strong> {heart_rate} bpm</p>
            <p><strong>Smoking:</strong> {smoking_status}</p>
            <p><strong>Alcohol:</strong> {alcohol_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk factor analysis
        st.markdown("### ⚡ Risk Factor Impact")
        
        # Calculate individual factor contributions for top disease
        if risks:
            top_disease = max(risks.items(), key=lambda x: x[1])
            disease_name, total_risk = top_disease
            disease_data = twb_data[disease_name]
            
            st.markdown(f"**Analysis for: {disease_name}**")
            
            # Individual factor contributions
            factors = []
            
            # Heart Rate
            hr_risk = disease_data.get(get_hr_category(heart_rate), 1.0)
            factors.append(("Heart Rate", hr_risk, f"{category_display[get_hr_category(heart_rate)]}"))
            
            # Age
            age_risk = disease_data.get('AGE', 1.0) ** age
            factors.append(("Age", age_risk, f"{age} years"))
            
            # Gender
            gender_risk = disease_data.get('FEMALE', 1.0) if gender == 'Female' else 1.0
            factors.append(("Gender", gender_risk, gender))
            
            # BMI
            bmi_risk = disease_data.get('BMI', 1.0) ** bmi
            factors.append(("BMI", bmi_risk, f"{bmi:.1f}"))
            
            # Smoking
            smoke_risk = 1.0
            if smoking_status == 'Current Smoker':
                smoke_risk = disease_data.get('Now_smoke', 1.0)
            elif smoking_status == 'Former Smoker':
                smoke_risk = disease_data.get('Ever_smoke', 1.0)
            factors.append(("Smoking", smoke_risk, smoking_status))
            
            # Alcohol
            alcohol_risk = 1.0
            if alcohol_status == 'Current Drinker':
                alcohol_risk = disease_data.get('Now_drink', 1.0)
            elif alcohol
