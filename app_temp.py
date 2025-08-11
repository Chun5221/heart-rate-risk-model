# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:53:22 2025

@author: chun5
"""

# -*- coding: utf-8 -*-
"""
Enhanced Heart Rate Risk Calculator using Cox Regression Model Results
Updated to use actual model coefficients from TWB_model1_sig.csv

@author: Enhanced with Cox regression model
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

# Load and parse the Cox regression model coefficients
@st.cache_data
def load_model_coefficients():
    """Load the Cox regression model coefficients from the CSV data"""
    # The CSV data from TWB_model1_sig.csv
    csv_data = """Daisease namw,Variable,Coef
Atrial Fibrillation,HR_cat60-69,REF
Atrial Fibrillation,HR_cat<60,0.245753
Atrial Fibrillation,HR_cat70-79,0.007172
Atrial Fibrillation,HR_cat80-89,0.285073
Atrial Fibrillation,HR_cat>=90,0.626423
Atrial Fibrillation,AGE,0.092884
Atrial Fibrillation,MALE,REF
Atrial Fibrillation,FERMALE,-0.826015
Atrial Fibrillation,BMI,0.052884
Atrial Fibrillation,Nerver_smoke,REF
Atrial Fibrillation,Ever_smoke,0.040906
Atrial Fibrillation,Now_smoke,-0.22748
Atrial Fibrillation,Nerver_drink,REF
Atrial Fibrillation,Ever_drink,0.114189
Atrial Fibrillation,Now_drink,0.079308
Anxiety,HR_cat60-69,REF
Anxiety,HR_cat<60,-0.0147946
Anxiety,HR_cat70-79,0.0331788
Anxiety,HR_cat80-89,0.079773
Anxiety,HR_cat>=90,0.0326789
Anxiety,AGE,0.0221697
Anxiety,MALE,REF
Anxiety,FERMALE,0.5610132
Anxiety,BMI,-0.0212976
Anxiety,Nerver_smoke,REF
Anxiety,Ever_smoke,0.1371077
Anxiety,Now_smoke,0.1912624
Anxiety,Nerver_drink,REF
Anxiety,Ever_drink,0.2342184
Anxiety,Now_drink,0.145487
Chronic Kidney Disease,HR_cat60-69,REF
Chronic Kidney Disease,HR_cat<60,-0.037829
Chronic Kidney Disease,HR_cat70-79,0.114072
Chronic Kidney Disease,HR_cat80-89,0.2957
Chronic Kidney Disease,HR_cat>=90,0.720731
Chronic Kidney Disease,AGE,0.067726
Chronic Kidney Disease,MALE,REF
Chronic Kidney Disease,FERMALE,-0.336082
Chronic Kidney Disease,BMI,0.081698
Chronic Kidney Disease,Nerver_smoke,REF
Chronic Kidney Disease,Ever_smoke,0.034987
Chronic Kidney Disease,Now_smoke,0.035351
Chronic Kidney Disease,Nerver_drink,REF
Chronic Kidney Disease,Ever_drink,0.283053
Chronic Kidney Disease,Now_drink,-0.097787
GERD,HR_cat60-69,REF
GERD,HR_cat<60,0.0173609
GERD,HR_cat70-79,0.106481
GERD,HR_cat80-89,0.1243943
GERD,HR_cat>=90,0.1830798
GERD,AGE,0.0110515
GERD,MALE,REF
GERD,FERMALE,0.1725073
GERD,BMI,0.0031696
GERD,Nerver_smoke,REF
GERD,Ever_smoke,0.1186117
GERD,Now_smoke,-0.0538687
GERD,Nerver_drink,REF
GERD,Ever_drink,0.0527654
GERD,Now_drink,0.0680244
Heart Failure,HR_cat60-69,REF
Heart Failure,HR_cat<60,0.094597
Heart Failure,HR_cat70-79,0.046528
Heart Failure,HR_cat80-89,0.281359
Heart Failure,HR_cat>=90,0.325456
Heart Failure,AGE,0.077657
Heart Failure,MALE,REF
Heart Failure,FERMALE,-0.053006
Heart Failure,BMI,0.10726
Heart Failure,Nerver_smoke,REF
Heart Failure,Ever_smoke,0.152996
Heart Failure,Now_smoke,0.254233
Heart Failure,Nerver_drink,REF
Heart Failure,Ever_drink,0.128193
Heart Failure,Now_drink,-0.019435
Myocardial Infarction,HR_cat60-69,REF
Myocardial Infarction,HR_cat<60,0.266526
Myocardial Infarction,HR_cat70-79,0.093003
Myocardial Infarction,HR_cat80-89,0.144828
Myocardial Infarction,HR_cat>=90,0.145092
Myocardial Infarction,AGE,0.079069
Myocardial Infarction,MALE,REF
Myocardial Infarction,FERMALE,-1.035229
Myocardial Infarction,BMI,0.088253
Myocardial Infarction,Nerver_smoke,REF
Myocardial Infarction,Ever_smoke,0.443004
Myocardial Infarction,Now_smoke,0.85316
Myocardial Infarction,Nerver_drink,REF
Myocardial Infarction,Ever_drink,0.236342
Myocardial Infarction,Now_drink,-0.37105
Type 2 Diabetes,HR_cat60-69,REF
Type 2 Diabetes,HR_cat<60,-0.115098
Type 2 Diabetes,HR_cat70-79,0.237972
Type 2 Diabetes,HR_cat80-89,0.504242
Type 2 Diabetes,HR_cat>=90,0.701339
Type 2 Diabetes,AGE,0.057474
Type 2 Diabetes,MALE,REF
Type 2 Diabetes,FERMALE,0.093901
Type 2 Diabetes,BMI,0.112338
Type 2 Diabetes,Nerver_smoke,REF
Type 2 Diabetes,Ever_smoke,0.135605
Type 2 Diabetes,Now_smoke,0.18522
Type 2 Diabetes,Nerver_drink,REF
Type 2 Diabetes,Ever_drink,0.168757
Type 2 Diabetes,Now_drink,0.122939
Anemia,HR_cat60-69,REF
Anemia,HR_cat<60,-0.0392645
Anemia,HR_cat70-79,0.0977869
Anemia,HR_cat80-89,0.1981423
Anemia,HR_cat>=90,0.0895871
Anemia,AGE,-0.0061833
Anemia,MALE,REF
Anemia,FERMALE,1.1290142
Anemia,BMI,-0.0073061
Anemia,Nerver_smoke,REF
Anemia,Ever_smoke,0.2113872
Anemia,Now_smoke,-0.0535957
Anemia,Nerver_drink,REF
Anemia,Ever_drink,0.1674907
Anemia,Now_drink,-0.1735583
Angina Pectoris,HR_cat60-69,REF
Angina Pectoris,HR_cat<60,0.14336
Angina Pectoris,HR_cat70-79,-0.072307
Angina Pectoris,HR_cat80-89,0.031692
Angina Pectoris,HR_cat>=90,-0.096771
Angina Pectoris,AGE,0.05689
Angina Pectoris,MALE,REF
Angina Pectoris,FERMALE,-0.184188
Angina Pectoris,BMI,0.055886
Angina Pectoris,Nerver_smoke,REF
Angina Pectoris,Ever_smoke,0.204652
Angina Pectoris,Now_smoke,0.206606
Angina Pectoris,Nerver_drink,REF
Angina Pectoris,Ever_drink,0.192616
Angina Pectoris,Now_drink,-0.005281
Asthma,HR_cat60-69,REF
Asthma,HR_cat<60,-0.106883
Asthma,HR_cat70-79,0.022602
Asthma,HR_cat80-89,0.053949
Asthma,HR_cat>=90,0.246947
Asthma,AGE,0.018771
Asthma,MALE,REF
Asthma,FERMALE,0.34897
Asthma,BMI,0.051065
Asthma,Nerver_smoke,REF
Asthma,Ever_smoke,0.180858
Asthma,Now_smoke,0.158275
Asthma,Nerver_drink,REF
Asthma,Ever_drink,0.048924
Asthma,Now_drink,-0.075188
Atherosclerosis,HR_cat60-69,REF
Atherosclerosis,HR_cat<60,0.143407
Atherosclerosis,HR_cat70-79,-0.046939
Atherosclerosis,HR_cat80-89,-0.010829
Atherosclerosis,HR_cat>=90,-0.193343
Atherosclerosis,AGE,0.076931
Atherosclerosis,MALE,REF
Atherosclerosis,FERMALE,-0.474543
Atherosclerosis,BMI,0.073382
Atherosclerosis,Nerver_smoke,REF
Atherosclerosis,Ever_smoke,0.218909
Atherosclerosis,Now_smoke,0.199181
Atherosclerosis,Nerver_drink,REF
Atherosclerosis,Ever_drink,0.250373
Atherosclerosis,Now_drink,0.049339
Cardiac Arrhythmia,HR_cat60-69,REF
Cardiac Arrhythmia,HR_cat<60,0.180933
Cardiac Arrhythmia,HR_cat70-79,0.067217
Cardiac Arrhythmia,HR_cat80-89,0.225652
Cardiac Arrhythmia,HR_cat>=90,0.416275
Cardiac Arrhythmia,AGE,0.033826
Cardiac Arrhythmia,MALE,REF
Cardiac Arrhythmia,FERMALE,0.200422
Cardiac Arrhythmia,BMI,0.003792
Cardiac Arrhythmia,Nerver_smoke,REF
Cardiac Arrhythmia,Ever_smoke,0.062005
Cardiac Arrhythmia,Now_smoke,-0.058892
Cardiac Arrhythmia,Nerver_drink,REF
Cardiac Arrhythmia,Ever_drink,0.152499
Cardiac Arrhythmia,Now_drink,0.055137
Depression,HR_cat60-69,REF
Depression,HR_cat<60,0.1195075
Depression,HR_cat70-79,0.1140151
Depression,HR_cat80-89,0.337861
Depression,HR_cat>=90,0.5705119
Depression,AGE,0.0123584
Depression,MALE,REF
Depression,FERMALE,0.5986412
Depression,BMI,-0.009728
Depression,Nerver_smoke,REF
Depression,Ever_smoke,0.3088563
Depression,Now_smoke,0.5593592
Depression,Nerver_drink,REF
Depression,Ever_drink,0.3865247
Depression,Now_drink,-0.0006955
Hypertension,HR_cat60-69,REF
Hypertension,HR_cat<60,-0.057906
Hypertension,HR_cat70-79,0.19495
Hypertension,HR_cat80-89,0.391184
Hypertension,HR_cat>=90,0.646424
Hypertension,AGE,0.058602
Hypertension,MALE,REF
Hypertension,FERMALE,-0.109502
Hypertension,BMI,0.115599
Hypertension,Nerver_smoke,REF
Hypertension,Ever_smoke,0.046438
Hypertension,Now_smoke,0.099012
Hypertension,Nerver_drink,REF
Hypertension,Ever_drink,0.185565
Hypertension,Now_drink,0.281362
Ischemic Heart Disease,HR_cat60-69,REF
Ischemic Heart Disease,HR_cat<60,0.176283
Ischemic Heart Disease,HR_cat70-79,0.024369
Ischemic Heart Disease,HR_cat80-89,-0.016693
Ischemic Heart Disease,HR_cat>=90,-0.01592
Ischemic Heart Disease,AGE,0.076265
Ischemic Heart Disease,MALE,REF
Ischemic Heart Disease,FERMALE,-0.201685
Ischemic Heart Disease,BMI,0.072083
Ischemic Heart Disease,Nerver_smoke,REF
Ischemic Heart Disease,Ever_smoke,0.110185
Ischemic Heart Disease,Now_smoke,0.2139
Ischemic Heart Disease,Nerver_drink,REF
Ischemic Heart Disease,Ever_drink,0.222689
Ischemic Heart Disease,Now_drink,-0.11555
Ischemic Stroke,HR_cat60-69,REF
Ischemic Stroke,HR_cat<60,0.096548
Ischemic Stroke,HR_cat70-79,0.27437
Ischemic Stroke,HR_cat80-89,0.434256
Ischemic Stroke,HR_cat>=90,0.503506
Ischemic Stroke,AGE,0.079883
Ischemic Stroke,MALE,REF
Ischemic Stroke,FERMALE,-0.182232
Ischemic Stroke,BMI,0.052259
Ischemic Stroke,Nerver_smoke,REF
Ischemic Stroke,Ever_smoke,0.087919
Ischemic Stroke,Now_smoke,0.408596
Ischemic Stroke,Nerver_drink,REF
Ischemic Stroke,Ever_drink,0.261755
Ischemic Stroke,Now_drink,0.224026
Migraine,HR_cat60-69,REF
Migraine,HR_cat<60,-0.048947
Migraine,HR_cat70-79,-0.005818
Migraine,HR_cat80-89,0.158086
Migraine,HR_cat>=90,0.225437
Migraine,AGE,-0.013665
Migraine,MALE,REF
Migraine,FERMALE,0.948374
Migraine,BMI,0.015827
Migraine,Nerver_smoke,REF
Migraine,Ever_smoke,0.167694
Migraine,Now_smoke,0.326955
Migraine,Nerver_drink,REF
Migraine,Ever_drink,-0.270397
Migraine,Now_drink,-0.253202
Parkinson's Disease,HR_cat60-69,REF
Parkinson's Disease,HR_cat<60,0.2167
Parkinson's Disease,HR_cat70-79,0.364965
Parkinson's Disease,HR_cat80-89,0.259543
Parkinson's Disease,HR_cat>=90,0.564756
Parkinson's Disease,AGE,0.133954
Parkinson's Disease,MALE,REF
Parkinson's Disease,FERMALE,-0.643991
Parkinson's Disease,BMI,0.030553
Parkinson's Disease,Nerver_smoke,REF
Parkinson's Disease,Ever_smoke,-0.462127
Parkinson's Disease,Now_smoke,-0.187978
Parkinson's Disease,Nerver_drink,REF
Parkinson's Disease,Ever_drink,0.002671
Parkinson's Disease,Now_drink,-0.688835"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    
    # Clean up the disease name column (there's a typo in the original)
    df.columns = ['Disease_Name', 'Variable', 'Coef']
    
    return df

def get_heart_rate_category(hr):
    """Categorize heart rate according to the model categories"""
    if hr < 60:
        return 'HR_cat<60'
    elif 60 <= hr < 70:
        return 'HR_cat60-69'
    elif 70 <= hr < 80:
        return 'HR_cat70-79'
    elif 80 <= hr < 90:
        return 'HR_cat80-89'
    else:
        return 'HR_cat>=90'

def calculate_cox_hazard_ratio(disease_name, age, gender, hr, bmi, smoking_status, drinking_status, model_df):
    """
    Calculate hazard ratio using Cox regression coefficients
    Following the algorithm: HR = exp(linear predictor)
    """
    try:
        # Filter coefficients for this disease
        disease_coefs = model_df[model_df['Disease_Name'] == disease_name].copy()
        
        if disease_coefs.empty:
            return None
        
        # Initialize linear predictor
        lp = 0.0
        
        # Heart Rate Category (reference is HR_cat60-69)
        hr_cat = get_heart_rate_category(hr)
        hr_coef = disease_coefs[disease_coefs['Variable'] == hr_cat]
        if not hr_coef.empty and hr_coef.iloc[0]['Coef'] != 'REF':
            lp += float(hr_coef.iloc[0]['Coef'])
        
        # Age (continuous variable)
        age_coef = disease_coefs[disease_coefs['Variable'] == 'AGE']
        if not age_coef.empty and age_coef.iloc[0]['Coef'] != 'REF':
            lp += float(age_coef.iloc[0]['Coef']) * age
        
        # Gender (reference is MALE)
        if gender == 'Female':
            gender_coef = disease_coefs[disease_coefs['Variable'] == 'FERMALE']
            if not gender_coef.empty and gender_coef.iloc[0]['Coef'] != 'REF':
                lp += float(gender_coef.iloc[0]['Coef'])
        
        # BMI (continuous variable)
        bmi_coef = disease_coefs[disease_coefs['Variable'] == 'BMI']
        if not bmi_coef.empty and bmi_coef.iloc[0]['Coef'] != 'REF':
            lp += float(bmi_coef.iloc[0]['Coef']) * bmi
        
        # Smoking Status (reference is Never_smoke)
        if smoking_status == 'Former Smoker':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        elif smoking_status == 'Current Smoker':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Now_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        
        # Drinking Status (reference is Never_drink)
        if drinking_status == 'Former Drinker':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        elif drinking_status == 'Current Drinker':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Now_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        
        # Calculate hazard ratio
        hazard_ratio = math.exp(lp)
        return hazard_ratio
    
    except Exception as e:
        st.error(f"Error calculating hazard ratio for {disease_name}: {str(e)}")
        return None

def get_risk_color(hazard_ratio):
    """Get color based on risk level"""
    if hazard_ratio < 1.2:
        return "#27ae60"  # Green
    elif hazard_ratio < 1.5:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(hazard_ratio):
    """Get risk level description"""
    if hazard_ratio < 1.2:
        return "Low Risk"
    elif hazard_ratio < 1.5:
        return "Moderate Risk"
    else:
        return "High Risk"

def categorize_diseases(disease_name):
    """Categorize diseases for filtering"""
    cardiovascular_diseases = [
        'Atrial Fibrillation', 'Heart Failure', 'Myocardial Infarction', 
        'Cardiac Arrhythmia', 'Ischemic Heart Disease', 'Angina Pectoris', 
        'Atherosclerosis', 'Hypertension', 'Ischemic Stroke'
    ]
    
    metabolic_diseases = [
        'Type 2 Diabetes', 'Chronic Kidney Disease'
    ]
    
    mental_health = [
        'Anxiety', 'Depression'
    ]
    
    other_conditions = [
        'GERD', 'Anemia', 'Asthma', 'Migraine', "Parkinson's Disease"
    ]
    
    if disease_name in cardiovascular_diseases:
        return 'Cardiovascular'
    elif disease_name in metabolic_diseases:
        return 'Metabolic'
    elif disease_name in mental_health:
        return 'Mental Health'
    else:
        return 'Other Conditions'

def main():
    # Load model coefficients
    model_df = load_model_coefficients()
    
    # Get unique diseases
    diseases = model_df['Disease_Name'].unique().tolist()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assess your disease risk based on Cox regression model results</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Input Parameters")
        
        # Personal information
        age = st.slider("Age", 20, 90, 65, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        bmi = st.slider("BMI", 15.0, 40.0, 24.0, step=0.1, help="Body Mass Index")
        
        # Heart rate
        st.markdown("### 💓 Heart Rate Information")
        current_hr = st.slider(
            "Current Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your current resting heart rate in beats per minute"
        )
        
        # Lifestyle factors
        st.markdown("### 🚬 Lifestyle Factors")
        smoking_status = st.selectbox(
            "Smoking Status", 
            ["Never Smoker", "Former Smoker", "Current Smoker"],
            help="Your smoking history"
        )
        
        drinking_status = st.selectbox(
            "Drinking Status",
            ["Never Drinker", "Former Drinker", "Current Drinker"],
            help="Your alcohol consumption history"
        )
        
        # Disease category filter
        st.markdown("### 🔍 Filter by Category")
        all_categories = ['Cardiovascular', 'Metabolic', 'Mental Health', 'Other Conditions']
        selected_categories = st.multiselect(
            "Select disease categories to display:",
            all_categories,
            default=all_categories
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Disease Risk Assessment Results")
        
        # Calculate hazard ratios for all diseases
        hazard_ratios = {}
        filtered_diseases = []
        
        for disease in diseases:
            category = categorize_diseases(disease)
            if category in selected_categories:
                filtered_diseases.append(disease)
                hr = calculate_cox_hazard_ratio(
                    disease, age, gender, current_hr, bmi, 
                    smoking_status, drinking_status, model_df
                )
                if hr is not None:
                    hazard_ratios[disease] = hr
        
        if hazard_ratios:
            # Create risk visualization
            diseases_list = list(hazard_ratios.keys())
            hr_values = list(hazard_ratios.values())
            colors = [get_risk_color(hr) for hr in hr_values]
            
            # Sort by hazard ratio for better visualization
            sorted_data = sorted(zip(diseases_list, hr_values, colors), key=lambda x: x[1], reverse=True)
            diseases_sorted, hr_sorted, colors_sorted = zip(*sorted_data)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=diseases_sorted,
                    x=hr_sorted,
                    orientation='h',
                    marker_color=colors_sorted,
                    text=[f"{hr:.2f}" for hr in hr_sorted],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Hazard Ratio: %{x:.2f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Hazard Ratios by Disease (Cox Regression Model)",
                xaxis_title="Hazard Ratio",
                yaxis_title="Diseases",
                height=max(400, len(diseases_list) * 25),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                         annotation_text="Baseline Risk (HR=1.0)", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level indicators
            st.markdown("### 🎯 Risk Level Summary")
            cols = st.columns(3)
            
            high_risk = sum(1 for hr in hr_values if hr >= 1.5)
            moderate_risk = sum(1 for hr in hr_values if 1.2 <= hr < 1.5)
            low_risk = sum(1 for hr in hr_values if hr < 1.2)
            
            with cols[0]:
                st.metric("🔴 High Risk Conditions", high_risk)
            with cols[1]:
                st.metric("🟡 Moderate Risk Conditions", moderate_risk)
            with cols[2]:
                st.metric("🟢 Low Risk Conditions", low_risk)
        else:
            st.warning("No valid hazard ratios calculated. Please check your input parameters.")
    
    with col2:
        st.markdown("### 💡 Heart Rate Status")
        
        # Heart rate category display
        hr_category = get_heart_rate_category(current_hr)
        category_display = {
            'HR_cat<60': '< 60 bpm (Bradycardia)',
            'HR_cat60-69': '60-69 bpm (Normal)',
            'HR_cat70-79': '70-79 bpm (Normal-High)',
            'HR_cat80-89': '80-89 bpm (Elevated)',
            'HR_cat>=90': '≥ 90 bpm (High)'
        }
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Your Heart Rate Category</h4>
            <p><strong>{current_hr} bpm</strong></p>
            <p>{category_display.get(hr_category, hr_category)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model information
        st.markdown("### 📚 Model Information")
        st.markdown("""
        <div class="info-box">
            <strong>Cox Regression Model:</strong><br>
            • Reference HR: 60-69 bpm<br>
            • Reference: Male, Never smoker/drinker<br>
            • Adjusted for age and BMI<br>
            • HR > 1.0 indicates increased risk
        </div>
        """, unsafe_allow_html=True)
        
        # Personal profile summary
        st.markdown("### 👤 Your Profile")
        st.markdown(f"""
        <div class="info-box">
            <strong>Personal Details:</strong><br>
            • Age: {age} years<br>
            • Gender: {gender}<br>
            • BMI: {bmi:.1f}<br>
            • Smoking: {smoking_status}<br>
            • Drinking: {drinking_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed risk breakdown
    if hazard_ratios:
        st.markdown("### 📋 Detailed Risk Analysis")
        
        # Create detailed table with categories
        risk_df = pd.DataFrame({
            'Disease': list(hazard_ratios.keys()),
            'Category': [categorize_diseases(disease) for disease in hazard_ratios.keys()],
            'Hazard Ratio': [f"{hr:.3f}" for hr in hazard_ratios.values()],
            'Risk Level': [get_risk_level(hr) for hr in hazard_ratios.values()],
            'Risk Interpretation': [
                f"{((hr-1)*100):+.1f}% change from baseline" if hr != 1.0 else "Baseline risk"
                for hr in hazard_ratios.values()
            ]
        })
        
        # Sort by hazard ratio
        risk_df = risk_df.sort_values('Hazard Ratio', ascending=False, key=lambda x: x.astype(float))
        
        # Color code the table
        def style_risk_level(val):
            if val == "High Risk":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "Moderate Risk":
                return 'background-color: #fff3e0; color: #ef6c00'
            else:
                return 'background-color: #e8f5e8; color: #2e7d32'
        
        styled_df = risk_df.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Heart rate sensitivity analysis
        st.markdown("### 📊 Heart Rate Sensitivity Analysis")
        
        # Select top 5 diseases with highest hazard ratios for trend analysis
        top_diseases = sorted(hazard_ratios.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if len(top_diseases) > 0:
            hr_range = np.arange(50, 101, 5)
            
            # Calculate trends for top diseases
            fig_trend = make_subplots(
                rows=2, cols=3,
                subplot_titles=[disease for disease, _ in top_diseases],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            for i, (disease, _) in enumerate(top_diseases):
                row = (i // 3) + 1
                col = (i % 3) + 1
                
                trend_hrs = []
                for test_hr in hr_range:
                    hr = calculate_cox_hazard_ratio(
                        disease, age, gender, test_hr, bmi, 
                        smoking_status, drinking_status, model_df
                    )
                    trend_hrs.append(hr if hr is not None else 1.0)
                
                # Add baseline reference line
                fig_trend.add_hline(
                    y=1.0, 
                    line_dash="dash", 
                    line_color="gray", 
                    opacity=0.5,
                    row=row, col=col
                )
                
                # Main trend line
                fig_trend.add_trace(
                    go.Scatter(
                        x=hr_range, 
                        y=trend_hrs,
                        mode='lines+markers',
                        name=disease,
                        line=dict(width=3),
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add current point
                current_risk = hazard_ratios[disease]
                fig_trend.add_trace(
                    go.Scatter(
                        x=[current_hr], 
                        y=[current_risk],
                        mode='markers',
                        name=f"Your Risk",
                        marker=dict(size=12, color='red', symbol='star'),
                        showlegend=False,
                        hovertemplate=f'<b>{disease}</b><br>HR: {current_hr} bpm<br>HR: {current_risk:.2f}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig_trend.update_layout(
                height=600,
                title_text="Heart Rate Sensitivity for Top Risk Diseases",
                showlegend=False,
                font=dict(size=10)
            )
            
            # Update axes for each subplot
            for i in range(len(top_diseases)):
                row = (i // 3) + 1
                col = (i % 3) + 1
                fig_trend.update_xaxes(title_text="Heart Rate (bpm)", row=row, col=col)
                fig_trend.update_yaxes(title_text="Hazard Ratio", row=row, col=col)
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Add legend explanation
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>📖 Chart Legend:</strong><br>
                • <span style="color: gray;">Gray dashed line</span>: Baseline risk (HR=1.0)<br>
                • <span style="color: red;">Red star</span>: Your current risk level<br>
                • Shows how risk changes with different heart rates while keeping other factors constant
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations based on results
    st.markdown("### 💊 Personalized Recommendations")
    
    if hazard_ratios:
        max_hr = max(hazard_ratios.values())
        high_risk_diseases = [disease for disease, hr in hazard_ratios.items() if hr >= 1.5]
        
        if max_hr >= 1.5:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ High Risk Detected</strong><br>
                You have elevated risk for: <strong>{', '.join(high_risk_diseases[:3])}</strong>
                {' and others' if len(high_risk_diseases) > 3 else ''}.<br><br>
                <strong>Recommendations:</strong><br>
                • Consult with a healthcare provider about your elevated heart rate<br>
                • Consider cardiovascular evaluation<br>
                • Lifestyle modifications may help reduce risk
            </div>
            """, unsafe_allow_html=True)
        elif max_hr >= 1.2:
            st.markdown("""
            <div class="warning-box">
                <strong>⚡ Moderate Risk</strong><br>
                Your profile shows moderate increased risk for some conditions.<br><br>
                <strong>Recommendations:</strong><br>
                • Regular exercise to improve cardiovascular fitness<br>
                • Maintain healthy weight and diet<br>
                • Consider stress management techniques<br>
                • Regular health check-ups
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>✅ Good Risk Profile</strong><br>
                Your current profile shows generally low risk across conditions.<br><br>
                <strong>Maintain:</strong><br>
                • Continue current healthy lifestyle<br>
                • Regular exercise and balanced nutrition<br>
                • Routine preventive care
            </div>
            """, unsafe_allow_html=True)
        
        # Specific lifestyle recommendations based on heart rate
        if current_hr >= 90:
            st.markdown("""
            <div class="warning-box">
                <strong>🏃 Heart Rate Specific Advice:</strong><br>
                Your resting heart rate (≥90 bpm) is elevated. Consider:<br>
                • Aerobic exercise training to improve cardiovascular fitness<br>
                • Stress reduction techniques (meditation, yoga)<br>
                • Adequate sleep (7-9 hours per night)<br>
                • Limit caffeine and stimulants<br>
                • Medical evaluation to rule out underlying conditions
            </div>
            """, unsafe_allow_html=True)
        elif current_hr >= 80:
            st.markdown("""
            <div class="info-box">
                <strong>💓 Heart Rate Optimization:</strong><br>
                Your heart rate is in the higher normal range. To optimize:<br>
                • Regular cardio exercise 150+ minutes per week<br>
                • Maintain healthy weight (BMI 18.5-24.9)<br>
                • Manage stress effectively<br>
                • Stay hydrated and get adequate rest
            </div>
            """, unsafe_allow_html=True)
    
    # Model methodology explanation
    with st.expander("📊 Model Methodology & Limitations"):
        st.markdown("""
        ### Cox Regression Model Details
        
        **Model Structure:**
        - **Heart Rate Categories:** <60, 60-69 (reference), 70-79, 80-89, ≥90 bpm
        - **Adjustments:** Age, gender, BMI, smoking status, drinking status
        - **Output:** Hazard Ratios (HR) representing relative risk compared to reference group
        
        **Interpretation:**
        - HR = 1.0: Same risk as reference group
        - HR > 1.0: Increased risk (e.g., HR = 1.5 means 50% higher risk)
        - HR < 1.0: Decreased risk (e.g., HR = 0.8 means 20% lower risk)
        
        **Important Limitations:**
        - Results are based on population-level associations
        - Individual risk may vary due to unmeasured factors
        - Model does not establish causation
        - Not a substitute for professional medical assessment
        - Based on specific study population characteristics
        
        **Data Source:**
        Taiwan Biobank Cox regression model results for disease risk prediction.
        """)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator uses Cox regression model results for educational purposes only. 
        The hazard ratios are based on population-level data and should not replace professional medical advice. 
        Individual risk factors and health conditions not included in the model may significantly affect your actual risk. 
        Always consult with a healthcare provider for personal health assessments and treatment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Risk calculations based on Cox regression model coefficients from Taiwan Biobank study.<br>
        Reference group: 60-69 bpm heart rate, male, never smoker/drinker.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
