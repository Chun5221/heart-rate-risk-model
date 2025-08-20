# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:55:53 2025
Author: Modified from original app3_HR.py

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
    
    .percentile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .high-risk-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(255,107,107,0.3);
    }
    
    .moderate-risk-card {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(255,167,38,0.3);
    }
    
    .low-risk-card {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(76,175,80,0.3);
    }
    
    .mortality-card {
        background: linear-gradient(135deg, #2c2c54 0%, #40407a 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(44,44,84,0.3);
        border: 2px solid #6c5ce7;
    }
    
    .demographic-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        text-align: center;
        max-width: 1100px;
        margin: 1rem auto;
        padding: 1.25rem 1.25rem;
    }
    
    .demographic-info h4 {
        font-size: 1.6rem;
        margin: 0.25rem 0 0.75rem 0;
        font-weight: 800;
    }
    
    .demographic-info p {
        font-size: 1.05rem;
        margin: 0.25rem 0;
    }
    
    .demographic-info > div {
        justify-content: center !important;
        gap: 0.75rem 1.5rem;
    }

    .bmi-info {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #27ae60;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .percentile-number {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .unit-toggle {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and parse the Cox regression model coefficients (all diseases including Death)
@st.cache_data
def load_model_coefficients():
    """Load the Cox regression model coefficients from the updated CSV data"""
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
Parkinson's Disease,Now_drink,-0.688835
Death,HR_cat60-69,REF
Death,HR_cat<60,0.034669
Death,HR_cat70-79,0.245161
Death,HR_cat80-89,0.55444
Death,HR_cat>=90,0.769643
Death,AGE,0.079282
Death,MALE,REF
Death,FERMALE,-0.674196
Death,BMI,0.036303
Death,Nerver_smoke,REF
Death,Ever_smoke,-0.01009
Death,Now_smoke,0.590807
Death,Nerver_drink,REF
Death,Ever_drink,0.491235
Death,Now_drink,0.246215"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    df.columns = ['Disease_Name', 'Variable', 'Coef']
    return df

# Load sample LP distribution data (placeholder - you'll replace this with real data)
@st.cache_data
def load_lp_distributions():
    """
    Placeholder for LP distribution data including Death model
    """
    
    # ALL DISEASES INCLUDING DEATH
    diseases = [
        'Atrial Fibrillation', 'Anxiety', 'Chronic Kidney Disease', 'GERD', 
        'Heart Failure', 'Myocardial Infarction', 'Type 2 Diabetes', 'Anemia',
        'Angina Pectoris', 'Asthma', 'Atherosclerosis', 'Cardiac Arrhythmia',
        'Depression', 'Hypertension', 'Ischemic Heart Disease', 'Ischemic Stroke',
        'Migraine', "Parkinson's Disease", 'Death'
    ]
    
    genders = ['Male', 'Female']
    age_groups = ['30-40', '40-50', '50-60', '60-70', '70-80']
    
    sample_data = []
    np.random.seed(42)  # For reproducible sample data
    
    for disease in diseases:
        for gender in genders:
            for age_group in age_groups:
                # Generate realistic sample LP distribution
                if disease == 'Death':
                    # Death/Mortality has unique characteristics
                    base_lp = np.random.normal(0.8, 1.0)  # Higher base risk, wider distribution
                elif disease in ['Atrial Fibrillation', 'Heart Failure', 'Myocardial Infarction']:
                    base_lp = np.random.normal(0.5, 0.8)  # Higher base risk
                elif disease in ['Anxiety', 'Depression', 'Migraine']:
                    base_lp = np.random.normal(0.2, 0.6)  # Moderate base risk
                else:
                    base_lp = np.random.normal(0.0, 0.7)  # Average base risk
                
                # Age and gender effects
                age_mid = int(age_group.split('-')[0]) + 5
                age_effect = (age_mid - 50) * 0.02  # Older = higher risk
                
                # Special gender effects for Death and specific diseases
                if disease == 'Death':
                    gender_effect = -0.3 if gender == 'Female' else 0  # Females generally lower mortality risk
                elif gender == 'Female' and disease in ['Migraine', 'Anxiety', 'Anemia']:
                    gender_effect = 0.2
                else:
                    gender_effect = -0.1 if gender == 'Female' else 0
                
                adjusted_base = base_lp + age_effect + gender_effect
                
                # Generate percentile thresholds
                percentiles = np.random.normal(adjusted_base, 0.6, 7)
                percentiles = np.sort(percentiles)
                
                sample_data.append({
                    'Disease_Name': disease,
                    'Gender': gender,
                    'Age_Group': age_group,
                    'P5': percentiles[0],
                    'P10': percentiles[1],
                    'P25': percentiles[2],
                    'P50': percentiles[3],
                    'P75': percentiles[4],
                    'P90': percentiles[5],
                    'P95': percentiles[6]
                })
    
    return pd.DataFrame(sample_data)

def calculate_bmi(height, weight, height_unit, weight_unit):
    """Calculate BMI from height and weight with unit conversion"""
    try:
        # Convert height to meters
        if height_unit == "cm":
            height_m = height / 100
        elif height_unit == "feet/inches":
            # Height is passed as total inches
            height_m = height * 0.0254
        else:  # meters
            height_m = height
        
        # Convert weight to kg
        if weight_unit == "lbs":
            weight_kg = weight * 0.453592
        else:  # kg
            weight_kg = weight
        
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)
    
    except (ZeroDivisionError, ValueError):
        return None

def get_bmi_category(bmi):
    """Categorize BMI according to WHO standards"""
    if bmi < 18.5:
        return "Underweight", "#3498db"
    elif bmi < 25:
        return "Normal weight", "#27ae60"
    elif bmi < 30:
        return "Overweight", "#f39c12"
    else:
        return "Obese", "#e74c3c"

def categorize_diseases(disease_name):
    """Categorize diseases for filtering - Death gets its own category"""
    if disease_name == 'Death':
        return 'Mortality Risk'
    
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

def get_age_group(age):
    """Convert age to age group for demographic comparison"""
    if age < 30:
        return '30-40'  # Minimum group
    elif age < 40:
        return '30-40'
    elif age < 50:
        return '40-50'
    elif age < 60:
        return '50-60'
    elif age < 70:
        return '60-70'
    else:
        return '70-80'

def calculate_linear_predictor(disease_name, age, gender, hr, bmi, smoking_status, drinking_status, model_df):
    """
    Calculate linear predictor (LP) using Cox regression coefficients
    LP = sum of (coefficient × variable_value)
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
        
        return lp
    
    except Exception as e:
        st.error(f"Error calculating LP for {disease_name}: {str(e)}")
        return None

def calculate_percentile_rank(user_lp, disease_name, gender, age_group, lp_dist_df):
    """
    Calculate user's percentile rank within their demographic group
    """
    try:
        # Find the matching demographic group
        demographic_data = lp_dist_df[
            (lp_dist_df['Disease_Name'] == disease_name) & 
            (lp_dist_df['Gender'] == gender) & 
            (lp_dist_df['Age_Group'] == age_group)
        ]
        
        if demographic_data.empty:
            return None
        
        # Get percentile thresholds
        percentiles = demographic_data.iloc[0]
        
        # Calculate where user falls in the distribution
        if user_lp <= percentiles['P5']:
            return 5  # Bottom 5%
        elif user_lp <= percentiles['P10']:
            return 10
        elif user_lp <= percentiles['P25']:
            return 25
        elif user_lp <= percentiles['P50']:
            return 50
        elif user_lp <= percentiles['P75']:
            return 75
        elif user_lp <= percentiles['P90']:
            return 90
        elif user_lp <= percentiles['P95']:
            return 95
        else:
            return 100  # Top 5% (highest risk)
        
    except Exception as e:
        st.error(f"Error calculating percentile: {str(e)}")
        return None

def get_risk_category_and_color(percentile, disease_name=''):
    """Get risk category and color based on percentile - special handling for Death"""
    if disease_name == 'Death':
        # Special styling for mortality risk
        if percentile >= 90:
            return "Critical Mortality Risk", "mortality-card", "#2c2c54"
        elif percentile >= 75:
            return "High Mortality Risk", "mortality-card", "#40407a"
        elif percentile >= 50:
            return "Moderate Mortality Risk", "mortality-card", "#6c5ce7"
        else:
            return "Lower Mortality Risk", "mortality-card", "#a29bfe"
    else:
        # Regular disease risk categories
        if percentile >= 90:
            return "High Risk", "high-risk-card", "#e74c3c"
        elif percentile >= 75:
            return "Moderate-High Risk", "moderate-risk-card", "#f39c12"
        elif percentile >= 50:
            return "Average Risk", "percentile-card", "#3498db"
        else:
            return "Lower Risk", "low-risk-card", "#27ae60"

def create_percentile_gauge(percentile, disease_name):
    """Create a gauge chart showing percentile position - special styling for Death"""
    
    # Special color scheme for Death/Mortality
    if disease_name == 'Death':
        if percentile >= 90:
            color = "#2c2c54"  # Dark purple
        elif percentile >= 75:
            color = "#40407a"  # Medium purple
        elif percentile >= 50:
            color = "#6c5ce7"  # Light purple
        else:
            color = "#a29bfe"  # Very light purple
        
        title_text = f"Mortality Risk<br>Percentile"
        steps = [
            {'range': [0, 50], 'color': "#ddd6fe"},
            {'range': [50, 75], 'color': "#c4b5fd"},
            {'range': [75, 90], 'color': "#a78bfa"},
            {'range': [90, 100], 'color': "#8b5cf6"}
        ]
    else:
        # Regular disease colors
        if percentile >= 90:
            color = "#e74c3c"  # Red
        elif percentile >= 75:
            color = "#f39c12"  # Orange
        elif percentile >= 50:
            color = "#3498db"  # Blue
        else:
            color = "#27ae60"  # Green
        
        title_text = f"{disease_name}<br>Risk Percentile"
        steps = [
            {'range': [0, 50], 'color': "#d5f4e6"},
            {'range': [50, 75], 'color': "#ffeaa7"},
            {'range': [75, 90], 'color': "#fdcb6e"},
            {'range': [90, 100], 'color': "#e17055"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentile,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title_text},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def main():
    # Load data
    model_df = load_model_coefficients()
    lp_dist_df = load_lp_distributions()
    
    # Get unique diseases
    diseases = model_df['Disease_Name'].unique().tolist()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Percentile Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Find out how your risk compares to people in your demographic group</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📋 Your Information")
        
        # Personal information
        age = st.slider("Age", 20, 90, 45, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # Height and Weight Section with BMI Calculator
        st.markdown("### 📏 Height & Weight")
        
        # Unit selection
        col1, col2 = st.columns(2)
        with col1:
            height_unit = st.selectbox(
                "Height unit", 
                ["cm", "feet/inches", "meters"],
                help="Select your preferred height measurement"
            )
        with col2:
            weight_unit = st.selectbox(
                "Weight unit", 
                ["kg", "lbs"],
                help="Select your preferred weight measurement"
            )
        
        # Height input based on unit
        if height_unit == "cm":
            height = st.slider("Height (cm)", 100, 220, 170, help="Your height in centimeters")
        elif height_unit == "feet/inches":
            feet = st.selectbox("Feet", list(range(3, 8)), index=2, help="Feet component of height")
            inches = st.selectbox("Inches", list(range(0, 12)), index=6, help="Inches component of height") 
            height = feet * 12 + inches  # Convert to total inches for calculation
            st.write(f"Height: {feet}'{inches}\"")
        else:  # meters
            height = st.slider("Height (m)", 1.0, 2.2, 1.7, step=0.01, help="Your height in meters")
        
        # Weight input based on unit
        if weight_unit == "kg":
            weight = st.slider("Weight (kg)", 30, 200, 70, help="Your weight in kilograms")
        else:  # lbs
            weight = st.slider("Weight (lbs)", 66, 440, 154, help="Your weight in pounds")
        
        # Calculate BMI automatically
        calculated_bmi = calculate_bmi(height, weight, height_unit, weight_unit)
        
        if calculated_bmi:
            bmi_category, bmi_color = get_bmi_category(calculated_bmi)
            st.markdown(f"""
            <div class="bmi-info">
                <h4>📊 Calculated BMI</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: {bmi_color};">
                    BMI: {calculated_bmi}
                </p>
                <p style="color: {bmi_color}; font-weight: bold;">
                    Category: {bmi_category}
                </p>
                <small>BMI = weight(kg) ÷ height(m)²</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Use calculated BMI
            bmi = calculated_bmi
        else:
            st.error("Could not calculate BMI. Please check your height and weight values.")
            bmi = 24.0  # Default fallback
        
        # Heart rate
        st.markdown("### 💗 Heart Rate")
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
        
        # Disease category filter - now including Mortality Risk
        st.markdown("### 🏥 Risk Categories")
        all_categories = ['Cardiovascular', 'Metabolic', 'Mental Health', 'Other Conditions', 'Mortality Risk']
        selected_categories = st.multiselect(
            "Select categories to display:",
            all_categories,
            default=all_categories,
            help="Choose which risk categories to show in your assessment"
        )
    
    # Determine user's demographic group
    age_group = get_age_group(age)
    
    # Display demographic info with BMI
    st.markdown(f"""
    <div class="demographic-info">
        <h4>👥 Your Profile & Demographic Group</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <p><strong>Comparing you to:</strong> {gender}s aged {age_group} years</p>
                <p><strong>Your Details:</strong> {age} years old, {gender}</p>
            </div>
            <div>
                <p><strong>Physical:</strong> BMI {bmi}, HR {current_hr} bpm</p>
                <p><strong>Lifestyle:</strong> {smoking_status.replace('Never Smoker', 'Non-smoker')}, {drinking_status.replace('Never Drinker', 'Non-drinker')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if True:  # Auto-calculate always
        st.markdown("### 🎯 Your Risk Percentiles")
        
        # Calculate percentiles for filtered diseases
        results = []
        
        # Filter diseases by selected categories
        for disease in diseases:
            category = categorize_diseases(disease)
            if category in selected_categories:
                # Calculate user's linear predictor
                user_lp = calculate_linear_predictor(
                    disease, age, gender, current_hr, bmi, 
                    smoking_status, drinking_status, model_df
                )
                
                if user_lp is not None:
                    # Calculate percentile rank
                    percentile = calculate_percentile_rank(
                        user_lp, disease, gender, age_group, lp_dist_df
                    )
                    
                    if percentile is not None:
                        risk_category, card_class, color = get_risk_category_and_color(percentile, disease)
                        results.append({
                            'disease': disease,
                            'category': category,
                            'percentile': percentile,
                            'risk_category': risk_category,
                            'card_class': card_class,
                            'color': color,
                            'lp': user_lp
                        })
        
        if results:
            # Separate Death/Mortality from other diseases for display
            mortality_result = [r for r in results if r['disease'] == 'Death']
            disease_results = [r for r in results if r['disease'] != 'Death']
            
            # Display Mortality Risk prominently if selected
            if mortality_result:
                st.markdown("### ⚰️ Mortality Risk Assessment")
                mort_result = mortality_result[0]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Create gauge chart for mortality
                    fig_mort = create_percentile_gauge(mort_result['percentile'], 'Death')
                    st.plotly_chart(fig_mort, use_container_width=True)
                
                with col2:
                    # Mortality risk interpretation
                    if mort_result['percentile'] >= 90:
                        interpretation = f"⚠️ **Critical: Top {100-mort_result['percentile']}% highest mortality risk**"
                        recommendation = "Immediate medical consultation strongly advised"
                        urgency_color = "#c0392b"
                    elif mort_result['percentile'] >= 75:
                        interpretation = f"🚨 **High mortality risk** (higher than {mort_result['percentile']}%)"
                        recommendation = "Schedule comprehensive health evaluation"
                        urgency_color = "#e67e22"
                    elif mort_result['percentile'] >= 50:
                        interpretation = f"⚡ **Moderate mortality risk** (top {100-mort_result['percentile']}%)"
                        recommendation = "Focus on preventive health measures"
                        urgency_color = "#8e44ad"
                    else:
                        interpretation = f"✅ **Lower mortality risk** (top {100-mort_result['percentile']}%)"
                        recommendation = "Continue healthy lifestyle practices"
                        urgency_color = "#27ae60"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {urgency_color} 0%, {mort_result['color']} 100%); 
                                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                        <h3>🎯 Mortality Risk: {mort_result['percentile']}th Percentile</h3>
                        <p style="font-size: 1.1rem; margin: 1rem 0;">{interpretation}</p>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p style="font-size: 1rem;"><strong>Recommendation:</strong> {recommendation}</p>
                        <p style="font-size: 0.9rem; opacity: 0.9;"><em>Based on population mortality data for {gender.lower()}s aged {age_group}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if disease_results:
                st.markdown("### 🏥 Disease-Specific Risk Assessment")
                
                # Sort by percentile (highest risk first)
                disease_results.sort(key=lambda x: x['percentile'], reverse=True)
                
                # Create dynamic grid layout based on number of diseases
                num_diseases = len(disease_results)
                if num_diseases <= 3:
                    cols_per_row = num_diseases
                elif num_diseases <= 6:
                    cols_per_row = 3
                elif num_diseases <= 9:
                    cols_per_row = 3
                else:
                    cols_per_row = 4
                
                # Display results in grid format
                for i in range(0, num_diseases, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < num_diseases:
                            result = disease_results[i + j]
                            with cols[j]:
                                # Create gauge chart
                                fig = create_percentile_gauge(result['percentile'], result['disease'])
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Risk interpretation
                                if result['percentile'] >= 90:
                                    interpretation = f"⚠️ **Top {100-result['percentile']}% highest risk**"
                                    recommendation = "Consider medical consultation"
                                elif result['percentile'] >= 75:
                                    interpretation = f"📈 **Higher than {result['percentile']}% of your demographic**"
                                    recommendation = "Monitor closely, lifestyle changes"
                                elif result['percentile'] >= 50:
                                    interpretation = f"📊 **Average risk** (top {100-result['percentile']}%)"
                                    recommendation = "Continue healthy habits"
                                else:
                                    interpretation = f"✅ **Lower risk** (top {100-result['percentile']}%)"
                                    recommendation = "Maintain current lifestyle"
                                
                                st.markdown(f"""
                                <div class="{result['card_class']}">
                                    <h4>{result['disease']}</h4>
                                    <div class="percentile-number">{result['percentile']}th</div>
                                    <p>Percentile</p>
                                    <p style="font-size: 0.8rem; opacity: 0.9;">{result['category']}</p>
                                    <hr style="border-color: rgba(255,255,255,0.3);">
                                    <p style="font-size: 0.9rem;">{interpretation}</p>
                                    <p style="font-size: 0.8rem;"><em>{recommendation}</em></p>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Category-wise summary (including Mortality Risk)
            st.markdown("### 📊 Risk Summary by Category")
            
            # Group results by category
            category_summary = {}
            for result in results:  # Include all results (mortality + diseases)
                cat = result['category']
                if cat not in category_summary:
                    category_summary[cat] = {'high': 0, 'moderate': 0, 'average': 0, 'low': 0, 'diseases': []}
                
                category_summary[cat]['diseases'].append(result['disease'])
                
                if result['percentile'] >= 90:
                    category_summary[cat]['high'] += 1
                elif result['percentile'] >= 75:
                    category_summary[cat]['moderate'] += 1
                elif result['percentile'] >= 50:
                    category_summary[cat]['average'] += 1
                else:
                    category_summary[cat]['low'] += 1
            
            # Display category summary
            if category_summary:
                summary_cols = st.columns(len(category_summary))
                for i, (category, summary) in enumerate(category_summary.items()):
                    with summary_cols[i]:
                        total = len(summary['diseases'])
                        high_pct = (summary['high'] / total) * 100
                        
                        # Special styling for Mortality Risk category
                        if category == 'Mortality Risk':
                            bg_color = "linear-gradient(135deg, #2c2c54 0%, #40407a 100%)"
                            icon = "⚰️"
                        elif high_pct >= 50:
                            bg_color = "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)"
                            icon = "🚨"
                        elif summary['moderate'] > 0:
                            bg_color = "linear-gradient(135deg, #ffa726 0%, #ff9800 100%)"
                            icon = "⚠️"
                        else:
                            bg_color = "linear-gradient(135deg, #66bb6a 0%, #4caf50 100%)"
                            icon = "✅"
                        
                        st.markdown(f"""
                        <div style="background: {bg_color}; padding: 1.5rem; border-radius: 15px; 
                                    color: white; text-align: center; margin: 0.5rem 0;">
                            <h4>{icon} {category}</h4>
                            <p><strong>{total}</strong> assessment{'s' if total > 1 else ''}</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p>🔴 High Risk: {summary['high']}</p>
                            <p>🟡 Moderate: {summary['moderate']}</p>
                            <p>🔵 Average: {summary['average']}</p>
                            <p>🟢 Lower: {summary['low']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed risk comparison table
            st.markdown("### 📋 Detailed Risk Comparison")
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Condition': [r['disease'] for r in results],
                'Category': [r['category'] for r in results],
                'Your Percentile': [f"{r['percentile']}th" for r in results],
                'Risk Level': [r['risk_category'] for r in results],
                'Interpretation': [
                    f"Higher risk than {r['percentile']}% of {gender.lower()}s aged {age_group}" 
                    if r['percentile'] > 50 
                    else f"Lower risk than {100-r['percentile']}% of {gender.lower()}s aged {age_group}"
                    for r in results
                ],
                'Linear Predictor': [f"{r['lp']:.3f}" for r in results]
            })
            
            # Style the dataframe
            def style_risk_level(val):
                if 'Critical' in val or 'High Risk' in val:
                    return 'background-color: #ffebee; color: #c62828'
                elif 'Moderate' in val:
                    return 'background-color: #fff3e0; color: #ef6c00'
                elif 'Average' in val:
                    return 'background-color: #e3f2fd; color: #1976d2'
                else:
                    return 'background-color: #e8f5e8; color: #2e7d32'
            
            def style_category(val):
                colors = {
                    'Cardiovascular': 'background-color: #ffebee; color: #c62828',
                    'Metabolic': 'background-color: #f3e5f5; color: #7b1fa2',
                    'Mental Health': 'background-color: #e8f5e8; color: #388e3c',
                    'Other Conditions': 'background-color: #e3f2fd; color: #1976d2',
                    'Mortality Risk': 'background-color: #e8eaf6; color: #3f51b5'
                }
                return colors.get(val, '')
            
            styled_df = comparison_df.style.applymap(style_risk_level, subset=['Risk Level'])\
                                           .applymap(style_category, subset=['Category'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("Could not calculate risk percentiles. Please check your inputs.")
    
    # Recommendations section
    if 'results' in locals() and results:
        st.markdown("### 💡 Personalized Recommendations")
        
        # Check for mortality risk first
        mortality_result = [r for r in results if r['disease'] == 'Death']
        if mortality_result and mortality_result[0]['percentile'] >= 75:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2c2c54 0%, #40407a 100%); 
                        padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;
                        border: 3px solid #6c5ce7;">
                <h4>🚨 Priority Alert: Elevated Mortality Risk</h4>
                <p>Your mortality risk percentile ({mortality_result[0]['percentile']}th) indicates you're in a higher risk group.</p>
                <ul>
                    <li>🏥 <strong>Schedule comprehensive medical evaluation immediately</strong></li>
                    <li>🩺 Discuss all risk factors with your healthcare provider</li>
                    <li>📋 Consider preventive screenings and health monitoring</li>
                    <li>💊 Review all medications and supplements with your doctor</li>
                    <li>🚨 Address modifiable risk factors urgently</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        high_risk_diseases = [r for r in results if r['percentile'] >= 90 and r['disease'] != 'Death']
        moderate_risk_diseases = [r for r in results if 75 <= r['percentile'] < 90 and r['disease'] != 'Death']
        
        if high_risk_diseases:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
                <h4>⚠️ High Priority Actions</h4>
                <p>You are in the <strong>top 10%</strong> risk group for: 
                   <strong>{', '.join([d['disease'] for d in high_risk_diseases])}</strong></p>
                <ul>
                    <li>🏥 Schedule a medical consultation soon</li>
                    <li>🔍 Discuss specific screening tests</li>
                    <li>💪 Implement immediate lifestyle changes</li>
                    <li>📅 Set up regular monitoring schedule</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if moderate_risk_diseases:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
                <h4>📈 Moderate Risk - Take Action</h4>
                <p>Higher than average risk for: 
                   <strong>{', '.join([d['disease'] for d in moderate_risk_diseases])}</strong></p>
                <ul>
                    <li>🏃 Increase physical activity</li>
                    <li>🥗 Optimize diet and weight management</li>
                    <li>😴 Ensure adequate sleep (7-9 hours)</li>
                    <li>🧘 Manage stress effectively</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Heart rate specific advice
        if current_hr >= 85:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%); 
                        padding: 1.5rem; border-radius: 15px; color: #2c3e50; margin: 1rem 0;">
                <h4>💗 Heart Rate Optimization</h4>
                <p>Your resting HR ({current_hr} bpm) could be improved:</p>
                <ul>
                    <li>🏃‍♂️ Cardio exercise 150+ min/week</li>
                    <li>🧘‍♀️ Stress management techniques</li>
                    <li>☕ Limit caffeine intake</li>
                    <li>💤 Prioritize quality sleep</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # BMI-specific advice
        bmi_category, _ = get_bmi_category(bmi)
        if bmi_category in ["Overweight", "Obese"]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
                <h4>⚖️ Weight Management Recommendations</h4>
                <p>Your BMI ({bmi}) is in the <strong>{bmi_category.lower()}</strong> range:</p>
                <ul>
                    <li>🎯 Target gradual weight loss (1-2 lbs/week)</li>
                    <li>🍽️ Focus on portion control and balanced nutrition</li>
                    <li>💧 Increase water intake</li>
                    <li>🚶‍♀️ Add more daily physical activity</li>
                    <li>📱 Consider using a food diary or app</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # BMI Information Section
    with st.expander("📊 Understanding BMI Categories"):
        st.markdown("""
        ### BMI (Body Mass Index) Categories
        
        **BMI Categories (WHO Standards):**
        - **Underweight:** BMI < 18.5
        - **Normal weight:** BMI 18.5-24.9
        - **Overweight:** BMI 25.0-29.9
        - **Obese:** BMI ≥ 30.0
        
        **BMI Calculation:**
        BMI = Weight (kg) ÷ Height (m)²
        
        **Important Notes:**
        - BMI is a screening tool, not a diagnostic measure
        - It doesn't account for muscle mass, bone density, or body composition
        - Athletes with high muscle mass may have high BMI but low body fat
        - Age, gender, and ethnicity can affect BMI interpretation
        - Consult healthcare providers for personalized assessment
        """)
    
    # Methodology explanation
    with st.expander("📚 How This Calculator Works"):
        st.markdown("""
        ### Percentile-Based Risk Assessment
        
        **What This Shows:**
        - Your risk percentile within people of your **same age group and gender**
        - For example, "85th percentile" means you have higher risk than 85% of people in your demographic
        
        **Risk Categories:**
        - **90th+ percentile:** Top 10% highest risk - medical consultation recommended
        - **75th-89th percentile:** Higher than average risk - lifestyle changes advised  
        - **50th-74th percentile:** Average risk - maintain healthy habits
        - **Below 50th percentile:** Lower than average risk - continue current lifestyle
        
        **Special Note on Mortality Risk:**
        - **Death/Mortality Risk** is assessed using the same percentile system
        - This represents your overall mortality risk compared to your demographic peers
        - Higher percentiles indicate elevated risk of mortality from all causes
        - This is **not a prediction** but a population-based risk comparison
        
        **Calculation Method:**
        1. Calculate your **Linear Predictor (LP)** using Cox model coefficients
        2. Compare your LP to the distribution of LPs in your demographic group
        3. Determine your percentile rank within that group
        
        **Factors Included:**
        - Age (continuous variable)
        - Gender (Male/Female)
        - **BMI (calculated from height/weight)**
        - Heart rate category (<60, 60-69, 70-79, 80-89, ≥90 bpm)
        - Smoking status (Never/Former/Current)
        - Drinking status (Never/Former/Current)
        
        **Data Source:**
        - Cox regression coefficients from Taiwan Biobank study
        - LP distributions calculated from population data (demographic-specific)
        - Includes 18 disease conditions plus mortality risk assessment
        
        **Important Notes:**
        - Results are population-based associations, not individual predictions
        - Multiple factors beyond those in the model may affect individual risk
        - This tool is for educational purposes and doesn't replace medical advice
        - Genetic factors and family history are not included in this model
        - **Mortality risk assessment should be discussed with healthcare providers**
        
        **Advantages of Percentile Approach:**
        - More intuitive than hazard ratios ("top 10%" vs "1.5x higher risk")
        - Age and gender-specific comparisons (fair demographic matching)
        - Clear action thresholds (90th percentile = high priority)
        - Better motivates preventive actions
        - Includes comprehensive mortality risk assessment
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This percentile risk calculator is for educational purposes only. 
        Risk percentiles are based on population data and Cox regression models. Individual risk may vary 
        significantly due to genetic factors, family history, and other variables not captured in the model. 
        High percentile rankings (especially 90th+) suggest discussing with a healthcare provider, but this 
        tool cannot diagnose conditions or replace professional medical evaluation. 
        <strong>Mortality risk assessments should be discussed with qualified healthcare professionals.</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Risk percentiles calculated using Cox regression model and demographic-specific LP distributions.<br>
        BMI automatically calculated from height ({height} {height_unit}) and weight ({weight} {weight_unit}).<br>
        Percentile rankings show your position relative to people of your same age group and gender.<br>
        Assessment includes {len(diseases)} conditions: 18 diseases + mortality risk.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
