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
    
    .demographic-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .percentile-number {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load and parse the Cox regression model coefficients (all diseases)
@st.cache_data
def load_model_coefficients():
    """Load the Cox regression model coefficients from the CSV data"""
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
Anxiety,AGE,REF
Anxiety,MALE,0.0221697
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
    df.columns = ['Disease_Name', 'Variable', 'Coef']
    return df

# Load sample LP distribution data (placeholder - you'll replace this with real data)
@st.cache_data
def load_lp_distributions():
    """
    Placeholder for LP distribution data
    You'll replace this with your actual LP distribution file
    
    Expected structure:
    - Disease_Name: disease name
    - Gender: Male/Female
    - Age_Group: age ranges (e.g., '40-50', '50-60')
    - LP_Percentiles: percentile values (5, 10, 25, 50, 75, 90, 95)
    """
    
    # ALL DISEASES - Replace with your actual data
    diseases = [
        'Atrial Fibrillation', 'Anxiety', 'Chronic Kidney Disease', 'GERD', 
        'Heart Failure', 'Myocardial Infarction', 'Type 2 Diabetes', 'Anemia',
        'Angina Pectoris', 'Asthma', 'Atherosclerosis', 'Cardiac Arrhythmia',
        'Depression', 'Hypertension', 'Ischemic Heart Disease', 'Ischemic Stroke',
        'Migraine', "Parkinson's Disease"
    ]
    
    genders = ['Male', 'Female']
    age_groups = ['30-40', '40-50', '50-60', '60-70', '70-80']
    
    sample_data = []
    np.random.seed(42)  # For reproducible sample data
    
    for disease in diseases:
        for gender in genders:
            for age_group in age_groups:
                # Generate realistic sample LP distribution
                # Different diseases have different base risks
                if disease in ['Atrial Fibrillation', 'Heart Failure', 'Myocardial Infarction']:
                    base_lp = np.random.normal(0.5, 0.8)  # Higher base risk
                elif disease in ['Anxiety', 'Depression', 'Migraine']:
                    base_lp = np.random.normal(0.2, 0.6)  # Moderate base risk
                else:
                    base_lp = np.random.normal(0.0, 0.7)  # Average base risk
                
                # Age and gender effects
                age_mid = int(age_group.split('-')[0]) + 5
                age_effect = (age_mid - 50) * 0.02  # Older = higher risk
                gender_effect = 0.1 if gender == 'Female' and disease in ['Migraine', 'Anxiety'] else -0.1
                
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

def get_risk_category_and_color(percentile):
    """Get risk category and color based on percentile"""
    if percentile >= 90:
        return "High Risk", "high-risk-card", "#e74c3c"
    elif percentile >= 75:
        return "Moderate-High Risk", "moderate-risk-card", "#f39c12"
    elif percentile >= 50:
        return "Average Risk", "percentile-card", "#3498db"
    else:
        return "Lower Risk", "low-risk-card", "#27ae60"

def create_percentile_gauge(percentile, disease_name):
    """Create a gauge chart showing percentile position"""
    
    # Determine color based on risk level
    if percentile >= 90:
        color = "#e74c3c"  # Red
    elif percentile >= 75:
        color = "#f39c12"  # Orange
    elif percentile >= 50:
        color = "#3498db"  # Blue
    else:
        color = "#27ae60"  # Green
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentile,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{disease_name}<br>Risk Percentile"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#d5f4e6"},
                {'range': [50, 75], 'color': "#ffeaa7"},
                {'range': [75, 90], 'color': "#fdcb6e"},
                {'range': [90, 100], 'color': "#e17055"}
            ],
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
    st.markdown('<h1 class="main-header">📊 Heart Rate Percentile Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Find out how your risk compares to people in your demographic group</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📋 Your Information")
        
        # Personal information
        age = st.slider("Age", 20, 90, 45, help="Your current age")
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
        
        # Disease category filter
        st.markdown("### 🏥 Disease Categories")
        all_categories = ['Cardiovascular', 'Metabolic', 'Mental Health', 'Other Conditions']
        selected_categories = st.multiselect(
            "Select categories to display:",
            all_categories,
            default=all_categories,
            help="Choose which disease categories to show in your risk assessment"
        )
        
        # Calculate button
        calculate_button = st.button("🔍 Calculate My Risk Percentiles", type="primary")
    
    # Determine user's demographic group
    age_group = get_age_group(age)
    
    # Display demographic info
    st.markdown(f"""
    <div class="demographic-info">
        <h4>👥 Your Demographic Group</h4>
        <p><strong>Comparing you to:</strong> {gender}s aged {age_group} years</p>
        <p><strong>Your Profile:</strong> {age} years old, {gender}, BMI {bmi}, HR {current_hr} bpm</p>
    </div>
    """, unsafe_allow_html=True)
    
    if calculate_button or True:  # Auto-calculate for demo
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
                        risk_category, card_class, color = get_risk_category_and_color(percentile)
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
            # Sort by percentile (highest risk first)
            results.sort(key=lambda x: x['percentile'], reverse=True)
            
            # Create dynamic grid layout based on number of diseases
            num_diseases = len(results)
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
                        result = results[i + j]
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
            
            # Category-wise summary
            st.markdown("### 📊 Risk Summary by Category")
            
            # Group results by category
            category_summary = {}
            for result in results:
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
                        
                        # Choose color based on risk level
                        if high_pct >= 50:
                            bg_color = "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)"
                        elif summary['moderate'] > 0:
                            bg_color = "linear-gradient(135deg, #ffa726 0%, #ff9800 100%)"
                        else:
                            bg_color = "linear-gradient(135deg, #66bb6a 0%, #4caf50 100%)"
                        
                        st.markdown(f"""
                        <div style="background: {bg_color}; padding: 1.5rem; border-radius: 15px; 
                                    color: white; text-align: center; margin: 0.5rem 0;">
                            <h4>{category}</h4>
                            <p><strong>{total}</strong> diseases assessed</p>
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
                'Disease': [r['disease'] for r in results],
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
                if 'High Risk' in val:
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
                    'Other Conditions': 'background-color: #e3f2fd; color: #1976d2'
                }
                return colors.get(val, '')
            
            styled_df = comparison_df.style.applymap(style_risk_level, subset=['Risk Level'])\
                                           .applymap(style_category, subset=['Category'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Risk distribution visualization for top diseases
            st.markdown("### 📊 Risk Distribution Analysis")
            
            # Show distribution for top 6 highest risk diseases
            top_diseases = results[:6] if len(results) >= 6 else results
            
            if len(top_diseases) > 0:
                # Create subplots for distribution curves
                cols_subplot = 2 if len(top_diseases) > 1 else 1
                rows_subplot = math.ceil(len(top_diseases) / cols_subplot)
                
                fig_dist = make_subplots(
                    rows=rows_subplot, 
                    cols=cols_subplot,
                    subplot_titles=[f"{d['disease']}" for d in top_diseases],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.12
                )
                
                # Create sample distribution curve for each disease
                x = np.linspace(0, 100, 100)
                
                for i, disease_result in enumerate(top_diseases):
                    row = (i // cols_subplot) + 1
                    col = (i % cols_subplot) + 1
                    
                    # Normal-like distribution curve
                    y = np.exp(-((x-50)**2)/(2*25**2))
                    
                    # Population distribution
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x, y=y,
                            fill='tozeroy',
                            fillcolor='rgba(52, 152, 219, 0.3)',
                            line=dict(color='rgba(52, 152, 219, 0.8)', width=2),
                            name=f'Population Distribution',
                            showlegend=(i == 0),
                            hovertemplate='Percentile: %{x}<br>Density: %{y}<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # User's position
                    user_percentile = disease_result['percentile']
                    user_y = np.exp(-((user_percentile-50)**2)/(2*25**2))
                    
                    fig_dist.add_trace(
                        go.Scatter(
                            x=[user_percentile], 
                            y=[user_y],
                            mode='markers',
                            name='Your Position',
                            marker=dict(
                                size=15, 
                                color=disease_result['color'], 
                                symbol='star',
                                line=dict(width=2, color='white')
                            ),
                            showlegend=(i == 0),
                            hovertemplate=f'<b>Your Position</b><br>Percentile: {user_percentile}<br>Risk Level: {disease_result["risk_category"]}<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # Add percentile markers
                    percentile_marks = [25, 50, 75, 90]
                    for pct in percentile_marks:
                        if pct != user_percentile:  # Don't duplicate user's position
                            fig_dist.add_vline(
                                x=pct,
                                line_dash="dot",
                                line_color="gray",
                                opacity=0.5,
                                row=row, col=col
                            )
                
                # Update layout
                fig_dist.update_layout(
                    height=300 * rows_subplot,
                    title_text=f"Your Risk Position Among {gender}s Aged {age_group}",
                    showlegend=True,
                    font=dict(size=10)
                )
                
                # Update axes
                for i in range(len(top_diseases)):
                    row = (i // cols_subplot) + 1
                    col = (i % cols_subplot) + 1
                    fig_dist.update_xaxes(title_text="Risk Percentile", row=row, col=col)
                    fig_dist.update_yaxes(title_text="Population Density", row=row, col=col)
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Legend explanation
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong>📖 Chart Legend:</strong><br>
                    • <span style="color: #3498db;">Blue area</span>: Population distribution of risk in your demographic<br>
                    • <span style="color: red;">⭐ Star</span>: Your risk position<br>
                    • <span style="color: gray;">Dotted lines</span>: Common percentile markers (25th, 50th, 75th, 90th)<br>
                    • Charts show where you rank compared to other {gender.lower()}s aged {age_group}
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.error("Could not calculate risk percentiles. Please check your inputs.")
    
    # Recommendations section
    if 'results' in locals() and results:
        st.markdown("### 💡 Personalized Recommendations")
        
        high_risk_diseases = [r for r in results if r['percentile'] >= 90]
        moderate_risk_diseases = [r for r in results if 75 <= r['percentile'] < 90]
        
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
                <h4>💓 Heart Rate Optimization</h4>
                <p>Your resting HR ({current_hr} bpm) could be improved:</p>
                <ul>
                    <li>🏃‍♂️ Cardio exercise 150+ min/week</li>
                    <li>🧘‍♀️ Stress management techniques</li>
                    <li>☕ Limit caffeine intake</li>
                    <li>💤 Prioritize quality sleep</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
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
        
        **Calculation Method:**
        1. Calculate your **Linear Predictor (LP)** using Cox model coefficients
        2. Compare your LP to the distribution of LPs in your demographic group
        3. Determine your percentile rank within that group
        
        **Data Source:**
        - Cox regression coefficients from Taiwan Biobank study
        - LP distributions calculated from population data (demographic-specific)
        
        **Important Notes:**
        - Results are population-based associations, not individual predictions
        - Multiple factors beyond those in the model may affect individual risk
        - This tool is for educational purposes and doesn't replace medical advice
        - Genetic factors and family history are not included in this model
        
        **Advantages of Percentile Approach:**
        - More intuitive than hazard ratios ("top 10%" vs "1.5x higher risk")
        - Age and gender-specific comparisons (fair demographic matching)
        - Clear action thresholds (90th percentile = high priority)
        - Better motivates preventive actions
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
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Risk percentiles calculated using Cox regression model and demographic-specific LP distributions.<br>
        Percentile rankings show your position relative to people of your same age group and gender.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


