# -*- coding: utf-8 -*-
"""
Updated Heart Rate Risk Percentile Calculator
Modified on Fri Aug 29 2025 use new model results with categorical BMI
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

# Load and parse the Cox regression model coefficients from the new file
@st.cache_data
def load_model_coefficients():
    """Load the Cox regression model coefficients from the new CSV data file"""
    # Read the uploaded file data directly
    data_content = """Disease,Variable,Coef
DEATH,HR_cat60-69,REF
DEATH,HR_cat<60,0.03559
DEATH,HR_cat70-79,0.245138
DEATH,HR_cat80-89,0.552592
DEATH,HR_cat>=90,0.760978
DEATH,AGE,0.079182
DEATH,MALE,REF
DEATH,FERMALE,-0.687091
DEATH,bmi_normal,REF
DEATH,bmi_underweight,0.47405
DEATH,bmi_overweight,0.157231
DEATH,bmi_obese,0.327973
DEATH,Nerver_smoke,REF
DEATH,Ever_smoke,-0.006856
DEATH,Now_smoke,0.589674
DEATH,Nerver_drink,REF
DEATH,Ever_drink,0.492511
DEATH,Now_drink,0.247009
t2d,HR_cat60-69,REF
t2d,HR_cat<60,-0.149824
t2d,HR_cat70-79,0.120603
t2d,HR_cat80-89,0.463608
t2d,HR_cat>=90,0.595579
t2d,AGE,0.065918
t2d,MALE,REF
t2d,FERMALE,0.066054
t2d,bmi_normal,REF
t2d,bmi_underweight,-0.903821
t2d,bmi_overweight,0.438925
t2d,bmi_obese,1.043188
t2d,Nerver_smoke,REF
t2d,Ever_smoke,0.203295
t2d,Now_smoke,0.266898
t2d,Nerver_drink,REF
t2d,Ever_drink,0.047062
t2d,Now_drink,0.160096
af,HR_cat60-69,REF
af,HR_cat<60,0.19999
af,HR_cat70-79,0.02885
af,HR_cat80-89,0.35315
af,HR_cat>=90,0.58248
af,AGE,0.09624
af,MALE,REF
af,FERMALE,-0.77577
af,bmi_normal,REF
af,bmi_underweight,-1.1248
af,bmi_overweight,0.13015
af,bmi_obese,0.38955
af,Nerver_smoke,REF
af,Ever_smoke,0.09146
af,Now_smoke,-0.30515
af,Nerver_drink,REF
af,Ever_drink,-0.03565
af,Now_drink,0.17075
anxiety,HR_cat60-69,REF
anxiety,HR_cat<60,-0.02914
anxiety,HR_cat70-79,0.06873
anxiety,HR_cat80-89,0.08684
anxiety,HR_cat>=90,0.05308
anxiety,AGE,0.02841
anxiety,MALE,REF
anxiety,FERMALE,0.52751
anxiety,bmi_normal,REF
anxiety,bmi_underweight,0.11908
anxiety,bmi_overweight,-0.08715
anxiety,bmi_obese,-0.20691
anxiety,Nerver_smoke,REF
anxiety,Ever_smoke,0.12562
anxiety,Now_smoke,0.26357
anxiety,Nerver_drink,REF
anxiety,Ever_drink,0.3835
anxiety,Now_drink,0.18204
ckd,HR_cat60-69,REF
ckd,HR_cat<60,0.009221
ckd,HR_cat70-79,0.197843
ckd,HR_cat80-89,0.428913
ckd,HR_cat>=90,0.996948
ckd,AGE,0.078225
ckd,MALE,REF
ckd,FERMALE,-0.448045
ckd,bmi_normal,REF
ckd,bmi_underweight,-0.236165
ckd,bmi_overweight,0.413529
ckd,bmi_obese,0.869457
ckd,Nerver_smoke,REF
ckd,Ever_smoke,0.061577
ckd,Now_smoke,0.090903
ckd,Nerver_drink,REF
ckd,Ever_drink,0.240892
ckd,Now_drink,-0.11281
gerd,HR_cat60-69,REF
gerd,HR_cat<60,0.044928
gerd,HR_cat70-79,0.097214
gerd,HR_cat80-89,0.13919
gerd,HR_cat>=90,0.143378
gerd,AGE,0.023791
gerd,MALE,REF
gerd,FERMALE,0.143428
gerd,bmi_normal,REF
gerd,bmi_underweight,0.057324
gerd,bmi_overweight,0.003157
gerd,bmi_obese,-0.024195
gerd,Nerver_smoke,REF
gerd,Ever_smoke,0.136964
gerd,Now_smoke,0.028098
gerd,Nerver_drink,REF
gerd,Ever_drink,-0.015931
gerd,Now_drink,-0.077732
heart_failure,HR_cat60-69,REF
heart_failure,HR_cat<60,0.23987
heart_failure,HR_cat70-79,0.12069
heart_failure,HR_cat80-89,0.49327
heart_failure,HR_cat>=90,0.40777
heart_failure,AGE,0.07843
heart_failure,MALE,REF
heart_failure,FERMALE,-0.25105
heart_failure,bmi_normal,REF
heart_failure,bmi_underweight,-0.79253
heart_failure,bmi_overweight,0.48228
heart_failure,bmi_obese,1.02165
heart_failure,Nerver_smoke,REF
heart_failure,Ever_smoke,0.21295
heart_failure,Now_smoke,0.28765
heart_failure,Nerver_drink,REF
heart_failure,Ever_drink,0.07138
heart_failure,Now_drink,-0.22844
anemias,HR_cat60-69,REF
anemias,HR_cat<60,-0.133022
anemias,HR_cat70-79,0.106695
anemias,HR_cat80-89,0.295665
anemias,HR_cat>=90,0.173965
anemias,AGE,-0.012707
anemias,MALE,REF
anemias,FERMALE,1.233641
anemias,bmi_normal,REF
anemias,bmi_underweight,-0.102467
anemias,bmi_overweight,-0.041169
anemias,bmi_obese,-0.064932
anemias,Nerver_smoke,REF
anemias,Ever_smoke,0.281375
anemias,Now_smoke,-0.192209
anemias,Nerver_drink,REF
anemias,Ever_drink,0.227121
anemias,Now_drink,-0.14395
asthma,HR_cat60-69,REF
asthma,HR_cat<60,-0.05073
asthma,HR_cat70-79,-0.02929
asthma,HR_cat80-89,0.25838
asthma,HR_cat>=90,0.26451
asthma,AGE,0.03481
asthma,MALE,REF
asthma,FERMALE,0.33305
asthma,bmi_normal,REF
asthma,bmi_underweight,-0.02926
asthma,bmi_overweight,0.24735
asthma,bmi_obese,0.53764
asthma,Nerver_smoke,REF
asthma,Ever_smoke,0.25093
asthma,Now_smoke,0.04096
asthma,Nerver_drink,REF
asthma,Ever_drink,0.12783
asthma,Now_drink,-0.25044
atheroscclerosis,HR_cat60-69,REF
atheroscclerosis,HR_cat<60,0.213643
atheroscclerosis,HR_cat70-79,-0.075494
atheroscclerosis,HR_cat80-89,0.078521
atheroscclerosis,HR_cat>=90,-0.28857
atheroscclerosis,AGE,0.082831
atheroscclerosis,MALE,REF
atheroscclerosis,FERMALE,-0.659888
atheroscclerosis,bmi_normal,REF
atheroscclerosis,bmi_underweight,-0.405545
atheroscclerosis,bmi_overweight,0.241369
atheroscclerosis,bmi_obese,0.632751
atheroscclerosis,Nerver_smoke,REF
atheroscclerosis,Ever_smoke,0.154472
atheroscclerosis,Now_smoke,0.195769
atheroscclerosis,Nerver_drink,REF
atheroscclerosis,Ever_drink,0.237052
atheroscclerosis,Now_drink,-0.044435
cardiac_arrhythmia,HR_cat60-69,REF
cardiac_arrhythmia,HR_cat<60,0.236325
cardiac_arrhythmia,HR_cat70-79,0.072188
cardiac_arrhythmia,HR_cat80-89,0.293397
cardiac_arrhythmia,HR_cat>=90,0.502355
cardiac_arrhythmia,AGE,0.054073
cardiac_arrhythmia,MALE,REF
cardiac_arrhythmia,FERMALE,0.234635
cardiac_arrhythmia,bmi_normal,REF
cardiac_arrhythmia,bmi_underweight,-0.024617
cardiac_arrhythmia,bmi_overweight,0.197351
cardiac_arrhythmia,bmi_obese,0.220454
cardiac_arrhythmia,Nerver_smoke,REF
cardiac_arrhythmia,Ever_smoke,0.072327
cardiac_arrhythmia,Now_smoke,0.122191
cardiac_arrhythmia,Nerver_drink,REF
cardiac_arrhythmia,Ever_drink,0.024056
cardiac_arrhythmia,Now_drink,-0.072795
dementia,HR_cat60-69,REF
dementia,HR_cat<60,0.424231
dementia,HR_cat70-79,0.150302
dementia,HR_cat80-89,0.143758
dementia,HR_cat>=90,0.420416
dementia,AGE,0.23123
dementia,MALE,REF
dementia,FERMALE,0.220448
dementia,bmi_normal,REF
dementia,bmi_underweight,0.274853
dementia,bmi_overweight,0.228676
dementia,bmi_obese,0.246473
dementia,Nerver_smoke,REF
dementia,Ever_smoke,-0.093189
dementia,Now_smoke,0.032187
dementia,Nerver_drink,REF
dementia,Ever_drink,0.191469
dementia,Now_drink,-0.578951
depression,HR_cat60-69,REF
depression,HR_cat<60,0.04088
depression,HR_cat70-79,0.078
depression,HR_cat80-89,0.34531
depression,HR_cat>=90,0.38877
depression,AGE,0.02033
depression,MALE,REF
depression,FERMALE,0.55122
depression,bmi_normal,REF
depression,bmi_underweight,0.34393
depression,bmi_overweight,-0.1258
depression,bmi_obese,0.02531
depression,Nerver_smoke,REF
depression,Ever_smoke,0.36363
depression,Now_smoke,0.63215
depression,Nerver_drink,REF
depression,Ever_drink,0.42167
depression,Now_drink,0.06226
hypertension,HR_cat60-69,REF
hypertension,HR_cat<60,-0.074567
hypertension,HR_cat70-79,0.236392
hypertension,HR_cat80-89,0.473373
hypertension,HR_cat>=90,0.817759
hypertension,AGE,0.060626
hypertension,MALE,REF
hypertension,FERMALE,-0.158935
hypertension,bmi_normal,REF
hypertension,bmi_underweight,-0.631423
hypertension,bmi_overweight,0.559873
hypertension,bmi_obese,1.16172
hypertension,Nerver_smoke,REF
hypertension,Ever_smoke,0.097167
hypertension,Now_smoke,0.081723
hypertension,Nerver_drink,REF
hypertension,Ever_drink,0.222135
hypertension,Now_drink,0.339044
ischemic_heart_disease,HR_cat60-69,REF
ischemic_heart_disease,HR_cat<60,0.184318
ischemic_heart_disease,HR_cat70-79,0.007331
ischemic_heart_disease,HR_cat80-89,0.027601
ischemic_heart_disease,HR_cat>=90,-0.25622
ischemic_heart_disease,AGE,0.086786
ischemic_heart_disease,MALE,REF
ischemic_heart_disease,FERMALE,-0.345756
ischemic_heart_disease,bmi_normal,REF
ischemic_heart_disease,bmi_underweight,-0.592953
ischemic_heart_disease,bmi_overweight,0.321841
ischemic_heart_disease,bmi_obese,0.617107
ischemic_heart_disease,Nerver_smoke,REF
ischemic_heart_disease,Ever_smoke,0.133943
ischemic_heart_disease,Now_smoke,0.279845
ischemic_heart_disease,Nerver_drink,REF
ischemic_heart_disease,Ever_drink,0.377737
ischemic_heart_disease,Now_drink,-0.069477
ischemic_stroke,HR_cat60-69,REF
ischemic_stroke,HR_cat<60,0.071456
ischemic_stroke,HR_cat70-79,0.279148
ischemic_stroke,HR_cat80-89,0.437323
ischemic_stroke,HR_cat>=90,0.837057
ischemic_stroke,AGE,0.09066
ischemic_stroke,MALE,REF
ischemic_stroke,FERMALE,-0.260478
ischemic_stroke,bmi_normal,REF
ischemic_stroke,bmi_underweight,-0.068895
ischemic_stroke,bmi_overweight,0.440064
ischemic_stroke,bmi_obese,0.53031
ischemic_stroke,Nerver_smoke,REF
ischemic_stroke,Ever_smoke,0.209785
ischemic_stroke,Now_smoke,0.657639
ischemic_stroke,Nerver_drink,REF
ischemic_stroke,Ever_drink,0.203407
ischemic_stroke,Now_drink,0.136869
migraine,HR_cat60-69,REF
migraine,HR_cat<60,0.1785802
migraine,HR_cat70-79,-0.0222494
migraine,HR_cat80-89,0.3703828
migraine,HR_cat>=90,0.492667
migraine,AGE,0.0005605
migraine,MALE,REF
migraine,FERMALE,1.2063488
migraine,bmi_normal,REF
migraine,bmi_underweight,-0.3796271
migraine,bmi_overweight,0.1365617
migraine,bmi_obese,0.1635858
migraine,Nerver_smoke,REF
migraine,Ever_smoke,0.30192
migraine,Now_smoke,0.2017923
migraine,Nerver_drink,REF
migraine,Ever_drink,0.3245442
migraine,Now_drink,-0.3845493"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(data_content))
    
    # Clean up disease names and standardize
    disease_name_map = {
        'DEATH': 'Death',
        't2d': 'Type 2 Diabetes',
        'af': 'Atrial Fibrillation',
        'anxiety': 'Anxiety',
        'ckd': 'Chronic Kidney Disease',
        'gerd': 'GERD',
        'heart_failure': 'Heart Failure',
        'anemias': 'Anemia',
        'asthma': 'Asthma',
        'atheroscclerosis': 'Atherosclerosis',
        'cardiac_arrhythmia': 'Cardiac Arrhythmia',
        'dementia': 'Dementia',
        'depression': 'Depression',
        'hypertension': 'Hypertension',
        'ischemic_heart_disease': 'Ischemic Heart Disease',
        'ischemic_stroke': 'Ischemic Stroke',
        'migraine': 'Migraine'
    }
    
    df['Disease'] = df['Disease'].map(disease_name_map)
    return df

# Load sample LP distribution data (updated for new diseases)
@st.cache_data
def load_lp_distributions():
    """
    LP distribution data for the updated disease list
    """
    diseases = [
        'Death', 'Type 2 Diabetes', 'Atrial Fibrillation', 'Anxiety', 
        'Chronic Kidney Disease', 'GERD', 'Heart Failure', 'Anemia',
        'Asthma', 'Atherosclerosis', 'Cardiac Arrhythmia', 'Dementia',
        'Depression', 'Hypertension', 'Ischemic Heart Disease', 
        'Ischemic Stroke', 'Migraine'
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
                    base_lp = np.random.normal(0.8, 1.0)  # Higher base risk, wider distribution
                elif disease in ['Atrial Fibrillation', 'Heart Failure']:
                    base_lp = np.random.normal(0.5, 0.8)  # Higher base risk
                elif disease in ['Anxiety', 'Depression', 'Migraine']:
                    base_lp = np.random.normal(0.2, 0.6)  # Moderate base risk
                else:
                    base_lp = np.random.normal(0.0, 0.7)  # Average base risk
                
                # Age and gender effects
                age_mid = int(age_group.split('-')[0]) + 5
                age_effect = (age_mid - 50) * 0.02  # Older = higher risk
                
                # Special gender effects
                if disease == 'Death':
                    gender_effect = -0.3 if gender == 'Female' else 0
                elif gender == 'Female' and disease in ['Migraine', 'Anxiety', 'Anemia']:
                    gender_effect = 0.2
                else:
                    gender_effect = -0.1 if gender == 'Female' else 0
                
                adjusted_base = base_lp + age_effect + gender_effect
                
                # Generate percentile thresholds
                percentiles = np.random.normal(adjusted_base, 0.6, 7)
                percentiles = np.sort(percentiles)
                
                sample_data.append({
                    'Disease': disease,
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
    """Categorize BMI according to the model's categories"""
    if bmi < 18.5:
        return "Underweight", "#3498db"
    elif bmi < 24:  # Changed from 25 to 24
        return "Normal weight", "#27ae60"
    elif bmi < 27:  # Changed from 30 to 27
        return "Overweight", "#f39c12"
    else:
        return "Obese", "#e74c3c"

def get_bmi_model_category(bmi):
    """Get BMI category for model calculation (matches new model structure)"""
    if bmi < 18.5:
        return 'bmi_underweight'
    elif bmi < 24:
        return 'bmi_normal'  # Reference category
    elif bmi < 27:
        return 'bmi_overweight'
    else:
        return 'bmi_obese'

def categorize_diseases(disease_name):
    """Categorize diseases for filtering"""
    if disease_name == 'Death':
        return 'Mortality Risk'
    
    cardiovascular_diseases = [
        'Atrial Fibrillation', 'Heart Failure', 'Cardiac Arrhythmia', 
        'Ischemic Heart Disease', 'Atherosclerosis', 'Hypertension', 
        'Ischemic Stroke'
    ]
    
    metabolic_diseases = [
        'Type 2 Diabetes', 'Chronic Kidney Disease'
    ]
    
    mental_health = [
        'Anxiety', 'Depression', 'Dementia'
    ]
    
    other_conditions = [
        'GERD', 'Anemia', 'Asthma', 'Migraine'
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
    Updated for categorical BMI model
    """
    try:
        # Filter coefficients for this disease
        disease_coefs = model_df[model_df['Disease'] == disease_name].copy()
        
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
        
        # BMI Category (reference is bmi_normal)
        bmi_cat = get_bmi_model_category(bmi)
        if bmi_cat != 'bmi_normal':  # Only add coefficient if not reference category
            bmi_coef = disease_coefs[disease_coefs['Variable'] == bmi_cat]
            if not bmi_coef.empty and bmi_coef.iloc[0]['Coef'] != 'REF':
                lp += float(bmi_coef.iloc[0]['Coef'])
        
        # Smoking Status (reference is Nerver_smoke)
        if smoking_status == 'Former Smoker':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        elif smoking_status == 'Current Smoker':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Now_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        
        # Drinking Status (reference is Nerver_drink)
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
            (lp_dist_df['Disease'] == disease_name) & 
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
    diseases = model_df['Disease'].unique().tolist()
    
    # Header
    st.markdown('<h1 class="main-header">Heart Rate Risk Percentile Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Find out how your risk compares to people in your demographic group</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### Your Information")
        
        # Personal information
        age = st.slider("Age", 20, 90, 45, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # Height and Weight Section with BMI Calculator
        st.markdown("### Height & Weight")
        
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
                <h4>Calculated BMI</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: {bmi_color};">
                    BMI: {calculated_bmi}
                </p>
                <p style="color: {bmi_color}; font-weight: bold;">
                    Category: {bmi_category}
                </p>
                <small>Updated BMI categories: Normal <24, Overweight 24-27, Obese ≥27</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Use calculated BMI
            bmi = calculated_bmi
        else:
            st.error("Could not calculate BMI. Please check your height and weight values.")
            bmi = 24.0  # Default fallback
        
        # Heart rate
        st.markdown("### Heart Rate")
        current_hr = st.slider(
            "Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your resting heart rate in beats per minute"
        )
        
        # Lifestyle factors
        st.markdown("### Lifestyle")
        smoking_status = st.selectbox(
            "Smoking Status", 
            ["Never Smoker", "Former Smoker", "Current Smoker"]
        )
        
        drinking_status = st.selectbox(
            "Drinking Status",
            ["Never Drinker", "Former Drinker", "Current Drinker"]
        )
        
        # Disease category filter
        st.markdown("### Risk Categories")
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
        <h4>Your Profile & Demographic Group</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <p><strong>Comparing you to:</strong> {gender}s aged {age_group} years</p>
                <p><strong>Your Details:</strong> {age} years old, {gender}</p>
            </div>
            <div>
                <p><strong>Physical:</strong> BMI {bmi} ({get_bmi_category(bmi)[0]}), HR {current_hr} bpm</p>
                <p><strong>Lifestyle:</strong> {smoking_status.replace('Never Smoker', 'Non-smoker')}, {drinking_status.replace('Never Drinker', 'Non-drinker')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if True:  # Auto-calculate always
        st.markdown("### Your Risk Percentiles")
        
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
                st.markdown("### Mortality Risk Assessment")
                mort_result = mortality_result[0]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Create gauge chart for mortality
                    fig_mort = create_percentile_gauge(mort_result['percentile'], 'Death')
                    st.plotly_chart(fig_mort, use_container_width=True)
                
                with col2:
                    # Mortality risk interpretation
                    if mort_result['percentile'] >= 90:
                        interpretation = f"Critical: Top {100-mort_result['percentile']}% highest mortality risk"
                        recommendation = "Immediate medical consultation strongly advised"
                        urgency_color = "#c0392b"
                    elif mort_result['percentile'] >= 75:
                        interpretation = f"High mortality risk (higher than {mort_result['percentile']}%)"
                        recommendation = "Schedule comprehensive health evaluation"
                        urgency_color = "#e67e22"
                    elif mort_result['percentile'] >= 50:
                        interpretation = f"Moderate mortality risk (top {100-mort_result['percentile']}%)"
                        recommendation = "Focus on preventive health measures"
                        urgency_color = "#8e44ad"
                    else:
                        interpretation = f"Lower mortality risk (top {100-mort_result['percentile']}%)"
                        recommendation = "Continue healthy lifestyle practices"
                        urgency_color = "#27ae60"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {urgency_color} 0%, {mort_result['color']} 100%); 
                                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                        <h3>Mortality Risk: {mort_result['percentile']}th Percentile</h3>
                        <p style="font-size: 1.1rem; margin: 1rem 0;">{interpretation}</p>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p style="font-size: 1rem;"><strong>Recommendation:</strong> {recommendation}</p>
                        <p style="font-size: 0.9rem; opacity: 0.9;"><em>Based on population mortality data for {gender.lower()}s aged {age_group}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if disease_results:
                st.markdown("### Disease-Specific Risk Assessment")
                
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
                                    interpretation = f"Top {100-result['percentile']}% highest risk"
                                    recommendation = "Consider medical consultation"
                                elif result['percentile'] >= 75:
                                    interpretation = f"Higher than {result['percentile']}% of your demographic"
                                    recommendation = "Monitor closely, lifestyle changes"
                                elif result['percentile'] >= 50:
                                    interpretation = f"Average risk (top {100-result['percentile']}%)"
                                    recommendation = "Continue healthy habits"
                                else:
                                    interpretation = f"Lower risk (top {100-result['percentile']}%)"
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
            
            # Detailed risk comparison table
            st.markdown("### Detailed Risk Comparison")
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Condition': [r['disease'] for r in results],
                'Category': [r['category'] for r in results],
                'Your Percentile': [f"{r['percentile']}th" for r in results],
                'Risk Level': [r['risk_category'] for r in results],
                'BMI Category': [get_bmi_model_category(bmi).replace('bmi_', '').title() for _ in results],
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
            
            styled_df = comparison_df.style.applymap(style_risk_level, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("Could not calculate risk percentiles. Please check your inputs.")
    
    # BMI Information Section
    with st.expander("Understanding BMI Categories (Updated)"):
        st.markdown("""
        ### BMI Categories (Updated Model)
        
        **New BMI Categories used in this model:**
        - **Underweight:** BMI < 18.5
        - **Normal weight:** BMI 18.5-23.9
        - **Overweight:** BMI 24.0-26.9  
        - **Obese:** BMI ≥ 27.0
        
        **Changes from WHO standards:**
        - Normal range reduced from <25 to <24
        - Overweight range is 24-27 instead of 25-30
        - Obesity threshold lowered from 30 to 27
        
        **Note:** These updated thresholds may be more appropriate for certain populations and reflect the model's training data.
        """)
    
    # Methodology explanation
    with st.expander("How This Calculator Works"):
        st.markdown("""
        ### Updated Model Features
        
        **Key Changes:**
        - **Categorical BMI:** BMI is now categorized (underweight/normal/overweight/obese) instead of continuous
        - **Updated Diseases:** Includes new conditions like Dementia
        - **Refined Categories:** Adjusted BMI thresholds (Normal <24, Overweight 24-27, Obese ≥27)
        
        **Variables in Model:**
        - Age (continuous)
        - Gender (Male/Female)
        - **BMI categories** (underweight/normal/overweight/obese)
        - Heart rate categories (<60, 60-69, 70-79, 80-89, ≥90 bpm)  
        - Smoking status (Never/Former/Current)
        - Drinking status (Never/Former/Current)
        
        **Included Conditions:**
        - **Mortality Risk (Death)**
        - Type 2 Diabetes
        - Atrial Fibrillation  
        - Anxiety
        - Chronic Kidney Disease
        - GERD
        - Heart Failure
        - Anemia
        - Asthma
        - Atherosclerosis
        - Cardiac Arrhythmia
        - **Dementia (New)**
        - Depression
        - Hypertension
        - Ischemic Heart Disease
        - Ischemic Stroke
        - Migraine
        
        **Important Notes:**
        - BMI categories may differ from standard WHO classifications
        - Model trained on specific population data (Taiwan Biobank)
        - Percentiles show your rank within same age/gender demographic
        - This tool is for educational purposes only
        """)

if __name__ == "__main__":
    main()
