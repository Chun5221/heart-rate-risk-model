# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:51:52 2025

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
    page_title="❤️ 心率風險計算器",
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
    
    # Clean up the disease name column (there's a typo in the original)
    df.columns = ['Disease_Name', 'Variable', 'Coef']
    
    return df

# Disease name translation dictionary
def get_disease_translations():
    """Get disease name translations from English to Traditional Chinese"""
    return {
        'Atrial Fibrillation': '心房顫動',
        'Anxiety': '焦慮症',
        'Chronic Kidney Disease': '慢性腎臟病',
        'GERD': '胃食道逆流',
        'Heart Failure': '心臟衰竭',
        'Myocardial Infarction': '心肌梗塞',
        'Type 2 Diabetes': '第二型糖尿病',
        'Anemia': '貧血',
        'Angina Pectoris': '心絞痛',
        'Asthma': '氣喘',
        'Atherosclerosis': '動脈粥樣硬化',
        'Cardiac Arrhythmia': '心律不整',
        'Depression': '憂鬱症',
        'Hypertension': '高血壓',
        'Ischemic Heart Disease': '缺血性心臟病',
        'Ischemic Stroke': '缺血性腦中風',
        'Migraine': '偏頭痛',
        "Parkinson's Disease": '帕金森氏症'
    }

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
        if gender == '女性':
            gender_coef = disease_coefs[disease_coefs['Variable'] == 'FERMALE']
            if not gender_coef.empty and gender_coef.iloc[0]['Coef'] != 'REF':
                lp += float(gender_coef.iloc[0]['Coef'])
        
        # BMI (continuous variable)
        bmi_coef = disease_coefs[disease_coefs['Variable'] == 'BMI']
        if not bmi_coef.empty and bmi_coef.iloc[0]['Coef'] != 'REF':
            lp += float(bmi_coef.iloc[0]['Coef']) * bmi
        
        # Smoking Status (reference is Never_smoke)
        if smoking_status == '曾經吸菸':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        elif smoking_status == '目前吸菸':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Now_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        
        # Drinking Status (reference is Never_drink)
        if drinking_status == '曾經飲酒':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        elif drinking_status == '目前飲酒':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Now_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        
        # Calculate hazard ratio
        hazard_ratio = math.exp(lp)
        return hazard_ratio
    
    except Exception as e:
        st.error(f"計算 {disease_name} 風險比時發生錯誤：{str(e)}")
        return None

def calculate_benchmark_comparison(disease_name, user_age, user_gender, user_hr, user_bmi, 
                                 user_smoking, user_drinking, model_df):
    """
    Calculate user's risk relative to a benchmark person
    Benchmark: Age=40, Male, HR=65 (60-69 category), BMI=22, Never smoker, Never drinker
    """
    # Define benchmark person
    benchmark_age = 40
    benchmark_gender = '男性'
    benchmark_hr = 65  # Falls in HR_cat60-69
    benchmark_bmi = 22.0
    benchmark_smoking = '從未吸菸'
    benchmark_drinking = '從未飲酒'
    
    # Calculate benchmark person's hazard ratio
    benchmark_hr_ratio = calculate_cox_hazard_ratio(
        disease_name, benchmark_age, benchmark_gender, benchmark_hr, 
        benchmark_bmi, benchmark_smoking, benchmark_drinking, model_df
    )
    
    # Calculate user's hazard ratio
    user_hr_ratio = calculate_cox_hazard_ratio(
        disease_name, user_age, user_gender, user_hr, 
        user_bmi, user_smoking, user_drinking, model_df
    )
    
    # Return relative risk compared to benchmark
    if benchmark_hr_ratio is not None and user_hr_ratio is not None and benchmark_hr_ratio != 0:
        return user_hr_ratio / benchmark_hr_ratio
    else:
        return None

def get_risk_color(relative_risk):
    """Get color based on relative risk compared to benchmark"""
    if relative_risk < 1.1:
        return "#27ae60"  # Green
    elif relative_risk < 1.3:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(relative_risk):
    """Get risk level description compared to benchmark"""
    if relative_risk < 1.1:
        return "相似風險"
    elif relative_risk < 1.3:
        return "中度風險"
    else:
        return "高度風險"

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
        return '心血管疾病'
    elif disease_name in metabolic_diseases:
        return '代謝性疾病'
    elif disease_name in mental_health:
        return '心理健康'
    else:
        return '其他疾病'

def main():
    # Load model coefficients
    model_df = load_model_coefficients()
    disease_translations = get_disease_translations()
    
    # Get unique diseases
    diseases = model_df['Disease_Name'].unique().tolist()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ 心率風險計算器</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基於Cox回歸模型評估您的疾病風險</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 輸入參數")
        
        # Personal information
        age = st.slider("年齡", 20, 90, 65, help="您目前的年齡")
        gender = st.selectbox("性別", ["男性", "女性"], help="生理性別")
        bmi = st.slider("身體質量指數 (BMI)", 15.0, 40.0, 24.0, step=0.1, help="身體質量指數")
        
        # Heart rate
        st.markdown("### 💓 心率資訊")
        current_hr = st.slider(
            "目前靜息心率 (每分鐘心跳數)", 
            40, 120, 72, 
            help="您目前的靜息心率（每分鐘心跳數）"
        )
        
        # Lifestyle factors
        st.markdown("### 🚬 生活習慣因子")
        smoking_status = st.selectbox(
            "吸菸狀況", 
            ["從未吸菸", "曾經吸菸", "目前吸菸"],
            help="您的吸菸歷史"
        )
        
        drinking_status = st.selectbox(
            "飲酒狀況",
            ["從未飲酒", "曾經飲酒", "目前飲酒"],
            help="您的飲酒歷史"
        )
        
        # Disease category filter
        st.markdown("### 🔍 依類別篩選")
        all_categories = ['心血管疾病', '代謝性疾病', '心理健康', '其他疾病']
        selected_categories = st.multiselect(
            "選擇要顯示的疾病類別：",
            all_categories,
            default=all_categories
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 疾病風險評估結果")
        
        # Calculate relative risks compared to benchmark for all diseases
        relative_risks = {}
        filtered_diseases = []
        
        for disease in diseases:
            category = categorize_diseases(disease)
            if category in selected_categories:
                filtered_diseases.append(disease)
                rel_risk = calculate_benchmark_comparison(
                    disease, age, gender, current_hr, bmi, 
                    smoking_status, drinking_status, model_df
                )
                if rel_risk is not None:
                    relative_risks[disease] = rel_risk
        
        if relative_risks:
            # Create risk visualization with Chinese disease names
            diseases_list = list(relative_risks.keys())
            diseases_chinese = [disease_translations.get(disease, disease) for disease in diseases_list]
            risk_values = list(relative_risks.values())
            colors = [get_risk_color(risk) for risk in risk_values]
            
            # Sort by relative risk for better visualization
            sorted_data = sorted(zip(diseases_chinese, risk_values, colors), key=lambda x: x[1], reverse=True)
            diseases_sorted, risks_sorted, colors_sorted = zip(*sorted_data)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=diseases_sorted,
                    x=risks_sorted,
                    orientation='h',
                    marker_color=colors_sorted,
                    text=[f"{risk:.2f}倍" for risk in risks_sorted],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>相對風險：%{x:.2f}倍 vs 基準人<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="您的風險 vs 健康基準人",
                xaxis_title="相對風險（vs 基準人）",
                yaxis_title="疾病",
                height=max(400, len(diseases_list) * 25),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                         annotation_text="與基準人相同 (1.0倍)", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level indicators
            st.markdown("### 🎯 風險等級摘要")
            cols = st.columns(3)
            
            high_risk = sum(1 for risk in risk_values if risk >= 1.3)
            moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.3)
            similar_risk = sum(1 for risk in risk_values if risk < 1.1)
            
            with cols[0]:
                st.metric("🔴 高度風險 vs 基準", high_risk)
            with cols[1]:
                st.metric("🟡 中度風險 vs 基準", moderate_risk)
            with cols[2]:
                st.metric("🟢 相似風險與基準", similar_risk)
        else:
            st.warning("無法計算有效的風險比較。請檢查您的輸入參數。")
    
    with col2:
        st.markdown("### 💡 心率狀況")
        
        # Heart rate category display
        hr_category = get_heart_rate_category(current_hr)
        category_display = {
            'HR_cat<60': '< 60 bpm (心搏過緩)',
            'HR_cat60-69': '60-69 bpm (正常)',
            'HR_cat70-79': '70-79 bpm (正常偏高)',
            'HR_cat80-89': '80-89 bpm (偏高)',
            'HR_cat>=90': '≥ 90 bpm (高)'
        }
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 您的心率類別</h4>
            <p><strong>{current_hr} bpm</strong></p>
            <p>{category_display.get(hr_category, hr_category)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model information
        st.markdown("### 📚 模型資訊")
        st.markdown("""
        <div class="info-box">
            <strong>基準人：</strong><br>
            • 年齡：40歲<br>
            • 性別：男性<br>
            • BMI：22.0<br>
            • 心率：65 bpm (60-69範圍)<br>
            • 吸菸：從未吸菸<br>
            • 飲酒：從未飲酒<br><br>
            <em>您的風險與此健康基準個體進行比較。</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Personal profile summary
        st.markdown("### 👤 您的個人檔案")
        st.markdown(f"""
        <div class="info-box">
            <strong>個人資料：</strong><br>
            • 年齡：{age} 歲<br>
            • 性別：{gender}<br>
            • BMI：{bmi:.1f}<br>
            • 吸菸：{smoking_status}<br>
            • 飲酒：{drinking_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed risk breakdown
    if relative_risks:
        st.markdown("### 📋 詳細風險分析")
        
        # Create detailed table with categories
        risk_df = pd.DataFrame({
            '疾病': [disease_translations.get(disease, disease) for disease in relative_risks.keys()],
            '類別': [categorize_diseases(disease) for disease in relative_risks.keys()],
            '相對風險 vs 基準': [f"{risk:.3f}倍" for risk in relative_risks.values()],
            '風險等級': [get_risk_level(risk) for risk in relative_risks.values()],
            '風險解釋': [
                f"{((risk-1)*100):+.1f}% vs 基準人" if risk != 1.0 else "與基準人相同"
                for risk in relative_risks.values()
            ]
        })
        
        # Sort by relative risk
        risk_df = risk_df.sort_values('相對風險 vs 基準', ascending=False, 
                                    key=lambda x: x.str.replace('倍', '').astype(float))
        
        # Color code the table
        def style_risk_level(val):
            if val == "高度風險":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "中度風險":
                return 'background-color: #fff3e0; color: #ef6c00'
            else:
                return 'background-color: #e8f5e8; color: #2e7d32'
        
        styled_df = risk_df.style.applymap(style_risk_level, subset=['風險等級'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Heart rate sensitivity analysis
        st.markdown("### 📊 心率敏感性分析")
        
        # Select top 6 diseases with highest relative risks for trend analysis
        top_diseases = sorted(relative_risks.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if len(top_diseases) > 0:
            hr_range = np.arange(50, 101, 5)
            
            # Calculate trends for top diseases
            fig_trend = make_subplots(
                rows=2, cols=3,
                subplot_titles=[disease_translations.get(disease, disease) for disease, _ in top_diseases],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            for i, (disease, _) in enumerate(top_diseases):
                row = (i // 3) + 1
                col = (i % 3) + 1
                
                trend_risks = []
                for test_hr in hr_range:
                    rel_risk = calculate_benchmark_comparison(
                        disease, age, gender, test_hr, bmi, 
                        smoking_status, drinking_status, model_df
                    )
                    trend_risks.append(rel_risk if rel_risk is not None else 1.0)
                
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
                        y=trend_risks,
                        mode='lines+markers',
                        name=disease_translations.get(disease, disease),
                        line=dict(width=3),
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add current point
                current_risk = relative_risks[disease]
                fig_trend.add_trace(
                    go.Scatter(
                        x=[current_hr], 
                        y=[current_risk],
                        mode='markers',
                        name=f"您的風險",
                        marker=dict(size=12, color='red', symbol='star'),
                        showlegend=False,
                        hovertemplate=f'<b>{disease_translations.get(disease, disease)}</b><br>心率：{current_hr} bpm<br>相對風險：{current_risk:.2f}倍<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig_trend.update_layout(
                height=600,
                title_text="心率敏感性 vs 基準人",
                showlegend=False,
                font=dict(size=10)
            )
            
            # Update axes for each subplot
            for i in range(len(top_diseases)):
                row = (i // 3) + 1
                col = (i % 3) + 1
                fig_trend.update_xaxes(title_text="心率 (bpm)", row=row, col=col)
                fig_trend.update_yaxes(title_text="相對風險 vs 基準", row=row, col=col)
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Add legend explanation
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>📖 圖表說明：</strong><br>
                • <span style="color: gray;">灰色虛線</span>：與基準人相同風險 (1.0倍)<br>
                • <span style="color: red;">紅色星號</span>：您目前的風險水準<br>
                • 顯示不同心率下您與基準人的風險變化
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations based on results
    st.markdown("### 💊 個人化建議")
    
    if relative_risks:
        max_risk = max(relative_risks.values())
        high_risk_diseases = [disease_translations.get(disease, disease) for disease, risk in relative_risks.items() if risk >= 1.3]
        
        if max_risk >= 1.3:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ 風險高於基準人</strong><br>
                相較於健康的40歲基準人，您在以下疾病的風險較高：
                <strong>{', '.join(high_risk_diseases[:3])}</strong>
                {'等疾病' if len(high_risk_diseases) > 3 else ''}。<br><br>
                <strong>建議：</strong><br>
                • 考慮諮詢醫療專業人員<br>
                • 專注於可改變的風險因子（運動、飲食、壓力管理）<br>
                • 定期健康監測和預防性護理
            </div>
            """, unsafe_allow_html=True)
        elif max_risk >= 1.1:
            st.markdown("""
            <div class="warning-box">
                <strong>⚡ 風險略高於基準人</strong><br>
                您的風險檔案顯示相較於基準人有中度較高的風險。<br><br>
                <strong>建議：</strong><br>
                • 繼續健康的生活習慣<br>
                • 定期體能活動和均衡營養<br>
                • 定期監測心血管健康<br>
                • 考慮壓力管理技巧
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>✅ 風險與健康基準人相似</strong><br>
                您的風險檔案與我們的健康基準人相似或更好。<br><br>
                <strong>請維持：</strong><br>
                • 繼續目前健康的生活方式<br>
                • 定期運動和均衡營養<br>
                • 例行預防性護理和健康監測
            </div>
            """, unsafe_allow_html=True)
        
        # Specific lifestyle recommendations based on heart rate
        if current_hr >= 90:
            st.markdown("""
            <div class="warning-box">
                <strong>🏃 心率特定建議：</strong><br>
                您的靜息心率（≥90 bpm）明顯高於基準人（65 bpm）。建議：<br>
                • 進行有氧運動訓練以改善心血管適能<br>
                • 壓力降低技巧（冥想、瑜珈）<br>
                • 充足睡眠（每晚7-9小時）<br>
                • 限制咖啡因和刺激物<br>
                • 醫療評估以排除潛在疾病
            </div>
            """, unsafe_allow_html=True)
        elif current_hr >= 80:
            st.markdown("""
            <div class="info-box">
                <strong>💓 心率優化：</strong><br>
                您的心率高於基準人（65 bpm）。優化建議：<br>
                • 每週150分鐘以上規律有氧運動<br>
                • 維持健康體重（BMI 18.5-24.9）<br>
                • 有效管理壓力<br>
                • 保持水分充足和充分休息
            </div>
            """, unsafe_allow_html=True)
    
    # Model methodology explanation
    with st.expander("📊 模型方法與限制"):
        st.markdown("""
        ### Cox回歸模型與基準比較
        
        **基準人（固定參考）：**
        - **年齡：** 40歲，**性別：** 男性，**BMI：** 22.0
        - **心率：** 65 bpm（60-69類別）
        - **生活習慣：** 從未吸菸，從未飲酒
        
        **模型結構：**
        - **心率類別：** <60, 60-69（參考組）, 70-79, 80-89, ≥90 bpm
        - **調整因子：** 年齡、性別、BMI、吸菸狀況、飲酒狀況
        - **輸出：** 您相對於基準人的風險
        
        **結果解釋：**
        - **1.0倍：** 與基準人相同風險
        - **>1.0倍：** 風險高於基準人（例如：1.5倍 = 風險高50%）
        - **<1.0倍：** 風險低於基準人（例如：0.8倍 = 風險低20%）
        
        **重要限制：**
        - 結果基於台灣生物資料庫的族群關聯性
        - 個人風險可能因遺傳因子和未測量變項而有所不同
        - 模型未建立因果關係，僅顯示關聯性
        - 不能替代專業醫療評估
        - 基準代表健康個體檔案
        
        **臨床意義：**
        基準人（40歲健康男性）代表低風險個體。
        與此基準比較有助於以實用方式了解您的相對風險。
        """)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ 醫療免責聲明：</strong> 此計算器使用Cox回歸模型結果將您的風險檔案與健康基準人進行比較，
        僅供教育目的使用。相對風險基於族群資料，不應取代專業醫療建議。
        模型中未包含的個別風險因子和健康狀況可能顯著影響您的實際風險。
        個人健康評估和治療決定請務必諮詢醫療專業人員。
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>風險計算基於台灣生物資料庫研究的Cox回歸模型係數。<br>
        基準人：40歲男性，BMI 22，心率65 bpm，從未吸菸/飲酒。</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()