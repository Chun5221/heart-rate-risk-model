# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:52:58 2025

@author: chun5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from scipy import stats

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
    
    .percentile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .high-risk-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .moderate-risk-card {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .low-risk-card {
        background: linear-gradient(135deg, #00b894 0%, #27ae60 100%);
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
    # The CSV data from TWB_model1_sig.csv (same as before)
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
Anxiety,Now_drink,0.145487"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    df.columns = ['Disease_Name', 'Variable', 'Coef']
    
    return df

# TODO: This will be replaced with actual LP distribution data
@st.cache_data
def load_lp_distributions():
    """
    Load Linear Predictor (LP) score distributions by age-gender cohorts.
    
    This is a MOCK function that generates sample data.
    Replace this with actual LP distribution data when available.
    
    Expected data structure:
    {
        'disease_name': {
            'age_group': {
                'gender': {
                    'lp_scores': [array of LP scores],
                    'percentiles': [1, 5, 10, 25, 50, 75, 90, 95, 99]
                }
            }
        }
    }
    """
    
    # Mock data structure - replace with actual data loading
    diseases = ['Atrial Fibrillation', 'Anxiety', 'Type 2 Diabetes', 'Heart Failure']
    age_groups = ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    genders = ['Male', 'Female']
    
    lp_distributions = {}
    
    for disease in diseases:
        lp_distributions[disease] = {}
        for age_group in age_groups:
            lp_distributions[disease][age_group] = {}
            for gender in genders:
                # Generate mock LP distribution
                # In reality, these would be calculated from actual patient data
                n_samples = 1000
                
                # Create realistic LP distributions with some variation by demographics
                base_mean = 0.0
                if gender == 'Female' and disease == 'Anxiety':
                    base_mean = 0.3  # Higher anxiety risk for females
                elif gender == 'Male' and disease == 'Atrial Fibrillation':
                    base_mean = 0.2  # Higher AF risk for males
                
                # Age effect
                age_mid = int(age_group.split('-')[0]) if '-' in age_group else 70
                age_effect = (age_mid - 40) * 0.01  # Increase with age
                
                mean_lp = base_mean + age_effect
                std_lp = 0.5
                
                # Generate LP scores
                lp_scores = np.random.normal(mean_lp, std_lp, n_samples)
                
                # Calculate percentiles
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(lp_scores, percentiles)
                
                lp_distributions[disease][age_group][gender] = {
                    'lp_scores': lp_scores,
                    'percentiles': dict(zip(percentiles, percentile_values)),
                    'mean': np.mean(lp_scores),
                    'std': np.std(lp_scores),
                    'n_samples': n_samples
                }
    
    return lp_distributions

def get_age_group(age):
    """Convert age to age group category"""
    if age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '70+'

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

def calculate_linear_predictor(disease_name, age, gender, hr, bmi, smoking_status, drinking_status, model_df):
    """
    Calculate linear predictor (LP) using Cox regression coefficients
    LP = β₁X₁ + β₂X₂ + ... + βₖXₖ
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
        st.error(f"Error calculating linear predictor for {disease_name}: {str(e)}")
        return None

def calculate_percentile_rank(user_lp, lp_distribution):
    """
    Calculate what percentile the user's LP score falls into
    within their demographic cohort
    """
    if user_lp is None or lp_distribution is None:
        return None
    
    lp_scores = lp_distribution['lp_scores']
    percentile_rank = stats.percentileofscore(lp_scores, user_lp, kind='rank')
    
    return percentile_rank

def get_risk_level_from_percentile(percentile_rank):
    """Categorize risk level based on percentile rank"""
    if percentile_rank >= 90:
        return "High Risk", "high-risk-card"
    elif percentile_rank >= 75:
        return "Moderate-High Risk", "moderate-risk-card"
    elif percentile_rank >= 50:
        return "Average Risk", "percentile-card"
    else:
        return "Below Average Risk", "low-risk-card"

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
    # Load model coefficients and LP distributions
    model_df = load_model_coefficients()
    lp_distributions = load_lp_distributions()
    
    # Get available diseases (intersection of model and LP data)
    model_diseases = set(model_df['Disease_Name'].unique())
    lp_diseases = set(lp_distributions.keys())
    available_diseases = list(model_diseases.intersection(lp_diseases))
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your risk percentile within your age-gender cohort</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Input Parameters")
        
        # Personal information
        age = st.slider("Age", 20, 90, 45, help="Your current age")
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
            default=['Cardiovascular', 'Mental Health']
        )
    
    # Determine user's demographic cohort
    age_group = get_age_group(age)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Your Risk Percentiles")
        st.markdown(f"**Your Demographic Cohort:** {gender}s aged {age_group}")
        
        # Calculate percentile ranks for all diseases
        percentile_results = {}
        filtered_diseases = []
        
        for disease in available_diseases:
            category = categorize_diseases(disease)
            if category in selected_categories:
                filtered_diseases.append(disease)
                
                # Calculate user's linear predictor
                user_lp = calculate_linear_predictor(
                    disease, age, gender, current_hr, bmi, 
                    smoking_status, drinking_status, model_df
                )
                
                # Get LP distribution for user's demographic
                if (disease in lp_distributions and 
                    age_group in lp_distributions[disease] and
                    gender in lp_distributions[disease][age_group]):
                    
                    lp_dist = lp_distributions[disease][age_group][gender]
                    percentile_rank = calculate_percentile_rank(user_lp, lp_dist)
                    
                    if percentile_rank is not None:
                        percentile_results[disease] = {
                            'percentile': percentile_rank,
                            'lp_score': user_lp,
                            'cohort_mean': lp_dist['mean'],
                            'cohort_std': lp_dist['std'],
                            'n_samples': lp_dist['n_samples']
                        }
        
        if percentile_results:
            # Create percentile visualization
            diseases_list = list(percentile_results.keys())
            percentiles = [percentile_results[disease]['percentile'] for disease in diseases_list]
            
            # Sort by percentile for better visualization
            sorted_data = sorted(zip(diseases_list, percentiles), key=lambda x: x[1], reverse=True)
            diseases_sorted, percentiles_sorted = zip(*sorted_data)
            
            # Color code based on risk level
            colors = []
            for percentile in percentiles_sorted:
                if percentile >= 90:
                    colors.append('#e74c3c')  # Red - High risk
                elif percentile >= 75:
                    colors.append('#f39c12')  # Orange - Moderate-high risk
                elif percentile >= 50:
                    colors.append('#3498db')  # Blue - Average risk
                else:
                    colors.append('#27ae60')  # Green - Below average risk
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=diseases_sorted,
                    x=percentiles_sorted,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{percentile:.1f}%" for percentile in percentiles_sorted],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Percentile: %{x:.1f}% in your cohort<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=f"Your Risk Percentiles vs {gender}s aged {age_group}",
                xaxis_title="Percentile Rank (%)",
                yaxis_title="Diseases",
                height=max(400, len(diseases_list) * 30),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            
            # Add reference lines
            fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                         annotation_text="Average (50%)", annotation_position="top")
            fig.add_vline(x=75, line_dash="dash", line_color="orange", 
                         annotation_text="75th percentile", annotation_position="top")
            fig.add_vline(x=90, line_dash="dash", line_color="red", 
                         annotation_text="90th percentile", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level summary
            st.markdown("### 🎯 Risk Level Summary")
            cols = st.columns(4)
            
            high_risk = sum(1 for p in percentiles if p >= 90)
            mod_high_risk = sum(1 for p in percentiles if 75 <= p < 90)
            average_risk = sum(1 for p in percentiles if 50 <= p < 75)
            below_avg_risk = sum(1 for p in percentiles if p < 50)
            
            with cols[0]:
                st.metric("🔴 High Risk (≥90%)", high_risk)
            with cols[1]:
                st.metric("🟡 Moderate-High (75-89%)", mod_high_risk)
            with cols[2]:
                st.metric("🔵 Average (50-74%)", average_risk)
            with cols[3]:
                st.metric("🟢 Below Average (<50%)", below_avg_risk)
            
            # Detailed percentile cards
            st.markdown("### 📋 Detailed Risk Assessment")
            
            for disease in diseases_sorted:
                result = percentile_results[disease]
                percentile = result['percentile']
                risk_level, card_class = get_risk_level_from_percentile(percentile)
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>{disease}</h3>
                    <h2>{percentile:.1f}th percentile</h2>
                    <p><strong>{risk_level}</strong></p>
                    <p>You are at higher risk than {percentile:.1f}% of {gender.lower()}s aged {age_group}</p>
                    <small>Based on {result['n_samples']:,} individuals in your demographic cohort</small>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("No valid percentile calculations available. Please check your input parameters.")
    
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
        
        # Demographic cohort information
        st.markdown("### 👥 Your Demographic Cohort")
        st.markdown(f"""
        <div class="info-box">
            <strong>Comparison Group:</strong><br>
            • Gender: {gender}<br>
            • Age Group: {age_group}<br>
            • Cohort Size: ~1,000 individuals<br><br>
            <em>Your percentile rank shows where you stand compared to others with similar demographics.</em>
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
            • Heart Rate: {current_hr} bpm<br>
            • Smoking: {smoking_status}<br>
            • Drinking: {drinking_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Percentile distribution visualization
    if percentile_results:
        st.markdown("### 📈 Risk Score Distributions")
        
        # Select a disease for detailed distribution view
        selected_disease = st.selectbox(
            "Select a disease to view detailed risk distribution:",
            list(percentile_results.keys())
        )
        
        if selected_disease:
            result = percentile_results[selected_disease]
            disease_lp_dist = lp_distributions[selected_disease][age_group][gender]
            
            # Create distribution plot
            fig = go.Figure()
            
            # Population distribution
            fig.add_trace(go.Histogram(
                x=disease_lp_dist['lp_scores'],
                nbinsx=50,
                name=f"{gender}s aged {age_group}",
                opacity=0.7,
                yaxis='y',
                histnorm='probability'
            ))
            
            # User's score
            fig.add_vline(
                x=result['lp_score'],
                line_dash="solid",
                line_color="red",
                line_width=3,
                annotation_text=f"Your Score<br>({result['percentile']:.1f}th percentile)",
                annotation_position="top"
            )
            
            # Percentile markers
            percentile_markers = [25, 50, 75, 90]
            for p in percentile_markers:
                p_value = np.percentile(disease_lp_dist['lp_scores'], p)
                fig.add_vline(
                    x=p_value,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"{p}%",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title=f"{selected_disease} - Risk Score Distribution<br><sub>{gender}s aged {age_group}</sub>",
                xaxis_title="Linear Predictor (Risk Score)",
                yaxis_title="Probability Density",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown(f"""
            <div class="info-box">
                <strong>📖 Interpretation for {selected_disease}:</strong><br>
                • Your risk score: {result['lp_score']:.3f}<br>
                • Cohort average: {result['cohort_mean']:.3f}<br>
                • You are at the {result['percentile']:.1f}th percentile<br>
                • This means you have higher risk than {result['percentile']:.1f}% of {gender.lower()}s in your age group
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations based on percentile results
    st.markdown("### 💊 Personalized Recommendations")
    
    if percentile_results:
        high_percentile_diseases = [
            disease for disease, result in percentile_results.items() 
            if result['percentile'] >= 90
        ]
        
        moderate_percentile_diseases = [
            disease for disease, result in percentile_results.items() 
            if 75 <= result['percentile'] < 90
        ]
        
        if high_percentile_diseases:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ High Risk Conditions (≥90th percentile)</strong><br>
                You are in the top 10% risk group for: <strong>{', '.join(high_percentile_diseases[:3])}</strong>
                {' and others' if len(high_percentile_diseases) > 3 else ''}.<br><br>
                <strong>Immediate Actions:</strong><br>
                • Consult with a healthcare provider promptly<br>
                • Consider specialized screening for high-risk conditions<br>
                • Discuss preventive treatment options<br>
                • Implement aggressive lifestyle modifications
            </div>
            """, unsafe_allow_html=True)
        elif moderate_percentile_diseases:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚡ Moderate-High Risk (75-89th percentile)</strong><br>
                You are in the top 25% risk group for: <strong>{', '.join(moderate_percentile_diseases[:3])}</strong>
                {' and others' if len(moderate_percentile_diseases) > 3 else ''}.<br><br>
                <strong>Recommended Actions:</strong><br>
                • Schedule regular health check-ups<br>
                • Focus on modifiable risk factors<br>
                • Consider enhanced monitoring for these conditions<br>
                • Maintain healthy lifestyle habits
            </div>
            """, unsafe_allow_html=True)
        else:
            avg_percentile = np.mean([result['percentile'] for result in percentile_results.values()])
            st.markdown(f"""
            <div class="info-box">
                <strong>✅ Average to Low Risk Profile</strong><br>
                Your average risk percentile is {avg_percentile:.1f}%, indicating relatively low risk compared to your demographic peers.<br><br>
                <strong>Maintain Current Health:</strong><br>
                • Continue current healthy lifestyle<br>
                • Regular preventive care and health screenings<br>
                • Stay physically active and maintain healthy diet<br>
                • Monitor key health metrics regularly
            </div>
            """, unsafe_allow_html=True)
        
        # Heart rate specific recommendations
        if current_hr >= 90:
            st.markdown("""
            <div class="warning-box">
                <strong>🏃 Heart Rate Optimization Priority</strong><br>
                Your resting heart rate (≥90 bpm) may be contributing to your higher percentile rankings. Focus on:<br>
                • Aerobic exercise training (150+ minutes/week)<br>
                • Stress management techniques (meditation, yoga)<br>
                • Adequate sleep (7-9 hours nightly)<br>
                • Limit caffeine and stimulants<br>
                • Medical evaluation for underlying causes
            </div>
            """, unsafe_allow_html=True)
        elif current_hr >= 80:
            st.markdown("""
            <div class="info-box">
                <strong>💓 Heart Rate Improvement Opportunity</strong><br>
                Optimizing your heart rate could improve your percentile rankings:<br>
                • Regular cardiovascular exercise<br>
                • Stress reduction techniques<br>
                • Maintain healthy weight<br>
                • Stay well-hydrated and rested
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed risk breakdown table
    if percentile_results:
        st.markdown("### 📋 Complete Risk Profile")
        
        # Create comprehensive results table
        results_data = []
        for disease, result in percentile_results.items():
            category = categorize_diseases(disease)
            risk_level, _ = get_risk_level_from_percentile(result['percentile'])
            
            # Calculate how many people user outranks
            outranks_count = int(result['percentile'] / 100 * result['n_samples'])
            
            results_data.append({
                'Disease': disease,
                'Category': category,
                'Your Percentile': f"{result['percentile']:.1f}%",
                'Risk Level': risk_level,
                'Outranks': f"{outranks_count:,} of {result['n_samples']:,}",
                'Your LP Score': f"{result['lp_score']:.3f}",
                'Cohort Average': f"{result['cohort_mean']:.3f}"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Sort by percentile
        results_df = results_df.sort_values('Your Percentile', ascending=False, 
                                          key=lambda x: x.str.replace('%', '').astype(float))
        
        # Style the dataframe
        def style_percentile_table(val):
            if 'High Risk' in str(val):
                return 'background-color: #ffebee; color: #c62828'
            elif 'Moderate-High' in str(val):
                return 'background-color: #fff3e0; color: #ef6c00'
            elif 'Average Risk' in str(val):
                return 'background-color: #e3f2fd; color: #1565c0'
            elif 'Below Average' in str(val):
                return 'background-color: #e8f5e8; color: #2e7d32'
            else:
                return ''
        
        styled_results = results_df.style.applymap(style_percentile_table, subset=['Risk Level'])
        st.dataframe(styled_results, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.markdown("### 📊 Cohort Comparison Summary")
        
        cols = st.columns(4)
        avg_percentile = np.mean([result['percentile'] for result in percentile_results.values()])
        max_percentile = max([result['percentile'] for result in percentile_results.values()])
        min_percentile = min([result['percentile'] for result in percentile_results.values()])
        
        with cols[0]:
            st.metric("Average Percentile", f"{avg_percentile:.1f}%")
        with cols[1]:
            st.metric("Highest Risk Percentile", f"{max_percentile:.1f}%")
        with cols[2]:
            st.metric("Lowest Risk Percentile", f"{min_percentile:.1f}%")
        with cols[3]:
            st.metric("Conditions Analyzed", len(percentile_results))
    
    # Heart rate sensitivity analysis
    if percentile_results:
        st.markdown("### 🔍 Heart Rate Impact Analysis")
        
        # Select top diseases for sensitivity analysis
        top_diseases = sorted(percentile_results.items(), key=lambda x: x[1]['percentile'], reverse=True)[:6]
        
        if len(top_diseases) > 0:
            hr_range = np.arange(50, 101, 5)
            
            # Calculate how percentiles change with different heart rates
            fig_sensitivity = make_subplots(
                rows=2, cols=3,
                subplot_titles=[disease for disease, _ in top_diseases],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )
            
            for i, (disease, current_result) in enumerate(top_diseases):
                row = (i // 3) + 1
                col = (i % 3) + 1
                
                percentile_trends = []
                for test_hr in hr_range:
                    # Calculate LP with different HR
                    test_lp = calculate_linear_predictor(
                        disease, age, gender, test_hr, bmi, 
                        smoking_status, drinking_status, model_df
                    )
                    
                    # Calculate percentile with this LP
                    if (disease in lp_distributions and 
                        age_group in lp_distributions[disease] and
                        gender in lp_distributions[disease][age_group]):
                        
                        lp_dist = lp_distributions[disease][age_group][gender]
                        test_percentile = calculate_percentile_rank(test_lp, lp_dist)
                        percentile_trends.append(test_percentile if test_percentile is not None else 50)
                    else:
                        percentile_trends.append(50)
                
                # Add reference lines
                fig_sensitivity.add_hline(
                    y=50, line_dash="dash", line_color="gray", 
                    opacity=0.3, row=row, col=col
                )
                fig_sensitivity.add_hline(
                    y=75, line_dash="dash", line_color="orange", 
                    opacity=0.3, row=row, col=col
                )
                fig_sensitivity.add_hline(
                    y=90, line_dash="dash", line_color="red", 
                    opacity=0.3, row=row, col=col
                )
                
                # Main trend line
                fig_sensitivity.add_trace(
                    go.Scatter(
                        x=hr_range, 
                        y=percentile_trends,
                        mode='lines+markers',
                        name=disease,
                        line=dict(width=3),
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Current point
                fig_sensitivity.add_trace(
                    go.Scatter(
                        x=[current_hr], 
                        y=[current_result['percentile']],
                        mode='markers',
                        name="Your Current Status",
                        marker=dict(size=12, color='red', symbol='star'),
                        showlegend=False,
                        hovertemplate=f'<b>{disease}</b><br>HR: {current_hr} bpm<br>Percentile: {current_result["percentile"]:.1f}%<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig_sensitivity.update_layout(
                height=600,
                title_text=f"How Heart Rate Affects Your Percentile Rank vs {gender}s aged {age_group}",
                showlegend=False,
                font=dict(size=10)
            )
            
            # Update axes
            for i in range(len(top_diseases)):
                row = (i // 3) + 1
                col = (i % 3) + 1
                fig_sensitivity.update_xaxes(title_text="Heart Rate (bpm)", row=row, col=col)
                fig_sensitivity.update_yaxes(title_text="Percentile Rank (%)", row=row, col=col)
            
            st.plotly_chart(fig_sensitivity, use_container_width=True)
            
            # Add interpretation
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>📖 Sensitivity Analysis Interpretation:</strong><br>
                • <span style="color: gray;">Gray line (50%)</span>: Average risk in your demographic<br>
                • <span style="color: orange;">Orange line (75%)</span>: Moderate-high risk threshold<br>
                • <span style="color: red;">Red line (90%)</span>: High risk threshold<br>
                • <span style="color: red;">Red star</span>: Your current position<br><br>
                This shows how optimizing your heart rate could potentially improve your percentile rankings within your demographic cohort.
            </div>
            """, unsafe_allow_html=True)
    
    # Data source and methodology
    with st.expander("📊 Methodology & Data Sources"):
        st.markdown(f"""
        ### Percentile-Based Risk Assessment Methodology
        
        **Your Demographic Cohort:**
        - **Age Group:** {age_group}
        - **Gender:** {gender}
        - **Comparison Population:** ~1,000 individuals with similar demographics
        
        **Calculation Process:**
        1. **Linear Predictor (LP) Calculation:** Your risk factors are combined using Cox regression coefficients
        2. **Cohort Comparison:** Your LP score is compared to the distribution of LP scores in your demographic cohort
        3. **Percentile Ranking:** Your position in the distribution determines your percentile rank
        
        **Percentile Interpretation:**
        - **90th percentile:** Higher risk than 90% of people in your demographic cohort
        - **75th percentile:** Higher risk than 75% of people in your demographic cohort
        - **50th percentile:** Average risk for your demographic cohort
        - **25th percentile:** Lower risk than 75% of people in your demographic cohort
        
        **Risk Categories:**
        - **High Risk (≥90th percentile):** Top 10% risk group in your cohort
        - **Moderate-High Risk (75-89th percentile):** Top 25% risk group
        - **Average Risk (50-74th percentile):** Above average but not high risk
        - **Below Average Risk (<50th percentile):** Lower than average risk
        
        **Model Components:**
        - **Heart Rate Categories:** <60, 60-69, 70-79, 80-89, ≥90 bpm
        - **Demographic Factors:** Age, gender, BMI
        - **Lifestyle Factors:** Smoking status, drinking status
        
        **Important Limitations:**
        - Percentiles are based on population-level data from Taiwan Biobank
        - Individual risk may vary due to genetic and unmeasured factors
        - Results show relative risk within demographic cohorts, not absolute risk
        - Not a substitute for professional medical assessment
        - Mock data used for demonstration - replace with actual LP distributions
        
        **Data Requirements for Implementation:**
        When implementing with real data, you will need:
        - Linear predictor score distributions for each disease by age-gender cohorts
        - Sufficient sample sizes in each demographic group (recommended: >500 per cohort)
        - Validation of model performance across different demographic groups
        """)
    
    # Implementation notes
    with st.expander("🔧 Implementation Notes for Developers"):
        st.markdown("""
        ### Key Changes from Benchmark Version
        
        **Data Structure Required:**
        ```python
        lp_distributions = {
            'disease_name': {
                'age_group': {  # '20-29', '30-39', '40-49', '50-59', '60-69', '70+'
                    'gender': {  # 'Male', 'Female'
                        'lp_scores': [array of LP scores from actual patients],
                        'percentiles': {1: value, 5: value, ..., 99: value},
                        'mean': mean_lp_score,
                        'std': std_lp_score,
                        'n_samples': number_of_patients
                    }
                }
            }
        }
        ```
        
        **Key Functions to Implement:**
        1. `load_lp_distributions()` - Load actual LP score distributions from your data
        2. `calculate_percentile_rank()` - Calculate user's percentile within their cohort
        3. Ensure sufficient sample sizes in each age-gender-disease combination
        
        **Data Processing Steps:**
        1. Calculate LP scores for all patients in your dataset using the Cox coefficients
        2. Group patients by age group and gender
        3. For each disease-demographic combination, store the LP score distribution
        4. Calculate and store percentile cutoffs (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
        
        **Validation Recommendations:**
        - Ensure each cohort has adequate sample size (minimum 100, recommended 500+)
        - Validate that LP score distributions are reasonable across demographic groups
        - Consider smoothing or modeling distributions if sample sizes are small
        - Test edge cases (very young/old, extreme risk factor combinations)
        """)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator shows your risk percentile compared to {gender.lower()}s aged {age_group} 
        using Cox regression model results for educational purposes only. Percentile rankings are based on 
        population-level data and should not replace professional medical advice. Individual risk factors 
        and health conditions not included in the model may significantly affect your actual risk. 
        Always consult with a healthcare provider for personal health assessments and treatment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown(f"""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Percentile calculations based on Cox regression model coefficients from Taiwan Biobank study.<br>
        Comparison cohort: {gender}s aged {age_group} (N≈1,000 per demographic group).</em><br>
        <strong>Note:</strong> Currently using mock LP distributions for demonstration. 
        Replace with actual patient data for production use.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()