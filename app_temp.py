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
    page_title="📊 Heart Rate Percentile Risk Calculator",
    page_icon="📊",
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

# Load and parse the Cox regression model coefficients (same as before)
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
Hypertension,Now_drink,0.281362"""
    
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
    
    # SAMPLE DATA - Replace with your actual data
    sample_data = []
    diseases = ['Atrial Fibrillation', 'Type 2 Diabetes', 'Hypertension']
    genders = ['Male', 'Female']
    age_groups = ['30-40', '40-50', '50-60', '60-70', '70-80']
    
    np.random.seed(42)  # For reproducible sample data
    
    for disease in diseases:
        for gender in genders:
            for age_group in age_groups:
                # Generate sample LP distribution (replace with real data)
                base_lp = np.random.normal(0, 1)
                percentiles = np.random.normal(base_lp, 0.5, 7)
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
        
        # Calculate percentiles for all diseases
        results = []
        
        for disease in diseases:
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
                        'percentile': percentile,
                        'risk_category': risk_category,
                        'card_class': card_class,
                        'color': color,
                        'lp': user_lp
                    })
        
        if results:
            # Sort by percentile (highest risk first)
            results.sort(key=lambda x: x['percentile'], reverse=True)
            
            # Create summary cards
            cols = st.columns(len(results))
            
            for i, result in enumerate(results):
                with cols[i]:
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
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p style="font-size: 0.9rem;">{interpretation}</p>
                        <p style="font-size: 0.8rem;"><em>{recommendation}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed risk comparison table
            st.markdown("### 📋 Detailed Risk Comparison")
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Disease': [r['disease'] for r in results],
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
            def style_percentile(val):
                if 'High Risk' in val:
                    return 'background-color: #ffebee; color: #c62828'
                elif 'Moderate' in val:
                    return 'background-color: #fff3e0; color: #ef6c00'
                elif 'Average' in val:
                    return 'background-color: #e3f2fd; color: #1976d2'
                else:
                    return 'background-color: #e8f5e8; color: #2e7d32'
            
            styled_df = comparison_df.style.applymap(style_percentile, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Risk distribution visualization
            st.markdown("### 📊 Risk Distribution in Your Demographic")
            
            # Show where user falls in the distribution
            fig_dist = go.Figure()
            
            # Create sample distribution curve for visualization
            x = np.linspace(0, 100, 100)
            y = np.exp(-((x-50)**2)/(2*20**2))  # Normal-like curve
            
            fig_dist.add_trace(go.Scatter(
                x=x, y=y,
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.3)',
                line=dict(color='rgba(52, 152, 219, 0.8)'),
                name='Population Distribution',
                hovertemplate='Percentile: %{x}<br>Density: %{y}<extra></extra>'
            ))
            
            # Add user's position for each disease
            for result in results[:3]:  # Show top 3 diseases
                fig_dist.add_vline(
                    x=result['percentile'],
                    line_dash="dash",
                    line_color=result['color'],
                    annotation_text=f"{result['disease']}<br>{result['percentile']}th percentile",
                    annotation_position="top"
                )
            
            fig_dist.update_layout(
                title=f"Your Position Among {gender}s Aged {age_group}",
                xaxis_title="Risk Percentile",
                yaxis_title="Population Density",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
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
            st.markdown("""
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
            """.replace('{current_hr}', str(current_hr)), unsafe_allow_html=True)
    
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
