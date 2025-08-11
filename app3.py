# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:19:31 2025

@author: chun5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="❤️ Heart Rate Risk Calculator - TWB Model",
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

@st.cache_data
def load_twb_data():
    """Load and process TWB model data"""
    # In a real app, you'd load this from a file
    # For now, I'll create the data structure based on your CSV
    twb_data = {
        'Atrial Fibrillation': {
            'HR_cat<60': 1.278584, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.007198,
            'HR_cat80-89': 1.329859, 'HR_cat>=90': 1.870907, 'category': 'Cardiovascular'
        },
        'Anxiety': {
            'HR_cat<60': 0.9853143, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.0337353,
            'HR_cat80-89': 1.0830412, 'HR_cat>=90': 1.0332187, 'category': 'Mental Health'
        },
        'Chronic Kidney Disease': {
            'HR_cat<60': 0.962877, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.120832,
            'HR_cat80-89': 1.344067, 'HR_cat>=90': 2.055936, 'category': 'Kidney'
        },
        'GERD': {
            'HR_cat<60': 1.0175125, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1123568,
            'HR_cat80-89': 1.1324623, 'HR_cat>=90': 1.2009102, 'category': 'Digestive'
        },
        'Heart Failure': {
            'HR_cat<60': 1.099216, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.047627,
            'HR_cat80-89': 1.324929, 'HR_cat>=90': 1.384662, 'category': 'Cardiovascular'
        },
        'Myocardial Infarction': {
            'HR_cat<60': 1.305421, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.097465,
            'HR_cat80-89': 1.15584, 'HR_cat>=90': 1.156146, 'category': 'Cardiovascular'
        },
        'Type 2 Diabetes': {
            'HR_cat<60': 0.891278, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.268673,
            'HR_cat80-89': 1.65573, 'HR_cat>=90': 2.01645, 'category': 'Metabolic'
        },
        'Anemia': {
            'HR_cat<60': 0.9614963, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1027278,
            'HR_cat80-89': 1.2191359, 'HR_cat>=90': 1.0937225, 'category': 'Blood'
        },
        'Angina Pectoris': {
            'HR_cat<60': 1.154145, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.930245,
            'HR_cat80-89': 1.032199, 'HR_cat>=90': 0.907764, 'category': 'Cardiovascular'
        },
        'Asthma': {
            'HR_cat<60': 0.898631, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.022859,
            'HR_cat80-89': 1.055431, 'HR_cat>=90': 1.280111, 'category': 'Respiratory'
        },
        'Atherosclerosis': {
            'HR_cat<60': 1.154199, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.954146,
            'HR_cat80-89': 0.989229, 'HR_cat>=90': 0.824199, 'category': 'Cardiovascular'
        },
        'Cardiac Arrhythmia': {
            'HR_cat<60': 1.198335, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.069528,
            'HR_cat80-89': 1.25314, 'HR_cat>=90': 1.516303, 'category': 'Cardiovascular'
        },
        'Depression': {
            'HR_cat<60': 1.1269417, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.1207691,
            'HR_cat80-89': 1.4019456, 'HR_cat>=90': 1.7691724, 'category': 'Mental Health'
        },
        'Hypertension': {
            'HR_cat<60': 0.943739, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.21525,
            'HR_cat80-89': 1.478731, 'HR_cat>=90': 1.908702, 'category': 'Cardiovascular'
        },
        'Ischemic Heart Disease': {
            'HR_cat<60': 1.192775, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.024668,
            'HR_cat80-89': 0.983446, 'HR_cat>=90': 0.984206, 'category': 'Cardiovascular'
        },
        'Ischemic Stroke': {
            'HR_cat<60': 1.101362, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.315702,
            'HR_cat80-89': 1.543814, 'HR_cat>=90': 1.654512, 'category': 'Cardiovascular'
        },
        'Migraine': {
            'HR_cat<60': 0.952232, 'HR_cat60-69': 1.0, 'HR_cat70-79': 0.994199,
            'HR_cat80-89': 1.171267, 'HR_cat>=90': 1.25287, 'category': 'Neurological'
        },
        'Parkinson\'s Disease': {
            'HR_cat<60': 1.241971, 'HR_cat60-69': 1.0, 'HR_cat70-79': 1.440464,
            'HR_cat80-89': 1.296338, 'HR_cat>=90': 1.759019, 'category': 'Neurological'
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

def get_hazard_ratio(disease, heart_rate, twb_data):
    """Get hazard ratio for a specific disease and heart rate"""
    hr_category = get_hr_category(heart_rate)
    return twb_data[disease].get(hr_category, 1.0)

def get_risk_color(hazard_ratio):
    """Get color based on risk level"""
    if hazard_ratio < 1.1:
        return "#27ae60"  # Green
    elif hazard_ratio < 1.3:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(hazard_ratio):
    """Get risk level description"""
    if hazard_ratio < 1.1:
        return "Low Risk"
    elif hazard_ratio < 1.3:
        return "Moderate Risk"
    else:
        return "High Risk"

def main():
    # Load TWB data
    twb_data = load_twb_data()
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Calculator - TWB Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assess your disease risk based on resting heart rate using Taiwan Biobank data</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📊 Input Parameters")
        
        # Age and gender for context
        age = st.slider("Age", 20, 90, 50, help="Your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
        
        # Heart rate inputs
        st.markdown("### 💓 Heart Rate Information")
        current_hr = st.slider(
            "Current Resting Heart Rate (bpm)", 
            40, 120, 72, 
            help="Your current resting heart rate in beats per minute"
        )
        
        # Show current HR category
        hr_category = get_hr_category(current_hr)
        category_display = {
            'HR_cat<60': '<60 bpm (Bradycardia)',
            'HR_cat60-69': '60-69 bpm (Normal)',
            'HR_cat70-79': '70-79 bpm (Normal)',
            'HR_cat80-89': '80-89 bpm (Upper Normal)',
            'HR_cat>=90': '≥90 bpm (Tachycardia)'
        }
        st.info(f"📊 Your HR Category: {category_display[hr_category]}")
        
        # Risk category filter
        st.markdown("### 🔍 Filter by Category")
        available_categories = list(set([data['category'] for data in twb_data.values()]))
        selected_categories = st.multiselect(
            "Select disease categories to display:",
            available_categories,
            default=available_categories
        )
        
        # Additional health context
        st.markdown("### 🏥 Health Context")
        has_conditions = st.multiselect(
            "Pre-existing conditions (optional)",
            ["Hypertension", "Diabetes", "Smoking", "Obesity", "Family History of Heart Disease"]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Disease Risk Assessment")
        
        # Calculate risks for all conditions
        risks = {}
        filtered_diseases = [disease for disease, data in twb_data.items() 
                           if data['category'] in selected_categories]
        
        for disease in filtered_diseases:
            risks[disease] = get_hazard_ratio(disease, current_hr, twb_data)
        
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
                    y=diseases,  # Horizontal bar chart for better readability
                    x=risk_values,
                    marker_color=colors,
                    text=[f"{risk:.2f}x" for risk in risk_values],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Hazard Ratio: %{x:.2f}x<extra></extra>',
                    orientation='h'
                )
            ])
            
            fig.update_layout(
                title="Hazard Ratios by Disease (TWB Model)",
                xaxis_title="Hazard Ratio",
                yaxis_title="Disease",
                height=max(400, len(diseases) * 25),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11),
                margin=dict(l=200, r=50, t=80, b=50)
            )
            
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", 
                         annotation_text="Baseline (1.0)", annotation_position="top")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level indicators
            st.markdown("### 🎯 Risk Level Summary")
            cols = st.columns(3)
            
            high_risk = sum(1 for risk in risk_values if risk >= 1.3)
            moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.3)
            low_risk = sum(1 for risk in risk_values if risk < 1.1)
            
            with cols[0]:
                st.metric("🔴 High Risk Diseases", high_risk)
            with cols[1]:
                st.metric("🟡 Moderate Risk Diseases", moderate_risk)
            with cols[2]:
                st.metric("🟢 Low Risk Diseases", low_risk)
        else:
            st.warning("Please select at least one category to view results.")
    
    with col2:
        st.markdown("### 💡 Heart Rate Analysis")
        
        # Heart rate status card
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Your Heart Rate Profile</h4>
            <p><strong>Current HR:</strong> {current_hr} bpm</p>
            <p><strong>Category:</strong> {category_display[hr_category]}</p>
            <p><strong>Reference:</strong> 60-69 bpm baseline</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 5 highest risks
        if risks:
            top_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)[:5]
            st.markdown("### 🔝 Top 5 Risk Conditions")
            
            for i, (disease, hr) in enumerate(top_risks, 1):
                risk_level = get_risk_level(hr)
                color = get_risk_color(hr)
                st.markdown(f"""
                <div style="background: {color}20; border-left: 4px solid {color}; padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px;">
                    <strong>{i}. {disease}</strong><br>
                    <span style="color: {color};">HR: {hr:.2f}x ({risk_level})</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Normal heart rate ranges
        st.markdown("### 📏 HR Categories")
        st.markdown("""
        <div class="info-box">
            <strong>Heart Rate Categories:</strong><br>
            • <60 bpm: Bradycardia<br>
            • 60-69 bpm: Normal (baseline)<br>
            • 70-79 bpm: Normal<br>
            • 80-89 bpm: Upper Normal<br>
            • ≥90 bpm: Tachycardia
        </div>
        """, unsafe_allow_html=True)
    
    if risks:
        # Detailed risk breakdown
        st.markdown("### 📋 Detailed Risk Analysis")
        
        # Create detailed table
        risk_df = pd.DataFrame({
            'Disease': list(risks.keys()),
            'Category': [twb_data[disease]['category'] for disease in risks.keys()],
            'Your HR Category': [category_display[hr_category]] * len(risks),
            'Hazard Ratio': [f"{hr:.3f}x" for hr in risks.values()],
            'Risk Level': [get_risk_level(hr) for hr in risks.values()],
            'Baseline (60-69 bpm)': ['1.000x'] * len(risks)
        })
        
        # Color code the table
        def style_risk_level(val):
            if val == "High Risk":
                return 'background-color: #ffebee; color: #c62828'
            elif val == "Moderate Risk":
                return 'background-color: #fff3e0; color: #ef6c00'
            else:
                return 'background-color: #e8f5e8; color: #2e7d32'
        
        styled_df = risk_df.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Heart Rate Category Comparison
        st.markdown("### 📊 Risk Across Different Heart Rate Categories")
        
        # Select a few key diseases for comparison
        key_diseases = ['Type 2 Diabetes', 'Hypertension', 'Heart Failure', 'Atrial Fibrillation', 'Chronic Kidney Disease']
        available_key_diseases = [d for d in key_diseases if d in filtered_diseases]
        
        if available_key_diseases:
            # Create comparison chart
            hr_categories = ['HR_cat<60', 'HR_cat60-69', 'HR_cat70-79', 'HR_cat80-89', 'HR_cat>=90']
            hr_labels = ['<60', '60-69', '70-79', '80-89', '≥90']
            
            fig_comparison = go.Figure()
            
            for disease in available_key_diseases[:5]:  # Limit to 5 diseases for readability
                hazard_ratios = []
                for cat in hr_categories:
                    hazard_ratios.append(twb_data[disease].get(cat, 1.0))
                
                fig_comparison.add_trace(go.Scatter(
                    x=hr_labels,
                    y=hazard_ratios,
                    mode='lines+markers',
                    name=disease,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            # Highlight current category
            current_idx = hr_categories.index(hr_category)
            fig_comparison.add_vline(
                x=current_idx,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your HR: {current_hr} bpm",
                annotation_position="top"
            )
            
            fig_comparison.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                                   annotation_text="Baseline (1.0)", annotation_position="bottom right")
            
            fig_comparison.update_layout(
                title="Disease Risk Across Heart Rate Categories",
                xaxis_title="Heart Rate Category (bpm)",
                yaxis_title="Hazard Ratio",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💊 Recommendations")
        
        max_risk = max(risks.values()) if risks else 1.0
        high_risk_diseases = [disease for disease, hr in risks.items() if hr >= 1.3]
        
        if max_risk >= 1.5:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Very High Risk Detected</strong><br>
                Your heart rate category shows significantly elevated risk for: <strong>{', '.join(high_risk_diseases[:3])}</strong>
                {f' and {len(high_risk_diseases)-3} others' if len(high_risk_diseases) > 3 else ''}.<br>
                <strong>Recommendation:</strong> Consult with a healthcare provider promptly for comprehensive evaluation.
            </div>
            """, unsafe_allow_html=True)
        elif max_risk >= 1.3:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ High Risk Detected</strong><br>
                Elevated risk for: <strong>{', '.join(high_risk_diseases[:2])}</strong>
                {f' and {len(high_risk_diseases)-2} others' if len(high_risk_diseases) > 2 else ''}.<br>
                Consider lifestyle modifications and medical evaluation.
            </div>
            """, unsafe_allow_html=True)
        elif max_risk >= 1.1:
            st.markdown("""
            <div class="warning-box">
                <strong>⚡ Moderate Risk</strong><br>
                Some conditions show elevated risk. Consider lifestyle improvements such as 
                regular exercise, stress management, and maintaining a healthy weight.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>✅ Good Heart Rate Range</strong><br>
                Your heart rate appears to be associated with relatively low disease risks. 
                Continue maintaining a healthy lifestyle with regular exercise and proper nutrition.
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator is for educational purposes only. 
        Results are based on Taiwan Biobank population data and should not replace professional 
        medical advice. Individual risk factors may vary significantly. Always consult with a 
        healthcare provider for personal health assessments and treatment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Data source
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>Risk calculations based on Taiwan Biobank Model 1 significant results<br>
        Heart rate categories: <60, 60-69 (baseline), 70-79, 80-89, ≥90 bpm</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
