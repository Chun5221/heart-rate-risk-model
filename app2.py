# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 09:07:48 2025

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
    page_title="Heart Rate Disease Risk Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Risk data from meta-analysis
RISK_DATA = {
    'Coronary Heart Disease': {
        'rr_per_10bpm': 1.07,
        'ci_lower': 1.05,
        'ci_upper': 1.10,
        'baseline_risk': 0.06,  # 6% baseline risk
        'description': 'Risk of coronary artery disease and heart attacks'
    },
    'Sudden Cardiac Death': {
        'rr_per_10bpm': 1.09,
        'ci_lower': 1.00,
        'ci_upper': 1.18,
        'baseline_risk': 0.001,  # 0.1% baseline risk
        'description': 'Risk of sudden cardiac arrest'
    },
    'Heart Failure': {
        'rr_per_10bpm': 1.18,
        'ci_lower': 1.10,
        'ci_upper': 1.27,
        'baseline_risk': 0.02,  # 2% baseline risk
        'description': 'Risk of heart failure development'
    },
    'Total Stroke': {
        'rr_per_10bpm': 1.06,
        'ci_lower': 1.02,
        'ci_upper': 1.10,
        'baseline_risk': 0.025,  # 2.5% baseline risk
        'description': 'Risk of any type of stroke'
    },
    'Cardiovascular Disease': {
        'rr_per_10bpm': 1.15,
        'ci_lower': 1.11,
        'ci_upper': 1.18,
        'baseline_risk': 0.08,  # 8% baseline risk
        'description': 'Overall cardiovascular disease risk'
    },
    'Total Cancer': {
        'rr_per_10bpm': 1.14,
        'ci_lower': 1.06,
        'ci_upper': 1.23,
        'baseline_risk': 0.12,  # 12% baseline risk
        'description': 'Risk of developing any type of cancer'
    },
    'All-Cause Mortality': {
        'rr_per_10bpm': 1.17,
        'ci_lower': 1.14,
        'ci_upper': 1.19,
        'baseline_risk': 0.01,  # 1% baseline risk per year
        'description': 'Risk of death from any cause'
    }
}

def calculate_relative_risk(rhr, reference_rhr=70):
    """Calculate relative risk based on heart rate difference from reference"""
    risks = {}
    hr_difference = rhr - reference_rhr
    
    for condition, data in RISK_DATA.items():
        # Calculate relative risk for the heart rate difference
        # RR = (RR_per_10bpm)^(hr_difference/10)
        relative_risk = data['rr_per_10bpm'] ** (hr_difference / 10)
        
        # Calculate absolute risk
        absolute_risk = data['baseline_risk'] * relative_risk
        
        # Calculate confidence intervals
        ci_lower_rr = data['ci_lower'] ** (hr_difference / 10)
        ci_upper_rr = data['ci_upper'] ** (hr_difference / 10)
        
        risks[condition] = {
            'relative_risk': relative_risk,
            'absolute_risk': absolute_risk,
            'ci_lower': ci_lower_rr,
            'ci_upper': ci_upper_rr,
            'description': data['description']
        }
    
    return risks

def get_risk_level(relative_risk):
    """Categorize risk level based on relative risk"""
    if relative_risk < 0.8:
        return "Low", "green"
    elif relative_risk < 1.2:
        return "Normal", "blue"
    elif relative_risk < 1.5:
        return "Moderate", "orange"
    else:
        return "High", "red"

def create_risk_chart(risks):
    """Create a horizontal bar chart showing relative risks"""
    conditions = list(risks.keys())
    relative_risks = [risks[condition]['relative_risk'] for condition in conditions]
    ci_lower = [risks[condition]['ci_lower'] for condition in conditions]
    ci_upper = [risks[condition]['ci_upper'] for condition in conditions]
    
    fig = go.Figure()
    
    # Add bars
    colors = ['red' if rr > 1.5 else 'orange' if rr > 1.2 else 'blue' if rr > 0.8 else 'green' 
              for rr in relative_risks]
    
    fig.add_trace(go.Bar(
        y=conditions,
        x=relative_risks,
        orientation='h',
        marker_color=colors,
        name='Relative Risk',
        error_x=dict(
            type='data',
            symmetric=False,
            array=[ci_upper[i] - relative_risks[i] for i in range(len(relative_risks))],
            arrayminus=[relative_risks[i] - ci_lower[i] for i in range(len(relative_risks))],
            color='black',
            thickness=1,
            width=3
        )
    ))
    
    # Add reference line at RR = 1
    fig.add_vline(x=1, line_dash="dash", line_color="black", 
                  annotation_text="Reference Risk")
    
    fig.update_layout(
        title="Relative Risk by Condition",
        xaxis_title="Relative Risk",
        yaxis_title="Condition",
        height=400,
        showlegend=False
    )
    
    return fig

def create_heart_rate_curve(condition):
    """Create a curve showing how risk changes with heart rate"""
    heart_rates = np.arange(50, 120, 5)
    reference_hr = 70
    
    data = RISK_DATA[condition]
    relative_risks = [data['rr_per_10bpm'] ** ((hr - reference_hr) / 10) for hr in heart_rates]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=heart_rates,
        y=relative_risks,
        mode='lines',
        name=condition,
        line=dict(width=3, color='blue')
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="black", 
                  annotation_text="Reference Risk")
    
    fig.update_layout(
        title=f"Risk Curve for {condition}",
        xaxis_title="Resting Heart Rate (bpm)",
        yaxis_title="Relative Risk",
        height=400
    )
    
    return fig

# Main app
def main():
    st.title("❤️ Heart Rate Disease Risk Calculator")
    st.markdown("### Calculate your disease risk based on resting heart rate")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Input Parameters")
        
        resting_hr = st.slider(
            "Resting Heart Rate (bpm)",
            min_value=40,
            max_value=120,
            value=70,
            step=1,
            help="Enter your resting heart rate in beats per minute"
        )
        
        reference_hr = st.slider(
            "Reference Heart Rate (bpm)",
            min_value=40,
            max_value=120,
            value=70,
            step=1,
            help="Reference heart rate for comparison (default: 70 bpm)"
        )
        
        st.markdown("---")
        st.markdown("**Normal Resting Heart Rate Ranges:**")
        st.markdown("- Adults: 60-100 bpm")
        st.markdown("- Athletes: 40-60 bpm")
        st.markdown("- Elderly: 60-100 bpm")
    
    # Calculate risks
    risks = calculate_relative_risk(resting_hr, reference_hr)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Risk Assessment Results")
        
        # Create risk summary table
        risk_df = pd.DataFrame({
            'Condition': list(risks.keys()),
            'Relative Risk': [f"{risks[condition]['relative_risk']:.2f}" for condition in risks.keys()],
            'Risk Level': [get_risk_level(risks[condition]['relative_risk'])[0] for condition in risks.keys()],
            'Absolute Risk (%)': [f"{risks[condition]['absolute_risk']*100:.2f}%" for condition in risks.keys()],
            'Description': [risks[condition]['description'] for condition in risks.keys()]
        })
        
        st.dataframe(risk_df, use_container_width=True)
        
        # Risk visualization
        st.plotly_chart(create_risk_chart(risks), use_container_width=True)
    
    with col2:
        st.subheader("Risk Interpretation")
        
        # Color-coded risk levels
        for condition, data in risks.items():
            risk_level, color = get_risk_level(data['relative_risk'])
            st.markdown(f"**{condition}:**")
            st.markdown(f"<span style='color:{color}'>{risk_level} Risk</span>", unsafe_allow_html=True)
            st.markdown(f"RR: {data['relative_risk']:.2f}")
            st.markdown("---")
    
    # Detailed analysis section
    st.subheader("Detailed Analysis")
    
    # Risk curve for selected condition
    selected_condition = st.selectbox(
        "Select condition to view risk curve:",
        list(RISK_DATA.keys())
    )
    
    if selected_condition:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_heart_rate_curve(selected_condition), use_container_width=True)
        
        with col2:
            st.markdown(f"**{selected_condition}**")
            st.markdown(f"Description: {RISK_DATA[selected_condition]['description']}")
            st.markdown(f"Relative Risk per 10 bpm increase: {RISK_DATA[selected_condition]['rr_per_10bpm']}")
            st.markdown(f"95% CI: {RISK_DATA[selected_condition]['ci_lower']:.2f} - {RISK_DATA[selected_condition]['ci_upper']:.2f}")
            
            current_risk = risks[selected_condition]
            st.markdown(f"**Your Risk:**")
            st.markdown(f"- Relative Risk: {current_risk['relative_risk']:.2f}")
            st.markdown(f"- Absolute Risk: {current_risk['absolute_risk']*100:.2f}%")
            st.markdown(f"- Risk Level: {get_risk_level(current_risk['relative_risk'])[0]}")
    
    # Information section
    st.subheader("Important Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Source:**")
        st.markdown("- Meta-analysis from PubMed")
        st.markdown("- Relative risk per 10 bpm increase")
        st.markdown("- Dose-response relationship observed")
        st.markdown("- J-shaped association for atrial fibrillation")
    
    with col2:
        st.markdown("**Disclaimer:**")
        st.markdown("- This tool is for educational purposes only")
        st.markdown("- Not a substitute for medical advice")
        st.markdown("- Consult healthcare provider for interpretation")
        st.markdown("- Individual factors may affect actual risk")
    
    # Footer
    st.markdown("---")
    st.markdown("*Based on meta-analysis data from peer-reviewed research. Results are estimates and should be interpreted by healthcare professionals.*")

if __name__ == "__main__":
    main()