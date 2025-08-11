# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:19:31 2025

@author: chun5
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 09:07:48 2025
Updated on Tue Jul 15 09:46 2025
Updated on Thu Jul 17 13:51 2025
Updated on Mon Aug 11 2025 - Modified to use CSV data

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

@st.cache_data
def load_risk_data(csv_file_path):
    """
    Load risk data from CSV file
    Expected columns in CSV:
    - condition/outcome: name of the health condition
    - category: category of the condition (e.g., 'Mortality', 'Cardiovascular', 'Kidney')
    - hazard_ratio/relative_risk: the risk ratio per unit increase
    - baseline_hr: baseline heart rate used in the study
    - per_unit: the unit of increase (e.g., 10 for per 10 bpm)
    - p_value: p-value for significance (optional)
    - ci_lower: lower confidence interval (optional)
    - ci_upper: upper confidence interval (optional)
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # Convert DataFrame to dictionary format similar to original RISK_DATA
        risk_data = {}
        for _, row in df.iterrows():
            # Adjust column names based on your actual CSV structure
            condition_name = row.get('condition', row.get('outcome', str(row.iloc[0])))
            
            risk_data[condition_name] = {
                'rr': float(row.get('hazard_ratio', row.get('relative_risk', row.get('hr', 1.0)))),
                'baseline': int(row.get('baseline_hr', row.get('baseline', 65))),
                'category': str(row.get('category', 'Other')),
                'per_unit': int(row.get('per_unit', 10)),  # per X bpm increase
                'p_value': float(row.get('p_value', 0.05)) if 'p_value' in row else None,
                'ci_lower': float(row.get('ci_lower', None)) if 'ci_lower' in row else None,
                'ci_upper': float(row.get('ci_upper', None)) if 'ci_upper' in row else None
            }
        
        return risk_data
        
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file_path}' not found. Using default risk data.")
        return get_default_risk_data()
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}. Using default risk data.")
        return get_default_risk_data()

def get_default_risk_data():
    """Fallback to original hard-coded risk data if CSV loading fails"""
    return {
        'All-cause Mortality': {'rr': 1.4, 'baseline': 65, 'category': 'Mortality', 'per_unit': 10},
        'Cardiovascular Mortality': {'rr': 1.42, 'baseline': 65, 'category': 'Mortality', 'per_unit': 10},
        'Heart Failure Mortality': {'rr': 1.67, 'baseline': 65, 'category': 'Mortality', 'per_unit': 10},
        'Coronary Heart Disease': {'rr': 1.18, 'baseline': 65, 'category': 'Cardiovascular', 'per_unit': 10},
        'Total Stroke': {'rr': 1.32, 'baseline': 65, 'category': 'Cardiovascular', 'per_unit': 10},
        'Hemorrhagic Stroke': {'rr': 1.29, 'baseline': 65, 'category': 'Cardiovascular', 'per_unit': 10},
        'Ischemic Stroke': {'rr': 1.28, 'baseline': 65, 'category': 'Cardiovascular', 'per_unit': 10},
        'ESRD (End-Stage Renal Disease)': {'rr': 1.14, 'baseline': 60, 'category': 'Kidney', 'per_unit': 10},
        'Diabetes Mortality': {'rr': 1.26, 'baseline': 60, 'category': 'Mortality', 'per_unit': 10},
        'Kidney Disease Mortality': {'rr': 1.24, 'baseline': 60, 'category': 'Mortality', 'per_unit': 10}
    }

def calculate_relative_risk(current_hr, risk_type, risk_data):
    """Calculate relative risk based on heart rate difference from study baseline"""
    rr_data = risk_data[risk_type]
    study_baseline = rr_data['baseline']
    per_unit = rr_data.get('per_unit', 10)  # Default to per 10 bpm
    
    # Calculate difference from study baseline
    hr_difference = current_hr - study_baseline
    hr_increase_per_unit = hr_difference / per_unit
    
    # Calculate relative risk
    relative_risk = rr_data['rr'] ** hr_increase_per_unit
    
    return relative_risk

def get_risk_color(relative_risk):
    """Get color based on risk level"""
    if relative_risk < 1.1:
        return "#27ae60"  # Green
    elif relative_risk < 1.3:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red

def get_risk_level(relative_risk):
    """Get risk level description"""
    if relative_risk < 1.1:
        return "Low Risk"
    elif relative_risk < 1.3:
        return "Moderate Risk"
    else:
        return "High Risk"

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Rate Risk Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Assess your cardiovascular disease risk based on resting heart rate</p>', unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.markdown("### 📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with risk data",
        type=['csv'],
        help="Upload your TWB_model1_sig.csv file or leave empty to use default data"
    )
    
    # Load risk data
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_risk_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        RISK_DATA = load_risk_data("temp_risk_data.csv")
        st.sidebar.success("✅ Custom risk data loaded successfully!")
        
        # Display data preview
        if st.sidebar.checkbox("Show data preview"):
            temp_df = pd.read_csv(uploaded_file)
            st.sidebar.dataframe(temp_df.head())
            
    else:
        # Try to load from file path (if file exists in same directory)
        RISK_DATA = load_risk_data("TWB_model1_sig.csv")
    
    # Get available categories from loaded data
    available_categories = list(set(data['category'] for data in RISK_DATA.values()))
    
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
        
        # Show study baselines for reference
        st.markdown("### 📚 Study Reference Points")
        baselines = set(data['baseline'] for data in RISK_DATA.values())
        baseline_text = ", ".join(f"{baseline} bpm" for baseline in sorted(baselines))
        st.info(f"📊 Study baselines: {baseline_text}")
        
        # Risk category filter
        st.markdown("### 🔍 Filter by Category")
        selected_categories = st.multiselect(
            "Select risk categories to display:",
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
        st.markdown("### 📈 Risk Assessment Results")
        
        # Calculate risks for all conditions
        risks = {}
        filtered_conditions = [cond for cond, data in RISK_DATA.items() 
                              if data['category'] in selected_categories]
        
        if not filtered_conditions:
            st.warning("No conditions found for selected categories. Please adjust your filters.")
            return
        
        for condition in filtered_conditions:
            risks[condition] = calculate_relative_risk(current_hr, condition, RISK_DATA)
        
        # Create risk visualization
        conditions = list(risks.keys())
        risk_values = list(risks.values())
        colors = [get_risk_color(risk) for risk in risk_values]
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=risk_values,
                marker_color=colors,
                text=[f"{risk:.2f}x" for risk in risk_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Relative Risk: %{y:.2f}x<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Relative Risk by Condition",
            xaxis_title="Health Conditions",
            yaxis_title="Relative Risk",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Baseline Risk (1.0)", annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level indicators
        st.markdown("### 🎯 Risk Level Summary")
        cols = st.columns(3)
        
        high_risk = sum(1 for risk in risk_values if risk >= 1.3)
        moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.3)
        low_risk = sum(1 for risk in risk_values if risk < 1.1)
        
        with cols[0]:
            st.metric("🔴 High Risk Conditions", high_risk)
        with cols[1]:
            st.metric("🟡 Moderate Risk Conditions", moderate_risk)
        with cols[2]:
            st.metric("🟢 Low Risk Conditions", low_risk)
    
    with col2:
        st.markdown("### 💡 Heart Rate Status")
        
        # Heart rate status card - dynamically show all baselines
        baseline_info = {}
        for condition, data in RISK_DATA.items():
            baseline = data['baseline']
            if baseline not in baseline_info:
                baseline_info[baseline] = []
            baseline_info[baseline].append(data['category'])
        
        status_text = ""
        for baseline in sorted(baseline_info.keys()):
            diff = current_hr - baseline
            categories = list(set(baseline_info[baseline]))
            status_text += f"<p><strong>{', '.join(categories)} studies:</strong> {diff:+d} bpm from {baseline} bpm baseline</p>"
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Status vs Study Baselines</h4>
            {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Normal heart rate ranges
        st.markdown("### 📏 Normal Ranges")
        st.markdown("""
        <div class="info-box">
            <strong>Normal Resting Heart Rate:</strong><br>
            • Adults: 60-100 bpm<br>
            • Athletes: 40-60 bpm<br>
            • Elderly: 60-100 bpm
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed risk breakdown
    st.markdown("### 📋 Detailed Risk Analysis")
    
    # Create detailed table with categories
    conditions = list(risks.keys())
    table_data = {
        'Condition': conditions,
        'Category': [RISK_DATA[cond]['category'] for cond in conditions],
        'Study Baseline': [f"{RISK_DATA[cond]['baseline']} bpm" for cond in conditions],
        f'Risk per {RISK_DATA[conditions[0]].get("per_unit", 10)} bpm': [f"{RISK_DATA[cond]['rr']:.2f}x" for cond in conditions],
        'Your Relative Risk': [f"{risks[cond]:.2f}x" for cond in conditions],
        'Risk Level': [get_risk_level(risks[cond]) for cond in conditions]
    }
    
    # Add confidence intervals if available
    if any('ci_lower' in RISK_DATA[cond] and RISK_DATA[cond]['ci_lower'] is not None for cond in conditions):
        table_data['95% CI'] = [
            f"({RISK_DATA[cond].get('ci_lower', 'N/A'):.2f}-{RISK_DATA[cond].get('ci_upper', 'N/A'):.2f})" 
            if RISK_DATA[cond].get('ci_lower') is not None else 'N/A'
            for cond in conditions
        ]
    
    # Add p-values if available
    if any('p_value' in RISK_DATA[cond] and RISK_DATA[cond]['p_value'] is not None for cond in conditions):
        table_data['P-value'] = [
            f"{RISK_DATA[cond].get('p_value', 'N/A'):.3f}" 
            if RISK_DATA[cond].get('p_value') is not None else 'N/A'
            for cond in conditions
        ]
    
    risk_df = pd.DataFrame(table_data)
    
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
    
    # Trend analysis
    st.markdown("### 📊 Heart Rate vs Risk Trend")
    
    # Create trend chart for all conditions in selected categories
    hr_range = np.arange(40, 121, 5)
    
    # Get all conditions for selected categories
    trend_conditions = [cond for cond, data in RISK_DATA.items() 
                       if data['category'] in selected_categories]
    
    if len(trend_conditions) > 0:
        # Calculate number of rows and columns needed
        n_conditions = len(trend_conditions)
        n_cols = min(3, n_conditions)
        n_rows = (n_conditions + n_cols - 1) // n_cols
        
        # Dynamic spacing based on number of rows and columns
        if n_rows == 1:
            vertical_spacing = 0.25
            horizontal_spacing = 0.15
        elif n_rows == 2:
            vertical_spacing = 0.20
            horizontal_spacing = 0.12
        elif n_rows == 3:
            vertical_spacing = 0.12
            horizontal_spacing = 0.08
        else:
            vertical_spacing = 0.08
            horizontal_spacing = 0.05
        
        fig_trend = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=trend_conditions,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        for i, condition in enumerate(trend_conditions):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            trend_risks = [calculate_relative_risk(hr, condition, RISK_DATA) for hr in hr_range]
            
            # Add baseline reference line
            baseline_hr = RISK_DATA[condition]['baseline']
            fig_trend.add_hline(
                y=1.0, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                row=row, col=col
            )
            
            # Add vertical line for study baseline
            fig_trend.add_vline(
                x=baseline_hr,
                line_dash="dot",
                line_color="blue",
                opacity=0.5,
                row=row, col=col
            )
            
            # Main trend line
            fig_trend.add_trace(
                go.Scatter(
                    x=hr_range, 
                    y=trend_risks,
                    mode='lines+markers',
                    name=condition,
                    line=dict(width=3),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add current point
            current_risk = calculate_relative_risk(current_hr, condition, RISK_DATA)
            fig_trend.add_trace(
                go.Scatter(
                    x=[current_hr], 
                    y=[current_risk],
                    mode='markers',
                    name=f"Your Risk",
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=False,
                    hovertemplate=f'<b>{condition}</b><br>HR: {current_hr} bpm<br>Risk: {current_risk:.2f}x<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Dynamic height and margins based on number of rows
        if n_rows == 1:
            chart_height = 400
            margin_dict = dict(l=50, r=50, t=80, b=50)
            title_standoff = 20
            font_size = 10
        elif n_rows == 2:
            chart_height = 600
            margin_dict = dict(l=45, r=45, t=75, b=45)
            title_standoff = 18
            font_size = 10
        elif n_rows == 3:
            chart_height = 900
            margin_dict = dict(l=35, r=35, t=65, b=35)
            title_standoff = 12
            font_size = 9
        else:
            chart_height = max(800, n_rows * 200)
            margin_dict = dict(l=35, r=35, t=65, b=35)
            title_standoff = 12
            font_size = 8
        
        fig_trend.update_layout(
            height=chart_height,
            title_text=f"Risk Trends for {', '.join(selected_categories)} Categories",
            showlegend=False,
            font=dict(size=font_size),
            margin=margin_dict
        )
        
        # Update axes for each subplot with dynamic spacing
        for i in range(n_conditions):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            fig_trend.update_xaxes(
                title_text="Heart Rate (bpm)", 
                row=row, col=col,
                title_standoff=title_standoff,
                tickfont=dict(size=font_size-1)
            )
            fig_trend.update_yaxes(
                title_text="Relative Risk", 
                row=row, col=col,
                title_standoff=title_standoff,
                tickfont=dict(size=font_size-1)
            )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Add legend explanation
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>📖 Chart Legend:</strong><br>
            • <span style="color: gray;">Gray dashed line</span>: Baseline risk (1.0x)<br>
            • <span style="color: blue;">Blue dotted line</span>: Study baseline heart rate<br>
            • <span style="color: red;">Red star</span>: Your current risk level
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("Please select at least one category to view trend analysis.")
    
    # Recommendations
    st.markdown("### 💊 Recommendations")
    
    if not risk_values:
        st.info("No risk data available for selected categories.")
    else:
        max_risk = max(risk_values)
        if max_risk >= 1.3:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ High Risk Detected</strong><br>
                Consider consulting with a healthcare provider about your elevated heart rate. 
                Lifestyle modifications and medical evaluation may be beneficial.
            </div>
            """, unsafe_allow_html=True)
        elif max_risk >= 1.1:
            st.markdown("""
            <div class="warning-box">
                <strong>⚡ Moderate Risk</strong><br>
                Your heart rate is slightly elevated. Consider lifestyle improvements such as 
                regular exercise, stress management, and maintaining a healthy weight.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>✅ Good Heart Rate Range</strong><br>
                Your heart rate appears to be in a healthy range. Continue maintaining 
                a healthy lifestyle with regular exercise and proper nutrition.
            </div>
            """, unsafe_allow_html=True)
    
    # Data source information
    st.markdown("### 📊 Data Source")
    if uploaded_file is not None:
        st.info(f"📁 Using uploaded file: {uploaded_file.name}")
    else:
        st.info("📁 Using TWB_model1_sig.csv (if available) or default risk data")
    
    # Show loaded data summary
    with st.expander("View Loaded Risk Data Summary"):
        summary_data = []
        for condition, data in RISK_DATA.items():
            summary_data.append({
                'Condition': condition,
                'Category': data['category'],
                'Hazard Ratio': data['rr'],
                'Baseline HR': f"{data['baseline']} bpm",
                'Per Unit': f"{data.get('per_unit', 10)} bpm"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This calculator is for educational purposes only. 
        The results are based on statistical analysis and should not replace 
        professional medical advice. Always consult with a healthcare provider for personal 
        health assessments and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()