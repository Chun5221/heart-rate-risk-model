# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 08:56:08 2025

@author: chun5
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from pathlib import Path
import json

## [Supabase 連接]
from supabase import create_client, Client
import uuid
from datetime import datetime, timezone, timedelta


# 初始化 Supabase Client（用 anon key 寫入）
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_ANON_KEY = st.secrets["supabase"]["anon_key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# 產生一個 session_id（每次重開頁面或重新評估都可共用）
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

APP_VERSION = "app_percentage_tw.py-2025-09-04"
MODEL_VERSION = "coef:2025-09-04; pct:2025-08-29"


# Page configuration
st.set_page_config(
    page_title="❤️ 個人化健康風險評估平台",
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
        background: linear-gradient(135deg, #6b9aff 0%, #667eea 100%);
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
    
    .profile-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #1565c0;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 45px rgba(33,150,243,0.15);
        border: 1px solid #90caf9;
    }
    
    .profile-info h3 {
        font-size: 2rem;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(21,101,192,0.3);
        color: #0d47a1;
    }
    
    .profile-detail {
        background: rgba(255,255,255,0.7);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(33,150,243,0.2);
    }
    
    .stats-dashboard {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    .stats-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stats-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .stats-card.high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    .stats-card.moderate-risk {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
    }
    
    .stats-card.average-risk {
        background: linear-gradient(135deg, #6b9aff 0%, #667eea 100%);
    }
    
    .stats-card.low-risk {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stats-label {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
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
    
    .category-header {
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        padding: 1rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        border: 1px solid #90caf9;
    }
    
</style>
""", unsafe_allow_html=True)

# Disease categorization
DISEASE_CATEGORIES = {
    "心血管及循環系統": [
        "Heart Failure",
        "Atrial Fibrillation", 
        "Cardiac Arrhythmia",
        "Ischemic Heart Disease",
        "Ischemic Stroke",
        "Atherosclerosis",
        "Hypertension"
    ],
    "代謝及內分泌": [
        "Type 2 Diabetes"
    ],
    "精神健康及神經系統": [
        "Anxiety",
        "Depression", 
        "Dementia",
        "Migraine"
    ],
    "其他疾病": [
        "Chronic Kidney Disease",
        "Gastroesophageal Reflux Disease",
        "Anemias",
        "Asthma"
    ],
    "死亡風險": [
        "Death"
    ]
}

# Chinese disease names mapping
DISEASE_CHINESE_NAMES = {
    "Heart Failure": "心臟衰竭",
    "Atrial Fibrillation": "心房顫動",
    "Cardiac Arrhythmia": "心律不整",
    "Ischemic Heart Disease": "缺血性心臟病",
    "Ischemic Stroke": "缺血性中風",
    "Atherosclerosis": "動脈粥狀硬化",
    "Hypertension": "高血壓",
    "Type 2 Diabetes": "第二型糖尿病",
    "Anxiety": "焦慮症",
    "Depression": "憂鬱症",
    "Dementia": "失智症",
    "Migraine": "偏頭痛",
    "Chronic Kidney Disease": "慢性腎臟病",
    "Gastroesophageal Reflux Disease": "胃食道逆流",
    "Anemias": "貧血",
    "Asthma": "氣喘",
    "Death": "死亡"
}

# Flatten categories for reverse lookup
DISEASE_TO_CATEGORY = {}
for category, diseases in DISEASE_CATEGORIES.items():
    for disease in diseases:
        DISEASE_TO_CATEGORY[disease] = category



## [讀取 GH 資料夾]
# === 版本預設（可用 manifest.json 覆蓋）===
DEFAULT_COEF_VER = "250829"
DEFAULT_PCT_VER  = "250829"

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "model"
COEF_DIR = ASSETS_DIR / "coefficients"
PCT_DIR  = ASSETS_DIR / "percentiles"
MANIFEST_FILE = ASSETS_DIR / "manifest.json"

def _get_versions():
    coef_ver, pct_ver = DEFAULT_COEF_VER, DEFAULT_PCT_VER
    try:
        if MANIFEST_FILE.exists():
            data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            coef_ver = data.get("coef_version", coef_ver)
            pct_ver  = data.get("pct_version",  pct_ver)
    except Exception as e:
        st.warning(f"讀取 manifest.json 失敗，改用程式內預設版本：{e}")
    return coef_ver, pct_ver

# 你的疾病名稱對照（沿用你原本的 mapping）
_DISEASE_MAP = {
    'DEATH': 'Death',
    't2d': 'Type 2 Diabetes',
    'af': 'Atrial Fibrillation',
    'anxiety': 'Anxiety',
    'ckd': 'Chronic Kidney Disease',
    'gerd': 'Gastroesophageal Reflux Disease',
    'heart_failure': 'Heart Failure',
    'anemias': 'Anemias',
    'asthma': 'Asthma',
    'atherosclerosis': 'Atherosclerosis',
    'cardiac_arrhythmia': 'Cardiac Arrhythmia',
    'dementia': 'Dementia',
    'depression': 'Depression',
    'hypertension': 'Hypertension',
    'ischemic_heart_disease': 'Ischemic Heart Disease',
    'ischemic_stroke': 'Ischemic Stroke',
    'migraine': 'Migraine',
}

# Load and parse the Cox regression model coefficients
@st.cache_data
def load_model_coefficients(coef_version: str | None = None) -> pd.DataFrame:
    """從 repo 檔案讀取 Cox 係數表（含 REF），保留與你原始程式一致的欄位與值。"""
    coef_ver, _ = _get_versions()
    if coef_version:
        coef_ver = coef_version

    path = COEF_DIR / f"coef_{coef_ver}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到係數檔：{path}")

    # 用 dtype=str 以保留 'REF'，後面再按需轉 float
    df = pd.read_csv(path, dtype=str)
    # 清理
    df.columns = df.columns.str.strip()
    for c in ["Disease", "Variable", "Coef"]:
        if c not in df.columns:
            raise ValueError(f"係數檔缺少欄位：{c}")
        df[c] = df[c].astype(str).str.strip()

    # 病名標準化（與你原本一致）
    df["Disease"] = df["Disease"].map(_DISEASE_MAP).fillna(df["Disease"])
    return df

# Load the percentile data from the uploaded file
@st.cache_data
def load_percentile_data(pct_version: str | None = None) -> pd.DataFrame:
    """從 repo 檔案讀取風險百分位表（Tab 分隔），並套用你原本的清理／轉碼。"""
    _, pct_ver = _get_versions()
    if pct_version:
        pct_ver = pct_version

    path = PCT_DIR / f"HR_quantile_{pct_ver}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"找不到百分位檔：{path}")

    # 讀取：Tab 分隔；不指定 dtype 讓數值列維持數字、字串列保字串
    df = pd.read_csv(path, sep="\t")

    # 欄名與欄位清理（維持你原本邏輯）
    # 讀完 df 後馬上清理欄名與字串欄位
    df.columns = df.columns.str.strip()
    # 重要：去除 Disease / AGE / SEX 可能的前後空白
    for col in ["Disease", "AGE"]:
        df[col] = df[col].astype(str).str.strip()
    # SEX 也可能讀到字串型態＋空白，先轉成數字
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    # Clean up disease names
    df["Disease"] = df["Disease"].map(_DISEASE_MAP).fillna(df["Disease"])
    # Map gender codes (1=Male, 2=Female)
    df["Gender"] = df["SEX"].map({1: "Male", 2: "Female"})
    # 確保百分位欄位都是數值
    pct_cols = ['1%', '3%', '5%', '10%', '15%', '20%', '30%', '40%', '50%',
                '60%', '70%', '80%', '85%', '90%', '95%', '98%', '100%']
    for c in pct_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 基本欄位檢查（避免部署時資料欄位異常）
    required_cols = {"Disease","Gender","AGE"} | set(pct_cols)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"百分位檔缺少欄位：{missing}")

    return df


def calculate_bmi(height, weight, height_unit, weight_unit):
    """Calculate BMI from height and weight with unit conversion"""
    try:
        if height_unit == "公分":
            height_m = height / 100
        elif height_unit == "英尺/英寸":
            height_m = height * 0.0254
        else:
            height_m = height
        
        if weight_unit == "磅":
            weight_kg = weight * 0.453592
        else:
            weight_kg = weight
        
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 1)
    
    except (ZeroDivisionError, ValueError):
        return None

def get_bmi_category(bmi):
    """Categorize BMI according to the model's categories"""
    if bmi < 18.5:
        return "體重過輕", "#3498db"
    elif bmi < 24:
        return "正常體重", "#27ae60"
    elif bmi < 27:
        return "體重過重", "#f39c12"
    else:
        return "肥胖", "#e74c3c"

def get_bmi_model_category(bmi):
    """Get BMI category for model calculation"""
    if bmi < 18.5:
        return 'bmi_underweight'
    elif bmi < 24:
        return 'bmi_normal'
    elif bmi < 27:
        return 'bmi_overweight'
    else:
        return 'bmi_obese'

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

def get_age_group_for_percentile(age):
    """Convert age to age group for percentile lookup matching the data file"""
    if age < 40:
        return '<40'
    elif age < 45:
        return '40-44'
    elif age < 50:
        return '45-49'
    elif age < 55:
        return '50-54'
    elif age < 60:
        return '55-59'
    else:
        return '>=60'

def calculate_linear_predictor(disease_name, age, gender, hr, bmi, smoking_status, drinking_status, model_df):
    """Calculate linear predictor (LP) using Cox regression coefficients"""
    try:
        disease_coefs = model_df[model_df['Disease'] == disease_name].copy()
        
        if disease_coefs.empty:
            return None
        
        lp = 0.0
        
        # Heart Rate Category
        hr_cat = get_heart_rate_category(hr)
        hr_coef = disease_coefs[disease_coefs['Variable'] == hr_cat]
        if not hr_coef.empty and hr_coef.iloc[0]['Coef'] != 'REF':
            lp += float(hr_coef.iloc[0]['Coef'])
        
        # Age
        age_coef = disease_coefs[disease_coefs['Variable'] == 'AGE']
        if not age_coef.empty and age_coef.iloc[0]['Coef'] != 'REF':
            lp += float(age_coef.iloc[0]['Coef']) * age
        
        # Gender
        if gender == 'Female':
            gender_coef = disease_coefs[disease_coefs['Variable'] == 'FEMALE']
            if not gender_coef.empty and gender_coef.iloc[0]['Coef'] != 'REF':
                lp += float(gender_coef.iloc[0]['Coef'])
        
        # BMI Category
        bmi_cat = get_bmi_model_category(bmi)
        if bmi_cat != 'bmi_normal':
            bmi_coef = disease_coefs[disease_coefs['Variable'] == bmi_cat]
            if not bmi_coef.empty and bmi_coef.iloc[0]['Coef'] != 'REF':
                lp += float(bmi_coef.iloc[0]['Coef'])
        
        # Smoking Status
        if smoking_status == '曾經吸菸':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        elif smoking_status == '目前吸菸':
            smoke_coef = disease_coefs[disease_coefs['Variable'] == 'Now_smoke']
            if not smoke_coef.empty and smoke_coef.iloc[0]['Coef'] != 'REF':
                lp += float(smoke_coef.iloc[0]['Coef'])
        
        # Drinking Status
        if drinking_status == '曾經飲酒':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Ever_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        elif drinking_status == '目前飲酒':
            drink_coef = disease_coefs[disease_coefs['Variable'] == 'Now_drink']
            if not drink_coef.empty and drink_coef.iloc[0]['Coef'] != 'REF':
                lp += float(drink_coef.iloc[0]['Coef'])
        
        return lp
    
    except Exception as e:
        st.error(f"計算 {disease_name} 的LP時發生錯誤: {str(e)}")
        return None

def calculate_percentile_rank(user_lp, disease_name, gender, age_group, percentile_df):
    """Calculate user's percentile rank using actual percentile data"""
    try:
        demographic_data = percentile_df[
            (percentile_df['Disease'] == disease_name) & 
            (percentile_df['Gender'] == gender) & 
            (percentile_df['AGE'] == age_group)
        ]
        
        if demographic_data.empty:
            return None, None
        
        row = demographic_data.iloc[0]
        
        # Percentile columns and their corresponding values
        percentile_cols = ['1%', '3%', '5%', '10%', '15%', '20%', '30%', '40%', 
                          '50%', '60%', '70%', '80%', '85%', '90%', '95%', '98%', '100%']
        percentile_values = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 100]
        
        # Find where user_lp falls
        for i, col in enumerate(percentile_cols):
            threshold = float(row[col])
            if user_lp <= threshold:
                exact_percentile = percentile_values[i]
                
                # Calculate more precise percentile if between thresholds
                if i > 0:
                    prev_threshold = float(row[percentile_cols[i-1]])
                    prev_percentile = percentile_values[i-1]
                    
                    # Linear interpolation
                    if threshold != prev_threshold:
                        ratio = (user_lp - prev_threshold) / (threshold - prev_threshold)
                        interpolated_percentile = prev_percentile + ratio * (exact_percentile - prev_percentile)
                        return int(round(interpolated_percentile)), exact_percentile
                
                return exact_percentile, exact_percentile
        
        # If higher than 100th percentile
        return 100, 100
        
    except Exception as e:
        st.error(f"計算百分位數時發生錯誤: {str(e)}")
        return None, None

def get_risk_category_and_color(percentile, disease_name=''):
    """Get risk category and color based on percentile"""
    # Use consistent risk categories for all diseases including Death
    if percentile >= 90:
        return "高風險", "high-risk-card", "#e74c3c"
    elif percentile >= 75:
        return "中高風險", "moderate-risk-card", "#f39c12"
    elif percentile >= 50:
        return "平均風險", "percentile-card", "#3498db"
    else:
        return "低風險", "low-risk-card", "#27ae60"

def create_percentile_gauge(percentile, disease_name):
    """Create a gauge chart showing percentile position"""
    # Use consistent colors for all diseases including Death
    if percentile >= 90:
        color = "#e74c3c"
    elif percentile >= 75:
        color = "#f39c12"
    elif percentile >= 50:
        color = "#3498db"
    else:
        color = "#27ae60"
    
    # Get Chinese disease name
    chinese_name = DISEASE_CHINESE_NAMES.get(disease_name, disease_name)
    title_text = f"{chinese_name}<br>風險百分位"
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

def create_risk_summary_chart(risk_counts):
    """Create a summary chart showing risk distribution"""
    # Define colors for each risk category (simplified since Death now uses same categories)
    colors = {
        '高風險': '#e74c3c',
        '中高風險': '#f39c12', 
        '平均風險': '#3498db',
        '低風險': '#27ae60'
    }
    
    # Simplified risk counting since all diseases use the same categories
    display_counts = risk_counts.copy()
    
    categories = list(display_counts.keys())
    values = list(display_counts.values())
    category_colors = [colors.get(cat, '#95a5a6') for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=category_colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="風險分佈摘要",
        xaxis_title="風險等級",
        yaxis_title="疾病數量",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def log_session_and_results(
    results, age, gender, bmi, current_hr, smoking_status, drinking_status, age_group,
    consent=False
    ):
    
    """把一整次評估寫入 Supabase：先寫 sessions，再批次 insert 各疾病結果。"""
    try:
        # 1) 先寫入/記錄一筆 session
        supabase.table("user_sessions").insert({
            "id": st.session_state["session_id"],
            "consent": bool(consent),
            "app_version": APP_VERSION,
            "client_hint": "streamlit",
        }, returning="minimal"   # ← 關鍵：不回傳資料，避免被 RLS 的 SELECT 擋
        ).execute()

        # 2) 準備「每個疾病一列」的資料
        rows = []
        for r in results:
            rows.append({
                "session_id": st.session_state["session_id"],
                "age": int(age),
                "gender": gender,
                "bmi": float(bmi),
                "current_hr": int(current_hr),
                "smoking_status": smoking_status,
                "drinking_status": drinking_status,
                "age_group": age_group,

                "disease": r["disease"],
                "category": r["category"],
                "lp": float(r["lp"]),
                "percentile": int(r["percentile"]),
                "exact_percentile": int(r["exact_percentile"]),
                "risk_category": r["risk_category"],

                "model_version": MODEL_VERSION,
                "timezone": "Asia/Taipei",
            })

        if rows:
            supabase.table("risk_events").insert(
                rows, returning="minimal"  # ← 同理這裡也關回傳
            ).execute()

        st.toast("✅ 已匿名記錄本次評估（寫入 Supabase）", icon="✅")
    except Exception as e:
        st.error(f"寫入 Supabase 發生錯誤：{e}")


try:
    model_df = load_model_coefficients()
    percentile_df = load_percentile_data()
except Exception as e:
    st.error(f"載入模型資料失敗：{e}")
    st.stop()

# 例：要求至少要有 Death 與 Type 2 Diabetes
must_have = {"Death", "Type 2 Diabetes"}
if not must_have.issubset(set(model_df["Disease"].unique())):
    st.error(f"係數檔疾病不完整，至少需包含：{must_have}")
    st.stop()

def main():
    # Load data
    model_df = load_model_coefficients()
    percentile_df = load_percentile_data()
    
    # Get unique diseases available in both datasets
    available_diseases = set(model_df['Disease'].unique()) & set(percentile_df['Disease'].unique())
    diseases = list(available_diseases)
    
    # Header
    st.markdown('<h1 class="main-header">❤️ 個人化健康風險評估平台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">使用實際人口數據將您的風險與同年齡層性別相同的人群進行比較</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs  👉 改為 form：只有按「確定」才提交
    with st.sidebar:
        st.markdown("### 您的資訊")
    
        # 用 session_state 保存「已提交」的值（第一次載入給預設）
        ss = st.session_state
        if "committed" not in ss:
            ss.committed = {
                "age": 43,
                "gender": "Male",
                "height_unit": "公分",
                "weight_unit": "公斤",
                "height": 170,   # 公分或公尺或英吋，依單位而異
                "weight": 75,    # 公斤或磅
                "current_hr": 72,
                "smoking_status": "從未吸菸",
                "drinking_status": "從未飲酒",
                "category_filters": {k: True for k in DISEASE_CATEGORIES.keys()}
            }
    
        # 建表單：只有按下提交才更新 ss.committed
        with st.form("user_inputs", clear_on_submit=False):
            age = st.slider("年齡", 20, 90, ss.committed["age"], help="您目前的年齡")
    
            gender = st.selectbox(
                "性別", ["Male", "Female"],
                index=(0 if ss.committed["gender"] == "Male" else 1),
                format_func=lambda x: "男性" if x == "Male" else "女性",
                help="生理性別"
            )
    
            st.markdown("### 身高體重")
            col1, col2 = st.columns(2)
            with col1:
                height_unit = st.selectbox(
                    "身高單位", ["公分", "英尺/英寸", "公尺"],
                    index=["公分","英尺/英寸","公尺"].index(ss.committed["height_unit"])
                )
            with col2:
                weight_unit = st.selectbox(
                    "體重單位", ["公斤", "磅"],
                    index=(0 if ss.committed["weight_unit"] == "公斤" else 1)
                )
    
            # 身高輸入（依單位）
            if height_unit == "公分":
                height = st.slider("身高 (公分)", 100, 220, int(ss.committed["height"]) if ss.committed["height_unit"]=="公分" else 170)
            elif height_unit == "英尺/英寸":
                # 若之前是英吋，轉回 feet/inches 的預設顯示；這裡簡化為固定起始 5'6"
                feet = st.selectbox("英尺", list(range(3, 8)), index=2)
                inches = st.selectbox("英寸", list(range(0, 12)), index=6)
                height = feet * 12 + inches  # 以總英吋存
                st.write(f"身高: {feet}'{inches}\"")
            else:
                height = st.slider("身高 (公尺)", 1.0, 2.2, float(ss.committed["height"]) if ss.committed["height_unit"]=="公尺" else 1.7, step=0.01)
    
            # 體重輸入
            if weight_unit == "公斤":
                weight = st.slider("體重 (公斤)", 30, 200, int(ss.committed["weight"]) if ss.committed["weight_unit"]=="公斤" else 75)
            else:
                weight = st.slider("體重 (磅)", 66, 440, int(ss.committed["weight"]) if ss.committed["weight_unit"]=="磅" else 165)
    
            # 預覽 BMI（僅表單內即時計算、未提交不會更新儀表板）
            preview_bmi = calculate_bmi(height, weight, height_unit, weight_unit)
            if preview_bmi:
                bmi_category, bmi_color = get_bmi_category(preview_bmi)
                st.markdown(f"""
                <div class="bmi-info">
                    <h4>（預覽）計算的BMI</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; color: {bmi_color};">
                        BMI: {preview_bmi}
                    </p>
                    <p style="color: {bmi_color}; font-weight: bold;">
                        分類: {bmi_category}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
            st.markdown("### 心率")
            current_hr = st.slider("靜息心率 (bpm)", 40, 120, ss.committed["current_hr"])
    
            st.markdown("### 生活習慣")
            smoking_status = st.selectbox(
                "吸菸狀況", ["從未吸菸", "曾經吸菸", "目前吸菸"],
                index=["從未吸菸","曾經吸菸","目前吸菸"].index(ss.committed["smoking_status"])
            )
            drinking_status = st.selectbox(
                "飲酒狀況", ["從未飲酒", "曾經飲酒", "目前飲酒"],
                index=["從未飲酒","曾經飲酒","目前飲酒"].index(ss.committed["drinking_status"])
            )
    
            st.markdown("### 疾病分類")
            st.markdown("選擇要分析的疾病類型：")
            category_filters_tmp = {}
            for category in DISEASE_CATEGORIES.keys():
                category_filters_tmp[category] = st.checkbox(
                    category,
                    value=ss.committed["category_filters"].get(category, True),
                    help=f"在分析中包含{category}"
                )
            
            st.markdown("---")
            consent = st.checkbox("✅ 我同意匿名記錄本次評估結果，用於日後研究分析（不含任何可識別個人資訊）", value=False)

            submitted = st.form_submit_button("✅ 確定（更新儀表板）")
    
        # 只有在提交時才「更新已提交的值」
        if submitted:
            ss.committed.update({
                "age": age,
                "gender": gender,
                "height_unit": height_unit,
                "weight_unit": weight_unit,
                "height": height,
                "weight": weight,
                "current_hr": current_hr,
                "smoking_status": smoking_status,
                "drinking_status": drinking_status,
                "category_filters": category_filters_tmp
            })
    
    # 從已提交的值讀取，下面所有顯示與計算都用 committed 值
    age = ss.committed["age"]
    gender = ss.committed["gender"]
    height_unit = ss.committed["height_unit"]
    weight_unit = ss.committed["weight_unit"]
    height = ss.committed["height"]
    weight = ss.committed["weight"]
    current_hr = ss.committed["current_hr"]
    smoking_status = ss.committed["smoking_status"]
    drinking_status = ss.committed["drinking_status"]
    category_filters = ss.committed["category_filters"]
    
    # 用「已提交」的值計算 BMI
    bmi = calculate_bmi(height, weight, height_unit, weight_unit) or 25.0

    
    # Determine user's demographic group
    age_group = get_age_group_for_percentile(age)
    
    # Enhanced profile section
    gender_chinese = "女性" if gender == "Female" else "男性"
    st.markdown(f"""
    <div class="profile-info">
        <h3>👤 您的健康檔案</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div class="profile-detail">
                <h4 style="margin: 0 0 0.5rem 0;">📊 比較群體</h4>
                <p style="margin: 0; font-size: 1.1rem;"><strong>您的比較對象是：</strong></p>
                <p style="margin: 0.25rem 0; font-size: 1.3rem; font-weight: bold;">{age_group} 歲的{gender_chinese}</p>
            </div>
            <div class="profile-detail">
                <h4 style="margin: 0 0 0.5rem 0;">🥼 您的詳細資料</h4>
                <p style="margin: 0;"><strong>年齡：</strong> {age} 歲</p>
                <p style="margin: 0;"><strong>身體狀況：</strong> BMI {bmi} ({get_bmi_category(bmi)[0]})</p>
                <p style="margin: 0;"><strong>心率：</strong> {current_hr} bpm</p>
                <p style="margin: 0;"><strong>生活方式：</strong> {smoking_status}，{drinking_status}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter diseases based on selected categories
    filtered_diseases = []
    for disease in diseases:
        disease_category = DISEASE_TO_CATEGORY.get(disease)
        if disease_category and category_filters.get(disease_category, False):
            filtered_diseases.append(disease)
    
    if not filtered_diseases:
        st.warning("請至少選擇一個疾病分類來進行分析。")
        return
    
    # Calculate percentiles for filtered diseases
    results = []
    
    for disease in filtered_diseases:
        user_lp = calculate_linear_predictor(
            disease, age, gender, current_hr, bmi, 
            smoking_status, drinking_status, model_df
        )
        
        if user_lp is not None:
            percentile, exact_percentile = calculate_percentile_rank(
                user_lp, disease, gender, age_group, percentile_df
            )
            
            if percentile is not None:
                risk_category, card_class, color = get_risk_category_and_color(percentile, disease)
                results.append({
                    'disease': disease,
                    'percentile': percentile,
                    'exact_percentile': exact_percentile,
                    'risk_category': risk_category,
                    'card_class': card_class,
                    'color': color,
                    'lp': user_lp,
                    'category': DISEASE_TO_CATEGORY.get(disease, '其他')
                })
    
    if results:
        # Create risk summary statistics
        risk_counts = {}
        for result in results:
            risk_category = result['risk_category']
            risk_counts[risk_category] = risk_counts.get(risk_category, 0) + 1
        
        # Display aggregated statistics dashboard
        st.markdown(f"""
        <div class="stats-dashboard">
            <h3 style="text-align: center; margin-bottom: 1.5rem; color: #2c3e50;">📈 風險評估摘要</h3>
            <p style="text-align: center; color: #7f8c8d; margin-bottom: 2rem;">
                針對 {len(results)} 種疾病與 {age_group} 歲{gender_chinese}進行分析比較
            </p>
        """, unsafe_allow_html=True)
        
        # Create statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Count diseases in each risk level (now simplified since Death uses same categories)
        high_risk = risk_counts.get('高風險', 0)
        moderate_risk = risk_counts.get('中高風險', 0)
        average_risk = risk_counts.get('平均風險', 0)
        low_risk = risk_counts.get('低風險', 0)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card high-risk">
                <p class="stats-number">{high_risk}</p>
                <p class="stats-label">高風險</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card moderate-risk">
                <p class="stats-number">{moderate_risk}</p>
                <p class="stats-label">中高風險</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card average-risk">
                <p class="stats-number">{average_risk}</p>
                <p class="stats-label">平均風險</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card low-risk">
                <p class="stats-number">{low_risk}</p>
                <p class="stats-label">低風險</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create and display risk distribution chart
        risk_chart = create_risk_summary_chart(risk_counts)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Sort results by percentile (highest risk first)
        results.sort(key=lambda x: x['percentile'], reverse=True)
        
        # Group results by category and display
        st.markdown("### 您的風險評估結果")
        
        # Get unique categories from results
        categories_with_results = list(set([result['category'] for result in results]))
        categories_with_results.sort()
        
        for category in categories_with_results:
            category_results = [r for r in results if r['category'] == category]
            
            if category_results:
                st.markdown(f'<div class="category-header">{category}</div>', unsafe_allow_html=True)
                
                # Display results in grid format
                cols = st.columns(min(len(category_results), 4))
                for i, result in enumerate(category_results):
                    with cols[i % len(cols)]:
                        # Create gauge chart
                        fig = create_percentile_gauge(result['percentile'], result['disease'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk interpretation in Chinese
                        chinese_disease_name = DISEASE_CHINESE_NAMES.get(result['disease'], result['disease'])
                        if result['percentile'] >= 90:
                            interpretation = f"風險高於{result['percentile']}%的同年齡層同性別者"
                            recommendation = "建議諮詢醫療專業人士"
                        elif result['percentile'] >= 75:
                            interpretation = f"風險高於{result['percentile']}%的同年齡層同性別者"
                            recommendation = "密切監控，調整生活方式"
                        elif result['percentile'] >= 50:
                            interpretation = f"平均風險（高於{result['percentile']}%的人）"
                            recommendation = "繼續保持健康習慣"
                        else:
                            interpretation = f"較低風險（高於{result['percentile']}%的人）"
                            recommendation = "維持現有的生活方式"
                        
                        st.markdown(f"""
                        <div class="{result['card_class']}">
                            <h4>{chinese_disease_name}</h4>
                            <div class="percentile-number">{result['percentile']}</div>
                            <p>百分位數</p>
                            <hr style="border-color: rgba(255,255,255,0.3);">
                            <p style="font-size: 0.9rem;">{interpretation}</p>
                            <p style="font-size: 0.8rem;"><em>{recommendation}</em></p>
                            <p style="font-size: 0.7rem;">線性預測值: {result['lp']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Detailed comparison table
        st.markdown("### 詳細結果表格")
        
        comparison_df = pd.DataFrame({
            '疾病': [DISEASE_CHINESE_NAMES.get(r['disease'], r['disease']) for r in results],
            '分類': [r['category'] for r in results],
            '您的百分位數': [f"{r['percentile']}" for r in results],
            '線性預測值': [f"{r['lp']:.3f}" for r in results],
            '風險等級': [r['risk_category'] for r in results],
            '人口統計組': [f"{gender_chinese}, {age_group}" for _ in results]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Summary insights
        st.markdown("### 💡 重點分析")
        
        total_conditions = len(results)
        high_risk_conditions = [r for r in results if r['percentile'] >= 90]
        moderate_risk_conditions = [r for r in results if 75 <= r['percentile'] < 90]
        
        if high_risk_conditions:
            disease_names_chinese = [DISEASE_CHINESE_NAMES.get(r['disease'], r['disease']) for r in high_risk_conditions]
            st.error(f"⚠️ **高優先級：** 您有{len(high_risk_conditions)}項疾病處於高風險類別（≥90百分位數）：{', '.join(disease_names_chinese)}")
        
        if moderate_risk_conditions:
            disease_names_chinese = [DISEASE_CHINESE_NAMES.get(r['disease'], r['disease']) for r in moderate_risk_conditions]
            st.warning(f"⚡ **密切監控：** 您有{len(moderate_risk_conditions)}項疾病處於中高風險類別（75-89百分位數）：{', '.join(disease_names_chinese)}")
        
        if not high_risk_conditions and not moderate_risk_conditions:
            st.success(f"✅ **好消息：** 您評估的{total_conditions}項疾病均未落入高風險類別！")
        
        st.markdown("""
        **注意：** 此計算器使用台灣生物資料庫的實際人口數據，以確定您計算的風險在同年齡層同性別群體中的位置。
        線性預測值（LP）是使用Cox回歸係數計算得出，您的百分位數顯示在您的人口統計組中有多少比例的人風險比您低。
        
        **免責聲明：** 此工具僅供教育目的使用，不應取代專業醫療建議。
        請諮詢醫療專業人士以獲得個人化的醫療指導。
        """)
    
    else:
        st.error("無法計算所選分類的風險百分位數。請檢查您的人口統計組是否有可用數據。")
    
    # 只有使用者勾選同意，才寫入
    if consent and results:
        log_session_and_results(
            results=results,
            age=age,
            gender=gender,
            bmi=bmi,
            current_hr=current_hr,
            smoking_status=smoking_status,
            drinking_status=drinking_status,
            age_group=age_group,
            consent=consent
        )


if __name__ == "__main__":
    main()
