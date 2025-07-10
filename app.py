# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:07:33 2025

@author: chun5
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 頁面設定
st.set_page_config(page_title="心率疾病風險儀表板", layout="centered")

# 標題與說明
st.title("❤️ 心率疾病風險評估")
st.write("請輸入您的心率，我們將估算與某些疾病相關的風險。")

# 使用者輸入
heart_rate = st.number_input("請輸入您的心率（bpm）", min_value=30, max_value=200, value=75)

# 簡易風險判斷模型（可替換為 ML 模型）
def assess_risk(hr):
    if hr < 60:
        return "偏低（可能與心律不整相關）", 0.3
    elif hr <= 100:
        return "正常", 0.1
    else:
        return "偏高（可能與高血壓或心臟疾病相關）", 0.8

risk_text, risk_score = assess_risk(heart_rate)

# 儀表板（Plotly）
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=heart_rate,
    title={"text": "目前心率（bpm）"},
    gauge={
        'axis': {'range': [30, 200]},
        'steps': [
            {'range': [30, 60], 'color': "lightblue"},
            {'range': [60, 100], 'color': "lightgreen"},
            {'range': [100, 200], 'color': "lightcoral"},
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': heart_rate
        }
    }
))

# 顯示圖表與風險評估
st.plotly_chart(fig)
st.subheader("風險分析")
st.write(f"根據您輸入的心率（{heart_rate} bpm），目前評估為：**{risk_text}**")
st.progress(min(risk_score, 1.0))
