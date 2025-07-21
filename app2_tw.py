'''
Created on 2025/07/21 09:34
'''

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 頁面配置
st.set_page_config(
    page_title="❤️ 心率風險計算器",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂CSS樣式
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

# 基於統合分析的風險計算資料
RISK_DATA = {
    # 來自論文1 (基準值 65 bpm)
    '總死亡率': {'rr': 1.4, 'baseline': 65, 'category': '死亡率'},
    '心血管疾病死亡率': {'rr': 1.42, 'baseline': 65, 'category': '死亡率'},
    '心臟衰竭死亡率': {'rr': 1.67, 'baseline': 65, 'category': '死亡率'},
    '冠心病': {'rr': 1.18, 'baseline': 65, 'category': '心血管疾病'},
    '總中風': {'rr': 1.32, 'baseline': 65, 'category': '心血管疾病'},
    '出血性中風': {'rr': 1.29, 'baseline': 65, 'category': '心血管疾病'},
    '缺血性中風': {'rr': 1.28, 'baseline': 65, 'category': '心血管疾病'},
    
    # 來自論文2 (基準值 60 bpm)
    '末期腎病': {'rr': 1.14, 'baseline': 60, 'category': '腎臟疾病'},
    '糖尿病死亡率': {'rr': 1.26, 'baseline': 60, 'category': '死亡率'},
    '腎臟病死亡率': {'rr': 1.24, 'baseline': 60, 'category': '死亡率'}
}

def calculate_relative_risk(current_hr, risk_type):
    """根據心率與研究基準值的差異計算相對風險"""
    rr_data = RISK_DATA[risk_type]
    study_baseline = rr_data['baseline']
    
    # 計算與研究基準值的差異
    hr_difference = current_hr - study_baseline
    hr_increase_per_10 = hr_difference / 10
    
    # 計算相對風險
    relative_risk = rr_data['rr'] ** hr_increase_per_10
    
    return relative_risk

def get_risk_color(relative_risk):
    """根據風險等級獲得顏色"""
    if relative_risk < 1.1:
        return "#27ae60"  # 綠色
    elif relative_risk < 1.3:
        return "#f39c12"  # 橙色
    else:
        return "#e74c3c"  # 紅色

def get_risk_level(relative_risk):
    """獲得風險等級描述"""
    if relative_risk < 1.1:
        return "低風險"
    elif relative_risk < 1.3:
        return "中等風險"
    else:
        return "高風險"

def main():
    # 標題
    st.markdown('<h1 class="main-header">❤️ 心率風險計算器</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">根據靜息心率評估您的心血管疾病風險</p>', unsafe_allow_html=True)
    
    # 側邊欄輸入
    with st.sidebar:
        st.markdown("### 📊 輸入參數")
        
        # 年齡和性別（作為背景資訊）
        age = st.slider("年齡", 20, 90, 50, help="您目前的年齡")
        gender = st.selectbox("性別", ["男性", "女性"], help="生理性別")
        
        # 心率輸入
        st.markdown("### 💓 心率資訊")
        current_hr = st.slider(
            "目前靜息心率 (每分鐘次數)", 
            40, 120, 72, 
            help="您目前的靜息心率，單位為每分鐘次數"
        )
        
        # 顯示研究基準值作為參考
        st.markdown("### 📚 研究參考基準")
        st.info("📊 研究基準值：65 bpm（心血管結果）和 60 bpm（腎臟/糖尿病結果）")
        
        # 風險類別篩選器
        st.markdown("### 🔍 依類別篩選")
        selected_categories = st.multiselect(
            "選擇要顯示的風險類別：",
            ["死亡率", "心血管疾病", "腎臟疾病"],
            default=["死亡率", "心血管疾病", "腎臟疾病"]
        )
        
        # 額外健康背景
        st.markdown("### 🏥 健康背景")
        has_conditions = st.multiselect(
            "既往病史（選填）",
            ["高血壓", "糖尿病", "吸菸", "肥胖", "心臟病家族史"]
        )
    
    # 主內容區域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 風險評估結果")
        
        # 計算所有疾病的風險
        risks = {}
        filtered_conditions = [cond for cond, data in RISK_DATA.items() 
                              if data['category'] in selected_categories]
        
        for condition in filtered_conditions:
            risks[condition] = calculate_relative_risk(current_hr, condition)
        
        # 創建風險視覺化
        conditions = list(risks.keys())
        risk_values = list(risks.values())
        colors = [get_risk_color(risk) for risk in risk_values]
        
        # 長條圖
        fig = go.Figure(data=[
            go.Bar(
                x=conditions,
                y=risk_values,
                marker_color=colors,
                text=[f"{risk:.2f}倍" for risk in risk_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>相對風險：%{y:.2f}倍<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="各疾病相對風險",
            xaxis_title="健康狀況",
            yaxis_title="相對風險",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="基準風險 (1.0)", annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 風險等級指標
        st.markdown("### 🎯 風險等級摘要")
        cols = st.columns(3)
        
        high_risk = sum(1 for risk in risk_values if risk >= 1.3)
        moderate_risk = sum(1 for risk in risk_values if 1.1 <= risk < 1.3)
        low_risk = sum(1 for risk in risk_values if risk < 1.1)
        
        with cols[0]:
            st.metric("🔴 高風險項目", high_risk)
        with cols[1]:
            st.metric("🟡 中等風險項目", moderate_risk)
        with cols[2]:
            st.metric("🟢 低風險項目", low_risk)
    
    with col2:
        st.markdown("### 💡 心率狀態")
        
        # 心率狀態卡片
        cardiovascular_baseline = 65
        kidney_baseline = 60
        
        cv_diff = current_hr - cardiovascular_baseline
        kidney_diff = current_hr - kidney_baseline
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 與研究基準值比較</h4>
            <p><strong>心血管研究：</strong> 比 65 bpm 基準值 {cv_diff:+d} bpm</p>
            <p><strong>腎臟/糖尿病研究：</strong> 比 60 bpm 基準值 {kidney_diff:+d} bpm</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 正常心率範圍
        st.markdown("### 📏 正常範圍")
        st.markdown("""
        <div class="info-box">
            <strong>正常靜息心率：</strong><br>
            • 成人：60-100 bpm<br>
            • 運動員：40-60 bpm<br>
            • 老年人：60-100 bpm
        </div>
        """, unsafe_allow_html=True)
    
    # 詳細風險分析
    st.markdown("### 📋 詳細風險分析")
    
    # 創建包含類別的詳細表格
    conditions = list(risks.keys())
    risk_df = pd.DataFrame({
        '疾病': conditions,
        '類別': [RISK_DATA[cond]['category'] for cond in conditions],
        '研究基準值': [f"{RISK_DATA[cond]['baseline']} bpm" for cond in conditions],
        '每增加10 bpm風險': [f"{RISK_DATA[cond]['rr']:.2f}倍" for cond in conditions],
        '您的相對風險': [f"{risks[cond]:.2f}倍" for cond in conditions],
        '風險等級': [get_risk_level(risks[cond]) for cond in conditions]
    })
    
    # 為表格著色
    def style_risk_level(val):
        if val == "高風險":
            return 'background-color: #ffebee; color: #c62828'
        elif val == "中等風險":
            return 'background-color: #fff3e0; color: #ef6c00'
        else:
            return 'background-color: #e8f5e8; color: #2e7d32'
    
    styled_df = risk_df.style.applymap(style_risk_level, subset=['風險等級'])
    st.dataframe(styled_df, use_container_width=True)
    
    # 趨勢分析
    st.markdown("### 📊 心率與風險趨勢")
    
    # 為選定類別中的所有疾病創建趨勢圖
    hr_range = np.arange(40, 121, 5)
    
    # 獲得所選類別的所有疾病
    trend_conditions = [cond for cond, data in RISK_DATA.items() 
                       if data['category'] in selected_categories]
    
    if len(trend_conditions) > 0:
        # 計算所需的行數和列數
        n_conditions = len(trend_conditions)
        n_cols = min(3, n_conditions)  # 最多3列
        n_rows = (n_conditions + n_cols - 1) // n_cols  # 向上取整
        
        # 根據行數動態調整間距
        if n_rows == 1:
            # 單行 - 使用寬鬆間距
            vertical_spacing = 0.25
            horizontal_spacing = 0.15
        elif n_rows == 2:
            # 兩行 - 使用中等間距
            vertical_spacing = 0.20
            horizontal_spacing = 0.12
        elif n_rows == 3:
            # 三行 - 使用較小間距但仍可讀（適用於3x3格式的9個條件）
            vertical_spacing = 0.12
            horizontal_spacing = 0.08
        else:
            # 四行或以上 - 使用最小間距
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
            
            trend_risks = [calculate_relative_risk(hr, condition) for hr in hr_range]
            
            # 添加基準參考線
            baseline_hr = RISK_DATA[condition]['baseline']
            fig_trend.add_hline(
                y=1.0, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                row=row, col=col
            )
            
            # 為研究基準添加垂直線
            fig_trend.add_vline(
                x=baseline_hr,
                line_dash="dot",
                line_color="blue",
                opacity=0.5,
                row=row, col=col
            )
            
            # 主趨勢線
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
            
            # 添加目前點位
            current_risk = calculate_relative_risk(current_hr, condition)
            fig_trend.add_trace(
                go.Scatter(
                    x=[current_hr], 
                    y=[current_risk],
                    mode='markers',
                    name=f"您的風險",
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=False,
                    hovertemplate=f'<b>{condition}</b><br>心率：{current_hr} bpm<br>風險：{current_risk:.2f}倍<extra></extra>'
                ),
                row=row, col=col
            )
        
        # 根據行數動態調整高度和邊距
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
            title_text=f"{', '.join(selected_categories)} 類別風險趨勢",
            showlegend=False,
            font=dict(size=font_size),
            margin=margin_dict
        )
        
        # 為每個子圖更新軸向，使用動態間距
        for i in range(n_conditions):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            fig_trend.update_xaxes(
                title_text="心率 (bpm)", 
                row=row, col=col,
                title_standoff=title_standoff,
                tickfont=dict(size=font_size-1)
            )
            fig_trend.update_yaxes(
                title_text="相對風險", 
                row=row, col=col,
                title_standoff=title_standoff,
                tickfont=dict(size=font_size-1)
            )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 添加圖例說明
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>📖 圖表圖例：</strong><br>
            • <span style="color: gray;">灰色虛線</span>：基準風險 (1.0倍)<br>
            • <span style="color: blue;">藍色點線</span>：研究基準心率<br>
            • <span style="color: red;">紅色星號</span>：您目前的風險水平
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("請至少選擇一個類別來查看趨勢分析。")
    
    # 建議
    st.markdown("### 💊 建議")
    
    max_risk = max(risk_values)
    if max_risk >= 1.3:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ 檢測到高風險</strong><br>
            建議就您的心率升高諮詢醫療保健提供者。生活方式的改變和醫療評估可能會有幫助。
        </div>
        """, unsafe_allow_html=True)
    elif max_risk >= 1.1:
        st.markdown("""
        <div class="warning-box">
            <strong>⚡ 中等風險</strong><br>
            您的心率稍微升高。建議考慮改善生活方式，如規律運動、壓力管理和保持健康體重。
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>✅ 良好的心率範圍</strong><br>
            您的心率似乎在健康範圍內。請繼續保持健康的生活方式，包括規律運動和適當營養。
        </div>
        """, unsafe_allow_html=True)
    
    # 頁尾免責聲明
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <strong>⚠️ 醫療免責聲明：</strong> 此計算器僅供教育目的使用。
        結果基於人群水平的統合分析資料，不應取代專業醫療建議。
        請務必諮詢醫療保健提供者進行個人健康評估和治療決策。
    </div>
    """, unsafe_allow_html=True)
    
    # 資料來源
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.8rem; margin-top: 1rem;">
        <em>風險計算基於統合分析研究：<br>
        • 心血管結果：基準值 65 bpm<br>
        • 腎臟和糖尿病結果：基準值 60 bpm</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
