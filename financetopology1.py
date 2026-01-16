import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 0. 全局設定與資產池 (Observatory Config)
# ==========================================
st.set_page_config(
    page_title="Posa 拓撲天文台 (Live Monitor)",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="collapsed" # 戰情室模式，預設收起側邊欄
)

# 注入戰情室風格 CSS
st.markdown("""
<style>
    .big-font { font-size: 20px !important; font-weight: bold; }
    .stMetric { background-color: #1E1E1E; border: 1px solid #444; border-radius: 5px; padding: 10px; }
    .status-ok { color: #00FF7F; font-weight: bold; }
    .status-warn { color: #FFD700; font-weight: bold; }
    .status-danger { color: #FF4B4B; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# 觀測名單 (The Golden 15 + Canary)
OBSERVATORY_ASSETS = {
    'Canary (金絲雀)': ['BAC'], # 系統性風險指標
    'Financials (金融)': ['JPM', 'WFC', 'XLF'],
    'Tech (科技)': ['NVDA', 'AMZN', 'GOOGL', 'TSLA', 'PLTR'],
    'Defensive (防禦)': ['KO', 'WMT', 'DIS', 'XLP'],
    'Macro (宏觀)': ['XLE', 'SPY']
}

ALL_TICKERS = [t for cat in OBSERVATORY_ASSETS.values() for t in cat]

# 拓撲參數
CONSTANTS = {
    "RF_TREES": 100,
    "LOOKBACK_YEARS": 2,
    "DEV_THRESHOLD_NORMAL": 0.05, # 一般股票 5% 警戒
    "DEV_THRESHOLD_CANARY": 0.02  # 金絲雀 2% 警戒 (更敏感)
}

# ==========================================
# 1. 核心數據引擎 (Real-time Data Sheaf)
# ==========================================
@st.cache_data(ttl=60) # 每 60 秒快取一次 (模擬即時)
def fetch_live_data(tickers):
    data = yf.download(tickers, period="2y", interval="1d", progress=False)
    # 處理 MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Close'].ffill()
        high = data['High'].ffill()
        low = data['Low'].ffill()
        volume = data['Volume'].ffill()
    else:
        adj_close = data['Close'].ffill()
        high = data['High'].ffill()
        low = data['Low'].ffill()
        volume = data['Volume'].ffill()
    return adj_close, high, low, volume

# ==========================================
# 2. 拓撲模型引擎 (The Model Core)
# ==========================================
def train_rf_model(series):
    try:
        df = pd.DataFrame({'Close': series})
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean()
        # 目標：預測"當下"的合理價 (用過去數據訓練)
        # 這裡我們做一個 "Nowcasting" 模型：用 t-1 的特徵預測 t 的價格
        df['Target'] = df['Close'] # 預測本身 (Auto-regressive)
        df['Prev_Close'] = df['Close'].shift(1)
        df = df.dropna()
        
        if len(df) < 60: return None
        
        X = df[['Prev_Close', 'Vol', 'SMA']]
        y = df['Target']
        
        model = RandomForestRegressor(n_estimators=CONSTANTS['RF_TREES'], max_depth=5, random_state=42)
        # 使用除了最後一天以外的數據訓練
        model.fit(X.iloc[:-1], y.iloc[:-1])
        
        # 預測最後一天 (今天) 的理論價
        predicted_price = model.predict(X.iloc[[-1]])[0]
        return predicted_price
    except: return None

def calculate_deviation(ticker, df_close, df_high, df_low):
    if ticker not in df_close.columns: return None
    
    # 1. 獲取現價
    price_real = df_close[ticker].iloc[-1]
    
    # 2. 計算模型理論價 (RF + ATR)
    # RF Component
    p_rf = train_rf_model(df_close[ticker])
    
    # ATR Component (波動率修正)
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # 綜合模型價 (RF 為主，ATR 為輔)
    if p_rf:
        p_model = p_rf 
        # 計算乖離率
        deviation = (price_real - p_model) / p_model
        return {
            "Price_Real": price_real,
            "Price_Model": p_model,
            "Deviation": deviation,
            "ATR": atr
        }
    return None

# ==========================================
# 3. 儀表板邏輯 (Dashboard Logic)
# ==========================================
def main():
    st.title("🔭 Posa 拓撲天文台 (Topological Observatory)")
    st.markdown("### 即時偏差監控與金絲雀警報系統")
    
    # 側邊欄：API Key (如果需要 FRED)
    with st.sidebar:
        st.write("🔧 系統設定")
        if st.button("🔄 刷新數據"):
            st.cache_data.clear()
            st.rerun()

    # 1. 獲取數據
    with st.spinner("🦅 正在掃描全市場拓撲結構..."):
        df_close, df_high, df_low, df_vol = fetch_live_data(ALL_TICKERS)
    
    # 2. 計算全市場偏差
    results = {}
    canary_status = "OK"
    
    for cat, tickers in OBSERVATORY_ASSETS.items():
        results[cat] = []
        for t in tickers:
            res = calculate_deviation(t, df_close, df_high, df_low)
            if res:
                # 燈號判定
                dev = res['Deviation']
                is_canary = (t == 'BAC')
                threshold = CONSTANTS['DEV_THRESHOLD_CANARY'] if is_canary else CONSTANTS['DEV_THRESHOLD_NORMAL']
                
                if abs(dev) > threshold * 1.5: status = "🔴 異常 (Anomaly)"
                elif abs(dev) > threshold: status = "🟡 警戒 (Warning)"
                else: status = "🟢 穩定 (Stable)"
                
                # 金絲雀檢查
                if is_canary and "🔴" in status: canary_status = "CRITICAL"
                elif is_canary and "🟡" in status: canary_status = "WARNING"
                
                results[cat].append({
                    "Ticker": t,
                    "Price": res['Price_Real'],
                    "Model": res['Price_Model'],
                    "Deviation": dev,
                    "Status": status
                })

    # 3. 頂部警報條 (The Canary Bar)
    if canary_status == "CRITICAL":
        st.error("🚨 【系統性警報】金絲雀 (BAC) 偵測到嚴重拓撲撕裂！全域流動性可能正在崩潰。建議立即執行 Hard Defense。")
    elif canary_status == "WARNING":
        st.warning("⚠️ 【流動性預警】金絲雀 (BAC) 出現異常波動。請密切關注板塊輪動。")
    else:
        st.success("✅ 【系統正常】全域流動性結構穩定。模型運作中。")
        
    st.markdown("---")

    # 4. 板塊監控儀表板 (Sector Monitors)
    # 使用 4 列佈局
    cols = st.columns(len(OBSERVATORY_ASSETS))
    
    for idx, (cat, data_list) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"#### {cat}")
            for item in data_list:
                # 視覺化偏差條
                dev_pct = item['Deviation'] * 100
                color = "green"
                if "🔴" in item['Status']: color = "red"
                elif "🟡" in item['Status']: color = "orange"
                
                st.markdown(f"""
                **{item['Ticker']}** 現價: ${item['Price']:.2f}  
                <span style='color:{color}; font-weight:bold'>乖離: {dev_pct:+.2f}%</span>  
                <progress value='{50 + dev_pct}' max='100' style='width:100%'></progress>
                <small>{item['Status']}</small>
                <hr style='margin: 5px 0'>
                """, unsafe_allow_html=True)

    # 5. 資金流向熱圖 (Sector Flow Heatmap)
    st.markdown("### 🌊 即時資金流向 (Real-time Flow)")
    
    # 準備熱圖數據
    heatmap_data = []
    for cat, items in results.items():
        avg_dev = np.mean([i['Deviation'] for i in items])
        heatmap_data.append({'Sector': cat, 'Avg_Deviation': avg_dev})
    
    hm_df = pd.DataFrame(heatmap_data)
    
    fig = px.bar(
        hm_df, x='Sector', y='Avg_Deviation',
        color='Avg_Deviation',
        color_continuous_scale=['red', 'yellow', 'green'],
        range_color=[-0.05, 0.05],
        title="板塊乖離率熱圖 (正值=資金流入/強於模型, 負值=資金流出/弱於模型)"
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()