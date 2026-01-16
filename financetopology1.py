import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 0. 全局設定與資產池
# ==========================================
st.set_page_config(
    page_title="Posa 拓撲天文台 (Alpha 14.0)",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="collapsed"
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
    
    /* 卡片式佈局 */
    .card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #555;
    }
    .card-title { font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }
    .card-value { font-size: 1.5em; font-weight: bold; }
    .card-sub { font-size: 0.9em; color: #AAA; }
    .pred-val { color: #00BFFF; }
    .acc-high { color: #00FF7F; }
    .acc-low { color: #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# 觀測名單
OBSERVATORY_ASSETS = {
    'Canary (金絲雀)': ['BAC'],
    'Financials (金融)': ['JPM', 'WFC', 'XLF'],
    'Tech (科技)': ['NVDA', 'AMZN', 'GOOGL', 'TSLA', 'PLTR'],
    'Defensive (防禦)': ['KO', 'WMT', 'DIS', 'XLP'],
    'Macro (宏觀)': ['XLE', 'SPY']
}

ALL_TICKERS = [t for cat in OBSERVATORY_ASSETS.values() for t in cat]

# 拓撲參數
CONSTANTS = {
    "RF_TREES": 100,
    "DEV_THRESHOLD_NORMAL": 0.05,
    "DEV_THRESHOLD_CANARY": 0.02,
    "LIQUIDITY_THRESHOLD": -0.137 
}

# ==========================================
# 1. 核心數據引擎
# ==========================================
@st.cache_data(ttl=60)
def fetch_live_data(tickers):
    data = yf.download(tickers, period="2y", interval="1d", progress=False)
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
# 2. 拓撲模型引擎 (雙向預測)
# ==========================================
def train_rf_model_dual(series, forecast_days=30):
    """
    同時訓練兩個模型：
    1. Backtest Model: 用 t-30 預測 t (驗證準確度)
    2. Forecast Model: 用 t 預測 t+30 (給出未來目標)
    """
    try:
        df = pd.DataFrame({'Close': series})
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean()
        
        # 特徵工程
        df['Target_Future'] = df['Close'].shift(-forecast_days) # 未來價格 (用於訓練預測模型)
        df['Target_Current'] = df['Close'] # 當前價格 (用於驗證過去預測)
        
        df = df.dropna()
        if len(df) < 100: return None, None
        
        # --- A. 準確度驗證 (Backtest) ---
        # 用 30 天前的數據特徵，來預測"今天"
        X_past = df[['Close', 'Vol', 'SMA']].shift(forecast_days).dropna()
        y_past = df['Target_Current'].reindex(X_past.index)
        
        # 取最近 30 筆來計算平均準確度
        recent_X = X_past.iloc[-30:]
        recent_y = y_past.iloc[-30:]
        
        model_back = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        # 用更早的數據訓練
        train_end = len(X_past) - 30
        model_back.fit(X_past.iloc[:train_end], y_past.iloc[:train_end])
        
        preds_past = model_back.predict(recent_X)
        errors = np.abs((preds_past - recent_y) / recent_y)
        avg_accuracy = 1 - errors.mean() # 平均準確度 (e.g., 98%)
        
        # --- B. 未來預測 (Forecast) ---
        # 用所有數據訓練，預測 30 天後
        X_now = df[['Close', 'Vol', 'SMA']]
        y_future = df['Target_Future'] # 這裡會有 NaN，因為最後 30 天沒未來
        
        valid_idx = y_future.dropna().index
        model_future = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_future.fit(X_now.loc[valid_idx], y_future.loc[valid_idx])
        
        # 預測未來
        last_features = X_now.iloc[[-1]]
        pred_future = model_future.predict(last_features)[0]
        
        return avg_accuracy, pred_future
        
    except Exception as e:
        return None, None

def calculate_metrics(ticker, df_close):
    if ticker not in df_close.columns: return None
    
    price_real = df_close[ticker].iloc[-1]
    
    # 執行雙向預測
    acc, pred_30d = train_rf_model_dual(df_close[ticker])
    
    if acc and pred_30d:
        # 簡單偏差 (Deviation)
        deviation = (price_real - pred_30d) / pred_30d # 這裡僅作參考，主要看準確度
        
        # 信心評分
        confidence = "HIGH" if acc > 0.95 else "LOW"
        
        return {
            "Price": price_real,
            "Pred_30d": pred_30d,
            "Accuracy": acc,
            "Confidence": confidence
        }
    return None

# ==========================================
# 3. 儀表板邏輯
# ==========================================
def main():
    st.title("🔭 Posa 拓撲天文台 (Alpha 14.0)")
    st.markdown("### 雙向監控：歷史準確度驗證 + 未來 30 天導航")
    
    with st.sidebar:
        if st.button("🔄 刷新數據"):
            st.cache_data.clear()
            st.rerun()

    # 1. 獲取數據
    with st.spinner("🦅 正在計算雙向拓撲軌跡..."):
        df_close, df_high, df_low, df_vol = fetch_live_data(ALL_TICKERS)
    
    # 2. 計算結果
    results = {}
    canary_status = "OK"
    
    for cat, tickers in OBSERVATORY_ASSETS.items():
        results[cat] = []
        for t in tickers:
            res = calculate_metrics(t, df_close)
            if res:
                # 金絲雀檢查
                if t == 'BAC' and res['Accuracy'] < 0.98: # 如果 BAC 準確度下降，代表模型失靈
                    canary_status = "WARNING"
                
                results[cat].append({
                    "Ticker": t,
                    "Data": res
                })

    # 3. 警報條
    if canary_status == "WARNING":
        st.warning("⚠️ 【金絲雀警示】BAC 預測準確度下降，全域流動性可能出現擾動。")
    else:
        st.success("✅ 【系統穩定】金絲雀 (BAC) 運行精準，模型可信度高。")
        
    st.markdown("---")

    # 4. 卡片式儀表板 (Card Dashboard)
    cols = st.columns(len(OBSERVATORY_ASSETS))
    
    for idx, (cat, items) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"#### {cat}")
            for item in items:
                t = item['Ticker']
                d = item['Data']
                
                # 樣式邏輯
                acc_fmt = f"{d['Accuracy']:.1%}"
                acc_class = "acc-high" if d['Accuracy'] > 0.95 else "acc-low"
                
                # 計算預期漲跌幅
                upside = (d['Pred_30d'] - d['Price']) / d['Price']
                upside_str = f"{upside:+.1%}"
                upside_color = "green" if upside > 0 else "red"
                
                # 拓撲修正註記 (模擬)
                # 在真實版本可加入 is_crunch 判斷
                
                st.markdown(f"""
                <div class="card" style="border-left-color: {upside_color};">
                    <div class="card-title">{t} <span style="font-size:0.8em; float:right;" class="{acc_class}">準度: {acc_fmt}</span></div>
                    <div class="card-value">${d['Price']:.2f}</div>
                    <div class="card-sub">
                        🎯 30天預測: <span class="pred-val">${d['Pred_30d']:.2f}</span><br>
                        📈 預期波動: <span style="color:{upside_color}">{upside_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()