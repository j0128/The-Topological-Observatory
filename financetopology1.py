import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from datetime import datetime

# ==========================================
# 0. 全局設定
# ==========================================
st.set_page_config(
    page_title="Posa 天文台 (Alpha 15.1 - Iron Dome)",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #555; }
    .card-title { font-size: 1.2em; font-weight: bold; }
    .card-value { font-size: 1.5em; font-weight: bold; }
    .fund-score-good { color: #00FF7F; font-weight: bold; }
    .fund-score-bad { color: #FF4B4B; font-weight: bold; }
    .fund-score-neutral { color: #FFD700; font-weight: bold; }
    .safe-harbor-header { color: #00BFFF; border-bottom: 2px solid #00BFFF; padding-bottom: 5px; margin-top: 30px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# 1. 主戰場名單 (您關注的高波動/成長股)
OBSERVATORY_ASSETS = {
    '🚨 Canary (系統金絲雀)': ['BAC'],
    '⚔️ Tech (進攻型)': ['NVDA', 'META', 'AMD', 'TSLA', 'PLTR'],
    '🚑 Distressed (觀察名單)': ['AMC', 'CLS']
}

# 2. 鐵壁防禦名單 (實驗 E 選出的 Top 15, 誤差 < 6%)
SAFE_HARBOR_LIST = [
    'XLP', 'TLT', 'XLV', 'KO', 'XLE', 
    'MMM', 'JNJ', 'MCD', 'XLF', 'RTX', 
    'XOM', 'CVX', 'MO', 'GILD', 'AMGN'
]

# 合併所有需要抓取的 Ticker
ALL_TICKERS = list(set([t for cat in OBSERVATORY_ASSETS.values() for t in cat] + SAFE_HARBOR_LIST))

# ==========================================
# 1. 基本面權重引擎
# ==========================================
@st.cache_data(ttl=3600*12) 
def get_fundamental_scalar(ticker):
    """
    計算基本面權重純量 (Scalar)。範圍：0.85 ~ 1.15
    """
    try:
        stock = yf.Ticker(ticker)
        fins = stock.quarterly_financials
        if fins.empty: fins = stock.financials
        
        # 對於 ETF (如 XLP, TLT)，通常抓不到財報，回傳 1.0 (中性)
        if fins.empty: 
            return 1.0, ["⚖️ ETF/無財報數據 (維持中性)"]

        score = 0
        details = []
        
        # A. 營收成長
        if 'Total Revenue' in fins.index and len(fins.columns) >= 2:
            r_now = fins.loc['Total Revenue'].iloc[0]
            r_prev = fins.loc['Total Revenue'].iloc[1]
            growth = (r_now - r_prev) / r_prev
            
            if growth > 0.10: 
                score += 1
                details.append(f"🔥 營收成長 (+{growth:.1%})")
            elif growth < -0.05: 
                score -= 1
                details.append(f"📉 營收衰退 ({growth:.1%})")
            else:
                details.append(f"⚪ 營收持平 ({growth:.1%})")
        
        # B. 獲利能力
        if 'Net Income' in fins.index:
            ni = fins.loc['Net Income'].iloc[0]
            if ni > 0: 
                score += 1
                details.append("💰 獲利為正")
            else: 
                score -= 1
                details.append("💸 處於虧損")
                
        scalar = 1.0 + (score * 0.05)
        scalar = max(0.85, min(1.15, scalar))
        
        return scalar, details
        
    except Exception as e:
        return 1.0, ["⚠️ 數據異常"]

# ==========================================
# 2. 技術模型引擎
# ==========================================
@st.cache_data(ttl=60)
def fetch_live_data(tickers):
    # 下載數據
    data = yf.download(tickers, period="2y", interval="1d", progress=False)
    # 處理 MultiIndex Column 問題
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close'].ffill()
    else:
        return data['Close'].ffill()

def train_rf_model(series):
    try:
        df = pd.DataFrame({'Close': series})
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Vol'] = df['Close'].pct_change().rolling(20).std()
        df['Target'] = df['Close'] # 預測合理價
        df = df.dropna()
        if len(df) < 50: return series.iloc[-1]
        
        X = df[['MA20', 'Vol']]
        y = df['Target']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except: return series.iloc[-1]

# ==========================================
# 3. 渲染卡片函數
# ==========================================
def render_card(t, price_now, tech_target, scalar, reasons):
    final_target = tech_target * scalar
    
    # 顏色邏輯
    scalar_pct = (scalar - 1) * 100
    if scalar > 1.0: s_color = "fund-score-good"; s_sign = "+"
    elif scalar < 1.0: s_color = "fund-score-bad"; s_sign = ""
    else: s_color = "fund-score-neutral"; s_sign = ""
    
    upside = (final_target - price_now) / price_now
    up_color = "#00FF7F" if upside > 0 else "#FF4B4B"
    border_color = up_color
    
    reasons_html = "<br>".join([f"<small>{r}</small>" for r in reasons])
    
    st.markdown(f"""
    <div class="card" style="border-left-color: {border_color};">
        <div class="card-title">{t} <span style="float:right; font-size:0.8em; color:#FFF">${price_now:.2f}</span></div>
        <div style="margin-top:5px; font-size:0.9em; color:#AAA;">
            技術價: ${tech_target:.2f}<br>
            <span class="{s_color}">財報權重: x{scalar:.2f} ({s_sign}{scalar_pct:.0f}%)</span>
        </div>
        <div class="card-value" style="color:{up_color}; margin-top:5px;">
            目標: ${final_target:.2f} <small>({upside:+.1%})</small>
        </div>
        <div style="color: #888; margin-top:5px; line-height:1.2;">
            {reasons_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. 主程式
# ==========================================
def main():
    st.title("🔭 Posa 拓撲天文台 (Alpha 15.1)")
    st.markdown("### 戰情室：技術模型 + 財報權重 + 鐵壁防禦")
    
    with st.sidebar:
        if st.button("🔄 刷新全域數據"):
            st.cache_data.clear()
            st.rerun()
            
    with st.spinner("🦅 正在掃描全市場拓撲結構 (Main + Safe Harbor)..."):
        df_close = fetch_live_data(ALL_TICKERS)
        
    # --- Part 1: 主戰場 (Observatory Assets) ---
    cols = st.columns(len(OBSERVATORY_ASSETS))
    for idx, (cat, tickers) in enumerate(OBSERVATORY_ASSETS.items()):
        with cols[idx]:
            st.markdown(f"#### {cat}")
            for t in tickers:
                if t not in df_close.columns: continue
                
                # 計算數據
                price_now = df_close[t].iloc[-1]
                tech_target = train_rf_model(df_close[t])
                scalar, reasons = get_fundamental_scalar(t)
                
                # 渲染
                render_card(t, price_now, tech_target, scalar, reasons)

    # --- Part 2: 鐵壁防禦陣列 (Safe Harbor) ---
    st.markdown("<h3 class='safe-harbor-header'>🛡️ Posa 鐵壁防禦陣列 (The Iron Dome)</h3>", unsafe_allow_html=True)
    st.markdown("以下 15 檔標的經實驗驗證，過去 12 個月模型預測誤差 **< 6%**。當市場動盪時，它們是資金的避風港。")
    
    # 使用 5 列佈局展示 15 支股票
    sh_cols = st.columns(5)
    
    for i, t in enumerate(SAFE_HARBOR_LIST):
        if t not in df_close.columns: continue
        
        col_idx = i % 5
        with sh_cols[col_idx]:
            price_now = df_close[t].iloc[-1]
            tech_target = train_rf_model(df_close[t])
            # 對於 Safe Harbor，我們同樣應用財報加權 (如果是 ETF 則為 1.0)
            scalar, reasons = get_fundamental_scalar(t)
            
            render_card(t, price_now, tech_target, scalar, reasons)

if __name__ == "__main__":
    main()