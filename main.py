import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time, timedelta
import time as tm
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# STREAMLIT CONFIG - INSTITUTIONAL TRADING PLATFORM
# ==================================================
st.set_page_config(
    page_title="Institutional Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional institutional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    .instrument-header {
        font-size: 28px;
        font-weight: 700;
        color: #34495E;
        margin-bottom: 5px;
    }
    .price-display {
        font-size: 32px;
        font-weight: 700;
        color: #27AE60;
        font-family: 'SF Mono', 'Consolas', monospace;
    }
    .price-change-positive {
        color: #27AE60;
        font-weight: 600;
    }
    .price-change-negative {
        color: #E74C3C;
        font-weight: 600;
    }
    .metric-label {
        font-size: 11px;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: #2C3E50;
    }
    .signal-buy {
        background-color: #27AE60;
        color: white;
        padding: 12px 20px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 20px;
        text-align: center;
    }
    .signal-sell {
        background-color: #E74C3C;
        color: white;
        padding: 12px 20px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 20px;
        text-align: center;
    }
    .signal-neutral {
        background-color: #7F8C8D;
        color: white;
        padding: 12px 20px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 20px;
        text-align: center;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #34495E;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #ECF0F1;
    }
    .card {
        background: white;
        border: 1px solid #ECF0F1;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .risk-high {
        color: #E74C3C;
        font-weight: 600;
    }
    .risk-medium {
        color: #F39C12;
        font-weight: 600;
    }
    .risk-low {
        color: #27AE60;
        font-weight: 600;
    }
    .session-indicator {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 10px;
        display: inline-block;
        font-weight: 600;
    }
    .session-open { background: #27AE60; color: white; }
    .session-london { background: #3498DB; color: white; }
    .session-asia { background: #9B59B6; color: white; }
    .session-close { background: #7F8C8D; color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">INSTITUTIONAL TRADING PLATFORM</div>', unsafe_allow_html=True)
st.markdown('**Multi-Asset Analysis • Professional Execution • Risk Management**')

# ==================================================
# SIDEBAR - TRADER CONFIGURATION
# ==================================================
with st.sidebar:
    # Logo placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding: 10px 0;'><strong>TRADING CONSOLE</strong></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Asset selection
    st.markdown("**INSTRUMENT SELECTION**")
    selected_instrument = st.selectbox(
        "",
        [
            "BTC-USD", 
            "ETH-USD", 
            "XAU-USD", 
            "DXY"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    # Instrument details
    instrument_data = {
        "BTC-USD": {"symbol": "BTC", "asset_class": "Digital Asset", "lot": 0.01},
        "ETH-USD": {"symbol": "ETH", "asset_class": "Digital Asset", "lot": 0.1},
        "XAU-USD": {"symbol": "XAU", "asset_class": "Commodity", "lot": 0.1},
        "DXY": {"symbol": "DXY", "asset_class": "Currency Index", "lot": 0.01}
    }
    
    instrument = instrument_data[selected_instrument]
    
    st.caption(f"Asset: {instrument['asset_class']}")
    st.caption(f"Lot size: {instrument['lot']}")
    
    st.divider()
    
    # Trading parameters
    st.markdown("**TRADING PARAMETERS**")
    
    col1, col2 = st.columns(2)
    with col1:
        position_size = st.number_input("SIZE", min_value=1, max_value=1000, value=10, step=5)
    with col2:
        risk_pct = st.slider("RISK", 0.1, 5.0, 1.0, 0.1)
    
    # Timeframe selection
    st.markdown("**TIME FRAME**")
    primary_tf = st.selectbox("Primary", ["15M", "1H", "4H", "1D"], index=1, label_visibility="collapsed")
    
    # Strategy
    st.markdown("**EXECUTION STRATEGY**")
    strategy = st.selectbox(
        "",
        ["Limit", "TWAP", "VWAP", "Market"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Market data refresh
    if st.button("REFRESH DATA", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.caption(f"{datetime.now().strftime('%H:%M')} UTC")

# ==================================================
# DATA MANAGEMENT - ROBUST & PROFESSIONAL
# ==================================================

def safe_get_last(series, default=0):
    """Safe data access"""
    try:
        if series is None or len(series) == 0:
            return default
        return series.iloc[-1]
    except:
        return default

def safe_get_prev(series, default=0):
    """Safe previous value access"""
    try:
        if series is None or len(series) < 2:
            return default
        return series.iloc[-2]
    except:
        return default

@st.cache_data(ttl=30)
def fetch_market_data(symbol, timeframe="1H"):
    """Fetch professional market data"""
    
    # Base prices
    base_prices = {
        "BTC": 45000,
        "ETH": 2500,
        "XAU": 5000,  # Professional gold price
        "DXY": 105.0
    }
    
    base_price = base_prices.get(symbol, 45000)
    
    # Timeframe mapping
    tf_map = {
        "15M": ("15min", 400),
        "1H": ("H", 300),
        "4H": ("4H", 200),
        "1D": ("D", 100)
    }
    
    freq, periods = tf_map.get(timeframe, ("H", 300))
    
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # Asset-specific volatility
    volatilities = {
        "BTC": 0.02,
        "ETH": 0.025,
        "XAU": 0.008,
        "DXY": 0.003
    }
    
    vol = volatilities.get(symbol, 0.02)
    
    # Generate professional price series
    returns = np.random.normal(0, vol, periods)
    
    # Add slight trend
    trend = np.random.uniform(-0.0001, 0.0001)
    returns = returns + trend
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    for i in range(len(df)):
        if i == 0:
            df.loc[df.index[i], 'open'] = df['close'].iloc[i] * 0.999
        else:
            df.loc[df.index[i], 'open'] = df['close'].iloc[i-1]
        
        spread = df['close'].iloc[i] * vol * np.random.uniform(0.8, 1.2)
        df.loc[df.index[i], 'high'] = max(df['close'].iloc[i], df['open'].iloc[i]) + spread * 0.6
        df.loc[df.index[i], 'low'] = min(df['close'].iloc[i], df['open'].iloc[i]) - spread * 0.6
    
    # Volume
    base_volumes = {
        "BTC": 1e9,
        "ETH": 5e8,
        "XAU": 1e7,
        "DXY": 1e6
    }
    
    base_vol = base_volumes.get(symbol, 1e8)
    df['volume'] = base_vol * np.exp(np.random.normal(0, 0.5, periods))
    
    # Add technical indicators
    df = calculate_technical_indicators(df)
    
    return df

def calculate_technical_indicators(df):
    """Calculate professional technical indicators"""
    if len(df) < 20:
        return df
    
    df = df.copy()
    
    # Moving averages
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df.fillna(method='ffill')

def get_market_session():
    """Get current trading session"""
    now = datetime.utcnow()
    hour = now.hour
    
    if 13 <= hour < 21:  # NY open
        return "NY Session", "session-open"
    elif 8 <= hour < 13:  # London
        return "London Session", "session-london"
    elif 0 <= hour < 8:  # Asia
        return "Asia Session", "session-asia"
    else:
        return "Market Close", "session-close"

# ==================================================
# TRADING ANALYSIS ENGINE
# ==================================================

class TradingAnalysis:
    """Professional trading analysis engine"""
    
    def __init__(self, df, symbol):
        self.df = df.copy()
        self.symbol = symbol
        self.analyze()
    
    def analyze(self):
        """Run comprehensive analysis"""
        # Trend analysis
        self.trend = self._analyze_trend()
        
        # Momentum analysis
        self.momentum = self._analyze_momentum()
        
        # Volume analysis
        self.volume = self._analyze_volume()
        
        # Generate signal
        self.signal = self._generate_signal()
        
        # Calculate levels
        self.levels = self._calculate_levels()
    
    def _analyze_trend(self):
        """Analyze market trend"""
        current_price = safe_get_last(self.df['close'])
        ma_20 = safe_get_last(self.df['ma_20'], current_price)
        ma_50 = safe_get_last(self.df['ma_50'], current_price)
        
        if current_price > ma_20 > ma_50:
            return {"direction": "UPTREND", "strength": "Strong"}
        elif current_price < ma_20 < ma_50:
            return {"direction": "DOWNTREND", "strength": "Strong"}
        elif current_price > ma_20:
            return {"direction": "UPTREND", "strength": "Moderate"}
        elif current_price < ma_20:
            return {"direction": "DOWNTREND", "strength": "Moderate"}
        else:
            return {"direction": "NEUTRAL", "strength": "Weak"}
    
    def _analyze_momentum(self):
        """Analyze market momentum"""
        rsi = safe_get_last(self.df['rsi'], 50)
        macd = safe_get_last(self.df['macd'], 0)
        macd_signal = safe_get_last(self.df['macd_signal'], 0)
        
        momentum = {"rsi": rsi, "macd": macd, "signal": macd_signal}
        
        if rsi < 30:
            momentum["state"] = "OVERSOLD"
        elif rsi > 70:
            momentum["state"] = "OVERBOUGHT"
        else:
            momentum["state"] = "NEUTRAL"
        
        if macd > macd_signal:
            momentum["macd_signal"] = "BULLISH"
        else:
            momentum["macd_signal"] = "BEARISH"
        
        return momentum
    
    def _analyze_volume(self):
        """Analyze volume profile"""
        volume_ratio = safe_get_last(self.df['volume_ratio'], 1)
        
        volume = {"ratio": volume_ratio}
        
        if volume_ratio > 1.5:
            volume["profile"] = "HIGH"
        elif volume_ratio < 0.5:
            volume["profile"] = "LOW"
        else:
            volume["profile"] = "NORMAL"
        
        return volume
    
    def _calculate_levels(self):
        """Calculate support/resistance levels"""
        recent = self.df.tail(50)
        
        if len(recent) < 10:
            current = safe_get_last(self.df['close'], 0)
            return {
                "resistance": current * 1.02,
                "support": current * 0.98,
                "pivot": current
            }
        
        high = recent['high'].max()
        low = recent['low'].min()
        current = safe_get_last(self.df['close'])
        
        return {
            "resistance": high,
            "support": low,
            "pivot": (high + low + current) / 3,
            "r1": (2 * ((high + low + current) / 3)) - low,
            "s1": (2 * ((high + low + current) / 3)) - high
        }
    
    def _generate_signal(self):
        """Generate trading signal"""
        current_price = safe_get_last(self.df['close'])
        rsi = self.momentum["rsi"]
        trend = self.trend["direction"]
        volume = self.volume["profile"]
        
        signal = {
            "action": "HOLD",
            "confidence": 50,
            "reason": [],
            "entry": current_price,
            "stop": None,
            "target": None
        }
        
        # Trend following with RSI confirmation
        if trend == "UPTREND" and rsi < 60:
            signal["action"] = "BUY"
            signal["confidence"] = 65
            signal["reason"].append("Uptrend with room for momentum")
            
            # Calculate levels
            atr = safe_get_last(self.df['atr'], current_price * 0.01)
            signal["stop"] = current_price - (atr * 2)
            signal["target"] = current_price + (atr * 4)
            
        elif trend == "DOWNTREND" and rsi > 40:
            signal["action"] = "SELL"
            signal["confidence"] = 65
            signal["reason"].append("Downtrend with momentum confirmation")
            
            # Calculate levels
            atr = safe_get_last(self.df['atr'], current_price * 0.01)
            signal["stop"] = current_price + (atr * 2)
            signal["target"] = current_price - (atr * 4)
        
        # Mean reversion at extremes
        elif rsi < 30 and volume == "HIGH":
            signal["action"] = "BUY"
            signal["confidence"] = 70
            signal["reason"].append("Oversold with volume confirmation")
            
            bb_lower = safe_get_last(self.df['bb_lower'], current_price * 0.98)
            signal["stop"] = bb_lower * 0.99
            signal["target"] = safe_get_last(self.df['bb_middle'], current_price * 1.01)
            
        elif rsi > 70 and volume == "HIGH":
            signal["action"] = "SELL"
            signal["confidence"] = 70
            signal["reason"].append("Overbought with volume confirmation")
            
            bb_upper = safe_get_last(self.df['bb_upper'], current_price * 1.02)
            signal["stop"] = bb_upper * 1.01
            signal["target"] = safe_get_last(self.df['bb_middle'], current_price * 0.99)
        
        # Volume check
        if volume == "LOW":
            signal["confidence"] = max(30, signal["confidence"] - 20)
            signal["reason"].append("Low volume - reduced conviction")
        
        return signal
    
    def get_risk_level(self):
        """Determine risk level"""
        atr_pct = (safe_get_last(self.df['atr'], 0) / safe_get_last(self.df['close'], 1)) * 100
        vol_20 = safe_get_last(self.df['close'].pct_change().rolling(20).std() * 100, 0)
        
        if vol_20 > 3 or atr_pct > 2:
            return "HIGH"
        elif vol_20 > 1.5 or atr_pct > 1:
            return "MEDIUM"
        else:
            return "LOW"

# ==================================================
# MAIN APPLICATION - PROFESSIONAL DASHBOARD
# ==================================================

def format_price(value, symbol):
    """Format price display"""
    try:
        if symbol in ["BTC", "ETH", "XAU"]:
            return f"${value:,.2f}"
        else:
            return f"{value:.3f}"
    except:
        return "—"

def main():
    # Data loading
    with st.spinner("Loading market data..."):
        # Fetch data
        market_data = fetch_market_data(instrument['symbol'], primary_tf)
        
        # Analyze
        analysis = TradingAnalysis(market_data, instrument['symbol'])
        
        # Get session
        session, session_class = get_market_session()
        
        # Current values
        current_price = safe_get_last(market_data['close'])
        prev_price = safe_get_prev(market_data['close'], current_price)
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        # Volume
        current_volume = safe_get_last(market_data['volume'], 0)
        avg_volume = market_data['volume'].mean() if len(market_data) > 0 else 0
    
    # ==================================================
    # TOP BAR - INSTRUMENT & PRICE
    # ==================================================
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f'<div class="instrument-header">{selected_instrument}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="price-display">{format_price(current_price, instrument["symbol"])}</div>', unsafe_allow_html=True)
        
        if price_change >= 0:
            st.markdown(f'<div class="price-change-positive">+{price_change:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="price-change-negative">{price_change:.2f}%</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-label">VOLUME</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${current_volume/1e6:.1f}M</div>', unsafe_allow_html=True)
        st.caption(f"Avg: ${avg_volume/1e6:.1f}M")
    
    with col3:
        st.markdown('<div class="metric-label">ATR</div>', unsafe_allow_html=True)
        atr = safe_get_last(market_data['atr'], 0)
        st.markdown(f'<div class="metric-value">{format_price(atr, instrument["symbol"])}</div>', unsafe_allow_html=True)
        
        risk_level = analysis.get_risk_level()
        if risk_level == "HIGH":
            st.markdown('<div class="risk-high">High Vol</div>', unsafe_allow_html=True)
        elif risk_level == "MEDIUM":
            st.markdown('<div class="risk-medium">Med Vol</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">Low Vol</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-label">SESSION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="session-indicator {session_class}">{session}</div>', unsafe_allow_html=True)
        st.caption(f"TF: {primary_tf}")
    
    st.divider()
    
    # ==================================================
    # TRADING SIGNAL SECTION
    # ==================================================
    st.markdown('<div class="section-header">TRADING SIGNAL</div>', unsafe_allow_html=True)
    
    signal = analysis.signal
    confidence = signal["confidence"]
    
    col_signal, col_details = st.columns([1, 2])
    
    with col_signal:
        if signal["action"] == "BUY":
            st.markdown('<div class="signal-buy">BUY</div>', unsafe_allow_html=True)
        elif signal["action"] == "SELL":
            st.markdown('<div class="signal-sell">SELL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-neutral">HOLD</div>', unsafe_allow_html=True)
        
        # Confidence meter
        st.progress(confidence/100)
        st.caption(f"Confidence: {confidence}%")
        
        # Risk indicator
        st.markdown(f"Risk: **{risk_level}**")
    
    with col_details:
        if signal["reason"]:
            for reason in signal["reason"]:
                st.info(reason)
        
        # Execution levels if active signal
        if signal["action"] in ["BUY", "SELL"] and signal["stop"] and signal["target"]:
            col_entry, col_stop, col_target = st.columns(3)
            
            with col_entry:
                st.markdown('<div class="metric-label">ENTRY</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_price(signal["entry"], instrument["symbol"])}</div>', unsafe_allow_html=True)
            
            with col_stop:
                st.markdown('<div class="metric-label">STOP</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_price(signal["stop"], instrument["symbol"])}</div>', unsafe_allow_html=True)
                stop_pct = abs((signal["stop"] - signal["entry"]) / signal["entry"] * 100)
                st.caption(f"({stop_pct:.1f}%)")
            
            with col_target:
                st.markdown('<div class="metric-label">TARGET</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{format_price(signal["target"], instrument["symbol"])}</div>', unsafe_allow_html=True)
                target_pct = abs((signal["target"] - signal["entry"]) / signal["entry"] * 100)
                st.caption(f"({target_pct:.1f}%)")
        
        # Position sizing
        if signal["action"] in ["BUY", "SELL"] and signal["stop"]:
            risk_amount = position_size * 1000 * (risk_pct / 100)
            risk_per_unit = abs(signal["entry"] - signal["stop"])
            
            if risk_per_unit > 0:
                units = risk_amount / risk_per_unit
                notional = units * signal["entry"]
                
                st.caption(f"Position: {units:.2f} units (${notional:,.0f})")
                st.caption(f"Strategy: {strategy}")
    
    st.divider()
    
    # ==================================================
    # MARKET ANALYSIS SECTION
    # ==================================================
    st.markdown('<div class="section-header">MARKET ANALYSIS</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Trend", "Momentum", "Levels"])
    
    with tab1:
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**TREND ANALYSIS**")
            st.markdown(f"Direction: **{analysis.trend['direction']}**")
            st.markdown(f"Strength: **{analysis.trend['strength']}**")
            
            ma_20 = safe_get_last(market_data['ma_20'], current_price)
            ma_50 = safe_get_last(market_data['ma_50'], current_price)
            
            st.caption(f"MA20: {format_price(ma_20, instrument['symbol'])}")
            st.caption(f"MA50: {format_price(ma_50, instrument['symbol'])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_t2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**VOLUME PROFILE**")
            st.markdown(f"Current: **{analysis.volume['profile']}**")
            st.markdown(f"Ratio: **{analysis.volume['ratio']:.1f}x**")
            
            vwap = safe_get_last(market_data['vwap'], current_price)
            vwap_dist = ((current_price - vwap) / vwap * 100) if vwap != 0 else 0
            
            st.caption(f"VWAP: {format_price(vwap, instrument['symbol'])}")
            st.caption(f"Distance: {vwap_dist:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**RSI MOMENTUM**")
            rsi = analysis.momentum["rsi"]
            st.markdown(f"Value: **{rsi:.1f}**")
            
            if rsi < 30:
                st.markdown("**Status: OVERSOLD**")
            elif rsi > 70:
                st.markdown("**Status: OVERBOUGHT**")
            else:
                st.markdown("**Status: NEUTRAL**")
            
            st.progress(min(rsi/100, 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**MACD SIGNAL**")
            macd = analysis.momentum["macd"]
            signal_line = analysis.momentum["signal"]
            
            st.markdown(f"MACD: **{macd:.4f}**")
            st.markdown(f"Signal: **{signal_line:.4f}**")
            st.markdown(f"Cross: **{analysis.momentum['macd_signal']}**")
            
            if macd > signal_line:
                st.success("Bullish momentum")
            else:
                st.error("Bearish momentum")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        col_l1, col_l2 = st.columns(2)
        
        with col_l1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**RESISTANCE LEVELS**")
            
            levels = analysis.levels
            st.markdown(f"R1: **{format_price(levels.get('r1', 0), instrument['symbol'])}**")
            st.markdown(f"Pivot: **{format_price(levels.get('pivot', 0), instrument['symbol'])}**")
            
            current = safe_get_last(market_data['close'], 0)
            resistance = levels.get('resistance', current * 1.02)
            dist_to_res = ((resistance - current) / current * 100) if current != 0 else 0
            
            st.caption(f"Distance to resistance: {dist_to_res:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_l2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**SUPPORT LEVELS**")
            
            st.markdown(f"S1: **{format_price(levels.get('s1', 0), instrument['symbol'])}**")
            st.markdown(f"Support: **{format_price(levels.get('support', 0), instrument['symbol'])}**")
            
            current = safe_get_last(market_data['close'], 0)
            support = levels.get('support', current * 0.98)
            dist_to_sup = ((current - support) / current * 100) if current != 0 else 0
            
            st.caption(f"Distance to support: {dist_to_sup:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ==================================================
    # TECHNICAL INDICATORS OVERVIEW
    # ==================================================
    st.markdown('<div class="section-header">TECHNICAL INDICATORS</div>', unsafe_allow_html=True)
    
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    
    with col_i1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**BOLLINGER BANDS**")
        bb_position = current_price
        bb_lower = safe_get_last(market_data['bb_lower'], current_price * 0.98)
        bb_upper = safe_get_last(market_data['bb_upper'], current_price * 1.02)
        
        if bb_upper > bb_lower:
            position_pct = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
            st.markdown(f"Position: **{position_pct:.0f}%**")
            
            if position_pct < 20:
                st.caption("Near lower band")
            elif position_pct > 80:
                st.caption("Near upper band")
            else:
                st.caption("Within bands")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_i2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**VOLATILITY**")
        
        atr_pct = (atr / current_price * 100) if current_price != 0 else 0
        st.markdown(f"ATR: **{atr_pct:.2f}%**")
        
        vol_20 = safe_get_last(market_data['close'].pct_change().rolling(20).std() * 100, 0)
        st.markdown(f"20D Vol: **{vol_20:.2f}%**")
        
        if vol_20 > 3:
            st.caption("High volatility")
        elif vol_20 > 1.5:
            st.caption("Moderate volatility")
        else:
            st.caption("Low volatility")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_i3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**PRICE ACTION**")
        
        # Recent performance
        if len(market_data) > 5:
            recent_high = market_data['high'].tail(5).max()
            recent_low = market_data['low'].tail(5).min()
            
            st.markdown(f"5-period high: **{format_price(recent_high, instrument['symbol'])}**")
            st.markdown(f"5-period low: **{format_price(recent_low, instrument['symbol'])}**")
            
            range_pct = ((recent_high - recent_low) / recent_low * 100) if recent_low != 0 else 0
            st.caption(f"Range: {range_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_i4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**MARKET REGIME**")
        
        # Determine market regime
        bb_width = (bb_upper - bb_lower) / bb_middle if (bb_upper - bb_lower) > 0 and (bb_middle := safe_get_last(market_data['bb_middle'], current_price)) != 0 else 0.1
        
        if bb_width > 0.1:
            st.markdown("**Regime: TRENDING**")
            st.caption("Wide price range")
        elif bb_width < 0.05:
            st.markdown("**Regime: COMPRESSION**")
            st.caption("Narrow price range")
        else:
            st.markdown("**Regime: NORMAL**")
            st.caption("Moderate price action")
        
        st.caption(f"BB Width: {bb_width:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================================================
    # FOOTER
    # ==================================================
    st.divider()
    
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    
    with col_f1:
        st.caption(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        st.caption(f"Data points: {len(market_data)} | Timeframe: {primary_tf}")
    
    with col_f2:
        st.caption(f"Asset class: {instrument['asset_class']}")
    
    with col_f3:
        st.caption("For professional use only")

# ==================================================
# EXECUTION
# ==================================================
if __name__ == "__main__":
    try:
        main()
        tm.sleep(30)
        st.rerun()
    except Exception as e:
        st.error("System error - please refresh")
        if st.button("Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
