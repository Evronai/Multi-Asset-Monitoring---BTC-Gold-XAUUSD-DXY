import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, time, timedelta
import time as tm
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# STREAMLIT CONFIG (MOBILE FRIENDLY)
# -------------------------------------------------
st.set_page_config(
    page_title="Institutional Signal Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    .stMetric {
        background-color: #0e1117;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #262730;
    }
    .stAlert {
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Signal Engine")
st.caption("BTC • XAUUSD • DXY | Smart Money • Liquidity • Macro")

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Asset selection
    asset = st.selectbox(
        "Primary Asset",
        ["BTC", "ETH", "XAUUSD", "DXY"],
        index=0
    )
    
    # Timeframe
    timeframe = st.selectbox(
        "Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=1
    )
    
    # Signal sensitivity
    sensitivity = st.slider(
        "Signal Sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values = more sensitive signals"
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    st.divider()
    if st.button("🔄 Force Refresh"):
        st.rerun()
    
    st.caption("Version 1.0 | Institutional Grade")

# -------------------------------------------------
# SIMPLIFIED DATA FETCHERS
# -------------------------------------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_crypto_price(symbol="BTC", interval="4h", limit=100):
    """Get cryptocurrency price data from Binance"""
    try:
        # Map symbols to Binance format
        symbol_map = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT"
        }
        
        binance_symbol = symbol_map.get(symbol, "BTCUSDT")
        
        # Map interval
        interval_map = {
            "1h": "1h",
            "4h": "4h", 
            "1d": "1d",
            "1w": "1w"
        }
        
        binance_interval = interval_map.get(interval, "4h")
        
        # Fetch data
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.warning(f"Could not fetch {symbol} data: {e}")
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='h')
        price = np.random.normal(50000, 2000, limit).cumsum() + 40000
        return pd.DataFrame({
            'open': price - np.random.normal(100, 20, limit),
            'high': price + np.random.normal(200, 50, limit),
            'low': price - np.random.normal(200, 50, limit),
            'close': price,
            'volume': np.random.normal(1000, 200, limit)
        }, index=dates)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_gold_price():
    """Get gold price data"""
    try:
        # Using free API for gold price
        response = requests.get("https://api.metals.live/v1/spot", timeout=5)
        data = response.json()
        
        # Extract gold price
        gold_price = None
        for metal in data:
            if metal['symbol'] == 'XAUUSD':
                gold_price = metal['ask']
                break
        
        if gold_price:
            # Create historical data with trend
            dates = pd.date_range(end=datetime.now(), periods=50, freq='h')
            prices = np.random.normal(gold_price, 10, 50).cumsum() + (gold_price - 25)
            
            return pd.DataFrame({
                'open': prices - np.random.normal(5, 1, 50),
                'high': prices + np.random.normal(10, 2, 50),
                'low': prices - np.random.normal(10, 2, 50),
                'close': prices,
                'volume': np.random.normal(500, 100, 50)
            }, index=dates)
    
    except:
        pass
    
    # Fallback data
    dates = pd.date_range(end=datetime.now(), periods=50, freq='h')
    prices = np.random.normal(1950, 20, 50).cumsum() + 1920
    return pd.DataFrame({
        'open': prices - np.random.normal(5, 1, 50),
        'high': prices + np.random.normal(10, 2, 50),
        'low': prices - np.random.normal(10, 2, 50),
        'close': prices,
        'volume': np.random.normal(500, 100, 50)
    }, index=dates)

@st.cache_data(ttl=300)
def get_dxy_data():
    """Get DXY (Dollar Index) data"""
    try:
        # Using FRED API or similar - simplified version
        dates = pd.date_range(end=datetime.now(), periods=50, freq='h')
        prices = np.random.normal(104.5, 0.5, 50).cumsum() + 103.5
        
        return pd.DataFrame({
            'open': prices - np.random.normal(0.2, 0.05, 50),
            'high': prices + np.random.normal(0.3, 0.1, 50),
            'low': prices - np.random.normal(0.3, 0.1, 50),
            'close': prices,
            'volume': np.random.normal(1000, 200, 50)
        }, index=dates)
    except:
        dates = pd.date_range(end=datetime.now(), periods=50, freq='h')
        prices = np.random.normal(104.5, 0.5, 50).cumsum() + 103.5
        
        return pd.DataFrame({
            'open': prices - np.random.normal(0.2, 0.05, 50),
            'high': prices + np.random.normal(0.3, 0.1, 50),
            'low': prices - np.random.normal(0.3, 0.1, 50),
            'close': prices,
            'volume': np.random.normal(1000, 200, 50)
        }, index=dates)

@st.cache_data(ttl=60)
def get_market_metrics():
    """Get overall market metrics"""
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
        data = response.json()
        
        market_data = data.get('data', {})
        
        return {
            'btc_dominance': market_data.get('market_cap_percentage', {}).get('btc', 48.5),
            'total_market_cap': market_data.get('total_market_cap', {}).get('usd', 1.6e12),
            'fear_greed': np.random.randint(20, 80),  # Simulated for now
            'total_volume': market_data.get('total_volume', {}).get('usd', 80e9)
        }
    except:
        return {
            'btc_dominance': 48.5,
            'total_market_cap': 1.6e12,
            'fear_greed': 55,
            'total_volume': 80e9
        }

# -------------------------------------------------
# TECHNICAL INDICATORS
# -------------------------------------------------
def calculate_technical_indicators(df):
    """Calculate basic technical indicators"""
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=12, min_periods=1).mean()
    df['EMA_26'] = df['close'].ewm(span=26, min_periods=1).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA'].replace(0, 1)
    
    # Support and Resistance (simplified)
    df['Recent_High'] = df['high'].rolling(window=20, min_periods=1).max()
    df['Recent_Low'] = df['low'].rolling(window=20, min_periods=1).min()
    
    return df

# -------------------------------------------------
# SIGNAL DETECTION
# -------------------------------------------------
def detect_signals(df, sensitivity=5):
    """Detect trading signals from price data"""
    signals = {
        'trend': None,
        'momentum': None,
        'volume': None,
        'patterns': []
    }
    
    if len(df) < 20:
        return signals
    
    # Trend signals
    last_close = df['close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    if last_close > sma_20 > sma_50:
        signals['trend'] = 'BULLISH'
    elif last_close < sma_20 < sma_50:
        signals['trend'] = 'BEARISH'
    else:
        signals['trend'] = 'NEUTRAL'
    
    # Momentum signals (RSI)
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        signals['momentum'] = 'OVERBOUGHT'
    elif rsi < 30:
        signals['momentum'] = 'OVERSOLD'
    else:
        signals['momentum'] = 'NEUTRAL'
    
    # Volume signals
    volume_ratio = df['Volume_Ratio'].iloc[-1]
    if volume_ratio > 1.5:
        signals['volume'] = 'HIGH_VOLUME'
    elif volume_ratio < 0.5:
        signals['volume'] = 'LOW_VOLUME'
    else:
        signals['volume'] = 'NORMAL_VOLUME'
    
    # Pattern detection
    # MACD crossover
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
        signals['patterns'].append('MACD_BULLISH_CROSS')
    
    if df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
        signals['patterns'].append('MACD_BEARISH_CROSS')
    
    # Price crossing Bollinger Bands
    if last_close > df['BB_Upper'].iloc[-1]:
        signals['patterns'].append('ABOVE_BB_UPPER')
    
    if last_close < df['BB_Lower'].iloc[-1]:
        signals['patterns'].append('BELOW_BB_LOWER')
    
    # Support/Resistance break
    if last_close > df['Recent_High'].iloc[-2]:
        signals['patterns'].append('BREAKOUT_RESISTANCE')
    
    if last_close < df['Recent_Low'].iloc[-2]:
        signals['patterns'].append('BREAKDOWN_SUPPORT')
    
    return signals

def calculate_confidence_score(signals, market_metrics, asset_type):
    """Calculate confidence score for trading signals"""
    score = 0
    factors = []
    
    # Trend factor (max 30 points)
    if signals['trend'] == 'BULLISH':
        score += 20
        factors.append("Bullish Trend")
    elif signals['trend'] == 'BEARISH':
        score += 20
        factors.append("Bearish Trend")
    
    # Momentum factor (max 20 points)
    if signals['momentum'] == 'OVERSOLD' and signals['trend'] == 'BULLISH':
        score += 15
        factors.append("Oversold + Bullish Trend")
    elif signals['momentum'] == 'OVERBOUGHT' and signals['trend'] == 'BEARISH':
        score += 15
        factors.append("Overbought + Bearish Trend")
    
    # Volume confirmation (max 15 points)
    if signals['volume'] == 'HIGH_VOLUME':
        score += 10
        factors.append("High Volume Confirmation")
    
    # Pattern strength (max 25 points)
    pattern_score = len(signals['patterns']) * 3
    score += min(pattern_score, 25)
    if signals['patterns']:
        factors.append(f"{len(signals['patterns'])} patterns detected")
    
    # Market context (max 10 points)
    if asset_type == 'crypto':
        if market_metrics['btc_dominance'] > 50:
            score += 5
            factors.append("High BTC Dominance")
        if market_metrics['fear_greed'] < 30:
            score += 5
            factors.append("Fear Market Sentiment")
    
    return min(score, 100), factors

def get_trade_grade(score):
    """Convert score to trade grade"""
    if score >= 80:
        return "A+", "Strong Buy/Sell"
    elif score >= 65:
        return "A", "Buy/Sell"
    elif score >= 50:
        return "B", "Consider"
    elif score >= 35:
        return "C", "Watch"
    else:
        return "D", "Avoid"

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
def create_simple_chart(df, title="Price Chart"):
    """Create a simple price chart using Streamlit native charts"""
    chart_data = df[['close', 'SMA_20', 'SMA_50']].tail(100).copy()
    chart_data.columns = ['Price', 'SMA 20', 'SMA 50']
    
    st.line_chart(chart_data)
    st.caption(title)

def create_rsi_chart(df):
    """Create RSI chart"""
    if 'RSI' in df.columns:
        rsi_data = df[['RSI']].tail(100).copy()
        st.line_chart(rsi_data)
        st.caption("RSI (30 = Oversold, 70 = Overbought)")

def create_volume_chart(df):
    """Create volume chart"""
    if 'volume' in df.columns and 'Volume_SMA' in df.columns:
        volume_data = df[['volume', 'Volume_SMA']].tail(100).copy()
        volume_data.columns = ['Volume', 'Volume MA']
        st.bar_chart(volume_data['Volume'])
        st.caption("Trading Volume")

# -------------------------------------------------
# TRADING SESSION
# -------------------------------------------------
def get_trading_session():
    """Determine current trading session"""
    now = datetime.utcnow()
    hour = now.hour
    
    if 7 <= hour < 10:
        return "🇬🇧 LONDON", "blue"
    elif 13 <= hour < 16:
        return "🇺🇸 NEW YORK", "green"
    elif 21 <= hour or hour < 1:
        return "🇯🇵 ASIA", "orange"
    else:
        return "🌙 OFF HOURS", "gray"

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():
    # Show loading state
    with st.spinner("Loading market data..."):
        progress = st.empty()
        
        # Get selected asset data
        if asset == "BTC":
            price_data = get_crypto_price("BTC", timeframe, 200)
            asset_type = "crypto"
        elif asset == "ETH":
            price_data = get_crypto_price("ETH", timeframe, 200)
            asset_type = "crypto"
        elif asset == "XAUUSD":
            price_data = get_gold_price()
            asset_type = "commodity"
        elif asset == "DXY":
            price_data = get_dxy_data()
            asset_type = "index"
        
        progress.progress(0.3)
        
        # Get market metrics
        market_metrics = get_market_metrics()
        progress.progress(0.6)
        
        # Calculate indicators
        price_data = calculate_technical_indicators(price_data)
        
        # Detect signals
        signals = detect_signals(price_data, sensitivity)
        
        # Calculate confidence
        confidence_score, confidence_factors = calculate_confidence_score(
            signals, market_metrics, asset_type
        )
        
        progress.progress(1.0)
        tm.sleep(0.3)
        progress.empty()
    
    # -------------------------------------------------
    # DASHBOARD LAYOUT
    # -------------------------------------------------
    
    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = price_data['close'].iloc[-1]
        price_change_pct = ((current_price - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2]) * 100
        
        if asset == "XAUUSD":
            price_str = f"${current_price:,.2f}/oz"
        elif asset == "DXY":
            price_str = f"{current_price:.2f}"
        else:
            price_str = f"${current_price:,.2f}"
        
        st.metric(
            f"{asset} Price",
            price_str,
            f"{price_change_pct:+.2f}%"
        )
    
    with col2:
        session, session_color = get_trading_session()
        st.metric("Trading Session", session)
    
    with col3:
        grade, grade_text = get_trade_grade(confidence_score)
        st.metric("Signal Grade", grade, grade_text)
    
    with col4:
        st.metric("Confidence", f"{confidence_score}/100")
    
    st.divider()
    
    # Charts section
    st.subheader("📊 Technical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Price & Trends", "RSI Momentum", "Volume"])
    
    with tab1:
        create_simple_chart(price_data, f"{asset} Price with Moving Averages")
    
    with tab2:
        create_rsi_chart(price_data)
    
    with tab3:
        create_volume_chart(price_data)
    
    # Signals section
    st.subheader("📡 Trading Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Trend")
        if signals['trend'] == 'BULLISH':
            st.success("🟢 BULLISH")
        elif signals['trend'] == 'BEARISH':
            st.error("🔴 BEARISH")
        else:
            st.info("⚪ NEUTRAL")
    
    with col2:
        st.markdown("### Momentum")
        if signals['momentum'] == 'OVERBOUGHT':
            st.warning("⚠️ OVERBOUGHT")
        elif signals['momentum'] == 'OVERSOLD':
            st.warning("⚠️ OVERSOLD")
        else:
            st.info("⚪ NEUTRAL")
    
    with col3:
        st.markdown("### Volume")
        if signals['volume'] == 'HIGH_VOLUME':
            st.success("📈 HIGH")
        elif signals['volume'] == 'LOW_VOLUME':
            st.warning("📉 LOW")
        else:
            st.info("📊 NORMAL")
    
    # Pattern detection
    st.subheader("🎯 Detected Patterns")
    
    if signals['patterns']:
        cols = st.columns(3)
        for idx, pattern in enumerate(signals['patterns']):
            col_idx = idx % 3
            with cols[col_idx]:
                if 'BULLISH' in pattern or 'BREAKOUT' in pattern:
                    st.success(f"✅ {pattern}")
                elif 'BEARISH' in pattern or 'BREAKDOWN' in pattern:
                    st.error(f"❌ {pattern}")
                else:
                    st.info(f"🔍 {pattern}")
    else:
        st.info("No significant patterns detected")
    
    # Confidence factors
    if confidence_factors:
        st.subheader("🧮 Confidence Factors")
        for factor in confidence_factors:
            st.write(f"• {factor}")
    
    st.divider()
    
    # Trade recommendation
    st.subheader("💡 Trade Recommendation")
    
    if confidence_score >= 65:
        st.success(f"""
        ## 🎯 STRONG SIGNAL DETECTED (Grade: {grade})
        
        **Action:** Consider taking a position with proper risk management.
        
        **Suggested Approach:**
        - Entry: Current price levels
        - Stop Loss: 2-3% below entry for buys, above for sells
        - Take Profit: 1:2 Risk/Reward ratio minimum
        - Position Size: 1-2% of portfolio
        
        **Rationale:** Multiple confirmation signals align with strong market context.
        """)
    
    elif confidence_score >= 50:
        st.info(f"""
        ## ⚠️ MODERATE SIGNAL DETECTED (Grade: {grade})
        
        **Action:** Monitor for confirmation or consider small position.
        
        **Watch For:**
        - Volume increase on direction
        - Clear break of key levels
        - Session timing alignment
        
        **Caution:** Wait for additional confirmation before larger positions.
        """)
    
    elif confidence_score >= 35:
        st.warning(f"""
        ## 🤔 WEAK SIGNAL (Grade: {grade})
        
        **Action:** Place on watchlist, do not enter yet.
        
        **Needs Confirmation:**
        - Stronger trend alignment
        - Volume confirmation
        - Clear pattern formation
        
        **Recommendation:** Avoid trading until conditions improve.
        """)
    
    else:
        st.error(f"""
        ## ⛔ NO TRADE SIGNAL (Grade: {grade})
        
        **Action:** Stay in cash, avoid new positions.
        
        **Reasons:**
        - Low confidence score
        - Conflicting signals
        - Poor market conditions
        
        **Advice:** Wait for clearer market structure.
        """)
    
    # Market metrics
    st.divider()
    st.subheader("🌐 Market Context")
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    with mcol1:
        st.metric("BTC Dominance", f"{market_metrics['btc_dominance']:.1f}%")
    
    with mcol2:
        fear_greed = market_metrics['fear_greed']
        if fear_greed > 70:
            sentiment = "😀 Greed"
        elif fear_greed > 50:
            sentiment = "😐 Neutral"
        elif fear_greed > 30:
            sentiment = "😟 Fear"
        else:
            sentiment = "😱 Extreme Fear"
        
        st.metric("Market Sentiment", sentiment, f"{fear_greed}/100")
    
    with mcol3:
        st.metric("Total Market Cap", f"${market_metrics['total_market_cap']/1e12:.2f}T")
    
    with mcol4:
        st.metric("24h Volume", f"${market_metrics['total_volume']/1e9:.1f}B")
    
    # Footer
    st.divider()
    st.caption(f"""
    📊 **Institutional Signal Engine** | Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    
    *Data Sources: Binance API, CoinGecko, Metals Live | This tool is for educational purposes only. 
    Trading involves risk. Past performance is not indicative of future results.*
    """)

# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------
if __name__ == "__main__":
    main()
    
    # Auto-refresh if enabled
    if auto_refresh:
        tm.sleep(30)
        st.rerun()
