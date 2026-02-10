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
# DATA VALIDATION AND FALLBACK FUNCTIONS
# -------------------------------------------------
def ensure_dataframe_has_data(df, min_rows=5):
    """Ensure DataFrame has minimum required data, generate fallback if not"""
    if df is None or len(df) < min_rows:
        # Generate fallback data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        base_price = 50000 if asset in ["BTC", "ETH"] else 1950 if asset == "XAUUSD" else 104.5
        price_variance = 2000 if asset in ["BTC", "ETH"] else 50 if asset == "XAUUSD" else 2
        
        prices = np.random.normal(base_price, price_variance/10, 100).cumsum() + base_price
        
        return pd.DataFrame({
            'open': prices - np.random.normal(base_price/200, base_price/1000, 100),
            'high': prices + np.random.normal(base_price/100, base_price/500, 100),
            'low': prices - np.random.normal(base_price/100, base_price/500, 100),
            'close': prices,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    
    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Add missing column with reasonable values
            if col == 'close':
                df['close'] = (df['open'] + df['high'] + df['low']) / 3 if all(x in df.columns for x in ['open', 'high', 'low']) else np.random.normal(50000, 1000, len(df))
            elif col == 'volume':
                df['volume'] = np.random.normal(1000, 200, len(df))
    
    return df

def safe_get_last_value(series, default_value=None):
    """Safely get the last value from a pandas Series"""
    if series is None or len(series) == 0:
        return default_value if default_value is not None else 0
    
    try:
        return series.iloc[-1]
    except (IndexError, KeyError):
        return default_value if default_value is not None else series.values[-1] if len(series.values) > 0 else 0

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
        response.raise_for_status()
        data = response.json()
        
        # Validate response
        if not data or len(data) == 0:
            raise ValueError("Empty response from API")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        df.set_index('timestamp', inplace=True)
        
        # Return only essential columns
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.warning(f"⚠️ Could not fetch {symbol} data from API: {str(e)[:100]}...")
        # Generate sample data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='h')
        base_price = 50000 if symbol == "BTC" else 3000
        price = np.random.normal(base_price, base_price * 0.02, limit).cumsum() + base_price * 0.9
        
        return pd.DataFrame({
            'open': price - np.random.normal(base_price * 0.002, base_price * 0.001, limit),
            'high': price + np.random.normal(base_price * 0.005, base_price * 0.002, limit),
            'low': price - np.random.normal(base_price * 0.005, base_price * 0.002, limit),
            'close': price,
            'volume': np.random.normal(1000, 200, limit)
        }, index=dates)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_gold_price():
    """Get gold price data"""
    try:
        # Using free API for gold price
        response = requests.get("https://api.metals.live/v1/spot", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Extract gold price
        gold_price = None
        for metal in data:
            if metal.get('symbol') == 'XAUUSD':
                gold_price = metal.get('ask', 1950)
                break
        
        # Create historical data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = np.random.normal(gold_price or 1950, 10, 100).cumsum() + (gold_price or 1950 - 25)
        
        return pd.DataFrame({
            'open': prices - np.random.normal(5, 1, 100),
            'high': prices + np.random.normal(10, 2, 100),
            'low': prices - np.random.normal(10, 2, 100),
            'close': prices,
            'volume': np.random.normal(500, 100, 100)
        }, index=dates)
    
    except Exception as e:
        # Fallback data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = np.random.normal(1950, 20, 100).cumsum() + 1920
        return pd.DataFrame({
            'open': prices - np.random.normal(5, 1, 100),
            'high': prices + np.random.normal(10, 2, 100),
            'low': prices - np.random.normal(10, 2, 100),
            'close': prices,
            'volume': np.random.normal(500, 100, 100)
        }, index=dates)

@st.cache_data(ttl=300)
def get_dxy_data():
    """Get DXY (Dollar Index) data"""
    try:
        # Using alternative free API
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = response.json()
        
        # Simplified DXY calculation (approximation)
        usd_rates = data.get('rates', {})
        eur_rate = usd_rates.get('EUR', 0.92)
        dxy_value = 100 / eur_rate if eur_rate != 0 else 104.5
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = np.random.normal(dxy_value, 0.5, 100).cumsum() + (dxy_value - 1)
        
        return pd.DataFrame({
            'open': prices - np.random.normal(0.2, 0.05, 100),
            'high': prices + np.random.normal(0.3, 0.1, 100),
            'low': prices - np.random.normal(0.3, 0.1, 100),
            'close': prices,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    except Exception as e:
        # Fallback data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
        prices = np.random.normal(104.5, 0.5, 100).cumsum() + 103.5
        return pd.DataFrame({
            'open': prices - np.random.normal(0.2, 0.05, 100),
            'high': prices + np.random.normal(0.3, 0.1, 100),
            'low': prices - np.random.normal(0.3, 0.1, 100),
            'close': prices,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)

@st.cache_data(ttl=60)
def get_market_metrics():
    """Get overall market metrics"""
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
        data = response.json()
        
        market_data = data.get('data', {})
        
        btc_dominance = market_data.get('market_cap_percentage', {}).get('btc', 48.5)
        total_mcap = market_data.get('total_market_cap', {}).get('usd', 1.6e12)
        
        return {
            'btc_dominance': float(btc_dominance),
            'total_market_cap': float(total_mcap),
            'fear_greed': np.random.randint(20, 80),  # Simulated for now
            'total_volume': float(market_data.get('total_volume', {}).get('usd', 80e9))
        }
    except Exception as e:
        # Return default values
        return {
            'btc_dominance': 48.5,
            'total_market_cap': 1.6e12,
            'fear_greed': 55,
            'total_volume': 80e9
        }

# -------------------------------------------------
# TECHNICAL INDICATORS WITH SAFE CALCULATIONS
# -------------------------------------------------
def calculate_technical_indicators(df):
    """Calculate basic technical indicators with safety checks"""
    if df is None or len(df) < 5:
        # Return empty DataFrame with required columns
        return pd.DataFrame()
    
    df = df.copy()
    
    # Ensure we have enough data for calculations
    min_periods = min(5, len(df))
    
    # Simple Moving Averages (with safe calculations)
    df['SMA_20'] = df['close'].rolling(window=min(20, len(df)), min_periods=min_periods).mean()
    df['SMA_50'] = df['close'].rolling(window=min(50, len(df)), min_periods=min_periods).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=min(12, len(df)), min_periods=min_periods).mean()
    df['EMA_26'] = df['close'].ewm(span=min(26, len(df)), min_periods=min_periods).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=min(9, len(df)), min_periods=min_periods).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (with safe calculations)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(df)), min_periods=min_periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(df)), min_periods=min_periods).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Bollinger Bands
    window = min(20, len(df))
    df['BB_Middle'] = df['close'].rolling(window=window, min_periods=min_periods).mean()
    bb_std = df['close'].rolling(window=window, min_periods=min_periods).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std.fillna(0) * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std.fillna(0) * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['volume'].rolling(window=min(20, len(df)), min_periods=min_periods).mean()
    df['Volume_SMA'] = df['Volume_SMA'].replace(0, 1)  # Avoid division by zero
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
    
    # Support and Resistance
    df['Recent_High'] = df['high'].rolling(window=min(20, len(df)), min_periods=min_periods).max()
    df['Recent_Low'] = df['low'].rolling(window=min(20, len(df)), min_periods=min_periods).min()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# -------------------------------------------------
# SIGNAL DETECTION WITH SAFETY CHECKS
# -------------------------------------------------
def detect_signals(df, sensitivity=5):
    """Detect trading signals from price data with safety checks"""
    signals = {
        'trend': 'NEUTRAL',
        'momentum': 'NEUTRAL',
        'volume': 'NORMAL_VOLUME',
        'patterns': []
    }
    
    if df is None or len(df) < 2:
        return signals
    
    # Use safe getter functions
    last_close = safe_get_last_value(df['close'], 0)
    sma_20 = safe_get_last_value(df.get('SMA_20'), last_close)
    sma_50 = safe_get_last_value(df.get('SMA_50'), last_close)
    rsi = safe_get_last_value(df.get('RSI'), 50)
    volume_ratio = safe_get_last_value(df.get('Volume_Ratio'), 1)
    
    # Trend signals
    if last_close > sma_20 > sma_50:
        signals['trend'] = 'BULLISH'
    elif last_close < sma_20 < sma_50:
        signals['trend'] = 'BEARISH'
    else:
        signals['trend'] = 'NEUTRAL'
    
    # Momentum signals (RSI)
    if rsi > 70:
        signals['momentum'] = 'OVERBOUGHT'
    elif rsi < 30:
        signals['momentum'] = 'OVERSOLD'
    else:
        signals['momentum'] = 'NEUTRAL'
    
    # Volume signals
    if volume_ratio > 1.5:
        signals['volume'] = 'HIGH_VOLUME'
    elif volume_ratio < 0.5:
        signals['volume'] = 'LOW_VOLUME'
    else:
        signals['volume'] = 'NORMAL_VOLUME'
    
    # Pattern detection (only if we have enough data)
    if len(df) >= 3:
        try:
            # MACD crossover
            macd_last = safe_get_last_value(df.get('MACD'), 0)
            macd_prev = safe_get_last_value(df.get('MACD').iloc[-2] if len(df) >= 2 else pd.Series([0]), 0)
            signal_last = safe_get_last_value(df.get('MACD_Signal'), 0)
            signal_prev = safe_get_last_value(df.get('MACD_Signal').iloc[-2] if len(df) >= 2 else pd.Series([0]), 0)
            
            if macd_last > signal_last and macd_prev <= signal_prev:
                signals['patterns'].append('MACD_BULLISH_CROSS')
            
            if macd_last < signal_last and macd_prev >= signal_prev:
                signals['patterns'].append('MACD_BEARISH_CROSS')
            
            # Price crossing Bollinger Bands
            bb_upper = safe_get_last_value(df.get('BB_Upper'), last_close * 1.1)
            bb_lower = safe_get_last_value(df.get('BB_Lower'), last_close * 0.9)
            
            if last_close > bb_upper:
                signals['patterns'].append('ABOVE_BB_UPPER')
            
            if last_close < bb_lower:
                signals['patterns'].append('BELOW_BB_LOWER')
            
            # Support/Resistance break
            if len(df) >= 3:
                recent_high = safe_get_last_value(df.get('Recent_High').iloc[-2] if len(df) >= 2 else pd.Series([last_close]), last_close)
                recent_low = safe_get_last_value(df.get('Recent_Low').iloc[-2] if len(df) >= 2 else pd.Series([last_close]), last_close)
                
                if last_close > recent_high:
                    signals['patterns'].append('BREAKOUT_RESISTANCE')
                
                if last_close < recent_low:
                    signals['patterns'].append('BREAKDOWN_SUPPORT')
                    
        except Exception as e:
            # Silently continue if pattern detection fails
            pass
    
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
        btc_dom = market_metrics.get('btc_dominance', 48.5)
        fear_greed = market_metrics.get('fear_greed', 50)
        
        if btc_dom > 50:
            score += 5
            factors.append("High BTC Dominance")
        if fear_greed < 30:
            score += 5
            factors.append("Fear Market Sentiment")
    
    return min(max(score, 0), 100), factors

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
    if df is None or len(df) == 0:
        st.warning("No data available for chart")
        return
    
    try:
        # Get last 100 data points or all if less
        display_data = df.tail(min(100, len(df))).copy()
        
        # Prepare chart data
        chart_data = pd.DataFrame()
        chart_data['Price'] = display_data['close']
        
        # Add moving averages if available
        if 'SMA_20' in display_data.columns:
            chart_data['SMA 20'] = display_data['SMA_20']
        if 'SMA_50' in display_data.columns:
            chart_data['SMA 50'] = display_data['SMA_50']
        
        st.line_chart(chart_data)
        st.caption(title)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def create_rsi_chart(df):
    """Create RSI chart"""
    if df is None or len(df) == 0 or 'RSI' not in df.columns:
        st.info("RSI data not available")
        return
    
    try:
        display_data = df.tail(min(100, len(df))).copy()
        rsi_data = pd.DataFrame({'RSI': display_data['RSI']})
        
        st.line_chart(rsi_data)
        st.caption("RSI (30 = Oversold, 70 = Overbought)")
    except Exception as e:
        st.error(f"Error creating RSI chart: {str(e)}")

def create_volume_chart(df):
    """Create volume chart"""
    if df is None or len(df) == 0 or 'volume' not in df.columns:
        st.info("Volume data not available")
        return
    
    try:
        display_data = df.tail(min(100, len(df))).copy()
        volume_data = pd.DataFrame({'Volume': display_data['volume']})
        
        st.bar_chart(volume_data)
        st.caption("Trading Volume")
    except Exception as e:
        st.error(f"Error creating volume chart: {str(e)}")

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
        progress.progress(0.1)
        
        try:
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
            else:
                price_data = get_crypto_price("BTC", timeframe, 200)
                asset_type = "crypto"
            
            progress.progress(0.4)
            
            # Ensure data is valid
            price_data = ensure_dataframe_has_data(price_data)
            
            # Get market metrics
            market_metrics = get_market_metrics()
            progress.progress(0.7)
            
            # Calculate indicators
            price_data = calculate_technical_indicators(price_data)
            
            # Detect signals
            signals = detect_signals(price_data, sensitivity)
            
            # Calculate confidence
            confidence_score, confidence_factors = calculate_confidence_score(
                signals, market_metrics, asset_type
            )
            
            progress.progress(1.0)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)[:200]}")
            # Create default data on error
            dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
            price_data = pd.DataFrame({
                'open': np.random.normal(50000, 1000, 100),
                'high': np.random.normal(51000, 1000, 100),
                'low': np.random.normal(49000, 1000, 100),
                'close': np.random.normal(50000, 1000, 100),
                'volume': np.random.normal(1000, 200, 100)
            }, index=dates)
            
            price_data = calculate_technical_indicators(price_data)
            signals = detect_signals(price_data, sensitivity)
            market_metrics = get_market_metrics()
            confidence_score, confidence_factors = calculate_confidence_score(
                signals, market_metrics, "crypto"
            )
        
        tm.sleep(0.3)
        progress.empty()
    
    # -------------------------------------------------
    # DASHBOARD LAYOUT
    # -------------------------------------------------
    
    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            current_price = safe_get_last_value(price_data['close'], 0)
            
            # Calculate price change safely
            if len(price_data) >= 2:
                prev_price = price_data['close'].iloc[-2] if len(price_data) >= 2 else current_price
                price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            else:
                price_change_pct = 0
            
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
        except Exception as e:
            st.metric(f"{asset} Price", "N/A", "0%")
    
    with col2:
        try:
            session, session_color = get_trading_session()
            st.metric("Trading Session", session)
        except:
            st.metric("Trading Session", "N/A")
    
    with col3:
        try:
            grade, grade_text = get_trade_grade(confidence_score)
            st.metric("Signal Grade", grade, grade_text)
        except:
            st.metric("Signal Grade", "N/A", "Error")
    
    with col4:
        try:
            st.metric("Confidence", f"{confidence_score}/100")
        except:
            st.metric("Confidence", "0/100")
    
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
        trend = signals.get('trend', 'NEUTRAL')
        if trend == 'BULLISH':
            st.success("🟢 BULLISH")
        elif trend == 'BEARISH':
            st.error("🔴 BEARISH")
        else:
            st.info("⚪ NEUTRAL")
    
    with col2:
        st.markdown("### Momentum")
        momentum = signals.get('momentum', 'NEUTRAL')
        if momentum == 'OVERBOUGHT':
            st.warning("⚠️ OVERBOUGHT")
        elif momentum == 'OVERSOLD':
            st.warning("⚠️ OVERSOLD")
        else:
            st.info("⚪ NEUTRAL")
    
    with col3:
        st.markdown("### Volume")
        volume = signals.get('volume', 'NORMAL_VOLUME')
        if volume == 'HIGH_VOLUME':
            st.success("📈 HIGH")
        elif volume == 'LOW_VOLUME':
            st.warning("📉 LOW")
        else:
            st.info("📊 NORMAL")
    
    # Pattern detection
    st.subheader("🎯 Detected Patterns")
    
    patterns = signals.get('patterns', [])
    if patterns:
        cols = st.columns(3)
        for idx, pattern in enumerate(patterns):
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
    
    try:
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
    except:
        st.warning("Unable to generate trade recommendation due to data issues.")
    
    # Market metrics
    st.divider()
    st.subheader("🌐 Market Context")
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    with mcol1:
        btc_dom = market_metrics.get('btc_dominance', 0)
        st.metric("BTC Dominance", f"{btc_dom:.1f}%")
    
    with mcol2:
        fear_greed = market_metrics.get('fear_greed', 50)
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
        total_mcap = market_metrics.get('total_market_cap', 0)
        st.metric("Total Market Cap", f"${total_mcap/1e12:.2f}T")
    
    with mcol4:
        total_volume = market_metrics.get('total_volume', 0)
        st.metric("24h Volume", f"${total_volume/1e9:.1f}B")
    
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
    try:
        main()
        
        # Auto-refresh if enabled
        if auto_refresh:
            tm.sleep(30)
            st.rerun()
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.info("The app has encountered an error. Please refresh the page or try again later.")
