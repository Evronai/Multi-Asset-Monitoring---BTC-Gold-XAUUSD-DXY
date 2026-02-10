import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time, timedelta
import time as tm
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# STREAMLIT CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Institutional Signal Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Institutional Signal Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BTC • XAUUSD • DXY | Smart Money • Liquidity • Macro Analysis</div>', unsafe_allow_html=True)

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset Selection
    selected_asset = st.selectbox(
        "Primary Asset",
        ["BTC/USD", "ETH/USD", "XAU/USD", "DXY"],
        index=0
    )
    
    # Timeframe Selection
    timeframe = st.selectbox(
        "Chart Timeframe",
        ["1 Hour", "4 Hours", "1 Day", "1 Week"],
        index=1
    )
    
    # Signal Settings
    st.subheader("Signal Parameters")
    sensitivity = st.slider("Sensitivity", 1, 10, 5)
    
    # Risk Management
    st.subheader("Risk Management")
    max_position_size = st.slider("Max Position Size (%)", 0.5, 10.0, 2.0, step=0.5)
    stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, step=0.5)
    
    # Refresh Control
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.caption(f"v1.0.0 • {datetime.now().strftime('%Y-%m-%d')}")

# ==================================================
# DATA FETCHING FUNCTIONS
# ==================================================
@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_market_data(asset, timeframe="4h", data_points=200):
    """Fetch market data with robust error handling"""
    
    # Map timeframe to API intervals
    interval_map = {
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d",
        "1 Week": "1w"
    }
    
    api_interval = interval_map.get(timeframe, "4h")
    
    try:
        if asset in ["BTC/USD", "ETH/USD"]:
            # Use Binance for crypto
            symbol = "BTCUSDT" if asset == "BTC/USD" else "ETHUSDT"
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": api_interval,
                "limit": data_points
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse data
            timestamps = [pd.to_datetime(x[0], unit='ms') for x in data]
            opens = [float(x[1]) for x in data]
            highs = [float(x[2]) for x in data]
            lows = [float(x[3]) for x in data]
            closes = [float(x[4]) for x in data]
            volumes = [float(x[5]) for x in data]
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            df.set_index('timestamp', inplace=True)
            return df
            
        elif asset == "XAU/USD":
            # Gold data - using simulated data for free API
            dates = pd.date_range(end=datetime.now(), periods=data_points, freq=api_interval[0])
            base_price = 1950
            volatility = 15
            
            prices = base_price + np.cumsum(np.random.randn(data_points) * volatility)
            
            df = pd.DataFrame({
                'open': prices - np.random.rand(data_points) * 10,
                'high': prices + np.random.rand(data_points) * 15,
                'low': prices - np.random.rand(data_points) * 15,
                'close': prices,
                'volume': np.random.rand(data_points) * 1000 + 500
            }, index=dates)
            
            return df
            
        elif asset == "DXY":
            # DXY data - simulated
            dates = pd.date_range(end=datetime.now(), periods=data_points, freq=api_interval[0])
            base_price = 104.5
            volatility = 0.3
            
            prices = base_price + np.cumsum(np.random.randn(data_points) * volatility)
            
            df = pd.DataFrame({
                'open': prices - np.random.rand(data_points) * 0.1,
                'high': prices + np.random.rand(data_points) * 0.15,
                'low': prices - np.random.rand(data_points) * 0.15,
                'close': prices,
                'volume': np.random.rand(data_points) * 500 + 250
            }, index=dates)
            
            return df
            
    except Exception as e:
        st.warning(f"⚠️ Data fetch error for {asset}: {str(e)}")
        # Return fallback data
        dates = pd.date_range(end=datetime.now(), periods=data_points, freq='h')
        
        if asset == "BTC/USD":
            base_price = 45000
        elif asset == "ETH/USD":
            base_price = 2500
        elif asset == "XAU/USD":
            base_price = 1950
        else:  # DXY
            base_price = 104.5
        
        prices = base_price + np.cumsum(np.random.randn(data_points) * (base_price * 0.01))
        
        df = pd.DataFrame({
            'open': prices - np.random.rand(data_points) * (base_price * 0.002),
            'high': prices + np.random.rand(data_points) * (base_price * 0.003),
            'low': prices - np.random.rand(data_points) * (base_price * 0.003),
            'close': prices,
            'volume': np.random.rand(data_points) * 1000 + 500
        }, index=dates)
        
        return df

@st.cache_data(ttl=120)
def fetch_market_indicators():
    """Fetch overall market indicators"""
    try:
        # Fetch BTC dominance and total market cap
        response = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        data = response.json()
        
        market_data = data.get('data', {})
        
        return {
            'btc_dominance': market_data.get('market_cap_percentage', {}).get('btc', 48.5),
            'total_mcap': market_data.get('total_market_cap', {}).get('usd', 1.6e12),
            'total_volume': market_data.get('total_volume', {}).get('usd', 80e9),
            'altcoin_mcap': market_data.get('total_market_cap', {}).get('usd', 1.6e12) * 
                           (1 - market_data.get('market_cap_percentage', {}).get('btc', 0.485))
        }
    except:
        return {
            'btc_dominance': 48.5,
            'total_mcap': 1.6e12,
            'total_volume': 80e9,
            'altcoin_mcap': 0.8e12
        }

# ==================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ==================================================
def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Ensure we have enough data
    if len(df) < 20:
        # Pad with simple calculations if insufficient data
        df['SMA_20'] = df['close']
        df['SMA_50'] = df['close']
        df['RSI'] = 50
        df['Volume_MA'] = df['volume']
        return df
    
    # Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
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
    df['BB_Std'] = df['close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Volume indicators
    df['Volume_MA'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, 1)
    
    # Support and Resistance
    df['Resistance_Level'] = df['high'].rolling(window=20, min_periods=1).max()
    df['Support_Level'] = df['low'].rolling(window=20, min_periods=1).min()
    
    # ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    
    return df

def analyze_trend(df):
    """Analyze market trend"""
    if df is None or len(df) < 2:
        return {
            'trend': 'NEUTRAL',
            'strength': 0,
            'direction': 'SIDEWAYS'
        }
    
    current_price = df['close'].iloc[-1] if len(df) > 0 else 0
    sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns and len(df) > 0 else current_price
    sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns and len(df) > 0 else current_price
    
    # Trend determination
    if current_price > sma_20 > sma_50:
        trend = 'BULLISH'
        strength = min(100, ((current_price - sma_50) / sma_50 * 100) * 2)
        direction = 'UPTREND'
    elif current_price < sma_20 < sma_50:
        trend = 'BEARISH'
        strength = min(100, ((sma_50 - current_price) / current_price * 100) * 2)
        direction = 'DOWNTREND'
    else:
        trend = 'NEUTRAL'
        strength = 0
        direction = 'SIDEWAYS'
    
    return {
        'trend': trend,
        'strength': strength,
        'direction': direction
    }

def detect_signals(df, sensitivity=5):
    """Detect trading signals"""
    if df is None or len(df) < 10:
        return {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': [],
            'risk_level': 'HIGH'
        }
    
    signals = {
        'buy_signals': [],
        'sell_signals': [],
        'neutral_signals': [],
        'risk_level': 'MEDIUM'
    }
    
    current_price = df['close'].iloc[-1]
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    
    # RSI signals
    if rsi < 30:
        signals['buy_signals'].append('RSI Oversold')
    elif rsi > 70:
        signals['sell_signals'].append('RSI Overbought')
    
    # MACD signals
    if len(df) >= 2 and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd_current = df['MACD'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2]
        signal_current = df['MACD_Signal'].iloc[-1]
        signal_prev = df['MACD_Signal'].iloc[-2]
        
        if macd_current > signal_current and macd_prev <= signal_prev:
            signals['buy_signals'].append('MACD Bullish Cross')
        elif macd_current < signal_current and macd_prev >= signal_prev:
            signals['sell_signals'].append('MACD Bearish Cross')
    
    # Bollinger Band signals
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        
        if current_price > bb_upper:
            signals['sell_signals'].append('Above BB Upper')
        elif current_price < bb_lower:
            signals['buy_signals'].append('Below BB Lower')
    
    # Volume confirmation
    if 'Volume_Ratio' in df.columns:
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if volume_ratio > 1.5:
            signals['buy_signals'].append('High Volume')
    
    # Determine risk level
    if len(signals['buy_signals']) > len(signals['sell_signals']) * 2:
        signals['risk_level'] = 'LOW'
    elif len(signals['sell_signals']) > len(signals['buy_signals']) * 2:
        signals['risk_level'] = 'HIGH'
    
    return signals

def calculate_confidence_score(signals, trend_analysis, market_data):
    """Calculate confidence score for trading decisions"""
    score = 50  # Start at neutral
    
    # Signal strength (max 30 points)
    buy_signal_count = len(signals.get('buy_signals', []))
    sell_signal_count = len(signals.get('sell_signals', []))
    
    if buy_signal_count > sell_signal_count:
        score += min(buy_signal_count * 5, 30)
    elif sell_signal_count > buy_signal_count:
        score -= min(sell_signal_count * 5, 30)
    
    # Trend alignment (max 20 points)
    trend = trend_analysis.get('trend', 'NEUTRAL')
    if trend == 'BULLISH' and buy_signal_count > 0:
        score += 15
    elif trend == 'BEARISH' and sell_signal_count > 0:
        score -= 15
    
    # Market context (max 15 points)
    btc_dominance = market_data.get('btc_dominance', 50)
    if btc_dominance > 55:  # BTC dominance high
        if selected_asset == "BTC/USD":
            score += 10
    elif btc_dominance < 45:  # Altcoin season
        if selected_asset == "ETH/USD":
            score += 10
    
    # Volume confirmation (max 10 points)
    if 'High Volume' in signals.get('buy_signals', []) or 'High Volume' in signals.get('sell_signals', []):
        score += 8
    
    # Risk adjustment
    risk_level = signals.get('risk_level', 'MEDIUM')
    if risk_level == 'LOW':
        score += 5
    elif risk_level == 'HIGH':
        score -= 10
    
    return max(0, min(100, score))

def get_trade_recommendation(confidence_score, signals):
    """Generate trade recommendation based on confidence"""
    buy_count = len(signals.get('buy_signals', []))
    sell_count = len(signals.get('sell_signals', []))
    
    if confidence_score >= 75:
        if buy_count > sell_count:
            return "STRONG BUY", "🟢", "Multiple confirmations with high confidence"
        else:
            return "STRONG SELL", "🔴", "Clear bearish signals with high confidence"
    
    elif confidence_score >= 60:
        if buy_count > sell_count:
            return "BUY", "🟢", "Good setup with moderate confidence"
        else:
            return "SELL", "🔴", "Bearish setup with moderate confidence"
    
    elif confidence_score >= 45:
        if buy_count > sell_count:
            return "CONSIDER BUY", "🟡", "Weak bullish signals, wait for confirmation"
        else:
            return "CONSIDER SELL", "🟡", "Weak bearish signals, wait for confirmation"
    
    elif confidence_score >= 30:
        return "HOLD/WATCH", "⚪", "Neutral signals, monitor for changes"
    
    else:
        return "AVOID", "⚫", "Low confidence, high risk"

# ==================================================
# VISUALIZATION FUNCTIONS
# ==================================================
def display_price_metrics(df):
    """Display price metrics"""
    if df is None or len(df) < 2:
        return None
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    high_24h = df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()
    low_24h = df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()
    
    return {
        'current': current_price,
        'change': price_change,
        'change_pct': price_change_pct,
        'high_24h': high_24h,
        'low_24h': low_24h,
        'range_pct': ((high_24h - low_24h) / low_24h * 100) if low_24h != 0 else 0
    }

def create_metrics_display(metrics, asset_name):
    """Create metrics display"""
    if metrics is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{asset_name} Price",
            f"${metrics['current']:,.2f}" if asset_name != "DXY" else f"{metrics['current']:.2f}",
            f"{metrics['change_pct']:+.2f}%"
        )
    
    with col2:
        st.metric("24h High", f"${metrics['high_24h']:,.2f}" if asset_name != "DXY" else f"{metrics['high_24h']:.2f}")
    
    with col3:
        st.metric("24h Low", f"${metrics['low_24h']:,.2f}" if asset_name != "DXY" else f"{metrics['low_24h']:.2f}")
    
    with col4:
        st.metric("24h Range", f"{metrics['range_pct']:.2f}%")

# ==================================================
# MAIN APPLICATION
# ==================================================
def main():
    # Initialize session state for caching
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Show loading state
    with st.spinner("🔍 Fetching market data and analyzing signals..."):
        progress_bar = st.progress(0)
        
        # Fetch data
        market_data = fetch_market_data(selected_asset, timeframe)
        progress_bar.progress(0.3)
        
        # Calculate indicators
        market_data = calculate_indicators(market_data)
        progress_bar.progress(0.6)
        
        # Analyze data
        price_metrics = display_price_metrics(market_data)
        trend_analysis = analyze_trend(market_data)
        signals = detect_signals(market_data, sensitivity)
        market_indicators = fetch_market_indicators()
        progress_bar.progress(0.9)
        
        # Calculate confidence
        confidence_score = calculate_confidence_score(signals, trend_analysis, market_indicators)
        
        # Get trade recommendation
        recommendation, rec_color, rec_reason = get_trade_recommendation(confidence_score, signals)
        
        progress_bar.progress(1.0)
        tm.sleep(0.3)
        progress_bar.empty()
    
    # ==================================================
    # DASHBOARD LAYOUT
    # ==================================================
    
    # Price Metrics
    st.markdown("### 📈 Price Overview")
    create_metrics_display(price_metrics, selected_asset.split('/')[0])
    
    st.divider()
    
    # Main Dashboard in Columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price Chart
        st.markdown("### 📊 Price Chart")
        if market_data is not None and len(market_data) > 0:
            chart_data = market_data[['close', 'SMA_20', 'SMA_50']].tail(100)
            st.line_chart(chart_data)
        else:
            st.info("No chart data available")
        
        # Technical Indicators
        st.markdown("### 🔧 Technical Indicators")
        
        if market_data is not None and len(market_data) > 0:
            tech_cols = st.columns(3)
            
            with tech_cols[0]:
                current_rsi = market_data['RSI'].iloc[-1] if 'RSI' in market_data.columns else 50
                st.metric("RSI", f"{current_rsi:.1f}")
                
                if current_rsi < 30:
                    st.success("Oversold")
                elif current_rsi > 70:
                    st.error("Overbought")
                else:
                    st.info("Neutral")
            
            with tech_cols[1]:
                if 'MACD' in market_data.columns and 'MACD_Signal' in market_data.columns:
                    macd_diff = market_data['MACD'].iloc[-1] - market_data['MACD_Signal'].iloc[-1]
                    st.metric("MACD", f"{macd_diff:.4f}")
                    
                    if macd_diff > 0:
                        st.success("Bullish")
                    else:
                        st.error("Bearish")
                else:
                    st.info("MACD: N/A")
            
            with tech_cols[2]:
                if 'Volume_Ratio' in market_data.columns:
                    volume_ratio = market_data['Volume_Ratio'].iloc[-1]
                    st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                    
                    if volume_ratio > 1.5:
                        st.success("High")
                    elif volume_ratio < 0.5:
                        st.warning("Low")
                    else:
                        st.info("Normal")
                else:
                    st.info("Volume: N/A")
    
    with col2:
        # Trade Signal Card
        st.markdown("### 🎯 Trade Signal")
        
        signal_card = st.container()
        with signal_card:
            st.markdown(f"### {rec_color} {recommendation}")
            st.markdown(f"**Confidence:** {confidence_score}/100")
            
            if confidence_score >= 70:
                st.success("High Confidence")
            elif confidence_score >= 50:
                st.info("Moderate Confidence")
            elif confidence_score >= 30:
                st.warning("Low Confidence")
            else:
                st.error("Very Low Confidence")
            
            st.caption(rec_reason)
        
        # Risk Management
        st.markdown("### 🛡️ Risk Management")
        
        risk_cols = st.columns(2)
        with risk_cols[0]:
            st.metric("Max Position", f"{max_position_size}%")
        with risk_cols[1]:
            st.metric("Stop Loss", f"{stop_loss}%")
        
        # Market Context
        st.markdown("### 🌍 Market Context")
        
        st.metric("BTC Dominance", f"{market_indicators['btc_dominance']:.1f}%")
        st.metric("Total Crypto MCap", f"${market_indicators['total_mcap']/1e12:.2f}T")
    
    st.divider()
    
    # Signals Breakdown
    st.markdown("### 📡 Detailed Signals Analysis")
    
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    
    with sig_col1:
        st.markdown("##### 🟢 Buy Signals")
        buy_signals = signals.get('buy_signals', [])
        if buy_signals:
            for signal in buy_signals:
                st.success(f"• {signal}")
        else:
            st.info("No buy signals detected")
    
    with sig_col2:
        st.markdown("##### 🔴 Sell Signals")
        sell_signals = signals.get('sell_signals', [])
        if sell_signals:
            for signal in sell_signals:
                st.error(f"• {signal}")
        else:
            st.info("No sell signals detected")
    
    with sig_col3:
        st.markdown("##### ⚪ Neutral/Info")
        st.info(f"• Trend: {trend_analysis['trend']}")
        st.info(f"• Trend Strength: {trend_analysis['strength']:.1f}")
        st.info(f"• Risk Level: {signals.get('risk_level', 'MEDIUM')}")
        
        if 'ATR' in market_data.columns:
            atr_value = market_data['ATR'].iloc[-1]
            atr_pct = (atr_value / price_metrics['current'] * 100) if price_metrics and price_metrics['current'] != 0 else 0
            st.info(f"• Volatility (ATR): {atr_pct:.2f}%")
    
    st.divider()
    
    # Trading Session Info
    current_time = datetime.utcnow()
    hour = current_time.hour
    
    session_info = ""
    if 7 <= hour < 10:
        session_info = "🇬🇧 **London Session** (7:00-10:00 UTC) - High volatility expected"
    elif 13 <= hour < 16:
        session_info = "🇺🇸 **New York Session** (13:00-16:00 UTC) - Peak trading hours"
    elif 21 <= hour or hour < 1:
        session_info = "🇯🇵 **Asian Session** (21:00-1:00 UTC) - Lower volatility"
    else:
        session_info = "🌙 **Off Hours** - Reduced market activity"
    
    st.info(f"**Current Session:** {session_info}")
    
    # Data Status
    st.caption(f"""
    📊 **Data Status:** {len(market_data)} data points loaded • 
    ⏰ **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} • 
    🔄 **Auto-refresh:** {'Enabled' if auto_refresh else 'Disabled'}
    """)
    
    st.caption("""
    *Disclaimer: This tool is for educational and informational purposes only. 
    Trading involves substantial risk of loss and is not suitable for every investor. 
    Past performance is not indicative of future results.*
    """)

# ==================================================
# APPLICATION ENTRY POINT
# ==================================================
if __name__ == "__main__":
    try:
        main()
        
        # Auto-refresh logic
        if auto_refresh:
            tm.sleep(30)
            st.rerun()
            
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or try again later.")
        
        # Show fallback interface
        st.markdown("### 🚨 Application Recovery Mode")
        st.warning("The application encountered an error but is still functional.")
        
        if st.button("Try to recover", type="primary"):
            st.cache_data.clear()
            st.rerun()
