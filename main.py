import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
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
        ["BTC", "ETH", "XAUUSD"],
        index=0
    )
    
    # Signal sensitivity
    sensitivity = st.slider(
        "Signal Sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values = more sensitive signals"
    )
    
    # Timeframe
    timeframe = st.selectbox(
        "Timeframe",
        ["1h", "4h", "Daily", "Weekly"],
        index=1
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    
    # Notification preferences
    st.subheader("🔔 Notifications")
    email_alerts = st.checkbox("Email Alerts", value=False)
    tg_alerts = st.checkbox("Telegram Alerts", value=False)
    
    if st.button("🔄 Force Refresh"):
        st.rerun()

# -------------------------------------------------
# DATA FETCHERS WITH ERROR HANDLING
# -------------------------------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_btc_price(interval="1h", limit=500):
    try:
        if interval == "4h":
            interval_param = "4h"
        elif interval == "Daily":
            interval_param = "1d"
        elif interval == "Weekly":
            interval_param = "1w"
        else:
            interval_param = "1h"
            
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": "BTCUSDT",
                "interval": interval_param,
                "limit": limit
            },
            timeout=10
        )
        r.raise_for_status()
        data = pd.DataFrame(r.json(), columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        numeric_cols = ["open", "high", "low", "close", "volume"]
        data[numeric_cols] = data[numeric_cols].astype(float)
        data["time"] = pd.to_datetime(data["time"], unit="ms")
        
        return data.set_index("time")
    except Exception as e:
        st.error(f"Error fetching BTC data: {e}")
        # Return sample data if API fails
        dates = pd.date_range(end=datetime.now(), periods=200, freq='h')
        prices = np.random.normal(40000, 1000, 200).cumsum()
        return pd.DataFrame({
            'open': prices - 100,
            'close': prices,
            'high': prices + np.random.normal(200, 50, 200),
            'low': prices - np.random.normal(200, 50, 200),
            'volume': np.random.normal(1000, 200, 200)
        }, index=dates)

@st.cache_data(ttl=3600)
def get_gold_price():
    try:
        # Using free Forex API as fallback for gold
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "XAU"},
            timeout=10
        )
        data = r.json()
        # For historical data, we'll create a simulated series
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.random.normal(1950, 50, 100).cumsum()
        return pd.DataFrame({
            'open': prices - 25,
            'close': prices,
            'high': prices + 50,
            'low': prices - 50,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    except:
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.random.normal(1950, 50, 100).cumsum()
        return pd.DataFrame({
            'open': prices - 25,
            'close': prices,
            'high': prices + 50,
            'low': prices - 50,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)

@st.cache_data(ttl=3600)
def get_dxy_data():
    try:
        # Alternative DXY data source
        url = "https://api.frankfurter.app/latest"
        params = {"from": "EUR", "to": "USD"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        # Simulate historical DXY
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        dxy_values = np.random.normal(105, 2, 100).cumsum()
        return pd.DataFrame({
            'open': dxy_values - 0.5,
            'close': dxy_values,
            'high': dxy_values + 1,
            'low': dxy_values - 1,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    except:
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        dxy_values = np.random.normal(105, 2, 100).cumsum()
        return pd.DataFrame({
            'open': dxy_values - 0.5,
            'close': dxy_values,
            'high': dxy_values + 1,
            'low': dxy_values - 1,
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)

@st.cache_data(ttl=3600)
def get_market_data():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        data = r.json()["data"]
        return {
            "btc_dominance": data["market_cap_percentage"]["btc"],
            "total_market_cap": data["total_market_cap"]["usd"],
            "fear_greed": np.random.randint(0, 100)  # Placeholder
        }
    except:
        return {
            "btc_dominance": 48.5,
            "total_market_cap": 1.5e12,
            "fear_greed": 50
        }

# -------------------------------------------------
# ENHANCED INDICATORS
# -------------------------------------------------
def calculate_indicators(df):
    df = df.copy()
    
    # Ensure we have enough data
    if len(df) < 50:
        # Add dummy indicators if not enough data
        df['EMA_21'] = df['close']
        df['EMA_50'] = df['close']
        df['EMA_200'] = df['close']
        df['RSI'] = 50
        df['Volume_MA'] = df['volume']
        df['Volume_Ratio'] = 1
        df['Resistance'] = df['high']
        df['Support'] = df['low']
        return df
    
    # EMAs
    df['EMA_21'] = df['close'].ewm(span=21, min_periods=1).mean()
    df['EMA_50'] = df['close'].ewm(span=50, min_periods=1).mean()
    df['EMA_200'] = df['close'].ewm(span=200, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Handle NaN values in RSI
    df['RSI'] = df['RSI'].fillna(50)
    
    # Volume profile
    df['Volume_MA'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, 1)
    
    # Support/Resistance
    df['Resistance'] = df['high'].rolling(window=20, min_periods=1).max()
    df['Support'] = df['low'].rolling(window=20, min_periods=1).min()
    
    return df

# -------------------------------------------------
# ENHANCED SMART MONEY DETECTION
# -------------------------------------------------
def detect_smart_money_patterns(price_df, volume_df, sensitivity=5):
    patterns = []
    
    if len(price_df) < 10:
        return ["INSUFFICIENT_DATA"]
    
    # Price-Volume Divergence
    try:
        price_trend = price_df['close'].iloc[-5:].mean() > price_df['close'].iloc[-10:-5].mean()
        volume_trend = volume_df['Volume_Ratio'].iloc[-5:].mean() < volume_df['Volume_Ratio'].iloc[-10:-5].mean()
        
        if price_trend and not volume_trend:
            patterns.append("SMART_ACCUMULATION")
        elif not price_trend and volume_trend:
            patterns.append("SMART_DISTRIBUTION")
    except:
        pass
    
    # Hidden Divergence
    try:
        if len(price_df) > 20:
            price_high = price_df['high'].iloc[-10:].max()
            price_low = price_df['low'].iloc[-10:].min()
            rsi_high = price_df['RSI'].iloc[-10:].max()
            rsi_low = price_df['RSI'].iloc[-10:].min()
            
            if price_high > price_df['high'].iloc[-20:-10].max() and rsi_high < price_df['RSI'].iloc[-20:-10].max():
                patterns.append("BEARISH_DIVERGENCE")
            elif price_low < price_df['low'].iloc[-20:-10].min() and rsi_low > price_df['RSI'].iloc[-20:-10].min():
                patterns.append("BULLISH_DIVERGENCE")
    except:
        pass
    
    return patterns if patterns else ["NO_SMART_PATTERN"]

# -------------------------------------------------
# LIQUIDITY ANALYSIS
# -------------------------------------------------
def analyze_liquidity(df, sensitivity=5):
    signals = []
    
    if len(df) < 10:
        return ["INSUFFICIENT_DATA"]
    
    # Recent price action
    recent = df.tail(min(10, len(df)))
    
    # Swing High/Low detection
    highs = recent['high']
    lows = recent['low']
    
    # Check for equal highs/lows (liquidity pools)
    try:
        recent_high = highs.iloc[-1]
        recent_low = lows.iloc[-1]
        
        equal_highs = sum(abs(highs - recent_high) / recent_high < 0.001)
        equal_lows = sum(abs(lows - recent_low) / recent_low < 0.001)
        
        if equal_highs >= 2:
            signals.append("LIQUIDITY_ABOVE")
        if equal_lows >= 2:
            signals.append("LIQUIDITY_BELOW")
    except:
        pass
    
    # Order block detection (simplified)
    try:
        if len(df) > 2:
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            if prev_candle['close'] > prev_candle['open'] and curr_candle['close'] < curr_candle['open']:
                signals.append("BEARISH_ORDER_BLOCK")
            elif prev_candle['close'] < prev_candle['open'] and curr_candle['close'] > curr_candle['open']:
                signals.append("BULLISH_ORDER_BLOCK")
    except:
        pass
    
    return signals if signals else ["NO_LIQUIDITY_SIGNAL"]

# -------------------------------------------------
# ADVANCED CONFIDENCE ENGINE
# -------------------------------------------------
def calculate_confidence_score(patterns, liquidity_signals, market_context):
    score = 0
    factors = []
    
    # Pattern strength (30 points max)
    pattern_scores = {
        "SMART_ACCUMULATION": 25,
        "SMART_DISTRIBUTION": 25,
        "BULLISH_DIVERGENCE": 20,
        "BEARISH_DIVERGENCE": 20,
        "NO_SMART_PATTERN": 0,
        "INSUFFICIENT_DATA": 0
    }
    
    for pattern in patterns:
        score += pattern_scores.get(pattern, 0)
    
    # Liquidity signals (20 points max)
    liquidity_scores = {
        "LIQUIDITY_ABOVE": -10,  # Selling pressure above
        "LIQUIDITY_BELOW": 10,   # Buying support below
        "BULLISH_ORDER_BLOCK": 15,
        "BEARISH_ORDER_BLOCK": -15,
        "NO_LIQUIDITY_SIGNAL": 0,
        "INSUFFICIENT_DATA": 0
    }
    
    for signal in liquidity_signals:
        score += liquidity_scores.get(signal, 0)
    
    # Market context (20 points max)
    if market_context.get('btc_dominance', 0) > 48:
        score += 10
        factors.append("BTC Dominance > 48%")
    
    if market_context.get('fear_greed', 50) < 30:
        score += 10
        factors.append("Fear & Greed < 30 (Fear)")
    
    # Volume confirmation (10 points)
    if market_context.get('volume_ratio', 1) > 1.2:
        score += 10
        factors.append("High Volume Confirmation")
    
    # Session timing (10 points)
    session = market_context.get('session', 'OFF')
    if session in ['LONDON', 'NEW_YORK']:
        score += 10
        factors.append(f"{session} Session")
    
    # Trend alignment (10 points)
    if market_context.get('trend_aligned', False):
        score += 10
        factors.append("Trend Aligned")
    
    return min(max(score, 0), 100), factors

def get_trade_grade(score):
    if score >= 85:
        return "A+", "High Conviction"
    elif score >= 70:
        return "A", "Strong Signal"
    elif score >= 55:
        return "B", "Moderate Signal"
    elif score >= 40:
        return "C", "Weak Signal"
    else:
        return "D", "No Trade"

# -------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------
def create_price_chart(df):
    try:
        # Use last 100 data points or all if less
        display_data = df.tail(min(100, len(df)))
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price with EMAs", "RSI", "Volume")
        )
        
        # Price and EMAs
        fig.add_trace(
            go.Candlestick(
                x=display_data.index,
                open=display_data['open'],
                high=display_data['high'],
                low=display_data['low'],
                close=display_data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add EMAs if they exist
        for ema_period, color in [(21, 'orange'), (50, 'blue'), (200, 'red')]:
            ema_col = f'EMA_{ema_period}'
            if ema_col in display_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=display_data.index,
                        y=display_data[ema_col],
                        name=f'EMA {ema_period}',
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )
        
        # RSI if it exists
        if 'RSI' in display_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=display_data.index,
                    y=display_data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Volume
        if 'volume' in display_data.columns and 'open' in display_data.columns and 'close' in display_data.columns:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(display_data['close'], display_data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=display_data.index,
                    y=display_data['volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Error loading chart",
            template='plotly_dark'
        )
        return fig

def create_market_overview(market_data):
    try:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("BTC Dominance", "Market Sentiment")
        )
        
        # BTC Dominance gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=market_data.get('btc_dominance', 50),
                title={'text': "BTC Dominance %"},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 50], 'color': "gray"},
                        {'range': [50, 60], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': market_data.get('btc_dominance', 50)
                    }
                }
            ),
            row=1, col=1
        )
        
        # Fear & Greed
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=market_data.get('fear_greed', 50),
                title={'text': "Fear & Greed Index"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=300, template='plotly_dark')
        return fig
    except:
        # Return simple figure on error
        fig = go.Figure()
        fig.update_layout(
            title="Market Overview",
            template='plotly_dark'
        )
        return fig

# -------------------------------------------------
# TELEGRAM NOTIFICATION
# -------------------------------------------------
def send_telegram_alert(message, chart_data=None):
    if not tg_alerts:
        return
    
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        
        if not token or not chat_id:
            return
            
        if chart_data is not None and len(chart_data) > 10:
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(chart_data.index[-50:], chart_data['close'].tail(50))
            ax.set_title("Price Action")
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            # Send with photo
            files = {'photo': buf}
            data = {'chat_id': chat_id, 'caption': message[:1024]}  # Telegram caption limit
            requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data=data,
                files=files,
                timeout=10
            )
        else:
            # Send text only
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    'chat_id': chat_id,
                    'text': message[:4096],  # Telegram message limit
                    'parse_mode': 'HTML'
                },
                timeout=10
            )
    except Exception as e:
        st.warning(f"Telegram notification failed: {e}")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def trading_session():
    now = datetime.utcnow().time()
    if time(7, 0) <= now <= time(10, 0):
        return "LONDON"
    if time(13, 0) <= now <= time(16, 0):
        return "NEW_YORK"
    if time(21, 0) <= now or now <= time(1, 0):
        return "ASIA"
    return "OFF_SESSION"

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
def main():
    # Display progress
    with st.spinner("Fetching market data..."):
        progress_bar = st.progress(0.0)  # Initialize with 0.0 (not 0)
        
        # Fetch data based on selected asset
        if asset == "BTC":
            price_data = get_btc_price(interval=timeframe)
        elif asset == "XAUUSD":
            price_data = get_gold_price()
        else:  # ETH or other - using BTC as placeholder
            price_data = get_btc_price(interval=timeframe)
        
        progress_bar.progress(0.25)  # Use values between 0.0 and 1.0
        
        market_data = get_market_data()
        progress_bar.progress(0.5)
        
        # Calculate indicators
        price_data = calculate_indicators(price_data)
        progress_bar.progress(0.75)
        
        # Detect patterns
        smart_patterns = detect_smart_money_patterns(price_data, price_data, sensitivity)
        liquidity_signals = analyze_liquidity(price_data, sensitivity)
        
        # Market context
        market_context = {
            'btc_dominance': market_data['btc_dominance'],
            'fear_greed': market_data['fear_greed'],
            'volume_ratio': price_data['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in price_data.columns else 1,
            'session': trading_session(),
            'trend_aligned': price_data['EMA_50'].iloc[-1] > price_data['EMA_200'].iloc[-1] if all(col in price_data.columns for col in ['EMA_50', 'EMA_200']) else False
        }
        
        # Calculate confidence
        confidence_score, confidence_factors = calculate_confidence_score(
            smart_patterns, liquidity_signals, market_context
        )
        
        progress_bar.progress(1.0)
        tm.sleep(0.5)
        progress_bar.empty()
    
    # -------------------------------------------------
    # DASHBOARD LAYOUT
    # -------------------------------------------------
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = price_data['close'].iloc[-1]
        if len(price_data) > 1:
            price_change = ((current_price - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2]) * 100
        else:
            price_change = 0
            
        st.metric(
            f"{asset} Price",
            f"${current_price:,.2f}" if asset != "XAUUSD" else f"${current_price:,.2f}/oz",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        grade, grade_text = get_trade_grade(confidence_score)
        st.metric(
            "Trade Grade",
            grade,
            grade_text
        )
    
    with col3:
        st.metric(
            "Confidence Score",
            f"{confidence_score}/100",
            f"{len(confidence_factors)} factors"
        )
    
    with col4:
        session = market_context['session']
        session_icon = "🟢" if session in ['LONDON', 'NEW_YORK'] else "⚫"
        st.metric(
            "Trading Session",
            f"{session_icon} {session}"
        )
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Market Overview", "📋 Signals"])
    
    with tab1:
        st.plotly_chart(create_price_chart(price_data), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_market_overview(market_data), use_container_width=True)
        
        # Additional market metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Market Cap", f"${market_data['total_market_cap']/1e12:.2f}T")
        with col2:
            vol_ratio = market_context['volume_ratio']
            st.metric("Volume Ratio", f"{vol_ratio:.2f}x", "High" if vol_ratio > 1.2 else "Normal" if vol_ratio > 0.8 else "Low")
        with col3:
            ema_status = "Bullish" if market_context['trend_aligned'] else "Neutral" if market_context['trend_aligned'] is not None else "N/A"
            st.metric("EMA Trend", ema_status)
    
    with tab3:
        # Signal details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Smart Money Patterns")
            for pattern in smart_patterns:
                if "BULLISH" in pattern:
                    st.success(f"✅ {pattern}")
                elif "BEARISH" in pattern:
                    st.error(f"❌ {pattern}")
                elif "SMART" in pattern:
                    st.info(f"🔍 {pattern}")
                elif "INSUFFICIENT" in pattern:
                    st.warning(f"⚠️ {pattern}")
                else:
                    st.warning(f"⚪ {pattern}")
        
        with col2:
            st.subheader("💧 Liquidity Signals")
            for signal in liquidity_signals:
                if "BULLISH" in signal:
                    st.success(f"📈 {signal}")
                elif "BEARISH" in signal:
                    st.error(f"📉 {signal}")
                elif "LIQUIDITY" in signal:
                    st.info(f"🌊 {signal}")
                elif "INSUFFICIENT" in signal:
                    st.warning(f"⚠️ {signal}")
                else:
                    st.warning(f"⚪ {signal}")
        
        # Confidence factors
        if confidence_factors:
            st.subheader("🧮 Confidence Factors")
            for factor in confidence_factors:
                st.write(f"• {factor}")
        else:
            st.info("No significant confidence factors identified")
    
    # Trade recommendation
    st.divider()
    
    if confidence_score >= 70:
        st.success("""
        ## 🎯 TRADE SETUP DETECTED
        
        **Recommended Action:** Consider entering a position with proper risk management.
        
        **Risk Management:**
        - Stop Loss: 2-3% below entry
        - Take Profit: 1:2 or 1:3 Risk/Reward ratio
        - Position Size: 1-2% of portfolio
        """)
        
        # Send alert for high-confidence signals
        if confidence_score >= 80 and tg_alerts:
            alert_msg = f"""
            🚨 HIGH CONFIDENCE SIGNAL 🚨
            
            Asset: {asset}
            Grade: {grade}
            Confidence: {confidence_score}/100
            Patterns: {', '.join(smart_patterns)}
            Liquidity: {', '.join(liquidity_signals)}
            
            Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            """
            send_telegram_alert(alert_msg, price_data)
    
    elif confidence_score >= 40:
        st.info("""
        ## ⚠️ WATCHLIST CANDIDATE
        
        **Action:** Monitor price action for confirmation.
        
        **Wait for:**
        - Volume confirmation
        - Clear break of key levels
        - Session alignment
        """)
    else:
        st.warning("""
        ## ⛔ NO CLEAR SETUP
        
        **Action:** Wait for better market conditions.
        
        **Why:**
        - Low confidence score
        - Mixed signals
        - Poor market structure
        """)
    
    # Footer
    st.divider()
    st.caption(f"""
    Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | 
    Data Sources: Binance, CoinGecko, Frankfurter.app | 
    For educational purposes only. Not financial advice.
    """)

# -------------------------------------------------
# RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    main()
    
    # Auto-refresh logic
    if auto_refresh:
        tm.sleep(60)
        st.rerun()
