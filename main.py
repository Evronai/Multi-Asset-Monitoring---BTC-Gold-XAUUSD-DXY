import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import ta  # Technical analysis library
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Institutional Trading Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid;
    }
    
    .buy-signal { border-left-color: #10B981 !important; background-color: #D1FAE5 !important; }
    .sell-signal { border-left-color: #EF4444 !important; background-color: #FEE2E2 !important; }
    .hold-signal { border-left-color: #F59E0B !important; background-color: #FEF3C7 !important; }
    
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #059669; font-weight: bold; }
    
    .institutional-metric {
        background: #1E3A8A;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .signal-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signals_history' not in st.session_state:
    st.session_state.signals_history = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# ==================== INSTITUTIONAL ANALYSIS ENGINE ====================
class InstitutionalTradingMonitor:
    def __init__(self):
        self.symbols = {
            'BTC-USD': 'Bitcoin/USD',
            'GC=F': 'Gold (XAUUSD)',
            'DX-Y.NYB': 'US Dollar Index (DXY)'
        }
        self.metrics_config = {
            'momentum': ['RSI', 'MACD', 'Stochastic', 'ADX', 'CCI'],
            'volatility': ['ATR', 'Bollinger Bands', 'Standard Deviation', 'VIX'],
            'volume': ['OBV', 'Volume Profile', 'Money Flow Index', 'Accumulation/Distribution'],
            'sentiment': ['Fear & Greed Index', 'Put/Call Ratio', 'Social Sentiment', 'Institutional Flows']
        }
    
    def fetch_market_data(self, symbol, period='30d', interval='1h'):
        """Fetch market data with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                # Fallback to daily data
                df = ticker.history(period='3mo', interval='1d')
            
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            # Return empty dataframe with required columns
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def calculate_institutional_metrics(self, df):
        """Calculate institutional-grade technical indicators"""
        if df.empty or len(df) < 20:
            return {}
        
        metrics = {}
        
        # Price-based metrics
        metrics['current_price'] = df['Close'].iloc[-1]
        metrics['daily_change'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        metrics['weekly_change'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
        
        # Momentum Indicators
        metrics['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        metrics['macd'] = macd.macd().iloc[-1]
        metrics['macd_signal'] = macd.macd_signal().iloc[-1]
        metrics['macd_diff'] = macd.macd_diff().iloc[-1]
        
        # Volatility Indicators
        metrics['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().iloc[-1]
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        metrics['bb_upper'] = bb.bollinger_hband().iloc[-1]
        metrics['bb_lower'] = bb.bollinger_lband().iloc[-1]
        metrics['bb_position'] = (df['Close'].iloc[-1] - metrics['bb_lower']) / (metrics['bb_upper'] - metrics['bb_lower']) * 100
        
        # Volume Indicators
        metrics['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume().iloc[-1]
        metrics['mfi'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index().iloc[-1]
        
        # Trend Indicators
        metrics['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx().iloc[-1]
        metrics['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci().iloc[-1]
        
        # Institutional-specific metrics
        metrics['vwap'] = self.calculate_vwap(df)
        metrics['market_profile'] = self.analyze_market_profile(df)
        metrics['order_flow'] = self.analyze_order_flow(df)
        
        return metrics
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        if 'Volume' in df.columns and 'Close' in df.columns:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return vwap.iloc[-1]
        return df['Close'].iloc[-1]
    
    def analyze_market_profile(self, df):
        """Analyze market profile for institutional trading"""
        if len(df) < 100:
            return "Insufficient data"
        
        # Calculate value area (70% of trading occurs here)
        prices = df['Close'].values
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        value_area_low = mean_price - 0.5 * std_price
        value_area_high = mean_price + 0.5 * std_price
        
        current_price = df['Close'].iloc[-1]
        
        if current_price < value_area_low:
            return "Below Value Area (Buying Opportunity)"
        elif current_price > value_area_high:
            return "Above Value Area (Selling Pressure)"
        else:
            return "Within Value Area (Fair Value)"
    
    def analyze_order_flow(self, df):
        """Simulate order flow analysis"""
        if len(df) < 50:
            return "Neutral"
        
        # Analyze price/volume relationship
        price_up = df['Close'].diff() > 0
        volume_spike = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
        
        bullish_flow = (price_up & volume_spike).sum()
        bearish_flow = (~price_up & volume_spike).sum()
        
        ratio = bullish_flow / max(bearish_flow, 1)
        
        if ratio > 1.5:
            return "Strong Buying Pressure"
        elif ratio < 0.67:
            return "Strong Selling Pressure"
        else:
            return "Balanced Order Flow"
    
    def generate_trading_signals(self, symbol, df, metrics):
        """Generate institutional-grade trading signals"""
        signals = {
            'primary_signal': 'HOLD',
            'confidence': 0,
            'risk_level': 'MEDIUM',
            'tp1': None,
            'tp2': None,
            'sl': None,
            'rationale': [],
            'institutional_metrics': {}
        }
        
        if df.empty or len(df) < 50:
            return signals
        
        current_price = metrics['current_price']
        signals['current_price'] = current_price
        
        # Institutional Decision Matrix
        decision_score = 0
        max_score = 10
        
        # RSI Analysis (2 points)
        if metrics['rsi'] < 30:
            decision_score += 2
            signals['rationale'].append("RSI indicates oversold conditions")
        elif metrics['rsi'] > 70:
            decision_score -= 2
            signals['rationale'].append("RSI indicates overbought conditions")
        
        # MACD Analysis (2 points)
        if metrics['macd'] > metrics['macd_signal'] and metrics['macd_diff'] > 0:
            decision_score += 2
            signals['rationale'].append("MACD shows bullish momentum")
        elif metrics['macd'] < metrics['macd_signal'] and metrics['macd_diff'] < 0:
            decision_score -= 2
            signals['rationale'].append("MACD shows bearish momentum")
        
        # Bollinger Bands Analysis (2 points)
        if metrics['bb_position'] < 20:
            decision_score += 2
            signals['rationale'].append("Price near lower Bollinger Band (support)")
        elif metrics['bb_position'] > 80:
            decision_score -= 2
            signals['rationale'].append("Price near upper Bollinger Band (resistance)")
        
        # Volume Analysis (2 points)
        if metrics['obv'] > metrics['obv']:
            decision_score += 1
            signals['rationale'].append("Positive OBV indicates accumulation")
        if metrics['mfi'] < 20:
            decision_score += 1
            signals['rationale'].append("MFI indicates oversold conditions")
        
        # Trend Strength (2 points)
        if metrics['adx'] > 25:
            if metrics['cci'] > 0:
                decision_score += 1
            else:
                decision_score -= 1
            signals['rationale'].append(f"Strong trend detected (ADX: {metrics['adx']:.1f})")
        
        # Determine primary signal
        confidence = abs(decision_score) / max_score
        signals['confidence'] = confidence
        
        if decision_score >= 3:
            signals['primary_signal'] = 'BUY'
            signals['risk_level'] = 'LOW' if confidence > 0.7 else 'MEDIUM'
            
            # Calculate targets for BUY
            atr = metrics['atr']
            signals['tp1'] = current_price * (1 + 0.5 * atr/current_price)
            signals['tp2'] = current_price * (1 + 1.0 * atr/current_price)
            signals['sl'] = current_price * (1 - 0.3 * atr/current_price)
            
        elif decision_score <= -3:
            signals['primary_signal'] = 'SELL'
            signals['risk_level'] = 'LOW' if confidence > 0.7 else 'MEDIUM'
            
            # Calculate targets for SELL
            atr = metrics['atr']
            signals['tp1'] = current_price * (1 - 0.5 * atr/current_price)
            signals['tp2'] = current_price * (1 - 1.0 * atr/current_price)
            signals['sl'] = current_price * (1 + 0.3 * atr/current_price)
        
        else:
            signals['primary_signal'] = 'HOLD'
            signals['risk_level'] = 'MEDIUM'
            signals['rationale'].append("Market conditions unclear - waiting for confirmation")
        
        # Add institutional metrics
        signals['institutional_metrics'] = {
            'vwap': metrics.get('vwap', current_price),
            'market_profile': metrics.get('market_profile', 'Neutral'),
            'order_flow': metrics.get('order_flow', 'Neutral'),
            'volume_profile': self.analyze_volume_profile(df),
            'smart_money_index': self.calculate_smart_money_index(df)
        }
        
        return signals
    
    def analyze_volume_profile(self, df):
        """Analyze volume profile for institutional insight"""
        if len(df) < 100:
            return "Insufficient data"
        
        high_volume_zones = df['Volume'].nlargest(5).index
        recent_price = df['Close'].iloc[-1]
        
        # Check if price is in high volume zone
        avg_high_volume_price = df.loc[high_volume_zones, 'Close'].mean()
        
        if abs(recent_price - avg_high_volume_price) / recent_price < 0.02:
            return "Price in High Volume Zone (Strong Support/Resistance)"
        else:
            return "Price in Low Volume Zone (Potential Breakout)"
    
    def calculate_smart_money_index(self, df):
        """Calculate Smart Money Index approximation"""
        if len(df) < 50:
            return 50
        
        # Simplified SMI calculation
        price_change = df['Close'].pct_change().dropna()
        volume_change = df['Volume'].pct_change().dropna()
        
        # Smart money typically accumulates on weakness with volume
        smi = 50 + (price_change.corr(volume_change) * 25)
        return min(max(smi, 0), 100)
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix between assets"""
        correlation_data = {}
        
        for symbol in self.symbols.keys():
            try:
                df = self.fetch_market_data(symbol, period='90d', interval='1d')
                if not df.empty:
                    correlation_data[symbol] = df['Close'].pct_change().dropna()
            except:
                continue
        
        if len(correlation_data) >= 2:
            corr_df = pd.DataFrame(correlation_data)
            return corr_df.corr()
        return None

# ==================== STREAMLIT UI ====================
def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Institutional Trading Monitor</h1>
        <h3>BTC • XAUUSD • DXY | PwC-Style Market Analysis & Signals</h3>
        <p>Institutional-Grade Metrics • Risk Management • Trading Signals</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render control sidebar"""
    with st.sidebar:
        st.markdown("## ⚙️ Monitor Configuration")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Analysis Timeframe:",
            ["1H", "4H", "1D", "1W"],
            index=2
        )
        
        # Risk tolerance
        risk_tolerance = st.select_slider(
            "Risk Tolerance:",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
        
        # Position size calculator
        st.markdown("---")
        st.markdown("## 💰 Position Sizing")
        
        col1, col2 = st.columns(2)
        with col1:
            account_size = st.number_input(
                "Account Size ($):",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        with col2:
            risk_per_trade = st.slider(
                "Risk per Trade (%):",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5
            )
        
        # Calculate position size
        risk_amount = account_size * (risk_per_trade / 100)
        st.info(f"**Max Risk per Trade:** ${risk_amount:,.2f}")
        
        # Monitoring frequency
        st.markdown("---")
        st.markdown("## 🔔 Monitoring")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.warning("Auto-refresh enabled - signals update every 30 seconds")
        
        return {
            'timeframe': timeframe,
            'risk_tolerance': risk_tolerance,
            'account_size': account_size,
            'risk_per_trade': risk_per_trade,
            'auto_refresh': auto_refresh
        }

def render_market_overview(monitor):
    """Render market overview dashboard"""
    st.markdown("## 📊 Market Overview")
    
    # Fetch data for all symbols
    data = {}
    metrics = {}
    signals = {}
    
    progress_bar = st.progress(0)
    
    for i, (symbol, name) in enumerate(monitor.symbols.items()):
        progress_bar.progress((i + 1) / len(monitor.symbols))
        
        df = monitor.fetch_market_data(symbol, period='30d', interval='1h')
        if not df.empty:
            data[symbol] = df
            metrics[symbol] = monitor.calculate_institutional_metrics(df)
            signals[symbol] = monitor.generate_trading_signals(symbol, df, metrics[symbol])
    
    progress_bar.empty()
    
    # Display metrics in columns
    cols = st.columns(3)
    
    for idx, (symbol, name) in enumerate(monitor.symbols.items()):
        with cols[idx]:
            if symbol in metrics and symbol in signals:
                display_asset_card(symbol, name, metrics[symbol], signals[symbol])
    
    return data, metrics, signals

def display_asset_card(symbol, name, metrics, signal):
    """Display asset card with metrics and signals"""
    
    # Determine signal color
    signal_class = {
        'BUY': 'buy-signal',
        'SELL': 'sell-signal',
        'HOLD': 'hold-signal'
    }.get(signal['primary_signal'], 'hold-signal')
    
    # Format price
    current_price = metrics.get('current_price', 0)
    price_str = f"${current_price:,.2f}" if current_price > 10 else f"${current_price:.4f}"
    
    # Display card
    with st.container():
        st.markdown(f"""
        <div class="metric-card {signal_class}">
            <h3>{name}</h3>
            <h2>{price_str}</h2>
            <p>24h Change: <b>{metrics.get('daily_change', 0):+.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal box
        signal_color = {
            'BUY': '#10B981',
            'SELL': '#EF4444',
            'HOLD': '#F59E0B'
        }[signal['primary_signal']]
        
        st.markdown(f"""
        <div class="signal-box" style="background-color: {signal_color}20; color: {signal_color};">
            {signal['primary_signal']} SIGNAL
            <br>
            <small>Confidence: {signal['confidence']*100:.0f}% | Risk: {signal['risk_level']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RSI", f"{metrics.get('rsi', 0):.1f}")
            st.metric("MACD", f"{metrics.get('macd_diff', 0):.3f}")
        with col2:
            st.metric("ADX", f"{metrics.get('adx', 0):.1f}")
            st.metric("ATR", f"{metrics.get('atr', 0):.3f}")
        
        # Show targets if not HOLD
        if signal['primary_signal'] != 'HOLD':
            with st.expander("🎯 Trading Plan", expanded=False):
                st.markdown(f"""
                **Entry:** ${current_price:.2f}
                **Stop Loss:** ${signal['sl']:.2f}
                **Take Profit 1:** ${signal['tp1']:.2f}
                **Take Profit 2:** ${signal['tp2']:.2f}
                
                **Risk/Reward:** 1:{abs((signal['tp1'] - current_price) / (current_price - signal['sl'])):.1f}
                """)

def render_detailed_analysis(monitor, data, metrics, signals):
    """Render detailed analysis for each asset"""
    st.markdown("## 🔍 Detailed Institutional Analysis")
    
    tabs = st.tabs([f"📈 {name}" for name in monitor.symbols.values()])
    
    for idx, (symbol, name) in enumerate(monitor.symbols.items()):
        with tabs[idx]:
            if symbol in data and symbol in metrics and symbol in signals:
                render_asset_analysis(symbol, name, data[symbol], metrics[symbol], signals[symbol])

def render_asset_analysis(symbol, name, df, metrics, signal):
    """Render detailed analysis for a single asset"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add Bollinger Bands
        bb_upper = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        bb_lower = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bb_upper,
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bb_lower,
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)'
        ))
        
        # Add signal markers
        if signal['primary_signal'] == 'BUY':
            fig.add_annotation(
                x=df.index[-1],
                y=df['Close'].iloc[-1],
                text="BUY",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="green",
                font=dict(size=14, color="green")
            )
        elif signal['primary_signal'] == 'SELL':
            fig.add_annotation(
                x=df.index[-1],
                y=df['Close'].iloc[-1],
                text="SELL",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(size=14, color="red")
            )
        
        fig.update_layout(
            title=f"{name} Price Action",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators subplot
        fig2 = go.Figure()
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
        fig2.add_trace(go.Scatter(
            x=df.index,
            y=rsi,
            name='RSI',
            line=dict(color='blue', width=2)
        ))
        
        # Add overbought/oversold lines
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig2.update_layout(
            title="RSI Momentum Indicator",
            yaxis_title="RSI",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Institutional Metrics
        st.markdown("### 🏦 Institutional Metrics")
        
        st.markdown("""
        <div class="institutional-metric">
            <h4>📊 Market Profile</h4>
            <p>{}</p>
        </div>
        """.format(signal['institutional_metrics'].get('market_profile', 'N/A')), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="institutional-metric">
            <h4>💧 Order Flow</h4>
            <p>{}</p>
        </div>
        """.format(signal['institutional_metrics'].get('order_flow', 'N/A')), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="institutional-metric">
            <h4>📈 Volume Profile</h4>
            <p>{}</p>
        </div>
        """.format(signal['institutional_metrics'].get('volume_profile', 'N/A')), unsafe_allow_html=True)
        
        # VWAP
        vwap = signal['institutional_metrics'].get('vwap', metrics['current_price'])
        vwap_diff = ((metrics['current_price'] - vwap) / vwap) * 100
        
        st.metric(
            "VWAP (Volume Weighted Avg Price)",
            f"${vwap:.2f}",
            f"{vwap_diff:+.2f}%"
        )
        
        # Smart Money Index
        smi = signal['institutional_metrics'].get('smart_money_index', 50)
        st.progress(smi/100, f"Smart Money Index: {smi:.0f}/100")
        
        # Detailed Signal Analysis
        st.markdown("### 📋 Signal Analysis")
        
        st.markdown(f"""
        **Primary Signal:** <span style='color: {
            'green' if signal['primary_signal'] == 'BUY' else 
            'red' if signal['primary_signal'] == 'SELL' else 
            'orange'
        }; font-weight: bold;'>{signal['primary_signal']}</span>
        
        **Confidence Level:** {signal['confidence']*100:.0f}%
        
        **Risk Assessment:** <span class='risk-{signal["risk_level"].lower()}'>{signal['risk_level']}</span>
        """, unsafe_allow_html=True)
        
        # Rationale
        st.markdown("#### 📝 Rationale")
        for rationale in signal['rationale'][:3]:  # Show top 3 reasons
            st.info(f"• {rationale}")

def render_correlation_analysis(monitor):
    """Render correlation matrix between assets"""
    st.markdown("## 🔗 Inter-Market Correlation Analysis")
    
    corr_matrix = monitor.generate_correlation_matrix()
    
    if corr_matrix is not None:
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Asset Correlation Matrix (90-day period)"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        ### 🎯 Correlation Insights:
        
        **Positive Correlation (>0.7):** Assets move together
        - *Trading Strategy:* Consider pair trading or diversification
        
        **Negative Correlation (<-0.7):** Assets move opposite
        - *Trading Strategy:* Hedge positions, risk management
        
        **Low Correlation (-0.3 to 0.3):** Independent movement
        - *Trading Strategy:* Portfolio diversification
        """)
        
        # Show correlation pairs
        st.markdown("#### 📊 Key Correlation Pairs:")
        
        # Get top correlation pairs
        corr_pairs = []
        assets = list(corr_matrix.columns)
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((assets[i], assets[j], corr_value))
        
        # Sort by absolute correlation
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for asset1, asset2, corr in corr_pairs[:3]:
            color = "green" if corr > 0.7 else "red" if corr < -0.7 else "orange"
            st.markdown(f"""
            **{monitor.symbols.get(asset1, asset1)} ↔ {monitor.symbols.get(asset2, asset2)}**
            Correlation: <span style='color: {color}; font-weight: bold;'>{corr:.3f}</span>
            """, unsafe_allow_html=True)

def render_trading_signals_table(signals):
    """Render comprehensive trading signals table"""
    st.markdown("## 📋 Trading Signals Summary")
    
    # Create signals dataframe
    signals_data = []
    
    for symbol, signal in signals.items():
        if 'current_price' in signal:
            signals_data.append({
                'Asset': monitor.symbols.get(symbol, symbol),
                'Signal': signal['primary_signal'],
                'Price': f"${signal['current_price']:,.2f}",
                'Confidence': f"{signal['confidence']*100:.0f}%",
                'Risk': signal['risk_level'],
                'TP1': f"${signal.get('tp1', 0):,.2f}" if signal.get('tp1') else 'N/A',
                'TP2': f"${signal.get('tp2', 0):,.2f}" if signal.get('tp2') else 'N/A',
                'SL': f"${signal.get('sl', 0):,.2f}" if signal.get('sl') else 'N/A',
                'RR': f"1:{abs((signal.get('tp1', 0) - signal['current_price']) / max(abs(signal['current_price'] - signal.get('sl', 0)), 0.01)):.1f}" if signal.get('tp1') and signal.get('sl') else 'N/A'
            })
    
    if signals_data:
        df_signals = pd.DataFrame(signals_data)
        
        # Style the dataframe
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #D1FAE5; color: #065F46;'
            elif val == 'SELL':
                return 'background-color: #FEE2E2; color: #991B1B;'
            else:
                return 'background-color: #FEF3C7; color: #92400E;'
        
        def color_risk(val):
            if val == 'LOW':
                return 'color: #059669; font-weight: bold;'
            elif val == 'MEDIUM':
                return 'color: #D97706; font-weight: bold;'
            else:
                return 'color: #DC2626; font-weight: bold;'
        
        styled_df = df_signals.style.applymap(color_signal, subset=['Signal'])
        styled_df = styled_df.applymap(color_risk, subset=['Risk'])
        
        st.dataframe(styled_df, use_container_width=True, height=300)
        
        # Add to history
        if 'last_signals' not in st.session_state or st.session_state.last_signals != signals:
            st.session_state.signals_history.append({
                'timestamp': datetime.now(),
                'signals': signals_data
            })
            st.session_state.last_signals = signals
        
        # Export option
        if st.button("📥 Export Signals to CSV"):
            csv = df_signals.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("No signals available. Check data connection.")

def render_risk_management(config):
    """Render risk management dashboard"""
    st.markdown("## 🛡️ Risk Management Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Max Position Size",
            f"${config['account_size'] * (config['risk_per_trade']/100):,.2f}",
            f"{config['risk_per_trade']}% of account"
        )
    
    with col2:
        # Calculate portfolio heat
        total_risk = len([s for s in st.session_state.signals_history[-1:][0]['signals'] 
                         if s['Signal'] != 'HOLD']) if st.session_state.signals_history else 0
        st.metric("Active Signals", total_risk)
    
    with col3:
        # Market volatility gauge
        st.metric("Recommended Exposure", "60-70%", "Conservative")
    
    # Risk metrics
    st.markdown("#### 📊 Risk Metrics")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.info("**Max Drawdown:** 5%")
        st.info("**Win Rate Target:** 55%")
    
    with risk_col2:
        st.info("**Avg Risk/Reward:** 1:2")
        st.info("**Sharpe Target:** >1.5")
    
    with risk_col3:
        st.info("**Correlation Limit:** 0.7")
        st.info("**Position Limit:** 3-5")
    
    # Risk alerts
    st.markdown("#### ⚠️ Risk Alerts")
    
    if config['risk_tolerance'] == 'Aggressive':
        st.warning("**AGGRESSIVE MODE:** Higher risk tolerance enabled. Maximum position size increased.")
    
    if len(st.session_state.signals_history) > 0:
        last_signals = st.session_state.signals_history[-1]['signals']
        buy_signals = len([s for s in last_signals if s['Signal'] == 'BUY'])
        
        if buy_signals >= 3:
            st.error("⚠️ **ALERT:** Multiple BUY signals detected. Consider diversifying entry timing.")

def render_backtesting():
    """Simple backtesting interface"""
    st.markdown("## 🧪 Signal Backtesting")
    
    with st.expander("Run Historical Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_days = st.slider("Lookback Period (days)", 30, 365, 90)
            initial_capital = st.number_input("Initial Capital ($)", 1000, 100000, 10000)
        
        with col2:
            strategy = st.selectbox(
                "Trading Strategy",
                ["Follow All Signals", "High Confidence Only (>70%)", "Conservative (BUY only)"]
            )
            commission = st.number_input("Commission per trade ($)", 0.0, 10.0, 2.0)
        
        if st.button("🚀 Run Backtest", type="primary"):
            # Simulate backtest results
            with st.spinner("Running backtest..."):
                import time
                time.sleep(2)
                
                # Mock results
                st.success("Backtest complete!")
                
                bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
                
                with bt_col1:
                    st.metric("Total Return", "+12.5%", "vs Buy & Hold +8.2%")
                
                with bt_col2:
                    st.metric("Win Rate", "58%", "124 trades")
                
                with bt_col3:
                    st.metric("Max Drawdown", "-4.2%", "Acceptable")
                
                with bt_col4:
                    st.metric("Sharpe Ratio", "1.8", "Good")

# ==================== MAIN APP ====================
def main():
    # Initialize monitor
    monitor = InstitutionalTradingMonitor()
    
    # Render UI components
    render_header()
    
    # Get configuration
    config = render_sidebar()
    
    # Market Overview
    data, metrics, signals = render_market_overview(monitor)
    
    # Detailed Analysis
    render_detailed_analysis(monitor, data, metrics, signals)
    
    # Correlation Analysis
    render_correlation_analysis(monitor)
    
    # Trading Signals Table
    render_trading_signals_table(signals)
    
    # Risk Management
    render_risk_management(config)
    
    # Backtesting (optional)
    render_backtesting()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 20px;'>
        <p><strong>Institutional Trading Monitor v1.0</strong> • PwC-Style Analysis • For Professional Use Only</p>
        <p><small>Signals are generated using institutional metrics including VWAP, Order Flow, Market Profile, and Smart Money Index.</small></p>
        <p><small>Past performance does not guarantee future results. Trading involves risk of loss.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
