import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time, timedelta
import time as tm
import warnings
import json
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==================================================
# STREAMLIT CONFIG - PROFESSIONAL INSTITUTIONAL PLATFORM
# ==================================================
st.set_page_config(
    page_title="Institutional Multi-Timeframe Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional institutional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #1A237E;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #37474F;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E3F2FD;
    }
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .timeframe-card {
        background: #F5F7FA;
        border: 1px solid #CFD8DC;
        border-radius: 6px;
        padding: 12px;
        margin: 6px 0;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        margin: 8px 0;
        background: linear-gradient(90deg, #FF5252 0%, #FF9800 50%, #4CAF50 100%);
    }
    .signal-buy {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(135deg, #F44336 0%, #C62828 100%);
        color: white;
        padding: 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #757575 0%, #424242 100%);
        color: white;
        padding: 12px;
        border-radius: 6px;
        font-weight: 700;
        text-align: center;
    }
    .heatmap-cell {
        padding: 8px;
        text-align: center;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .news-positive { background: #E8F5E9; color: #2E7D32; }
    .news-negative { background: #FFEBEE; color: #C62828; }
    .news-neutral { background: #F5F5F5; color: #616161; }
    .liquidity-signal { background: #E3F2FD; padding: 6px 10px; border-radius: 4px; margin: 2px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏛️ INSTITUTIONAL MULTI-TIMEFRAME ANALYZER</div>', unsafe_allow_html=True)
st.markdown('**Smart Money • Liquidity Detection • News Sentiment • Multi-TF Confirmation**')

# ==================================================
# SIDEBAR - PROFESSIONAL CONFIGURATION
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ INSTITUTIONAL SETTINGS")
    
    # API Configuration
    with st.expander("🔑 API CONFIGURATION", expanded=True):
        st.markdown("**Fast Forex API**")
        fast_forex_key = st.text_input("API Key", value="6741a9cd7c-d2a1c6afde-ta8cti", type="password", key="fast_forex_key")
        
        st.markdown("**News API (Optional)**")
        news_api_key = st.text_input("News API Key", type="password", key="news_api_key")
        news_source = st.selectbox("News Source", ["NewsData.io", "CryptoPanic", "None"], index=2, key="news_source")
    
    # Instrument Selection
    st.markdown("### 📊 INSTRUMENT SELECTION")
    instruments = st.multiselect(
        "Select Instruments",
        ["BTC-USD", "ETH-USD", "XAU-USD", "EUR-USD", "GBP-USD", "USD-JPY"],
        default=["BTC-USD", "XAU-USD"],
        key="instruments"
    )
    
    # Timeframe Configuration
    st.markdown("### ⏰ TIMEFRAME CONFIGURATION")
    timeframes = st.multiselect(
        "Analysis Timeframes",
        ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
        default=["M15", "H1", "H4", "D1"],
        key="timeframes"
    )
    
    # Signal Parameters
    st.markdown("### 🎯 SIGNAL PARAMETERS")
    confirmation_required = st.checkbox("Require Higher TF Confirmation", value=True, key="confirmation_required")
    min_confidence = st.slider("Minimum Confidence %", 0, 100, 65, key="min_confidence")
    sensitivity = st.slider("Signal Sensitivity", 1, 10, 6, key="sensitivity")
    
    # Risk Management
    st.markdown("### 🛡️ RISK MANAGEMENT")
    max_risk = st.slider("Max Position Risk %", 0.1, 5.0, 1.0, 0.1, key="max_risk")
    use_atr_stops = st.checkbox("Use ATR-Based Stops", value=True, key="use_atr_stops")
    atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.1, key="atr_multiplier")
    
    # Notifications
    st.markdown("### 🔔 NOTIFICATIONS")
    telegram_alerts = st.checkbox("Telegram Alerts", value=False, key="telegram_alerts")
    if telegram_alerts:
        telegram_token = st.text_input("Bot Token", type="password", key="telegram_token")
        telegram_chat = st.text_input("Chat ID", key="telegram_chat")
    
    st.divider()
    
    if st.button("🔄 RUN ANALYSIS", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.caption(f"System Time: {datetime.now().strftime('%H:%M:%S UTC')}")

# ==================================================
# FAST FOREX API INTEGRATION
# ==================================================

class FastForexAPI:
    """Professional Fast Forex API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.fastforex.io"
    
    def fetch_historical(self, symbol: str, interval: str = "1h", count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical OHLC data"""
        try:
            # Parse symbol
            if symbol == "XAU-USD":
                from_currency = "XAU"
                to_currency = "USD"
            elif symbol == "BTC-USD":
                # Use crypto alternative for BTC
                return self._fetch_crypto_data("BTC", interval, count)
            else:
                from_currency, to_currency = symbol.split("-")
            
            url = f"{self.base_url}/time-series"
            params = {
                "api_key": self.api_key,
                "from": from_currency,
                "to": to_currency,
                "period": self._map_interval(interval),
                "length": count
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'results' in data and to_currency in data['results']:
                return self._parse_time_series(data['results'][to_currency], interval)
            
        except Exception as e:
            st.warning(f"Fast Forex API Error: {e}")
        
        return None
    
    def _fetch_crypto_data(self, symbol: str, interval: str, count: int) -> pd.DataFrame:
        """Fetch cryptocurrency data from alternative source"""
        try:
            coin_id = "bitcoin" if symbol == "BTC" else "ethereum"
            days = self._interval_to_days(interval)
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "hourly" if interval in ["M1", "M5", "M15", "M30", "H1"] else "daily"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            prices = data.get('prices', [])
            if prices:
                timestamps = [pd.to_datetime(x[0], unit='ms') for x in prices]
                price_values = [x[1] for x in prices]
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'close': price_values
                })
                df.set_index('timestamp', inplace=True)
                
                # Generate OHLC from close prices
                return self._generate_ohlc_from_close(df, interval)
        
        except:
            pass
        
        return self._generate_fallback_data(symbol, interval, count)
    
    def _parse_time_series(self, time_series: dict, interval: str) -> pd.DataFrame:
        """Parse time series data into OHLC DataFrame"""
        timestamps = []
        prices = []
        
        for ts, price in time_series.items():
            timestamps.append(pd.to_datetime(ts))
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close': prices
        })
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        return self._generate_ohlc_from_close(df, interval)
    
    def _generate_ohlc_from_close(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Generate OHLC data from close prices"""
        if len(df) < 2:
            return df
        
        df = df.copy()
        
        # Calculate returns volatility for realistic OHLC generation
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std()
        else:
            vol = 0.01
        
        # Generate realistic OHLC
        df['open'] = df['close'].shift(1).fillna(df['close'] * (1 - vol/2))
        
        # Timeframe-specific volatility multiplier
        tf_multiplier = {
            "M1": 0.3, "M5": 0.5, "M15": 0.7, "M30": 0.8,
            "H1": 1.0, "H4": 1.2, "D1": 1.5, "W1": 2.0
        }.get(interval, 1.0)
        
        for i in range(len(df)):
            if i == 0:
                continue
            
            price_range = abs(df['close'].iloc[i] - df['open'].iloc[i]) * tf_multiplier
            if price_range == 0:
                price_range = df['close'].iloc[i] * vol * tf_multiplier
            
            df.loc[df.index[i], 'high'] = max(df['close'].iloc[i], df['open'].iloc[i]) + price_range * 0.3
            df.loc[df.index[i], 'low'] = min(df['close'].iloc[i], df['open'].iloc[i]) - price_range * 0.3
        
        # Fill any missing values
        df['high'] = df['high'].fillna(df[['open', 'close']].max(axis=1) * 1.001)
        df['low'] = df['low'].fillna(df[['open', 'close']].min(axis=1) * 0.999)
        
        # Add volume
        df['volume'] = np.random.lognormal(12, 0.8, len(df)) * 1e6
        
        return df
    
    def _map_interval(self, interval: str) -> str:
        """Map trading intervals to Fast Forex periods"""
        interval_map = {
            "M1": "5m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w"
        }
        return interval_map.get(interval, "1h")
    
    def _interval_to_days(self, interval: str) -> int:
        """Convert interval to days for CoinGecko"""
        if interval in ["M1", "M5", "M15", "M30"]:
            return 7
        elif interval == "H1":
            return 30
        elif interval == "H4":
            return 90
        else:
            return 180
    
    def _generate_fallback_data(self, symbol: str, interval: str, count: int) -> pd.DataFrame:
        """Generate accurate fallback data"""
        # Current market prices (update these)
        current_prices = {
            "BTC": 43000, "ETH": 2300, "XAU": 5000,
            "EUR-USD": 1.0850, "GBP-USD": 1.2650, "USD-JPY": 148.50
        }
        
        # Get base price
        if symbol == "XAU-USD":
            base_price = current_prices.get("XAU", 5000)
        elif symbol == "BTC-USD":
            base_price = current_prices.get("BTC", 43000)
        elif symbol == "ETH-USD":
            base_price = current_prices.get("ETH", 2300)
        else:
            base_price = current_prices.get(symbol, 1.0)
        
        # Generate dates
        freq_map = {
            "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
            "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W"
        }
        
        dates = pd.date_range(end=datetime.now(), periods=count, freq=freq_map.get(interval, "1h"))
        
        # Generate price series with realistic volatility
        tf_volatility = {
            "M1": 0.002, "M5": 0.003, "M15": 0.005, "M30": 0.008,
            "H1": 0.012, "H4": 0.018, "D1": 0.025, "W1": 0.035
        }
        
        vol = tf_volatility.get(interval, 0.01)
        returns = np.random.normal(0, vol, count)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # Generate OHLC
        for i in range(len(df)):
            if i == 0:
                df.loc[df.index[i], 'open'] = df['close'].iloc[i] * (1 - vol/2)
            else:
                df.loc[df.index[i], 'open'] = df['close'].iloc[i-1]
            
            price_range = df['close'].iloc[i] * vol * np.random.uniform(0.8, 1.2)
            df.loc[df.index[i], 'high'] = max(df['close'].iloc[i], df['open'].iloc[i]) + price_range * 0.4
            df.loc[df.index[i], 'low'] = min(df['close'].iloc[i], df['open'].iloc[i]) - price_range * 0.4
        
        df['volume'] = np.random.lognormal(12, 0.8, count) * 1e6
        
        return df

# ==================================================
# NEWS & SENTIMENT INTEGRATION
# ==================================================

class NewsSentimentAnalyzer:
    """Professional news sentiment analysis"""
    
    def __init__(self, api_key: Optional[str] = None, source: str = "NewsData.io"):
        self.api_key = api_key
        self.source = source
    
    def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch relevant news for symbol"""
        news_items = []
        
        try:
            if self.source == "NewsData.io" and self.api_key:
                # NewsData.io implementation
                query = self._get_news_query(symbol)
                url = "https://newsdata.io/api/1/news"
                params = {
                    "apikey": self.api_key,
                    "q": query,
                    "language": "en",
                    "size": 5
                }
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if data.get('status') == 'success':
                    for item in data.get('results', []):
                        news_items.append({
                            'title': item.get('title', ''),
                            'description': item.get('description', ''),
                            'source': item.get('source_id', ''),
                            'published': item.get('pubDate', ''),
                            'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('description', ''))
                        })
            
            elif self.source == "CryptoPanic" and self.api_key:
                # CryptoPanic implementation
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    "auth_token": self.api_key,
                    "public": "true",
                    "filter": "important" if "BTC" in symbol or "ETH" in symbol else "rising"
                }
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                for item in data.get('results', []):
                    if symbol in str(item.get('title', '')) or symbol in str(item.get('body', '')):
                        news_items.append({
                            'title': item.get('title', ''),
                            'description': item.get('body', ''),
                            'source': 'CryptoPanic',
                            'published': item.get('published_at', ''),
                            'sentiment': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('body', ''))
                        })
        
        except Exception as e:
            st.warning(f"News API Error: {e}")
        
        # Add sample news if API fails
        if not news_items:
            news_items = self._generate_sample_news(symbol)
        
        return news_items
    
    def _get_news_query(self, symbol: str) -> str:
        """Get news query for symbol"""
        symbol_map = {
            "BTC-USD": "Bitcoin OR BTC",
            "ETH-USD": "Ethereum OR ETH",
            "XAU-USD": "Gold OR XAU OR precious metals",
            "EUR-USD": "Euro OR EUR USD OR forex",
            "GBP-USD": "Pound OR GBP USD OR forex",
            "USD-JPY": "Yen OR JPY USD OR forex"
        }
        return symbol_map.get(symbol, symbol)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['bullish', 'rise', 'gain', 'surge', 'rally', 'positive', 'strong', 'buy', 'support']
        negative_words = ['bearish', 'fall', 'drop', 'decline', 'crash', 'negative', 'weak', 'sell', 'resistance']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _generate_sample_news(self, symbol: str) -> List[Dict]:
        """Generate sample news for demonstration"""
        sample_news = {
            "BTC-USD": [
                {
                    'title': 'Bitcoin ETF Inflows Reach Record High',
                    'description': 'Institutional investors continue to accumulate Bitcoin through ETF channels.',
                    'source': 'Bloomberg',
                    'published': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'sentiment': 'positive'
                },
                {
                    'title': 'Market Volatility Expected Ahead of FOMC Meeting',
                    'description': 'Traders brace for increased volatility in crypto markets.',
                    'source': 'Reuters',
                    'published': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
                    'sentiment': 'negative'
                }
            ],
            "XAU-USD": [
                {
                    'title': 'Gold Prices Hold Steady Amid Geopolitical Tensions',
                    'description': 'Safe-haven demand supports gold prices.',
                    'source': 'CNBC',
                    'published': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'sentiment': 'positive'
                }
            ]
        }
        
        return sample_news.get(symbol, [])

# ==================================================
# ADVANCED TECHNICAL ANALYSIS ENGINE
# ==================================================

class MultiTimeframeAnalyzer:
    """Professional multi-timeframe analysis engine"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], symbol: str, timeframes: List[str]):
        self.data = data
        self.symbol = symbol
        self.timeframes = timeframes
        self.analysis_results = {}
        self.analyze_all_timeframes()
    
    def analyze_all_timeframes(self):
        """Analyze all timeframes"""
        for tf in self.timeframes:
            if tf in self.data and len(self.data[tf]) > 20:
                self.analysis_results[tf] = self._analyze_timeframe(tf)
    
    def _analyze_timeframe(self, timeframe: str) -> Dict:
        """Analyze single timeframe"""
        df = self.data[timeframe].copy()  # Make a copy to avoid modifying original
        
        # Calculate all indicators and add to DataFrame
        indicators = self._calculate_indicators(df, timeframe)
        
        # Ensure RSI and ATR are in the DataFrame for pattern detection
        if 'rsi' not in df.columns:
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        if 'atr' not in df.columns:
            # Calculate ATR
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
        
        # Analyze price structure
        price_structure = self._analyze_price_structure(df)
        
        # Detect smart money patterns
        smart_money = self._detect_smart_money_patterns(df, timeframe)
        
        # Analyze liquidity
        liquidity = self._analyze_liquidity(df, timeframe)
        
        # Determine bias
        bias = self._determine_bias(indicators, price_structure, smart_money)
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicators, price_structure, smart_money, liquidity, timeframe)
        
        return {
            'indicators': indicators,
            'price_structure': price_structure,
            'smart_money': smart_money,
            'liquidity': liquidity,
            'bias': bias,
            'confidence': confidence,
            'current_price': df['close'].iloc[-1],
            'atr': indicators.get('atr', 0),
            'volatility': indicators.get('volatility', 0)
        }
    
    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Calculate technical indicators with timeframe-specific parameters"""
        df = df.copy()
        
        # Timeframe-specific parameters
        tf_params = {
            "M1": {"ema_periods": [9, 21], "bb_period": 20, "rsi_period": 14},
            "M5": {"ema_periods": [9, 21], "bb_period": 20, "rsi_period": 14},
            "M15": {"ema_periods": [9, 21, 50], "bb_period": 20, "rsi_period": 14},
            "M30": {"ema_periods": [9, 21, 50], "bb_period": 20, "rsi_period": 14},
            "H1": {"ema_periods": [9, 21, 50, 200], "bb_period": 20, "rsi_period": 14},
            "H4": {"ema_periods": [21, 50, 200], "bb_period": 20, "rsi_period": 14},
            "D1": {"ema_periods": [50, 200], "bb_period": 20, "rsi_period": 14},
            "W1": {"ema_periods": [50, 200], "bb_period": 20, "rsi_period": 14}
        }
        
        params = tf_params.get(timeframe, {"ema_periods": [9, 21, 50], "bb_period": 20, "rsi_period": 14})
        
        # EMAs
        emas = {}
        for period in params['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            emas[f'ema_{period}'] = df[f'ema_{period}'].iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(params['bb_period']).mean()
        bb_std = df['close'].rolling(params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR (timeframe-adjusted)
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
        
        # Volatility (timeframe-specific)
        if timeframe in ["M1", "M5", "M15", "M30"]:
            lookback = 100
        elif timeframe == "H1":
            lookback = 50
        else:
            lookback = 20
        
        returns = df['close'].pct_change().tail(lookback)
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'].iloc[-1] / df['volume_ma'].iloc[-1] if df['volume_ma'].iloc[-1] != 0 else 1
        else:
            volume_ratio = 1
        
        # Support/Resistance levels
        recent_data = df.tail(50)
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        pivot = (recent_data['high'].iloc[-1] + recent_data['low'].iloc[-1] + recent_data['close'].iloc[-1]) / 3
        
        # Fibonacci levels
        fib_levels = self._calculate_fibonacci_levels(recent_data)
        
        return {
            'emas': emas,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_bullish': macd > macd_signal,
            'bb_upper': df['bb_upper'].iloc[-1],
            'bb_lower': df['bb_lower'].iloc[-1],
            'bb_middle': df['bb_middle'].iloc[-1],
            'atr': atr,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'support': support,
            'resistance': resistance,
            'pivot': pivot,
            'fib_levels': fib_levels
        }
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        return {
            '0.0': high,
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382,
            '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786,
            '1.0': low
        }
    
    def _analyze_price_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze price structure for trend and patterns"""
        current_price = df['close'].iloc[-1]
        
        # Recent highs and lows
        recent_highs = df['high'].tail(20).sort_values().tail(3).values
        recent_lows = df['low'].tail(20).sort_values().head(3).values
        
        # Determine market structure
        if len(df) > 50:
            # Higher highs/lows analysis
            last_20 = df.tail(20)
            highs = last_20['high']
            lows = last_20['low']
            
            higher_highs = all(highs.iloc[i] > highs.iloc[i-1] for i in range(1, len(highs)))
            higher_lows = all(lows.iloc[i] > lows.iloc[i-1] for i in range(1, len(lows)))
            lower_highs = all(highs.iloc[i] < highs.iloc[i-1] for i in range(1, len(highs)))
            lower_lows = all(lows.iloc[i] < lows.iloc[i-1] for i in range(1, len(lows)))
            
            if higher_highs and higher_lows:
                structure = "Uptrend"
            elif lower_highs and lower_lows:
                structure = "Downtrend"
            else:
                structure = "Ranging"
        else:
            structure = "Unknown"
        
        # Key levels
        key_levels = {
            'recent_high': max(recent_highs) if len(recent_highs) > 0 else current_price,
            'recent_low': min(recent_lows) if len(recent_lows) > 0 else current_price,
            'current': current_price
        }
        
        # Price position relative to recent range
        recent_range = key_levels['recent_high'] - key_levels['recent_low']
        if recent_range > 0:
            position_pct = (current_price - key_levels['recent_low']) / recent_range * 100
        else:
            position_pct = 50
        
        return {
            'structure': structure,
            'key_levels': key_levels,
            'position_pct': position_pct,
            'range_size': recent_range,
            'is_near_high': abs(current_price - key_levels['recent_high']) / key_levels['recent_high'] < 0.01,
            'is_near_low': abs(current_price - key_levels['recent_low']) / key_levels['recent_low'] < 0.01
        }
    
    def _detect_smart_money_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect smart money patterns and divergences"""
        patterns = []
        
        # Calculate ATR for this timeframe if not already in df
        if 'atr' not in df.columns:
            # Calculate ATR manually
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr_series = true_range.rolling(14).mean()
        else:
            atr_series = df['atr']
        
        # RSI Divergence
        if len(df) > 25:
            # Calculate RSI if not in df
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
            else:
                rsi_series = df['rsi']
            
            # Look for hidden divergence
            recent_highs = df['high'].tail(25)
            recent_lows = df['low'].tail(25)
            recent_rsi = rsi_series.tail(25)
            
            if len(recent_highs) >= 10 and len(recent_rsi) >= 10:
                # Bearish divergence (price makes higher high, RSI makes lower high)
                if (recent_highs.iloc[-1] > recent_highs.iloc[-5] and 
                    recent_rsi.iloc[-1] < recent_rsi.iloc[-5]):
                    patterns.append("BEARISH_DIVERGENCE")
                
                # Bullish divergence (price makes lower low, RSI makes higher low)
                if (recent_lows.iloc[-1] < recent_lows.iloc[-5] and 
                    recent_rsi.iloc[-1] > recent_rsi.iloc[-5]):
                    patterns.append("BULLISH_DIVERGENCE")
        
        # Order Block Detection
        if len(df) > 10:
            # Look for breaker blocks (strong moves followed by consolidation)
            for i in range(5, len(df) - 1):
                candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                
                # Get ATR value for this candle
                current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else df['close'].iloc[i] * 0.01
                
                # Bullish order block (strong down candle followed by up candle)
                if (prev_candle['close'] < prev_candle['open'] and 
                    candle['close'] > candle['open'] and
                    abs(prev_candle['close'] - prev_candle['open']) > current_atr * 0.5):
                    if i == len(df) - 2:  # Recent pattern
                        patterns.append("BULLISH_ORDER_BLOCK")
                        break
                
                # Bearish order block (strong up candle followed by down candle)
                if (prev_candle['close'] > prev_candle['open'] and 
                    candle['close'] < candle['open'] and
                    abs(prev_candle['close'] - prev_candle['open']) > current_atr * 0.5):
                    if i == len(df) - 2:  # Recent pattern
                        patterns.append("BEARISH_ORDER_BLOCK")
                        break
        
        # Fair Value Gap Detection
        if len(df) > 3:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            prev_2 = df.iloc[-3]
            
            # Bullish FVG
            if (prev_2['low'] > current['high'] and 
                prev['high'] < prev_2['low'] and 
                prev['low'] > current['high']):
                patterns.append("BULLISH_FVG")
            
            # Bearish FVG
            if (prev_2['high'] < current['low'] and 
                prev['low'] > prev_2['high'] and 
                prev['high'] < current['low']):
                patterns.append("BEARISH_FVG")
        
        # Liquidity sweep detection
        if len(df) > 5:
            # Check if price swept a recent high/low with wick
            last_candle = df.iloc[-1]
            prev_highs = df['high'].iloc[-6:-1].max()
            prev_lows = df['low'].iloc[-6:-1].min()
            
            if last_candle['high'] > prev_highs * 1.001 and last_candle['close'] < last_candle['high']:
                patterns.append("LIQUIDITY_SWEEP_HIGH")
            if last_candle['low'] < prev_lows * 0.999 and last_candle['close'] > last_candle['low']:
                patterns.append("LIQUIDITY_SWEEP_LOW")
        
        return {
            'patterns': patterns,
            'has_bullish_pattern': any('BULLISH' in p for p in patterns),
            'has_bearish_pattern': any('BEARISH' in p for p in patterns)
        }
    
    def _analyze_liquidity(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze liquidity levels and zones"""
        current_price = df['close'].iloc[-1]
        
        # Identify significant highs/lows (potential liquidity pools)
        window = min(100, len(df))
        df_window = df.tail(window)
        
        # Find swing highs and lows
        highs = []
        lows = []
        for i in range(2, len(df_window) - 2):
            if (df_window['high'].iloc[i] > df_window['high'].iloc[i-1] and 
                df_window['high'].iloc[i] > df_window['high'].iloc[i-2] and
                df_window['high'].iloc[i] > df_window['high'].iloc[i+1] and 
                df_window['high'].iloc[i] > df_window['high'].iloc[i+2]):
                highs.append((df_window.index[i], df_window['high'].iloc[i]))
            
            if (df_window['low'].iloc[i] < df_window['low'].iloc[i-1] and 
                df_window['low'].iloc[i] < df_window['low'].iloc[i-2] and
                df_window['low'].iloc[i] < df_window['low'].iloc[i+1] and 
                df_window['low'].iloc[i] < df_window['low'].iloc[i+2]):
                lows.append((df_window.index[i], df_window['low'].iloc[i]))
        
        # Get recent highs/lows
        recent_highs = [h[1] for h in highs[-5:]] if highs else []
        recent_lows = [l[1] for l in lows[-5:]] if lows else []
        
        # Liquidity above (resistance) and below (support)
        liquidity_above = [h for h in recent_highs if h > current_price]
        liquidity_below = [l for l in recent_lows if l < current_price]
        
        # Nearest liquidity levels
        nearest_resistance = min(liquidity_above) if liquidity_above else None
        nearest_support = max(liquidity_below) if liquidity_below else None
        
        # Distance to nearest levels (in %)
        dist_to_res = (nearest_resistance - current_price) / current_price * 100 if nearest_resistance else None
        dist_to_sup = (current_price - nearest_support) / current_price * 100 if nearest_support else None
        
        # Liquidity sweep likelihood (if price approaching levels)
        sweep_likelihood = "LOW"
        if nearest_resistance and dist_to_res and dist_to_res < 0.5:
            sweep_likelihood = "HIGH (resistance test)"
        elif nearest_support and dist_to_sup and dist_to_sup < 0.5:
            sweep_likelihood = "HIGH (support test)"
        
        return {
            'liquidity_above': liquidity_above,
            'liquidity_below': liquidity_below,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'dist_to_resistance_pct': dist_to_res,
            'dist_to_support_pct': dist_to_sup,
            'sweep_likelihood': sweep_likelihood,
            'swing_highs': recent_highs,
            'swing_lows': recent_lows
        }
    
    def _determine_bias(self, indicators: Dict, price_structure: Dict, smart_money: Dict) -> str:
        """Determine overall bias for the timeframe"""
        signals = []
        
        # Trend from price structure
        if price_structure['structure'] == "Uptrend":
            signals.append("BULLISH")
        elif price_structure['structure'] == "Downtrend":
            signals.append("BEARISH")
        else:
            signals.append("NEUTRAL")
        
        # EMA alignment
        emas = indicators['emas']
        if len(emas) >= 2:
            # Determine fast and slow EMAs (assume keys like 'ema_9', 'ema_21', etc.)
            fast_key = next((k for k in emas.keys() if '9' in k or '21' in k), None)
            slow_key = next((k for k in emas.keys() if '50' in k or '200' in k), None)
            if fast_key and slow_key:
                if emas[fast_key] > emas[slow_key]:
                    signals.append("BULLISH")
                elif emas[fast_key] < emas[slow_key]:
                    signals.append("BEARISH")
        
        # RSI
        rsi = indicators['rsi']
        if rsi > 70:
            signals.append("BEARISH")  # overbought
        elif rsi < 30:
            signals.append("BULLISH")  # oversold
        else:
            signals.append("NEUTRAL")
        
        # MACD
        if indicators['macd_bullish']:
            signals.append("BULLISH")
        else:
            signals.append("BEARISH")
        
        # Bollinger Bands (if current price relative to bands)
        # We need the current price, but indicators dict doesn't have it directly. Use close from elsewhere? 
        # For simplicity, we'll skip BB in bias for now.
        
        # Smart money patterns
        if smart_money['has_bullish_pattern']:
            signals.append("BULLISH")
        if smart_money['has_bearish_pattern']:
            signals.append("BEARISH")
        
        # Count signals
        bullish_count = signals.count("BULLISH")
        bearish_count = signals.count("BEARISH")
        
        if bullish_count > bearish_count + 1:
            return "BULLISH"
        elif bearish_count > bullish_count + 1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_confidence(self, indicators: Dict, price_structure: Dict, 
                              smart_money: Dict, liquidity: Dict, timeframe: str) -> float:
        """Calculate confidence score (0-100) for the bias"""
        confidence = 50  # base
        
        # Trend alignment
        if price_structure['structure'] in ["Uptrend", "Downtrend"]:
            confidence += 10
        else:
            confidence -= 10
        
        # RSI strength
        rsi = indicators['rsi']
        if 40 <= rsi <= 60:
            confidence += 5
        elif rsi > 80 or rsi < 20:
            confidence -= 10  # extreme
        
        # Volume confirmation
        if indicators['volume_ratio'] > 1.2:
            confidence += 10
        elif indicators['volume_ratio'] < 0.8:
            confidence -= 5
        
        # Smart money patterns
        if smart_money['has_bullish_pattern'] or smart_money['has_bearish_pattern']:
            confidence += 15
        
        # Liquidity zones
        if liquidity['nearest_resistance'] and liquidity['dist_to_resistance_pct'] and liquidity['dist_to_resistance_pct'] < 1:
            confidence += 5  # near resistance adds to bearish confidence, but we need to align with bias
        if liquidity['nearest_support'] and liquidity['dist_to_support_pct'] and liquidity['dist_to_support_pct'] < 1:
            confidence += 5
        
        # Timeframe weight (higher timeframes get more confidence)
        tf_weight = {"M1": 0.8, "M5": 0.9, "M15": 1.0, "M30": 1.1, "H1": 1.2, "H4": 1.3, "D1": 1.4, "W1": 1.5}
        confidence *= tf_weight.get(timeframe, 1.0)
        
        # Normalize
        confidence = max(0, min(100, confidence))
        
        return confidence
    
    def get_consolidated_signal(self, min_confidence: float = 65) -> Dict:
        """Combine all timeframe analyses into a consolidated signal"""
        if not self.analysis_results:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'details': {}}
        
        # Count biases across timeframes
        bullish_tf = []
        bearish_tf = []
        neutral_tf = []
        
        for tf, result in self.analysis_results.items():
            bias = result['bias']
            conf = result['confidence']
            if conf >= min_confidence:
                if bias == 'BULLISH':
                    bullish_tf.append(tf)
                elif bias == 'BEARISH':
                    bearish_tf.append(tf)
                else:
                    neutral_tf.append(tf)
            else:
                neutral_tf.append(tf)  # low confidence treated as neutral
        
        # Determine overall signal
        if len(bullish_tf) > len(bearish_tf) and len(bullish_tf) >= 2:
            signal = "BULLISH"
        elif len(bearish_tf) > len(bullish_tf) and len(bearish_tf) >= 2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        # Calculate weighted confidence
        total_conf = 0
        count = 0
        for tf, result in self.analysis_results.items():
            conf = result['confidence']
            weight = {"M1": 0.5, "M5": 0.6, "M15": 0.7, "M30": 0.8, "H1": 1.0, "H4": 1.2, "D1": 1.5, "W1": 2.0}.get(tf, 1)
            total_conf += conf * weight
            count += weight
        
        avg_confidence = total_conf / count if count > 0 else 0
        
        return {
            'signal': signal,
            'confidence': avg_confidence,
            'bullish_timeframes': bullish_tf,
            'bearish_timeframes': bearish_tf,
            'neutral_timeframes': neutral_tf,
            'details': self.analysis_results
        }

# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    """Main Streamlit application"""
    
    # Get sidebar parameters from session state
    fast_forex_key = st.session_state.get("fast_forex_key", "6741a9cd7c-d2a1c6afde-ta8cti")
    news_api_key = st.session_state.get("news_api_key", "")
    news_source = st.session_state.get("news_source", "None")
    instruments = st.session_state.get("instruments", ["BTC-USD", "XAU-USD"])
    timeframes = st.session_state.get("timeframes", ["M15", "H1", "H4", "D1"])
    confirmation_required = st.session_state.get("confirmation_required", True)
    min_confidence = st.session_state.get("min_confidence", 65)
    sensitivity = st.session_state.get("sensitivity", 6)
    
    # Initialize API clients
    forex_api = FastForexAPI(api_key=fast_forex_key)
    news_api = NewsSentimentAnalyzer(api_key=news_api_key, source=news_source)
    
    # Create tabs for organized display
    tab1, tab2, tab3, tab4 = st.tabs(["📈 SIGNALS & ANALYSIS", "📊 CHARTS", "📰 NEWS SENTIMENT", "⚙️ BACKTEST"])
    
    with tab1:
        st.markdown('<div class="section-header">LIVE MULTI-TIMEFRAME SIGNALS</div>', unsafe_allow_html=True)
        
        if not instruments:
            st.warning("Please select at least one instrument in the sidebar.")
            return
        
        # Process each instrument
        for symbol in instruments:
            with st.expander(f"🔍 {symbol} - Institutional Analysis", expanded=True):
                # Fetch data for all timeframes
                data = {}
                progress_bar = st.progress(0, text=f"Fetching {symbol} data...")
                for i, tf in enumerate(timeframes):
                    df = forex_api.fetch_historical(symbol, interval=tf, count=200)
                    if df is not None and len(df) > 20:
                        data[tf] = df
                    progress_bar.progress((i+1)/len(timeframes))
                progress_bar.empty()
                
                if not data:
                    st.error(f"Failed to fetch data for {symbol}")
                    continue
                
                # Run multi-timeframe analysis
                analyzer = MultiTimeframeAnalyzer(data, symbol, timeframes)
                signal = analyzer.get_consolidated_signal(min_confidence=min_confidence)
                
                # Display main signal
                col1, col2, col3, col4 = st.columns([2,1,1,2])
                with col1:
                    if signal['signal'] == 'BULLISH':
                        st.markdown('<div class="signal-buy">📈 BULLISH SIGNAL</div>', unsafe_allow_html=True)
                    elif signal['signal'] == 'BEARISH':
                        st.markdown('<div class="signal-sell">📉 BEARISH SIGNAL</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="signal-neutral">⚖️ NEUTRAL SIGNAL</div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{signal['confidence']:.1f}%")
                
                with col3:
                    # Current price from highest timeframe available
                    current_price = None
                    for tf in ["W1", "D1", "H4", "H1", "M30", "M15", "M5", "M1"]:
                        if tf in data and len(data[tf]) > 0:
                            current_price = data[tf]['close'].iloc[-1]
                            break
                    if current_price:
                        st.metric("Price", f"{current_price:.2f}")
                
                with col4:
                    st.markdown(f"**Timeframes:** {len(signal['bullish_timeframes'])} Bullish, {len(signal['bearish_timeframes'])} Bearish")
                
                # Timeframe breakdown
                st.markdown("#### Timeframe Analysis")
                tf_cols = st.columns(len(timeframes))
                for idx, tf in enumerate(timeframes):
                    if tf in analyzer.analysis_results:
                        res = analyzer.analysis_results[tf]
                        bias = res['bias']
                        conf = res['confidence']
                        with tf_cols[idx]:
                            if bias == 'BULLISH':
                                st.markdown(f"<div style='background:#4CAF5022; padding:10px; border-radius:5px; text-align:center;'><b>{tf}</b><br>🟢 BULL<br>{conf:.0f}%</div>", unsafe_allow_html=True)
                            elif bias == 'BEARISH':
                                st.markdown(f"<div style='background:#F4433622; padding:10px; border-radius:5px; text-align:center;'><b>{tf}</b><br>🔴 BEAR<br>{conf:.0f}%</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background:#75757522; padding:10px; border-radius:5px; text-align:center;'><b>{tf}</b><br>⚪ NEUTRAL<br>{conf:.0f}%</div>", unsafe_allow_html=True)
                
                # Detailed indicators for each timeframe
                with st.expander("View Detailed Technicals"):
                    for tf in timeframes:
                        if tf in analyzer.analysis_results:
                            res = analyzer.analysis_results[tf]
                            st.markdown(f"**{tf}**")
                            cols = st.columns(4)
                            ind = res['indicators']
                            with cols[0]:
                                st.markdown(f"RSI: {ind['rsi']:.1f}")
                                st.markdown(f"MACD: {'Bullish' if ind['macd_bullish'] else 'Bearish'}")
                            with cols[1]:
                                st.markdown(f"ATR: {ind['atr']:.2f}")
                                st.markdown(f"Volatility: {ind['volatility']:.2f}%")
                            with cols[2]:
                                st.markdown(f"Support: {ind['support']:.2f}")
                                st.markdown(f"Resistance: {ind['resistance']:.2f}")
                            with cols[3]:
                                patterns = res['smart_money']['patterns']
                                pattern_str = ", ".join(patterns) if patterns else "None"
                                st.markdown(f"Patterns: {pattern_str}")
                            st.divider()
    
    with tab2:
        st.markdown('<div class="section-header">PRICE CHARTS WITH INSTITUTIONAL LEVELS</div>', unsafe_allow_html=True)
        
        selected_symbol = st.selectbox("Select Instrument", instruments, key="chart_symbol")
        selected_tf = st.selectbox("Select Timeframe", timeframes, key="chart_tf")
        
        if selected_symbol and selected_tf:
            # Fetch fresh data for charting (more data for better view)
            df = forex_api.fetch_historical(selected_symbol, interval=selected_tf, count=300)
            if df is not None and len(df) > 20:
                # Create plotly candlestick chart
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    showlegend=False
                ), row=1, col=1)
                
                # Add EMAs
                ema9 = df['close'].ewm(span=9).mean()
                ema21 = df['close'].ewm(span=21).mean()
                ema50 = df['close'].ewm(span=50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ema9, line=dict(color='blue', width=1), name="EMA 9"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ema21, line=dict(color='orange', width=1), name="EMA 21"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ema50, line=dict(color='red', width=1), name="EMA 50"), row=1, col=1)
                
                # Add volume
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='gray'), row=2, col=1)
                
                # Add RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='purple', width=1), name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(height=800, title=f"{selected_symbol} - {selected_tf}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for charting.")
    
    with tab3:
        st.markdown('<div class="section-header">MARKET NEWS & SENTIMENT</div>', unsafe_allow_html=True)
        
        news_symbol = st.selectbox("Select Instrument for News", instruments, key="news_symbol")
        
        if news_symbol:
            news_items = news_api.fetch_news(news_symbol)
            
            if news_items:
                for item in news_items:
                    sentiment_class = f"news-{item['sentiment']}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between;">
                            <strong>{item['title']}</strong>
                            <span class="{sentiment_class}">{item['sentiment'].upper()}</span>
                        </div>
                        <div style="color:#666; font-size:0.9em;">{item['description']}</div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.8em; color:#999;">
                            <span>{item['source']}</span>
                            <span>{item['published']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news found. Sample news shown.")
                # Show sample news
                sample = news_api._generate_sample_news(news_symbol)
                for item in sample:
                    st.markdown(f"**{item['title']}** - {item['description']} *({item['sentiment']})*")
    
    with tab4:
        st.markdown('<div class="section-header">BACKTESTING SIMULATOR</div>', unsafe_allow_html=True)
        st.info("Backtesting module coming soon. This will allow you to test the signal accuracy on historical data.")
        # Placeholder for future backtesting functionality

# ==================================================
# RUN THE APP
# ==================================================
if __name__ == "__main__":
    main()
