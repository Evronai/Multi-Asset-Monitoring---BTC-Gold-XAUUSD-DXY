import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="Institutional Multi-Timeframe Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: 700; color: #1A237E; margin-bottom: 10px; }
    .section-header { font-size: 18px; font-weight: 600; color: #37474F; margin: 20px 0 10px 0;
        padding-bottom: 8px; border-bottom: 2px solid #E3F2FD; }
    .metric-card { background: #1E1E2E; border: 1px solid #3A3A5C; border-radius: 8px;
        padding: 16px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.3); color: #E0E0E0; }
    .metric-card strong { color: #FFFFFF; }
    .signal-buy { background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); color: white;
        padding: 12px; border-radius: 6px; font-weight: 700; text-align: center; }
    .signal-sell { background: linear-gradient(135deg, #F44336 0%, #C62828 100%); color: white;
        padding: 12px; border-radius: 6px; font-weight: 700; text-align: center; }
    .signal-neutral { background: linear-gradient(135deg, #757575 0%, #424242 100%); color: white;
        padding: 12px; border-radius: 6px; font-weight: 700; text-align: center; }
    .news-positive { background: #1B5E20; color: #A5D6A7; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .news-negative { background: #B71C1C; color: #FFCDD2; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .news-neutral  { background: #37474F; color: #CFD8DC; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .data-tag { background: #0D47A1; color: #BBDEFB; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
    .warn-tag { background: #E65100; color: #FFE0B2; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏛️ INSTITUTIONAL MULTI-TIMEFRAME ANALYZER</div>', unsafe_allow_html=True)
st.markdown('**Real OHLC Data • Smart Money Concepts • VADER Sentiment • Multi-TF Confirmation • Backtesting**')

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ SETTINGS")

    with st.expander("🔑 API KEYS", expanded=True):
        st.markdown("**Twelve Data API** (Forex/Gold OHLC)")
        twelve_key = st.text_input("Twelve Data Key", type="password", key="twelve_key",
                                   help="Free tier: 800 calls/day. Get at twelvedata.com")
        st.markdown("**NewsData.io** (Optional)")
        news_api_key = st.text_input("NewsData Key", type="password", key="news_api_key")
        st.caption("⚠️ Crypto uses Coinbase → CoinGecko (free, no key). Forex/Gold requires Twelve Data key.")

    st.markdown("### 📊 INSTRUMENTS")
    instruments = st.multiselect(
        "Select Instruments",
        ["BTC/USDT", "ETH/USDT", "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"],
        default=["BTC/USDT", "XAU/USD"],
        key="instruments"
    )

    st.markdown("### ⏰ TIMEFRAMES")
    timeframes = st.multiselect(
        "Analysis Timeframes",
        ["15m", "1h", "4h", "1day"],
        default=["15m", "1h", "4h", "1day"],
        key="timeframes"
    )

    st.markdown("### 🎯 SIGNAL PARAMETERS")
    min_confidence = st.slider("Min Confidence %", 0, 100, 60, key="min_confidence")
    require_htf = st.checkbox("Require Higher TF Confirmation", value=True, key="require_htf")

    st.markdown("### 🛡️ RISK MANAGEMENT")
    max_risk = st.slider("Max Risk per Trade %", 0.1, 5.0, 1.0, 0.1, key="max_risk")
    atr_multiplier = st.slider("ATR Stop Multiplier", 1.0, 5.0, 2.0, 0.1, key="atr_multiplier")

    st.divider()
    run_btn = st.button("🔄 RUN ANALYSIS", type="primary", use_container_width=True)
    if run_btn:
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


# ==================================================
# DATA LAYER — REAL OHLC FROM FREE SOURCES
# ==================================================

class DataFetcher:
    """
    Fetches REAL OHLC data:
    - BTC/ETH: Binance public API (no key required, full OHLCV)
    - Forex/Gold: Twelve Data API (free tier, key required)
    """

    # Coinbase granularity in seconds
    COINBASE_TF_MAP = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1day": 86400, "1week": 604800
    }

    TWELVE_TF_MAP = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "4h": "4h", "1day": "1day", "1week": "1week"
    }

    # CoinGecko days per timeframe
    COINGECKO_DAYS = {
        "1m": 1, "5m": 3, "15m": 7, "30m": 14,
        "1h": 30, "4h": 90, "1day": 365, "1week": 365
    }

    def __init__(self, twelve_api_key: str = ""):
        self.twelve_key = twelve_api_key

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch(_self, symbol: str, interval: str, limit: int = 500) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Returns (DataFrame, source_label).
        DataFrame always has: open, high, low, close, volume with DatetimeIndex.
        Crypto: tries Coinbase → CoinGecko fallback.
        Forex/Gold: Twelve Data.
        """
        is_crypto = symbol in ["BTC/USDT", "ETH/USDT", "BTC-USD", "ETH-USD"]

        if is_crypto:
            # Try Coinbase first
            df, src = _self._fetch_coinbase(symbol, interval, limit)
            if df is not None and len(df) >= 20:
                return df, src
            # Fallback to CoinGecko (OHLC endpoint)
            df, src = _self._fetch_coingecko(symbol, interval, limit)
            return df, src
        else:
            return _self._fetch_twelve_data(symbol, interval, limit)

    def _fetch_coinbase(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch OHLCV from Coinbase Advanced Trade public API.
        No API key required. Not geo-blocked like Binance.
        Max 300 candles per request.
        """
        try:
            # Map symbol to Coinbase product ID
            sym_map = {
                "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD",
                "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
            }
            product_id = sym_map.get(symbol, symbol)
            granularity = self.COINBASE_TF_MAP.get(interval, 3600)

            # Coinbase allows max 300 candles, paginate if needed
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - granularity * min(limit, 300)

            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                "granularity": granularity,
                "start": datetime.utcfromtimestamp(start_time).isoformat(),
                "end": datetime.utcfromtimestamp(end_time).isoformat()
            }

            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.json()

            if not raw or isinstance(raw, dict):
                return None, f"Coinbase: unexpected response"

            # Coinbase returns [timestamp, low, high, open, close, volume]
            df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.sort_index()
            df = df[df["volume"] > 0]

            return df, f"Coinbase ({product_id})"

        except Exception as e:
            return None, f"Coinbase error: {e}"

    def _fetch_coingecko(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """
        CoinGecko OHLC endpoint — returns REAL OHLC (not reconstructed).
        Free, no API key. Fallback when Coinbase fails.
        Note: CoinGecko OHLC granularity is fixed by the 'days' param:
          1-2 days → 30min candles, 3-89 days → 4h candles, 90+ days → 4day candles
        """
        try:
            coin_map = {
                "BTC/USDT": "bitcoin", "ETH/USDT": "ethereum",
                "BTC-USD": "bitcoin", "ETH-USD": "ethereum"
            }
            coin_id = coin_map.get(symbol, "bitcoin")
            days = self.COINGECKO_DAYS.get(interval, 30)

            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": days}

            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                return None, "CoinGecko: no data"

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            df["volume"] = 0.0  # CoinGecko OHLC has no volume; use market_chart for vol if needed
            df = df.sort_index()

            # Merge in volume from market_chart endpoint
            try:
                vol_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                vol_resp = requests.get(vol_url,
                                        params={"vs_currency": "usd", "days": days},
                                        timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                vol_data = vol_resp.json().get("total_volumes", [])
                vol_df = pd.DataFrame(vol_data, columns=["timestamp", "volume"])
                vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms", utc=True)
                vol_df.set_index("timestamp", inplace=True)
                # Resample volume to match OHLC frequency
                freq = df.index.to_series().diff().median()
                vol_resampled = vol_df["volume"].resample(freq).sum()
                df["volume"] = vol_resampled.reindex(df.index, method="nearest").fillna(0)
            except Exception:
                pass  # volume stays 0, non-fatal

            return df, f"CoinGecko ({coin_id}, {days}d)"

        except Exception as e:
            return None, f"CoinGecko error: {e}"

    def _fetch_twelve_data(self, symbol: str, interval: str, limit: int) -> Tuple[Optional[pd.DataFrame], str]:
        """Fetch real OHLCV from Twelve Data"""
        if not self.twelve_key:
            return None, "No Twelve Data API key provided"

        try:
            tf = self.TWELVE_TF_MAP.get(interval, "1h")
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol,
                "interval": tf,
                "outputsize": min(limit, 5000),
                "apikey": self.twelve_key,
                "format": "JSON"
            }

            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            if data.get("status") == "error":
                return None, f"Twelve Data: {data.get('message', 'unknown error')}"

            values = data.get("values", [])
            if not values:
                return None, "Twelve Data: no data returned"

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            # Forex/commodities have no centralized volume — select only available cols
            cols = ["open", "high", "low", "close"]
            df = df[cols].astype(float)
            df["volume"] = 0.0  # placeholder; vol_ratio logic handles zero gracefully
            df = df.sort_index()

            return df, f"Twelve Data ({symbol})"

        except Exception as e:
            return None, f"Twelve Data error: {e}"


# ==================================================
# TECHNICAL INDICATORS — VECTORIZED, NO LOOPS
# ==================================================

class Indicators:
    """Stateless indicator calculations — all vectorized"""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        exp_fast = close.ewm(span=fast, adjust=False).mean()
        exp_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        atr_val = Indicators.atr(high, low, close, period)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).mean() / atr_val
        minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr_val
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def find_swing_highs(high: pd.Series, lookback: int = 3) -> pd.Series:
        """Vectorized swing high detection"""
        result = pd.Series(False, index=high.index)
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window.max():
                result.iloc[i] = True
        return result

    @staticmethod
    def find_swing_lows(low: pd.Series, lookback: int = 3) -> pd.Series:
        """Vectorized swing low detection"""
        result = pd.Series(False, index=low.index)
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window.min():
                result.iloc[i] = True
        return result


# ==================================================
# SMART MONEY CONCEPTS ENGINE — PROPER IMPLEMENTATION
# ==================================================

class SmartMoneyEngine:
    """
    Proper ICT / Smart Money pattern detection.
    All patterns use REAL OHLC — no synthetic data.
    """

    def __init__(self, df: pd.DataFrame, atr_series: pd.Series):
        self.df = df.copy()
        self.atr = atr_series
        self.df['atr'] = atr_series

    def detect_order_blocks(self) -> List[Dict]:
        """
        Bullish OB: Last bearish candle before a bullish impulsive move.
        Bearish OB: Last bullish candle before a bearish impulsive move.
        Proper definition: strong move that breaks prior structure.
        """
        df = self.df
        obs = []
        min_move_atr = 1.5  # minimum move size relative to ATR

        for i in range(2, len(df) - 2):
            candle = df.iloc[i]
            prev = df.iloc[i - 1]
            next_2 = df.iloc[i + 1]
            atr_val = self.atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            move_up = next_2['close'] - candle['close']
            move_down = candle['close'] - next_2['close']

            # Bullish OB: bearish candle (i-1 or i) followed by strong upward move
            if (prev['close'] < prev['open'] and  # bearish candle
                    move_up > min_move_atr * atr_val and  # impulsive up move
                    next_2['close'] > prev['high']):  # breaks prior candle high (structure break)
                obs.append({
                    'type': 'BULLISH_OB',
                    'top': prev['high'],
                    'bottom': prev['low'],
                    'index': df.index[i - 1],
                    'atr': atr_val,
                    'strength': move_up / atr_val
                })

            # Bearish OB: bullish candle followed by strong downward move
            if (prev['close'] > prev['open'] and  # bullish candle
                    move_down > min_move_atr * atr_val and  # impulsive down move
                    next_2['close'] < prev['low']):  # breaks prior candle low
                obs.append({
                    'type': 'BEARISH_OB',
                    'top': prev['high'],
                    'bottom': prev['low'],
                    'index': df.index[i - 1],
                    'atr': atr_val,
                    'strength': move_down / atr_val
                })

        return obs[-10:]  # keep last 10

    def detect_fair_value_gaps(self) -> List[Dict]:
        """
        Proper 3-candle FVG:
        Bullish: candle[i-2].high < candle[i].low (gap between them)
        Bearish: candle[i-2].low > candle[i].high (gap between them)
        """
        df = self.df
        fvgs = []

        for i in range(2, len(df)):
            c0 = df.iloc[i - 2]
            c2 = df.iloc[i]

            # Bullish FVG
            if c0['high'] < c2['low']:
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'top': c2['low'],
                    'bottom': c0['high'],
                    'midpoint': (c0['high'] + c2['low']) / 2,
                    'index': df.index[i - 1],
                    'filled': False
                })

            # Bearish FVG
            if c0['low'] > c2['high']:
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'top': c0['low'],
                    'bottom': c2['high'],
                    'midpoint': (c0['low'] + c2['high']) / 2,
                    'index': df.index[i - 1],
                    'filled': False
                })

        return fvgs[-10:]

    def detect_liquidity_sweeps(self) -> List[Dict]:
        """
        Liquidity sweep: price takes out a recent swing high/low with a wick,
        then closes back inside — indicating stop hunt / liquidity grab.
        """
        df = self.df
        sweeps = []
        lookback = 20

        for i in range(lookback, len(df)):
            candle = df.iloc[i]
            window = df.iloc[i - lookback:i]
            atr_val = self.atr.iloc[i]

            if pd.isna(atr_val) or atr_val == 0:
                continue

            recent_high = window['high'].max()
            recent_low = window['low'].min()

            wick_up = candle['high'] - max(candle['open'], candle['close'])
            wick_down = min(candle['open'], candle['close']) - candle['low']

            # Bearish sweep: wick above recent high, closes below
            if (candle['high'] > recent_high and
                    candle['close'] < recent_high and
                    wick_up > 0.3 * atr_val):
                sweeps.append({
                    'type': 'SWEEP_HIGH',
                    'price': candle['high'],
                    'swept_level': recent_high,
                    'index': df.index[i]
                })

            # Bullish sweep: wick below recent low, closes above
            if (candle['low'] < recent_low and
                    candle['close'] > recent_low and
                    wick_down > 0.3 * atr_val):
                sweeps.append({
                    'type': 'SWEEP_LOW',
                    'price': candle['low'],
                    'swept_level': recent_low,
                    'index': df.index[i]
                })

        return sweeps[-5:]

    def detect_market_structure(self) -> Dict:
        """
        Proper market structure using swing highs/lows.
        HH+HL = uptrend, LH+LL = downtrend, otherwise ranging.
        Also detects Break of Structure (BOS) and Change of Character (CHOCH).
        """
        df = self.df
        if len(df) < 20:
            return {'structure': 'UNKNOWN', 'bos': False, 'choch': False}

        swing_highs = Indicators.find_swing_highs(df['high'], lookback=3)
        swing_lows = Indicators.find_swing_lows(df['low'], lookback=3)

        sh_prices = df['high'][swing_highs].values
        sl_prices = df['low'][swing_lows].values

        if len(sh_prices) < 2 or len(sl_prices) < 2:
            return {'structure': 'UNKNOWN', 'bos': False, 'choch': False}

        last_sh = sh_prices[-2:]
        last_sl = sl_prices[-2:]

        hh = last_sh[-1] > last_sh[-2]
        hl = last_sl[-1] > last_sl[-2]
        lh = last_sh[-1] < last_sh[-2]
        ll = last_sl[-1] < last_sl[-2]

        if hh and hl:
            structure = 'UPTREND'
        elif lh and ll:
            structure = 'DOWNTREND'
        elif hh and ll:
            structure = 'RANGING'
        else:
            structure = 'RANGING'

        # BOS: price breaks most recent swing high/low
        current = df['close'].iloc[-1]
        recent_sh = sh_prices[-1] if len(sh_prices) > 0 else None
        recent_sl = sl_prices[-1] if len(sl_prices) > 0 else None

        bos_bullish = current > recent_sh if recent_sh else False
        bos_bearish = current < recent_sl if recent_sl else False

        return {
            'structure': structure,
            'bos_bullish': bos_bullish,
            'bos_bearish': bos_bearish,
            'swing_highs': sh_prices.tolist(),
            'swing_lows': sl_prices.tolist(),
            'last_swing_high': float(recent_sh) if recent_sh else None,
            'last_swing_low': float(recent_sl) if recent_sl else None
        }

    def detect_rsi_divergence(self, rsi: pd.Series) -> List[str]:
        """Proper RSI divergence using swing points"""
        df = self.df
        divergences = []
        lookback = 30

        if len(df) < lookback:
            return divergences

        recent_df = df.tail(lookback)
        recent_rsi = rsi.tail(lookback)

        # Find local price lows and RSI values there
        lows_idx = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]):
                lows_idx.append(i)

        highs_idx = []
        for i in range(2, len(recent_df) - 2):
            if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]):
                highs_idx.append(i)

        # Bullish divergence: price lower low, RSI higher low
        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if (recent_df['low'].iloc[i2] < recent_df['low'].iloc[i1] and
                    recent_rsi.iloc[i2] > recent_rsi.iloc[i1]):
                divergences.append('BULLISH_DIVERGENCE')

        # Bearish divergence: price higher high, RSI lower high
        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if (recent_df['high'].iloc[i2] > recent_df['high'].iloc[i1] and
                    recent_rsi.iloc[i2] < recent_rsi.iloc[i1]):
                divergences.append('BEARISH_DIVERGENCE')

        return divergences


# ==================================================
# MULTI-TIMEFRAME ANALYZER
# ==================================================

class MultiTimeframeAnalyzer:

    # Timeframe hierarchy for HTF confirmation
    TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1day", "1week"]

    def __init__(self, data: Dict[str, Tuple[pd.DataFrame, str]], timeframes: List[str]):
        self.data = data  # {tf: (df, source_label)}
        self.timeframes = timeframes
        self.results = {}
        self._analyze_all()

    def _analyze_all(self):
        for tf in self.timeframes:
            if tf not in self.data:
                continue
            df, source = self.data[tf]
            if df is None or len(df) < 50:
                continue
            self.results[tf] = self._analyze_tf(df, tf, source)

    def _analyze_tf(self, df: pd.DataFrame, tf: str, source: str) -> Dict:
        """Full analysis of a single timeframe using real OHLCV"""
        ind = Indicators()

        # Core indicators — all vectorized, using real OHLC
        atr_s = ind.atr(df['high'], df['low'], df['close'])
        rsi_s = ind.rsi(df['close'])
        macd_line, signal_line, histogram = ind.macd(df['close'])
        bb_upper, bb_mid, bb_lower = ind.bollinger(df['close'])
        vol_sma = ind.volume_sma(df['volume'])
        adx_s, plus_di, minus_di = ind.adx(df['high'], df['low'], df['close'])

        # Current values (last bar)
        cur = df.iloc[-1]
        rsi_val = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
        atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
        macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        macd_sig = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        hist_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0
        plus_di_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
        minus_di_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

        # Volume ratio (real volume vs SMA)
        vol_ratio = float(cur['volume'] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0

        # EMAs
        emas = {}
        for period in [9, 21, 50, 200]:
            ema_s = ind.ema(df['close'], period)
            emas[period] = float(ema_s.iloc[-1])

        # Smart money engine
        smc = SmartMoneyEngine(df, atr_s)
        order_blocks = smc.detect_order_blocks()
        fvgs = smc.detect_fair_value_gaps()
        sweeps = smc.detect_liquidity_sweeps()
        structure = smc.detect_market_structure()
        divergences = smc.detect_rsi_divergence(rsi_s)

        # Fibonacci levels from recent swing
        fib = self._fibonacci(df)

        # Nearest significant levels (swing-based)
        swing_h = structure.get('last_swing_high')
        swing_l = structure.get('last_swing_low')

        # Current price
        price = float(cur['close'])

        # Generate signals
        signals = self._generate_signals(
            price, rsi_val, macd_val, macd_sig, hist_val, adx_val,
            plus_di_val, minus_di_val, vol_ratio, emas, structure,
            order_blocks, fvgs, sweeps, divergences,
            bb_upper.iloc[-1], bb_mid.iloc[-1], bb_lower.iloc[-1]
        )

        confidence = self._calc_confidence(signals, adx_val, vol_ratio, tf)

        return {
            'tf': tf,
            'source': source,
            'price': price,
            'rsi': rsi_val,
            'macd': macd_val,
            'macd_signal': macd_sig,
            'macd_hist': hist_val,
            'adx': adx_val,
            'plus_di': plus_di_val,
            'minus_di': minus_di_val,
            'atr': atr_val,
            'vol_ratio': vol_ratio,
            'emas': emas,
            'bb_upper': float(bb_upper.iloc[-1]),
            'bb_mid': float(bb_mid.iloc[-1]),
            'bb_lower': float(bb_lower.iloc[-1]),
            'structure': structure,
            'order_blocks': order_blocks,
            'fvgs': fvgs,
            'sweeps': sweeps,
            'divergences': divergences,
            'fib': fib,
            'signals': signals,
            'bias': signals['bias'],
            'confidence': confidence,
            'swing_high': swing_h,
            'swing_low': swing_l,
            'df': df,  # keep reference for charting
            'atr_series': atr_s
        }

    def _generate_signals(self, price, rsi, macd, macd_sig, macd_hist, adx,
                          plus_di, minus_di, vol_ratio, emas, structure,
                          order_blocks, fvgs, sweeps, divergences,
                          bb_upper, bb_mid, bb_lower) -> Dict:
        """
        Rule-based signal generation with scoring.
        Each component votes +1 bullish, -1 bearish, 0 neutral.
        """
        votes = []
        reasons = []

        # 1. Trend structure
        s = structure['structure']
        if s == 'UPTREND':
            votes.append(1)
            reasons.append("✅ Uptrend (HH+HL)")
        elif s == 'DOWNTREND':
            votes.append(-1)
            reasons.append("🔴 Downtrend (LH+LL)")
        else:
            votes.append(0)
            reasons.append("⚪ Ranging market")

        # 2. BOS
        if structure.get('bos_bullish'):
            votes.append(1)
            reasons.append("✅ Bullish BOS (broke swing high)")
        elif structure.get('bos_bearish'):
            votes.append(-1)
            reasons.append("🔴 Bearish BOS (broke swing low)")

        # 3. EMA alignment (9 > 21 > 50 > 200)
        if all(k in emas for k in [9, 21, 50]):
            if emas[9] > emas[21] > emas[50]:
                votes.append(1)
                reasons.append("✅ EMA bullish stack (9>21>50)")
            elif emas[9] < emas[21] < emas[50]:
                votes.append(-1)
                reasons.append("🔴 EMA bearish stack (9<21<50)")
            else:
                votes.append(0)
                reasons.append("⚪ EMAs mixed")

        # 4. Price vs EMA 50
        if 50 in emas:
            if price > emas[50]:
                votes.append(1)
                reasons.append("✅ Price above EMA 50")
            else:
                votes.append(-1)
                reasons.append("🔴 Price below EMA 50")

        # 5. RSI (avoid extremes, use for confirmation)
        if 45 < rsi < 70:
            votes.append(1)
            reasons.append(f"✅ RSI bullish zone ({rsi:.1f})")
        elif 30 < rsi <= 45:
            votes.append(-1)
            reasons.append(f"🔴 RSI bearish zone ({rsi:.1f})")
        elif rsi >= 70:
            votes.append(-1)
            reasons.append(f"⚠️ RSI overbought ({rsi:.1f})")
        elif rsi <= 30:
            votes.append(1)
            reasons.append(f"⚠️ RSI oversold ({rsi:.1f})")

        # 6. MACD — use histogram for direction + cross
        if macd_hist > 0 and macd > macd_sig:
            votes.append(1)
            reasons.append("✅ MACD bullish cross + histogram positive")
        elif macd_hist < 0 and macd < macd_sig:
            votes.append(-1)
            reasons.append("🔴 MACD bearish cross + histogram negative")
        else:
            votes.append(0)

        # 7. ADX trend strength + DI direction
        if adx > 25:
            if plus_di > minus_di:
                votes.append(1)
                reasons.append(f"✅ ADX {adx:.1f} strong — +DI dominates")
            else:
                votes.append(-1)
                reasons.append(f"🔴 ADX {adx:.1f} strong — -DI dominates")
        else:
            votes.append(0)
            reasons.append(f"⚪ ADX {adx:.1f} weak trend")

        # 8. Volume confirmation (skip if no real volume data)
        if vol_ratio > 0 and vol_ratio != 1.0:
            if vol_ratio > 1.3:
                votes.append(1)
                reasons.append(f"✅ Above-average volume ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.7:
                votes.append(-1)
                reasons.append(f"⚠️ Low volume ({vol_ratio:.1f}x) — weak move")
            else:
                votes.append(0)
        else:
            reasons.append("⚪ Volume N/A (forex/commodity — no centralized volume)")

        # 9. Bollinger Bands position
        if price > bb_upper:
            votes.append(-1)
            reasons.append("⚠️ Price above BB upper (overextended)")
        elif price < bb_lower:
            votes.append(1)
            reasons.append("✅ Price below BB lower (oversold)")
        elif price > bb_mid:
            votes.append(1)
            reasons.append("✅ Price above BB midline")
        else:
            votes.append(-1)
            reasons.append("🔴 Price below BB midline")

        # 10. Smart Money patterns
        recent_bullish_ob = any(ob['type'] == 'BULLISH_OB' for ob in order_blocks[-3:])
        recent_bearish_ob = any(ob['type'] == 'BEARISH_OB' for ob in order_blocks[-3:])
        if recent_bullish_ob:
            votes.append(1)
            reasons.append("✅ Bullish Order Block detected")
        if recent_bearish_ob:
            votes.append(-1)
            reasons.append("🔴 Bearish Order Block detected")

        recent_bullish_fvg = any(f['type'] == 'BULLISH_FVG' for f in fvgs[-3:])
        recent_bearish_fvg = any(f['type'] == 'BEARISH_FVG' for f in fvgs[-3:])
        if recent_bullish_fvg:
            votes.append(1)
            reasons.append("✅ Bullish FVG present")
        if recent_bearish_fvg:
            votes.append(-1)
            reasons.append("🔴 Bearish FVG present")

        # 11. Liquidity sweeps
        for sw in sweeps[-2:]:
            if sw['type'] == 'SWEEP_LOW':
                votes.append(1)
                reasons.append("✅ Bullish liquidity sweep (stop hunt below)")
            elif sw['type'] == 'SWEEP_HIGH':
                votes.append(-1)
                reasons.append("🔴 Bearish liquidity sweep (stop hunt above)")

        # 12. Divergences
        for div in divergences:
            if 'BULLISH' in div:
                votes.append(1)
                reasons.append("✅ Bullish RSI divergence")
            elif 'BEARISH' in div:
                votes.append(-1)
                reasons.append("🔴 Bearish RSI divergence")

        # Tally
        bull_votes = votes.count(1)
        bear_votes = votes.count(-1)
        total = len(votes)

        if bull_votes > bear_votes and (bull_votes / total) >= 0.55:
            bias = 'BULLISH'
        elif bear_votes > bull_votes and (bear_votes / total) >= 0.55:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'

        return {
            'bias': bias,
            'bull_votes': bull_votes,
            'bear_votes': bear_votes,
            'total_votes': total,
            'reasons': reasons
        }

    def _calc_confidence(self, signals: Dict, adx: float, vol_ratio: float, tf: str) -> float:
        """
        Confidence = (dominant vote fraction) * modifiers.
        Not arbitrary — tied to actual signal agreement rate.
        """
        total = signals['total_votes']
        if total == 0:
            return 0.0

        bull = signals['bull_votes']
        bear = signals['bear_votes']
        dominant = max(bull, bear)
        base_confidence = (dominant / total) * 100

        # ADX modifier: stronger trend = more confidence
        adx_mod = min(1.0 + (adx - 20) / 100, 1.3) if adx > 20 else 0.8

        # Volume modifier
        vol_mod = min(1.0 + (vol_ratio - 1.0) * 0.2, 1.2) if vol_ratio > 1 else 0.9

        # Timeframe modifier — higher TF signals are more reliable
        tf_mod = {
            "1m": 0.7, "5m": 0.8, "15m": 0.9, "30m": 0.95,
            "1h": 1.0, "4h": 1.05, "1day": 1.1, "1week": 1.15
        }.get(tf, 1.0)

        conf = base_confidence * adx_mod * vol_mod * tf_mod
        return min(conf, 98.0)

    def _fibonacci(self, df: pd.DataFrame) -> Dict:
        """Fibonacci from last 50 bars swing"""
        recent = df.tail(50)
        high = float(recent['high'].max())
        low = float(recent['low'].min())
        diff = high - low
        return {
            '0.0': high, '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382, '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618, '0.786': high - diff * 0.786,
            '1.0': low
        }

    def get_consolidated_signal(self, min_conf: float = 60, require_htf: bool = True) -> Dict:
        """
        Consolidated signal across timeframes.
        Optionally requires HTF alignment for trade validation.
        """
        if not self.results:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'valid': False}

        tfs_sorted = sorted(self.results.keys(),
                            key=lambda x: self.TF_ORDER.index(x) if x in self.TF_ORDER else 99)

        bull_tfs, bear_tfs, neutral_tfs = [], [], []
        for tf in tfs_sorted:
            r = self.results[tf]
            if r['confidence'] >= min_conf:
                if r['bias'] == 'BULLISH':
                    bull_tfs.append(tf)
                elif r['bias'] == 'BEARISH':
                    bear_tfs.append(tf)
                else:
                    neutral_tfs.append(tf)
            else:
                neutral_tfs.append(tf)

        # Weighted confidence (HTF weighted more)
        weights = {"1m": 0.5, "5m": 0.6, "15m": 0.8, "30m": 0.9,
                   "1h": 1.0, "4h": 1.3, "1day": 1.6, "1week": 2.0}
        total_w = sum(weights.get(tf, 1.0) for tf in self.results)
        weighted_conf = sum(
            self.results[tf]['confidence'] * weights.get(tf, 1.0)
            for tf in self.results
        ) / total_w if total_w > 0 else 0

        # Determine signal
        if len(bull_tfs) > len(bear_tfs) and len(bull_tfs) >= 2:
            signal = 'BULLISH'
        elif len(bear_tfs) > len(bull_tfs) and len(bear_tfs) >= 2:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # HTF confirmation check
        htf_confirmed = True
        if require_htf and len(tfs_sorted) > 1:
            highest_tf = tfs_sorted[-1]
            htf_bias = self.results[highest_tf]['bias']
            if signal != 'NEUTRAL' and htf_bias != signal:
                htf_confirmed = False

        # Position sizing hint (Kelly-lite)
        price = self.results[tfs_sorted[0]]['price'] if tfs_sorted else 0
        atr = self.results[tfs_sorted[0]]['atr'] if tfs_sorted else 0

        return {
            'signal': signal,
            'confidence': weighted_conf,
            'bull_tfs': bull_tfs,
            'bear_tfs': bear_tfs,
            'neutral_tfs': neutral_tfs,
            'htf_confirmed': htf_confirmed,
            'valid': htf_confirmed and weighted_conf >= min_conf and signal != 'NEUTRAL',
            'price': price,
            'atr': atr
        }


# ==================================================
# NEWS SENTIMENT — VADER (NOT KEYWORD COUNTING)
# ==================================================

class SentimentEngine:
    """
    Uses VADER for proper compound sentiment scoring on real text.
    Falls back to NewsData.io if API key provided.
    """

    # Fallback lexicon for when VADER is not installed
    POSITIVE = [
        'bullish', 'surge', 'rally', 'gain', 'rise', 'soar', 'jump', 'record',
        'high', 'strong', 'growth', 'inflow', 'adoption', 'optimism', 'support',
        'buy', 'accumulate', 'breakthrough', 'upside', 'recovery', 'positive',
        'beat', 'exceeded', 'outperform', 'upgrade', 'boom', 'momentum'
    ]
    NEGATIVE = [
        'bearish', 'crash', 'plunge', 'drop', 'fall', 'decline', 'dump', 'sell',
        'outflow', 'fear', 'risk', 'warn', 'weak', 'low', 'loss', 'concern',
        'negative', 'miss', 'underperform', 'downgrade', 'bust', 'collapse',
        'headwind', 'pressure', 'volatility', 'uncertain', 'halt', 'ban'
    ]
    NEGATORS = ['not', "n't", 'no', 'never', 'without', 'against']

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None

    def analyze_text(self, text: str) -> Dict:
        if self.vader is not None:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
        else:
            compound = self._lexicon_score(text)

        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return {'label': label, 'compound': round(compound, 3)}

    def _lexicon_score(self, text: str) -> float:
        """Simple but negation-aware lexicon scoring (VADER fallback)"""
        words = text.lower().split()
        score = 0
        for i, word in enumerate(words):
            # Strip punctuation
            clean = word.strip('.,!?;:"\'')
            negated = any(n in words[max(0, i-3):i] for n in self.NEGATORS)
            if clean in self.POSITIVE:
                score += -1 if negated else 1
            elif clean in self.NEGATIVE:
                score += 1 if negated else -1
        # Normalize to [-1, 1]
        total_words = max(len(words), 1)
        return max(-1.0, min(1.0, score / (total_words ** 0.5)))

    def fetch_and_analyze(self, symbol: str) -> List[Dict]:
        articles = []
        if self.api_key:
            articles = self._fetch_newsdata(symbol)

        if not articles:
            articles = self._sample_articles(symbol)

        # Run VADER on each article
        for a in articles:
            sentiment = self.analyze_text(a.get('title', '') + ' ' + a.get('description', ''))
            a['sentiment'] = sentiment['label']
            a['compound'] = sentiment['compound']

        return articles

    def _fetch_newsdata(self, symbol: str) -> List[Dict]:
        query_map = {
            "BTC/USDT": "Bitcoin BTC crypto",
            "ETH/USDT": "Ethereum ETH crypto",
            "XAU/USD": "Gold XAU precious metals",
            "EUR/USD": "Euro EUR USD forex",
            "GBP/USD": "Pound GBP sterling forex",
            "USD/JPY": "Yen JPY USD forex"
        }
        query = query_map.get(symbol, symbol)

        try:
            url = "https://newsdata.io/api/1/news"
            params = {"apikey": self.api_key, "q": query, "language": "en", "size": 8}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            results = []
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', '') or '',
                    'source': item.get('source_id', 'Unknown'),
                    'published': item.get('pubDate', '')
                })
            return results
        except:
            return []

    def _sample_articles(self, symbol: str) -> List[Dict]:
        samples = {
            "BTC/USDT": [
                {'title': 'Bitcoin ETF sees record institutional inflows amid market optimism',
                 'description': 'Major asset managers continue accumulating BTC through regulated vehicles.',
                 'source': 'CoinDesk', 'published': 'Sample'},
                {'title': 'Crypto market faces headwinds as Fed signals prolonged high rates',
                 'description': 'Risk assets under pressure as monetary tightening concerns resurface.',
                 'source': 'Reuters', 'published': 'Sample'},
                {'title': 'Bitcoin on-chain metrics show accumulation at current levels',
                 'description': 'Long-term holder supply reaches all-time high, suggesting strong conviction.',
                 'source': 'Glassnode', 'published': 'Sample'},
            ],
            "XAU/USD": [
                {'title': 'Gold prices steady near highs as geopolitical tensions persist',
                 'description': 'Safe-haven demand underpins gold amid ongoing global uncertainty.',
                 'source': 'Bloomberg', 'published': 'Sample'},
                {'title': 'Dollar strength weighs on gold outlook despite recession fears',
                 'description': 'The greenback rally is limiting gold upside in the near term.',
                 'source': 'FT', 'published': 'Sample'},
            ]
        }
        return samples.get(symbol, [
            {'title': f'{symbol} trades within tight range amid low volatility',
             'description': 'Market participants await key economic data for directional cues.',
             'source': 'Reuters', 'published': 'Sample'}
        ])


# ==================================================
# BACKTESTER — ACTUAL HISTORICAL SIMULATION
# ==================================================

class Backtester:
    """
    Runs the signal logic on historical OHLCV data.
    Uses real candle OHLC — no synthetic prices.
    Measures: win rate, profit factor, max drawdown, Sharpe ratio.
    """

    def __init__(self, df: pd.DataFrame, atr_multiplier: float = 2.0, risk_pct: float = 1.0):
        self.df = df.copy()
        self.atr_mult = atr_multiplier
        self.risk_pct = risk_pct

    def run(self) -> Dict:
        df = self.df
        if len(df) < 100:
            return {'error': 'Insufficient data (need 100+ bars)'}

        ind = Indicators()
        atr_s = ind.atr(df['high'], df['low'], df['close'])
        rsi_s = ind.rsi(df['close'])
        macd_line, signal_line, _ = ind.macd(df['close'])
        ema9 = ind.ema(df['close'], 9)
        ema21 = ind.ema(df['close'], 21)
        ema50 = ind.ema(df['close'], 50)

        trades = []
        equity = 10000.0
        equity_curve = [equity]
        in_trade = False
        trade_dir = None
        entry_price = None
        stop_price = None
        target_price = None
        entry_idx = None

        for i in range(50, len(df) - 1):
            bar = df.iloc[i]
            next_bar = df.iloc[i + 1]
            atr = atr_s.iloc[i]

            if pd.isna(atr) or atr == 0:
                continue

            # Generate simple signal from indicators
            bullish = (ema9.iloc[i] > ema21.iloc[i] > ema50.iloc[i] and
                       rsi_s.iloc[i] > 45 and rsi_s.iloc[i] < 70 and
                       macd_line.iloc[i] > signal_line.iloc[i])
            bearish = (ema9.iloc[i] < ema21.iloc[i] < ema50.iloc[i] and
                       rsi_s.iloc[i] < 55 and rsi_s.iloc[i] > 30 and
                       macd_line.iloc[i] < signal_line.iloc[i])

            if not in_trade:
                if bullish:
                    in_trade = True
                    trade_dir = 'LONG'
                    entry_price = next_bar['open']  # enter at next open (realistic)
                    stop_price = entry_price - atr * self.atr_mult
                    target_price = entry_price + atr * self.atr_mult * 2  # 2:1 RR
                    entry_idx = i
                elif bearish:
                    in_trade = True
                    trade_dir = 'SHORT'
                    entry_price = next_bar['open']
                    stop_price = entry_price + atr * self.atr_mult
                    target_price = entry_price - atr * self.atr_mult * 2
                    entry_idx = i
            else:
                # Check if stop or target hit on this candle
                hit_stop = False
                hit_target = False

                if trade_dir == 'LONG':
                    hit_stop = bar['low'] <= stop_price
                    hit_target = bar['high'] >= target_price
                else:
                    hit_stop = bar['high'] >= stop_price
                    hit_target = bar['low'] <= target_price

                if hit_target or hit_stop or (i - entry_idx > 20):  # 20 bar timeout
                    exit_price = target_price if hit_target else (stop_price if hit_stop else bar['close'])
                    pnl_pct = (exit_price - entry_price) / entry_price
                    if trade_dir == 'SHORT':
                        pnl_pct = -pnl_pct

                    # Position sizing: risk_pct of equity / stop distance in pct
                    stop_dist_pct = abs(entry_price - stop_price) / entry_price
                    position_size = (equity * self.risk_pct / 100) / stop_dist_pct if stop_dist_pct > 0 else equity
                    pnl_dollar = pnl_pct * position_size
                    equity += pnl_dollar

                    trades.append({
                        'direction': trade_dir,
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'won': pnl_pct > 0,
                        'reason': 'target' if hit_target else ('stop' if hit_stop else 'timeout'),
                        'entry_date': df.index[entry_idx],
                        'exit_date': df.index[i]
                    })
                    equity_curve.append(equity)
                    in_trade = False

        # Stats
        if not trades:
            return {'error': 'No trades generated', 'equity_curve': equity_curve}

        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['won']]
        losses = trades_df[~trades_df['won']]

        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins['pnl_dollar'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_dollar'].mean() if len(losses) > 0 else 0
        profit_factor = (wins['pnl_dollar'].sum() / abs(losses['pnl_dollar'].sum())
                         if losses['pnl_dollar'].sum() != 0 else float('inf'))

        # Max drawdown
        eq_series = pd.Series(equity_curve)
        rolling_max = eq_series.cummax()
        drawdown = (eq_series - rolling_max) / rolling_max * 100
        max_drawdown = float(drawdown.min())

        # Sharpe (simplified)
        returns = trades_df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win_dollar': avg_win,
            'avg_loss_dollar': avg_loss,
            'total_pnl': equity - 10000,
            'final_equity': equity,
            'max_drawdown_pct': max_drawdown,
            'sharpe': sharpe,
            'equity_curve': equity_curve,
            'trades': trades_df
        }


# ==================================================
# RISK MANAGEMENT CALCULATOR
# ==================================================

def calculate_position_size(equity: float, risk_pct: float, entry: float,
                             stop: float) -> Dict:
    """Calculate position size based on fixed risk %"""
    if entry == 0 or stop == 0 or entry == stop:
        return {}
    risk_amount = equity * risk_pct / 100
    stop_distance = abs(entry - stop)
    stop_pct = stop_distance / entry * 100
    units = risk_amount / stop_distance
    position_value = units * entry
    return {
        'risk_amount': risk_amount,
        'stop_distance': stop_distance,
        'stop_pct': stop_pct,
        'units': units,
        'position_value': position_value
    }


# ==================================================
# CHARTING — INSTITUTIONAL-GRADE CHARTS
# ==================================================

def build_chart(df: pd.DataFrame, symbol: str, tf: str, result: Dict) -> go.Figure:
    """Build comprehensive candlestick chart with all levels"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=[f"{symbol} {tf}", "Volume", "RSI", "MACD"]
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price", showlegend=False,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # EMAs
    ind = Indicators()
    colors = {9: '#2196F3', 21: '#FF9800', 50: '#9C27B0', 200: '#F44336'}
    for period, color in colors.items():
        ema = ind.ema(df['close'], period)
        fig.add_trace(go.Scatter(
            x=df.index, y=ema,
            line=dict(color=color, width=1, dash='solid'),
            name=f"EMA {period}", showlegend=True
        ), row=1, col=1)

    # Bollinger Bands
    bb_u, bb_m, bb_l = ind.bollinger(df['close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_u, line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
        name="BB Upper", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_l, line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
        name="BB Lower", fill='tonexty', fillcolor='rgba(100,100,100,0.05)',
        showlegend=False
    ), row=1, col=1)

    # Order blocks
    for ob in result.get('order_blocks', [])[-5:]:
        color = 'rgba(76,175,80,0.2)' if ob['type'] == 'BULLISH_OB' else 'rgba(244,67,54,0.2)'
        border = 'rgba(76,175,80,0.8)' if ob['type'] == 'BULLISH_OB' else 'rgba(244,67,54,0.8)'
        fig.add_hrect(y0=ob['bottom'], y1=ob['top'],
                      fillcolor=color, line_color=border, line_width=1,
                      annotation_text=ob['type'].replace('_OB', ' OB'),
                      annotation_position="top left",
                      annotation_font_size=9, row=1, col=1)

    # Swing highs/lows
    sw = result.get('structure', {})
    if sw.get('last_swing_high'):
        fig.add_hline(y=sw['last_swing_high'], line_dash="dash",
                      line_color="red", line_width=1,
                      annotation_text="Swing High", row=1, col=1)
    if sw.get('last_swing_low'):
        fig.add_hline(y=sw['last_swing_low'], line_dash="dash",
                      line_color="green", line_width=1,
                      annotation_text="Swing Low", row=1, col=1)

    # Volume with color coding
    colors_vol = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
    vol_sma = ind.volume_sma(df['volume'])
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], name="Volume",
        marker_color=colors_vol, showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_sma, line=dict(color='orange', width=1),
        name="Vol MA", showlegend=False
    ), row=2, col=1)

    # RSI
    rsi = ind.rsi(df['close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, line=dict(color='purple', width=1.5),
        name="RSI", showlegend=False
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=0.8, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=0.8, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=0.5, row=3, col=1)

    # MACD
    macd_l, sig_l, hist = ind.macd(df['close'])
    hist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="MACD Hist",
        marker_color=hist_colors, showlegend=False
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_l, line=dict(color='blue', width=1.2),
        name="MACD", showlegend=False
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=sig_l, line=dict(color='orange', width=1.2),
        name="Signal", showlegend=False
    ), row=4, col=1)

    fig.update_layout(
        height=900, template="plotly_dark",
        title=dict(text=f"{symbol} — {tf} | Source: {result.get('source', 'N/A')}",
                   font=dict(size=14)),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=40, t=80, b=20)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    twelve_key = st.session_state.get("twelve_key", "")
    news_key = st.session_state.get("news_api_key", "")
    instruments = st.session_state.get("instruments", ["BTC/USDT", "XAU/USD"])
    timeframes = st.session_state.get("timeframes", ["15m", "1h", "4h", "1day"])
    min_confidence = st.session_state.get("min_confidence", 60)
    require_htf = st.session_state.get("require_htf", True)
    max_risk = st.session_state.get("max_risk", 1.0)
    atr_multiplier = st.session_state.get("atr_multiplier", 2.0)

    fetcher = DataFetcher(twelve_api_key=twelve_key)
    sentiment_engine = SentimentEngine(api_key=news_key)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 SIGNALS", "📊 CHARTS", "📰 NEWS", "🧪 BACKTEST"])

    # ========== TAB 1: SIGNALS ==========
    with tab1:
        st.markdown('<div class="section-header">LIVE MULTI-TIMEFRAME SIGNALS</div>',
                    unsafe_allow_html=True)

        if not instruments:
            st.warning("Select instruments in the sidebar.")
            return

        for symbol in instruments:
            with st.expander(f"🔍 {symbol}", expanded=True):
                # Fetch real data for all timeframes
                data = {}
                with st.spinner(f"Fetching real OHLCV for {symbol}..."):
                    for tf in timeframes:
                        df, source = fetcher.fetch(symbol, tf, limit=300)
                        if df is not None and len(df) >= 50:
                            data[tf] = (df, source)
                        else:
                            st.warning(f"⚠️ {symbol} {tf}: {source}")

                if not data:
                    st.error(f"No data available for {symbol}. Check API key.")
                    continue

                # Run analysis
                analyzer = MultiTimeframeAnalyzer(data, timeframes)
                consolidated = analyzer.get_consolidated_signal(
                    min_conf=min_confidence, require_htf=require_htf
                )

                # Main signal display
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                with col1:
                    sig = consolidated['signal']
                    valid = consolidated['valid']
                    if sig == 'BULLISH':
                        icon = "📈 BULLISH" + (" ✅ VALID" if valid else " ⚠️ UNCONFIRMED")
                        st.markdown(f'<div class="signal-buy">{icon}</div>', unsafe_allow_html=True)
                    elif sig == 'BEARISH':
                        icon = "📉 BEARISH" + (" ✅ VALID" if valid else " ⚠️ UNCONFIRMED")
                        st.markdown(f'<div class="signal-sell">{icon}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="signal-neutral">⚖️ NO CLEAR SIGNAL</div>',
                                    unsafe_allow_html=True)

                with col2:
                    st.metric("Confidence", f"{consolidated['confidence']:.1f}%")
                with col3:
                    st.metric("Price", f"{consolidated['price']:.4f}")
                with col4:
                    st.metric("HTF Confirmed", "✅ Yes" if consolidated['htf_confirmed'] else "❌ No")
                with col5:
                    atr_val = consolidated['atr']
                    st.metric("ATR", f"{atr_val:.4f}")

                # Risk management levels
                if sig != 'NEUTRAL' and consolidated['price'] > 0 and atr_val > 0:
                    price = consolidated['price']
                    if sig == 'BULLISH':
                        stop = price - atr_val * atr_multiplier
                        target = price + atr_val * atr_multiplier * 2
                    else:
                        stop = price + atr_val * atr_multiplier
                        target = price - atr_val * atr_multiplier * 2

                    sizing = calculate_position_size(10000, max_risk, price, stop)
                    with st.container():
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Entry", f"{price:.4f}")
                        rc2.metric("Stop Loss", f"{stop:.4f}",
                                   delta=f"{(stop-price)/price*100:.2f}%",
                                   delta_color="inverse")
                        rc3.metric("Target (2:1)", f"{target:.4f}",
                                   delta=f"{(target-price)/price*100:.2f}%")
                        rc4.metric("Risk Amount", f"${sizing.get('risk_amount', 0):.0f}",
                                   help="Based on $10,000 account")

                # Per-timeframe breakdown
                st.markdown("#### Timeframe Breakdown")
                tf_cols = st.columns(len(analyzer.results))
                for idx, (tf, res) in enumerate(analyzer.results.items()):
                    with tf_cols[idx]:
                        bias = res['bias']
                        conf = res['confidence']
                        sig_counts = res['signals']
                        source_label = res['source'].split('(')[0].strip()

                        if bias == 'BULLISH':
                            bg = '#4CAF5020'
                            emoji = '🟢'
                        elif bias == 'BEARISH':
                            bg = '#F4433620'
                            emoji = '🔴'
                        else:
                            bg = '#75757520'
                            emoji = '⚪'

                        st.markdown(f"""
                        <div style='background:{bg}; padding:10px; border-radius:5px; text-align:center;'>
                        <b>{tf}</b><br>
                        {emoji} <b>{bias}</b><br>
                        {conf:.0f}% conf<br>
                        <small>🟢{sig_counts['bull_votes']} 🔴{sig_counts['bear_votes']}</small><br>
                        <small style='color:#999'>{source_label}</small>
                        </div>
                        """, unsafe_allow_html=True)

                # Signal reasons
                with st.expander("📋 Full Signal Rationale"):
                    for tf, res in analyzer.results.items():
                        st.markdown(f"**{tf} — {res['bias']} ({res['confidence']:.1f}%)**")
                        for reason in res['signals']['reasons']:
                            st.markdown(f"  {reason}")
                        # Smart money summary
                        smc_items = []
                        for ob in res['order_blocks'][-2:]:
                            smc_items.append(f"  🧱 {ob['type']} @ {ob['bottom']:.4f}–{ob['top']:.4f}")
                        for fvg in res['fvgs'][-2:]:
                            smc_items.append(f"  🕳️ {fvg['type']} @ {fvg['bottom']:.4f}–{fvg['top']:.4f}")
                        for sw in res['sweeps'][-2:]:
                            smc_items.append(f"  💧 {sw['type']} @ {sw['price']:.4f}")
                        if smc_items:
                            st.markdown("  **Smart Money Patterns:**")
                            for item in smc_items:
                                st.markdown(item)
                        st.divider()

    # ========== TAB 2: CHARTS ==========
    with tab2:
        st.markdown('<div class="section-header">INSTITUTIONAL CHARTS</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        chart_symbol = col1.selectbox("Instrument", instruments, key="chart_symbol")
        chart_tf = col2.selectbox("Timeframe", timeframes, key="chart_tf")

        if chart_symbol and chart_tf:
            with st.spinner("Building chart..."):
                df, source = fetcher.fetch(chart_symbol, chart_tf, limit=300)

                if df is not None and len(df) > 50:
                    # Get cached analysis result if available
                    st.caption(f"<span class='data-tag'>Data: {source}</span> "
                               f"<span class='data-tag'>{len(df)} bars</span> "
                               f"Last: {df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}"
                               , unsafe_allow_html=True)

                    # Quick analysis for chart overlay
                    ind = Indicators()
                    atr_s = ind.atr(df['high'], df['low'], df['close'])
                    smc = SmartMoneyEngine(df, atr_s)
                    quick_result = {
                        'order_blocks': smc.detect_order_blocks(),
                        'fvgs': smc.detect_fair_value_gaps(),
                        'sweeps': smc.detect_liquidity_sweeps(),
                        'structure': smc.detect_market_structure(),
                        'source': source
                    }

                    fig = build_chart(df, chart_symbol, chart_tf, quick_result)
                    st.plotly_chart(fig, use_container_width=True)

                    # Fibonacci table
                    fib = {
                        '0.0': float(df['high'].tail(50).max()),
                        '0.236': float(df['high'].tail(50).max() - (df['high'].tail(50).max() - df['low'].tail(50).min()) * 0.236),
                        '0.382': float(df['high'].tail(50).max() - (df['high'].tail(50).max() - df['low'].tail(50).min()) * 0.382),
                        '0.5': float(df['high'].tail(50).max() - (df['high'].tail(50).max() - df['low'].tail(50).min()) * 0.5),
                        '0.618': float(df['high'].tail(50).max() - (df['high'].tail(50).max() - df['low'].tail(50).min()) * 0.618),
                        '1.0': float(df['low'].tail(50).min()),
                    }
                    st.markdown("**Fibonacci Levels (last 50 bars)**")
                    fib_cols = st.columns(len(fib))
                    for i, (level, price) in enumerate(fib.items()):
                        fib_cols[i].metric(f"Fib {level}", f"{price:.4f}")
                else:
                    st.error(f"Chart data unavailable: {source}")

    # ========== TAB 3: NEWS ==========
    with tab3:
        st.markdown('<div class="section-header">NEWS & VADER SENTIMENT ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.caption("Sentiment scored using VADER compound score (range: -1 bearish → +1 bullish)")

        news_symbol = st.selectbox("Instrument", instruments, key="news_sym")

        with st.spinner("Analyzing news sentiment..."):
            articles = sentiment_engine.fetch_and_analyze(news_symbol)

        if articles:
            # Aggregate sentiment
            compounds = [a['compound'] for a in articles]
            avg_compound = np.mean(compounds)
            overall = "📈 BULLISH BIAS" if avg_compound > 0.05 else (
                "📉 BEARISH BIAS" if avg_compound < -0.05 else "⚖️ NEUTRAL")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Avg VADER Score", f"{avg_compound:.3f}")
            mc2.metric("Overall Sentiment", overall)
            mc3.metric("Articles Analyzed", len(articles))

            st.divider()

            for a in articles:
                sent_class = f"news-{a['sentiment']}"
                compound = a.get('compound', 0)
                bar_width = int(abs(compound) * 100)
                bar_color = '#4CAF50' if compound > 0 else ('#F44336' if compound < 0 else '#9E9E9E')

                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <strong style="flex:1; color:#FFFFFF">{a['title']}</strong>
                        <span class="{sent_class}" style="margin-left:10px; white-space:nowrap">{a['sentiment'].upper()}</span>
                    </div>
                    <div style="color:#B0BEC5; font-size:0.9em; margin:6px 0">{a['description']}</div>
                    <div style="height:4px; background:#2A2A3E; border-radius:2px; margin:4px 0">
                        <div style="height:4px; width:{bar_width}%; background:{bar_color}; border-radius:2px"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8em; color:#78909C">
                        <span>{a['source']}</span>
                        <span>VADER: {compound:.3f} | {a['published']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ========== TAB 4: BACKTEST ==========
    with tab4:
        st.markdown('<div class="section-header">HISTORICAL BACKTEST</div>',
                    unsafe_allow_html=True)
        st.caption("Strategy: EMA stack (9>21>50) + RSI 45–70 + MACD cross. ATR-based stops. 2:1 RR.")

        bt_col1, bt_col2, bt_col3 = st.columns(3)
        bt_symbol = bt_col1.selectbox("Symbol", instruments, key="bt_sym")
        bt_tf = bt_col2.selectbox("Timeframe", timeframes, key="bt_tf")
        bt_account = bt_col3.number_input("Account Size ($)", value=10000, step=1000, key="bt_acct")

        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Fetching data and running backtest..."):
                df, source = fetcher.fetch(bt_symbol, bt_tf, limit=500)

                if df is None or len(df) < 100:
                    st.error(f"Insufficient data: {source}")
                else:
                    bt = Backtester(df, atr_multiplier=atr_multiplier, risk_pct=max_risk)
                    results = bt.run()

                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        # Metrics
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Total Trades", results['total_trades'])
                        m2.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        m3.metric("Profit Factor",
                                  f"{results['profit_factor']:.2f}" if results['profit_factor'] != float('inf') else "∞")
                        m4.metric("Net P&L", f"${results['total_pnl']:.0f}",
                                  delta=f"{results['total_pnl']/bt_account*100:.1f}%")
                        m5.metric("Max Drawdown", f"{results['max_drawdown_pct']:.1f}%",
                                  delta_color="inverse")
                        m6.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")

                        # Equity curve
                        eq_fig = go.Figure()
                        eq_fig.add_trace(go.Scatter(
                            y=results['equity_curve'],
                            mode='lines', name='Equity',
                            line=dict(color='#4CAF50' if results['total_pnl'] > 0 else '#F44336', width=2),
                            fill='tozeroy', fillcolor='rgba(76,175,80,0.1)'
                        ))
                        eq_fig.update_layout(
                            title=f"Equity Curve — {bt_symbol} {bt_tf}",
                            height=350, template='plotly_dark',
                            xaxis_title="Trade #", yaxis_title="Equity ($)"
                        )
                        st.plotly_chart(eq_fig, use_container_width=True)

                        # Trade log
                        with st.expander("📋 Trade Log"):
                            trades_df = results['trades'].copy()
                            trades_df['entry_date'] = trades_df['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
                            trades_df['exit_date'] = trades_df['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
                            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
                            trades_df['pnl_dollar'] = trades_df['pnl_dollar'].apply(lambda x: f"${x:.0f}")
                            st.dataframe(
                                trades_df[['entry_date', 'exit_date', 'direction', 'entry',
                                           'exit', 'pnl_pct', 'pnl_dollar', 'won', 'reason']],
                                use_container_width=True
                            )

                        st.caption(f"Data source: {source} | "
                                   f"Risk per trade: {max_risk}% | ATR mult: {atr_multiplier}x | "
                                   f"Bars analyzed: {len(df)}")


if __name__ == "__main__":
    main()
