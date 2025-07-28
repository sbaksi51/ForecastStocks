import pandas as pd
import numpy as np

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
    
    def calculate_indicators(self, hist_data):
        """Calculate all technical indicators for the given historical data"""
        if hist_data.empty:
            return {}
        
        df = hist_data.copy()
        
        # Price-based indicators
        indicators = {
            'sma_20': self._calculate_sma(df, 20),
            'sma_50': self._calculate_sma(df, 50),
            'ema_12': self._calculate_ema(df, 12),
            'ema_26': self._calculate_ema(df, 26),
            'rsi': self._calculate_rsi(df),
            'macd': self._calculate_macd(df),
            'bollinger_bands': self._calculate_bollinger_bands(df),
            'stochastic': self._calculate_stochastic(df),
            'atr': self._calculate_atr(df),
            'obv': self._calculate_obv(df),
            'vwap': self._calculate_vwap(df),
            'support_resistance': self._calculate_support_resistance(df)
        }
        
        # Add derived signals
        indicators['signals'] = self._generate_signals(indicators, df)
        
        return indicators
    
    def _calculate_sma(self, df, period):
        """Simple Moving Average"""
        if len(df) < period:
            return {'value': 0, 'series': pd.Series(), 'trend': 'neutral'}
        
        sma = df['Close'].rolling(window=period).mean()
        return {
            'value': sma.iloc[-1] if not sma.empty else 0,
            'series': sma,
            'trend': 'bullish' if df['Close'].iloc[-1] > sma.iloc[-1] else 'bearish'
        }
    
    def _calculate_ema(self, df, period):
        """Exponential Moving Average"""
        if len(df) < period:
            return {'value': 0, 'series': pd.Series()}
        
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        return {
            'value': ema.iloc[-1] if not ema.empty else 0,
            'series': ema
        }
    
    def _calculate_rsi(self, df, period=14):
        """Relative Strength Index"""
        if len(df) < period + 1:
            return {'value': 50, 'series': pd.Series(), 'signal': 'neutral'}
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        return {
            'value': current_rsi,
            'series': rsi,
            'signal': 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'
        }
    
    def _calculate_macd(self, df):
        """MACD - Moving Average Convergence Divergence"""
        if len(df) < 26:
            return {
                'macd': 0,
                'signal': 0,
                'histogram': 0,
                'bullish_crossover': False,
                'bearish_crossover': False
            }
        
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': macd_histogram.iloc[-1],
            'bullish_crossover': len(df) > 1 and macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2],
            'bearish_crossover': len(df) > 1 and macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]
        }
    
    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Bollinger Bands"""
        if len(df) < period:
            return {
                'upper': 0,
                'middle': 0,
                'lower': 0,
                'price_position': 'within_bands',
                'bandwidth': 0
            }
        
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = df['Close'].iloc[-1]
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1],
            'price_position': 'above_upper' if current_price > upper_band.iloc[-1] else 'below_lower' if current_price < lower_band.iloc[-1] else 'within_bands',
            'bandwidth': ((upper_band.iloc[-1] - lower_band.iloc[-1]) / sma.iloc[-1]) * 100 if sma.iloc[-1] != 0 else 0
        }
    
    def _calculate_stochastic(self, df, period=14):
        """Stochastic Oscillator"""
        if len(df) < period:
            return {
                'k': 50,
                'd': 50,
                'signal': 'neutral'
            }
        
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k': k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50,
            'd': d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50,
            'signal': 'oversold' if k_percent.iloc[-1] < 20 else 'overbought' if k_percent.iloc[-1] > 80 else 'neutral'
        }
    
    def _calculate_atr(self, df, period=14):
        """Average True Range - Volatility indicator"""
        if len(df) < period:
            return {
                'value': 0,
                'volatility': 'low'
            }
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return {
            'value': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
            'volatility': 'high' if atr.iloc[-1] > atr.mean() else 'low'
        }
    
    def _calculate_obv(self, df):
        """On-Balance Volume"""
        if len(df) < 20:
            return {
                'value': 0,
                'trend': 'neutral'
            }
        
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_sma = obv.rolling(window=20).mean()
        
        return {
            'value': obv.iloc[-1],
            'trend': 'bullish' if obv.iloc[-1] > obv_sma.iloc[-1] else 'bearish'
        }
    
    def _calculate_vwap(self, df):
        """Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        
        if cumulative_volume.iloc[-1] == 0:
            return {
                'value': df['Close'].iloc[-1],
                'price_position': 'at'
            }
        
        vwap = (typical_price * df['Volume']).cumsum() / cumulative_volume
        
        return {
            'value': vwap.iloc[-1],
            'price_position': 'above' if df['Close'].iloc[-1] > vwap.iloc[-1] else 'below'
        }
    
    def _calculate_support_resistance(self, df, window=20):
        """Calculate support and resistance levels"""
        if len(df) < window:
            current_price = df['Close'].iloc[-1]
            return {
                'resistance': current_price * 1.05,
                'support': current_price * 0.95,
                'fibonacci': {},
                'current_level': 'N/A'
            }
        
        # Find local maxima and minima
        highs = df['High'].rolling(window=window).max()
        lows = df['Low'].rolling(window=window).min()
        
        # Get recent levels
        recent_high = df['High'].iloc[-window:].max()
        recent_low = df['Low'].iloc[-window:].min()
        
        # Fibonacci retracement levels
        diff = recent_high - recent_low
        fib_levels = {
            '0.236': recent_high - (diff * 0.236),
            '0.382': recent_high - (diff * 0.382),
            '0.5': recent_high - (diff * 0.5),
            '0.618': recent_high - (diff * 0.618),
            '0.786': recent_high - (diff * 0.786)
        }
        
        return {
            'resistance': recent_high,
            'support': recent_low,
            'fibonacci': fib_levels,
            'current_level': self._find_nearest_level(df['Close'].iloc[-1], fib_levels)
        }
    
    def _find_nearest_level(self, price, levels):
        """Find the nearest Fibonacci level to current price"""
        if not levels:
            return 'N/A'
        
        min_diff = float('inf')
        nearest_level = None
        
        for level_name, level_value in levels.items():
            diff = abs(price - level_value)
            if diff < min_diff:
                min_diff = diff
                nearest_level = level_name
        
        return nearest_level
    
    def _generate_signals(self, indicators, df):
        """Generate buy/sell signals based on multiple indicators"""
        signals = []
        strength = 0
        
        # RSI signals
        if indicators['rsi']['signal'] == 'oversold':
            signals.append('RSI Oversold - Potential Buy')
            strength += 1
        elif indicators['rsi']['signal'] == 'overbought':
            signals.append('RSI Overbought - Potential Sell')
            strength -= 1
        
        # MACD signals
        if indicators['macd']['bullish_crossover']:
            signals.append('MACD Bullish Crossover')
            strength += 2
        elif indicators['macd']['bearish_crossover']:
            signals.append('MACD Bearish Crossover')
            strength -= 2
        
        # Moving average signals
        sma_20_value = indicators['sma_20']['value']
        sma_50_value = indicators['sma_50']['value'] 
        current_price = df['Close'].iloc[-1]
        
        if sma_20_value > 0 and sma_50_value > 0:
            if current_price > sma_20_value > sma_50_value:
                signals.append('Price above moving averages - Uptrend')
                strength += 1
            elif current_price < sma_20_value < sma_50_value:
                signals.append('Price below moving averages - Downtrend')
                strength -= 1
        
        # Bollinger Bands signals
        if indicators['bollinger_bands']['price_position'] == 'below_lower':
            signals.append('Price at lower Bollinger Band - Oversold')
            strength += 1
        elif indicators['bollinger_bands']['price_position'] == 'above_upper':
            signals.append('Price at upper Bollinger Band - Overbought')
            strength -= 1
        
        # Volume signals
        if indicators['obv']['trend'] == 'bullish':
            signals.append('Volume trend bullish')
            strength += 1
        else:
            signals.append('Volume trend bearish')
            strength -= 1
        
        # Overall signal
        if strength >= 3:
            overall = 'STRONG BUY'
        elif strength >= 1:
            overall = 'BUY'
        elif strength <= -3:
            overall = 'STRONG SELL'
        elif strength <= -1:
            overall = 'SELL'
        else:
            overall = 'HOLD'
        
        return {
            'signals': signals,
            'strength': strength,
            'overall': overall
        } 