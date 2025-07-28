import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
    
    def fetch_stock_data(self, symbol, period='3mo', interval='1d'):
        """
        Fetch stock data from Yahoo Finance with caching
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            Dictionary containing stock information and historical data
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.info(f"Returning cached data for {symbol}")
                return cached_data
        
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = stock.history(period=period, interval=interval)
            
            if hist_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Get current info
            info = stock.info
            
            # Prepare the data
            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'current_price': info.get('currentPrice', hist_data['Close'].iloc[-1]),
                'previous_close': info.get('previousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else 0),
                'market_cap': info.get('marketCap', 0),
                'volume': info.get('volume', hist_data['Volume'].iloc[-1]),
                'pe_ratio': info.get('forwardPE', info.get('trailingPE', 0)),
                'dividend_yield': info.get('dividendYield', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', hist_data['High'].max()),
                '52_week_low': info.get('fiftyTwoWeekLow', hist_data['Low'].min()),
                'historical_data': hist_data,
                'change_percent': self._calculate_change_percent(hist_data)
            }
            
            # Cache the data
            self.cache[cache_key] = (stock_data, datetime.now())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _calculate_change_percent(self, hist_data):
        """Calculate percentage change from previous close"""
        if len(hist_data) < 2:
            return 0
        
        current = hist_data['Close'].iloc[-1]
        previous = hist_data['Close'].iloc[-2]
        
        if previous == 0:
            return 0
        
        return ((current - previous) / previous) * 100
    
    def prepare_training_data(self, hist_data, lookback=60):
        """
        Prepare data for model training
        
        Args:
            hist_data: Historical stock data
            lookback: Number of days to look back for prediction
        
        Returns:
            Tuple of (features, target, scaler)
        """
        # Extract relevant features
        features = pd.DataFrame({
            'open': hist_data['Open'],
            'high': hist_data['High'],
            'low': hist_data['Low'],
            'close': hist_data['Close'],
            'volume': hist_data['Volume'],
            'returns': hist_data['Close'].pct_change(),
            'volatility': hist_data['Close'].rolling(window=20).std(),
            'sma_20': hist_data['Close'].rolling(window=20).mean(),
            'sma_50': hist_data['Close'].rolling(window=50).mean(),
            'volume_sma': hist_data['Volume'].rolling(window=20).mean()
        })
        
        # Remove NaN values
        features = features.dropna()
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_features[i, 3])  # Predict close price
        
        return np.array(X), np.array(y), scaler
    
    def fetch_multiple_stocks(self, symbols, period='3mo'):
        """Fetch data for multiple stocks"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_stock_data(symbol, period)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
    
    def get_sector_performance(self, sector_stocks):
        """Calculate sector performance metrics"""
        performances = []
        
        for symbol in sector_stocks:
            try:
                data = self.fetch_stock_data(symbol, period='1mo')
                hist = data['historical_data']
                
                performance = {
                    'symbol': symbol,
                    'name': data['name'],
                    '1d_return': self._calculate_return(hist, 1),
                    '1w_return': self._calculate_return(hist, 5),
                    '1m_return': self._calculate_return(hist, 20),
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252)
                }
                performances.append(performance)
                
            except Exception as e:
                logger.error(f"Error calculating performance for {symbol}: {str(e)}")
        
        return performances
    
    def _calculate_return(self, hist_data, days):
        """Calculate return over specified days"""
        if len(hist_data) < days + 1:
            return 0
        
        current = hist_data['Close'].iloc[-1]
        past = hist_data['Close'].iloc[-(days+1)]
        
        if past == 0:
            return 0
        
        return ((current - past) / past) * 100 