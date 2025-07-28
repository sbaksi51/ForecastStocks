import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_path='models/trained/'):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.lookback = 20
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_data(self, hist_data, technical_indicators=None):
        """Prepare data for prediction"""
        # Create feature dataframe
        features = pd.DataFrame({
            'open': hist_data['Open'],
            'high': hist_data['High'],
            'low': hist_data['Low'],
            'close': hist_data['Close'],
            'volume': hist_data['Volume'],
            'returns': hist_data['Close'].pct_change(),
            'volatility': hist_data['Close'].rolling(window=20).std()
        })
        
        # Add technical indicators if provided
        if technical_indicators:
            for key, value in technical_indicators.items():
                if isinstance(value, dict) and 'value' in value:
                    features[key] = value['value']
                elif isinstance(value, (int, float)):
                    features[key] = value
        
        # Remove NaN values
        features = features.dropna()
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        return scaled_features, scaler, features.index
    
    def predict(self, hist_data, technical_indicators=None, days=1):
        """Make stock price predictions"""
        try:
            # Simple prediction based on technical analysis and trends
            return self._simple_prediction(hist_data, technical_indicators, days)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Fallback to simple prediction
            return self._simple_prediction(hist_data, technical_indicators, days)
    
    def _simple_prediction(self, hist_data, technical_indicators, days):
        """Simple prediction based on technical analysis and trends"""
        current_price = hist_data['Close'].iloc[-1]
        returns = hist_data['Close'].pct_change().dropna()
        
        # Calculate trend
        sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = hist_data['Close'].rolling(window=50).mean().iloc[-1] if len(hist_data) > 50 else sma_20
        
        # Momentum indicator
        momentum = (current_price - hist_data['Close'].iloc[-20]) / hist_data['Close'].iloc[-20] if len(hist_data) > 20 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Simple prediction logic
        trend_factor = 1.0
        if current_price > sma_20 > sma_50:
            trend_factor = 1.02  # Bullish
        elif current_price < sma_20 < sma_50:
            trend_factor = 0.98  # Bearish
        
        # Add momentum influence
        momentum_factor = 1 + (momentum * 0.3)
        
        # Calculate predicted prices with some randomness based on volatility
        predicted_prices = []
        price = current_price
        
        for day in range(days):
            daily_return = (trend_factor - 1) * momentum_factor
            daily_volatility = volatility / np.sqrt(252)
            random_shock = np.random.normal(0, daily_volatility * 0.5)
            
            price = price * (1 + daily_return + random_shock)
            predicted_prices.append(price)
        
        return self._format_prediction(current_price, predicted_prices, days)
    
    def _format_prediction(self, current_price, predicted_prices, days):
        """Format prediction results"""
        final_price = predicted_prices[-1]
        price_change = final_price - current_price
        return_percentage = (price_change / current_price) * 100
        
        # Confidence based on prediction horizon and volatility
        confidence = max(0.3, min(0.9, 0.9 - (days * 0.02)))
        
        return {
            'current_price': round(current_price, 2),
            'predicted_price': round(final_price, 2),
            'price_change': round(price_change, 2),
            'return_percentage': round(return_percentage, 2),
            'confidence': round(confidence, 2),
            'prediction_days': days,
            'price_trajectory': [round(p, 2) for p in predicted_prices],
            'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    
    def train_model(self, symbol, hist_data):
        """Train model on historical data (placeholder for demo)"""
        logger.info(f"Training model for {symbol}...")
        # In production, implement full training pipeline
        pass 