"""
Quantitative Trading Models and Strategies
Includes statistical arbitrage, mean reversion, momentum, and risk parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Quantitative Finance
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from arch import arch_model

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Portfolio Optimization
import cvxpy as cp

# Backtesting
import backtrader as bt
import empyrical as emp

logger = logging.getLogger(__name__)

class QuantitativeAnalyzer:
    """Advanced quantitative analysis and trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.strategies = {}
        
    def analyze_mean_reversion(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Analyze mean reversion opportunities"""
        
        # Calculate rolling statistics
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        
        # Calculate z-score
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        # Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(data['close'].dropna())
        
        # Half-life of mean reversion
        spread = data['close'] - rolling_mean
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # OLS regression
        spread_lag_clean = spread_lag.dropna()
        spread_diff_clean = spread_diff.dropna()
        
        if len(spread_lag_clean) > 0:
            model = sm.OLS(spread_diff_clean, spread_lag_clean)
            results = model.fit()
            half_life = -np.log(2) / results.params[0] if results.params[0] < 0 else np.inf
        else:
            half_life = np.inf
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['z_score'] = z_score
        signals['signal'] = 0
        signals.loc[z_score < -2, 'signal'] = 1  # Buy signal
        signals.loc[z_score > 2, 'signal'] = -1  # Sell signal
        
        return {
            'z_score_current': z_score.iloc[-1] if not z_score.empty else 0,
            'mean': rolling_mean.iloc[-1] if not rolling_mean.empty else 0,
            'std': rolling_std.iloc[-1] if not rolling_std.empty else 0,
            'half_life_days': half_life,
            'is_stationary': adf_result[1] < 0.05,  # p-value < 0.05
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'current_signal': signals['signal'].iloc[-1],
            'signals': signals
        }
    
    def analyze_momentum(self, data: pd.DataFrame, 
                        short_window: int = 10, 
                        long_window: int = 30) -> Dict:
        """Analyze momentum indicators and generate signals"""
        
        # Price momentum
        returns = data['close'].pct_change()
        momentum_short = returns.rolling(window=short_window).mean()
        momentum_long = returns.rolling(window=long_window).mean()
        
        # Rate of Change (ROC)
        roc = (data['close'] - data['close'].shift(short_window)) / data['close'].shift(short_window) * 100
        
        # Relative Strength Index (already calculated in technical indicators)
        # Moving Average Convergence Divergence (MACD)
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # Volume momentum
        volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['momentum_signal'] = np.where(momentum_short > momentum_long, 1, -1)
        signals['macd_signal'] = np.where(macd > signal_line, 1, -1)
        signals['volume_confirm'] = np.where(volume_ratio > 1.2, 1, 0)
        
        # Combined signal
        signals['combined'] = signals['momentum_signal'] * (1 + signals['volume_confirm'] * 0.5)
        
        return {
            'momentum_short': momentum_short.iloc[-1] if not momentum_short.empty else 0,
            'momentum_long': momentum_long.iloc[-1] if not momentum_long.empty else 0,
            'roc': roc.iloc[-1] if not roc.empty else 0,
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'macd_signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'volume_ratio': volume_ratio.iloc[-1] if not volume_ratio.empty else 0,
            'current_signal': signals['combined'].iloc[-1],
            'signal_strength': abs(signals['combined'].iloc[-1]),
            'signals': signals
        }
    
    def pairs_trading_analysis(self, stock1_data: pd.DataFrame, 
                             stock2_data: pd.DataFrame, 
                             window: int = 60) -> Dict:
        """Analyze pairs trading opportunities using cointegration"""
        
        # Ensure same index
        common_index = stock1_data.index.intersection(stock2_data.index)
        s1 = stock1_data.loc[common_index, 'close']
        s2 = stock2_data.loc[common_index, 'close']
        
        # Test for cointegration
        coint_result = coint(s1, s2)
        
        # Calculate hedge ratio using OLS
        model = sm.OLS(s1, sm.add_constant(s2))
        results = model.fit()
        hedge_ratio = results.params[1]
        
        # Calculate spread
        spread = s1 - hedge_ratio * s2
        
        # Calculate z-score of spread
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate trading signals
        signals = pd.DataFrame(index=common_index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # Entry signals
        signals.loc[z_score < -2, 'position'] = 1  # Long stock1, short stock2
        signals.loc[z_score > 2, 'position'] = -1  # Short stock1, long stock2
        
        # Exit signals
        signals.loc[abs(z_score) < 0.5, 'position'] = 0
        
        return {
            'is_cointegrated': coint_result[1] < 0.05,
            'pvalue': coint_result[1],
            'hedge_ratio': hedge_ratio,
            'current_spread': spread.iloc[-1],
            'current_z_score': z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0,
            'current_signal': signals['position'].iloc[-1],
            'spread_mean': spread_mean.iloc[-1] if not pd.isna(spread_mean.iloc[-1]) else 0,
            'spread_std': spread_std.iloc[-1] if not pd.isna(spread_std.iloc[-1]) else 0,
            'signals': signals
        }
    
    def volatility_analysis(self, data: pd.DataFrame, window: int = 30) -> Dict:
        """Analyze volatility using GARCH models"""
        
        returns = data['close'].pct_change().dropna() * 100  # Percentage returns
        
        # Realized volatility
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # GARCH(1,1) model
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Forecast volatility
            forecast = model_fit.forecast(horizon=5)
            predicted_vol = np.sqrt(forecast.variance.values[-1, :])
            
            garch_success = True
            garch_params = {
                'omega': model_fit.params['omega'],
                'alpha': model_fit.params['alpha[1]'],
                'beta': model_fit.params['beta[1]']
            }
        except:
            garch_success = False
            predicted_vol = realized_vol.iloc[-1] if not realized_vol.empty else 0
            garch_params = {}
        
        # VIX-style calculation (simplified)
        high_low_vol = np.log(data['high'] / data['low']).rolling(window=window).std() * np.sqrt(252)
        
        # Volatility regime
        vol_percentile = stats.percentileofscore(realized_vol.dropna(), realized_vol.iloc[-1])
        
        if vol_percentile > 80:
            regime = 'high_volatility'
        elif vol_percentile < 20:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
        
        return {
            'realized_volatility': realized_vol.iloc[-1] if not realized_vol.empty else 0,
            'garch_forecast': predicted_vol.tolist() if garch_success else [],
            'garch_params': garch_params,
            'high_low_volatility': high_low_vol.iloc[-1] if not high_low_vol.empty else 0,
            'volatility_regime': regime,
            'volatility_percentile': vol_percentile,
            'vol_change': (realized_vol.iloc[-1] / realized_vol.iloc[-20] - 1) * 100 if len(realized_vol) > 20 else 0
        }
    
    def risk_parity_allocation(self, returns_data: pd.DataFrame, 
                             target_vol: float = 0.15) -> Dict:
        """Calculate risk parity portfolio allocation"""
        
        # Calculate covariance matrix
        cov_matrix = returns_data.cov() * 252  # Annualized
        
        # Number of assets
        n_assets = len(returns_data.columns)
        
        # Risk parity optimization
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - contrib.mean())**2)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        risk_parity_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(risk_parity_weights, returns_data.mean() * 252)
        portfolio_vol = np.sqrt(np.dot(risk_parity_weights.T, 
                                      np.dot(cov_matrix, risk_parity_weights)))
        
        # Risk contributions
        marginal_contrib = np.dot(cov_matrix, risk_parity_weights) / portfolio_vol
        risk_contributions = risk_parity_weights * marginal_contrib
        
        # Scale to target volatility
        scale_factor = target_vol / portfolio_vol
        scaled_weights = risk_parity_weights * scale_factor
        
        return {
            'weights': dict(zip(returns_data.columns, risk_parity_weights)),
            'scaled_weights': dict(zip(returns_data.columns, scaled_weights)),
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol,
            'risk_contributions': dict(zip(returns_data.columns, risk_contributions)),
            'target_vol': target_vol,
            'leverage': scale_factor
        }
    
    def ml_price_prediction(self, data: pd.DataFrame, 
                          features: List[str] = None,
                          target_days: int = 5) -> Dict:
        """Machine learning based price prediction"""
        
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Feature engineering
        df = data.copy()
        
        # Technical features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        
        # Price features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Target variable (future returns)
        df['target'] = df['close'].shift(-target_days) / df['close'] - 1
        
        # Remove NaN values
        df = df.dropna()
        
        # Select features
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 
                       'high_low_ratio', 'close_open_ratio', 'volume_ratio']
        
        X = df[feature_cols]
        y = df['target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svm': SVR(kernel='rbf', C=1.0),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01)
        }
        
        results = {}
        predictions = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = np.mean((y_test - y_pred)**2)
            r2 = model.score(X_test_scaled, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            results[name] = {
                'mse': mse,
                'r2': r2,
                'cv_score': -cv_scores.mean(),
                'model': model
            }
            
            # Predict next period
            last_features = scaler.transform(X.iloc[-1:])
            next_return = model.predict(last_features)[0]
            predictions[name] = next_return
        
        # Ensemble prediction
        ensemble_prediction = np.mean(list(predictions.values()))
        
        # Current price and predicted prices
        current_price = data['close'].iloc[-1]
        predicted_prices = {
            name: current_price * (1 + pred) 
            for name, pred in predictions.items()
        }
        predicted_prices['ensemble'] = current_price * (1 + ensemble_prediction)
        
        return {
            'models_performance': results,
            'predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'predicted_prices': predicted_prices,
            'target_days': target_days,
            'feature_importance': self._get_feature_importance(
                results['random_forest']['model'], feature_cols
            ) if 'random_forest' in results else {}
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD line"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        return ema_12 - ema_26
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Get feature importance from tree-based models"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {} 