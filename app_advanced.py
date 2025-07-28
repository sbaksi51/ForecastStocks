from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all components
from src.models.stock_predictor import StockPredictor
from src.data_processing.stock_data import StockDataProcessor
from src.utils.technical_indicators import TechnicalAnalysis
from src.utils.recommendations import RecommendationEngine

# Import AI components
from src.ai.llm_integration import LLMAnalyzer
from src.ai.time_series_models import TimeSeriesPredictor
from src.ai.quant_models import QuantitativeAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG = {
    # LLM API Keys (from environment variables)
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
    'google_api_key': os.getenv('GOOGLE_API_KEY'),
    
    # Data source API keys
    'news_api_key': os.getenv('NEWS_API_KEY'),
    'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
    'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
    'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    
    # Model settings
    'model_dir': 'models/trained/',
    'cache_dir': 'cache/',
    
    # Trading parameters
    'lookback_period': 60,
    'prediction_days': [1, 7, 30],
    'confidence_threshold': 0.7
}

# Initialize components
stock_predictor = StockPredictor()
data_processor = StockDataProcessor()
tech_analysis = TechnicalAnalysis()
recommendation_engine = RecommendationEngine()

# Initialize AI components
llm_analyzer = LLMAnalyzer(CONFIG)
time_series_predictor = TimeSeriesPredictor(CONFIG)
quant_analyzer = QuantitativeAnalyzer(CONFIG)

# Sector mappings
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'CVS', 'ABT', 'MRK', 'TMO', 'ABBV'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP'],
    'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO'],
    'Industrial': ['BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM']
}

@app.route('/')
def index():
    return render_template('index_advanced.html')

@app.route('/api/stocks/<sector>')
def get_sector_stocks_advanced(sector):
    """Get stocks with comprehensive AI analysis"""
    try:
        if sector not in SECTORS:
            return jsonify({'error': 'Invalid sector'}), 400
        
        stocks_data = []
        
        for symbol in SECTORS[sector]:
            # Fetch stock data
            stock_info = data_processor.fetch_stock_data(symbol, period='6mo')
            
            # Technical indicators
            indicators = tech_analysis.calculate_indicators(stock_info['historical_data'])
            
            # Basic prediction
            basic_prediction = stock_predictor.predict(stock_info['historical_data'], indicators)
            
            # LLM sentiment analysis
            sentiment_analysis = llm_analyzer.analyze_market_sentiment(
                symbol, stock_info['name']
            )
            
            # Quantitative analysis
            quant_results = {
                'mean_reversion': quant_analyzer.analyze_mean_reversion(
                    stock_info['historical_data'][['close']]
                ),
                'momentum': quant_analyzer.analyze_momentum(
                    stock_info['historical_data'][['close', 'volume']]
                ),
                'volatility': quant_analyzer.volatility_analysis(
                    stock_info['historical_data']
                )
            }
            
            # Enhanced recommendation combining all analyses
            enhanced_recommendation = _generate_enhanced_recommendation(
                basic_prediction, indicators, sentiment_analysis, quant_results
            )
            
            stocks_data.append({
                'symbol': symbol,
                'name': stock_info['name'],
                'current_price': stock_info['current_price'],
                'change_percent': stock_info['change_percent'],
                'prediction': basic_prediction,
                'sentiment': sentiment_analysis['combined_sentiment'],
                'quant_signals': {
                    'mean_reversion': quant_results['mean_reversion']['current_signal'],
                    'momentum': quant_results['momentum']['current_signal'],
                    'volatility_regime': quant_results['volatility']['volatility_regime']
                },
                'recommendation': enhanced_recommendation,
                'indicators': {
                    'rsi': indicators['rsi']['value'],
                    'macd': indicators['macd']['macd'],
                    'bollinger': indicators['bollinger_bands']['price_position']
                }
            })
        
        return jsonify({
            'sector': sector,
            'stocks': stocks_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/advanced')
def get_advanced_analysis(symbol):
    """Get comprehensive AI-powered analysis for a specific stock"""
    try:
        # Fetch comprehensive data
        stock_info = data_processor.fetch_stock_data(symbol, period='1y')
        hist_data = stock_info['historical_data']
        
        # Rename columns for consistency
        hist_data_clean = hist_data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Technical indicators
        indicators = tech_analysis.calculate_indicators(stock_info['historical_data'])
        
        # Time series predictions (check if we have enough data)
        time_series_predictions = {}
        if len(hist_data_clean) >= 100:  # Need sufficient data for LSTM
            try:
                # Train or load existing models
                if f"{symbol}_lstm_model.h5" not in os.listdir(CONFIG['model_dir']):
                    # Train models (this would typically be done offline)
                    time_series_predictions = {
                        'status': 'training_required',
                        'message': 'Models need to be trained first'
                    }
                else:
                    # Make predictions
                    time_series_predictions = time_series_predictor.predict_ensemble(
                        symbol, hist_data_clean, days_ahead=5
                    )
            except Exception as e:
                time_series_predictions = {'error': str(e)}
        
        # LLM sentiment and market analysis
        sentiment_analysis = llm_analyzer.analyze_market_sentiment(
            symbol, stock_info['name']
        )
        
        # Quantitative strategies
        quant_analysis = {
            'mean_reversion': quant_analyzer.analyze_mean_reversion(hist_data_clean),
            'momentum': quant_analyzer.analyze_momentum(hist_data_clean),
            'volatility': quant_analyzer.volatility_analysis(hist_data_clean),
            'ml_prediction': quant_analyzer.ml_price_prediction(hist_data_clean)
        }
        
        # Risk metrics
        returns = hist_data_clean['close'].pct_change().dropna()
        risk_metrics = {
            'daily_var_95': np.percentile(returns, 5),
            'daily_cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': (hist_data_clean['close'] / hist_data_clean['close'].cummax() - 1).min()
        }
        
        # Comprehensive recommendation
        final_recommendation = _generate_comprehensive_recommendation(
            stock_info, indicators, sentiment_analysis, quant_analysis, 
            time_series_predictions, risk_metrics
        )
        
        return jsonify({
            'symbol': symbol,
            'company_info': {
                'name': stock_info['name'],
                'sector': stock_info['sector'],
                'market_cap': stock_info['market_cap'],
                'pe_ratio': stock_info['pe_ratio'],
                'dividend_yield': stock_info['dividend_yield']
            },
            'current_metrics': {
                'price': stock_info['current_price'],
                'change_percent': stock_info['change_percent'],
                'volume': stock_info['volume'],
                '52_week_high': stock_info['52_week_high'],
                '52_week_low': stock_info['52_week_low']
            },
            'technical_analysis': indicators,
            'sentiment_analysis': sentiment_analysis,
            'quantitative_analysis': quant_analysis,
            'time_series_predictions': time_series_predictions,
            'risk_metrics': risk_metrics,
            'recommendation': final_recommendation,
            'chart_data': hist_data_clean.tail(180).to_dict('records')  # Last 6 months
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio allocation using risk parity"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        target_vol = data.get('target_volatility', 0.15)
        
        if len(symbols) < 2:
            return jsonify({'error': 'Need at least 2 symbols for portfolio optimization'}), 400
        
        # Fetch returns data
        returns_data = pd.DataFrame()
        
        for symbol in symbols:
            stock_data = data_processor.fetch_stock_data(symbol, period='1y')
            returns = stock_data['historical_data']['Close'].pct_change()
            returns_data[symbol] = returns
        
        returns_data = returns_data.dropna()
        
        # Risk parity optimization
        optimization_result = quant_analyzer.risk_parity_allocation(
            returns_data, target_vol
        )
        
        return jsonify(optimization_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pairs-trading', methods=['POST'])
def analyze_pairs_trading():
    """Analyze pairs trading opportunity"""
    try:
        data = request.json
        symbol1 = data.get('symbol1')
        symbol2 = data.get('symbol2')
        
        # Fetch data for both stocks
        stock1_data = data_processor.fetch_stock_data(symbol1, period='6mo')
        stock2_data = data_processor.fetch_stock_data(symbol2, period='6mo')
        
        # Prepare data
        hist1 = stock1_data['historical_data'].rename(columns={'Close': 'close'})
        hist2 = stock2_data['historical_data'].rename(columns={'Close': 'close'})
        
        # Pairs trading analysis
        pairs_result = quant_analyzer.pairs_trading_analysis(hist1, hist2)
        
        return jsonify({
            'pair': f"{symbol1}/{symbol2}",
            'analysis': pairs_result,
            'recommendation': _generate_pairs_recommendation(pairs_result)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-models/<symbol>', methods=['POST'])
def train_models(symbol):
    """Train time series models for a specific stock"""
    try:
        # Fetch training data
        stock_data = data_processor.fetch_stock_data(symbol, period='2y')
        hist_data = stock_data['historical_data'].rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Train ensemble models
        training_results = time_series_predictor.train_ensemble(
            symbol, hist_data, epochs=30
        )
        
        # Train Prophet model
        prophet_model = time_series_predictor.train_prophet_model(symbol, hist_data)
        
        return jsonify({
            'symbol': symbol,
            'models_trained': list(training_results.keys()),
            'training_complete': True,
            'message': f'Successfully trained models for {symbol}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _generate_enhanced_recommendation(prediction, indicators, sentiment, quant_results):
    """Generate enhanced recommendation combining all analyses"""
    
    # Base scores
    price_score = prediction['return_percentage'] / 10  # Normalize to -1 to 1
    technical_score = indicators['signals']['strength'] / 5  # Normalize
    sentiment_score = sentiment['combined_sentiment']['score']
    
    # Quant scores
    mean_reversion_score = quant_results['mean_reversion']['current_signal']
    momentum_score = quant_results['momentum']['current_signal'] / 2
    
    # Volatility adjustment
    vol_regime = quant_results['volatility']['volatility_regime']
    vol_multiplier = 0.7 if vol_regime == 'high_volatility' else 1.0
    
    # Combined score
    combined_score = (
        price_score * 0.2 +
        technical_score * 0.25 +
        sentiment_score * 0.3 +
        mean_reversion_score * 0.15 +
        momentum_score * 0.1
    ) * vol_multiplier
    
    # Determine action
    if combined_score > 0.4:
        action = 'STRONG BUY'
    elif combined_score > 0.2:
        action = 'BUY'
    elif combined_score < -0.4:
        action = 'STRONG SELL'
    elif combined_score < -0.2:
        action = 'SELL'
    else:
        action = 'HOLD'
    
    # Confidence based on agreement between signals
    signals = [price_score > 0, technical_score > 0, sentiment_score > 0, 
               mean_reversion_score > 0, momentum_score > 0]
    agreement = sum(signals) / len(signals)
    confidence = 0.5 + abs(agreement - 0.5)
    
    return {
        'action': action,
        'score': round(combined_score, 3),
        'confidence': round(confidence, 2),
        'components': {
            'price_prediction': round(price_score, 2),
            'technical': round(technical_score, 2),
            'sentiment': round(sentiment_score, 2),
            'mean_reversion': round(mean_reversion_score, 2),
            'momentum': round(momentum_score, 2)
        }
    }

def _generate_comprehensive_recommendation(stock_info, indicators, sentiment, 
                                         quant_analysis, time_series, risk_metrics):
    """Generate comprehensive recommendation from all analyses"""
    
    recommendations = []
    total_score = 0
    weights = {
        'technical': 0.2,
        'sentiment': 0.25,
        'quantitative': 0.25,
        'time_series': 0.2,
        'risk': 0.1
    }
    
    # Technical analysis score
    tech_signal = indicators['signals']['overall']
    tech_score = {'STRONG BUY': 2, 'BUY': 1, 'HOLD': 0, 'SELL': -1, 'STRONG SELL': -2}.get(tech_signal, 0)
    recommendations.append(f"Technical indicators suggest: {tech_signal}")
    total_score += tech_score * weights['technical']
    
    # Sentiment score
    sent_score = sentiment['combined_sentiment']['score']
    recommendations.append(f"Market sentiment is {sentiment['combined_sentiment']['sentiment']}")
    total_score += sent_score * weights['sentiment']
    
    # Quantitative score
    quant_score = (
        quant_analysis['mean_reversion']['current_signal'] * 0.3 +
        quant_analysis['momentum']['current_signal'] * 0.4 +
        (1 if quant_analysis['ml_prediction']['ensemble_prediction'] > 0 else -1) * 0.3
    )
    recommendations.append(f"Quantitative models signal: {quant_score > 0 and 'Positive' or 'Negative'}")
    total_score += quant_score * weights['quantitative']
    
    # Time series prediction score (if available)
    if 'predictions' in time_series:
        ts_return = (time_series['predictions']['ensemble'][0] / stock_info['current_price'] - 1) * 100
        ts_score = ts_return / 5  # Normalize
        recommendations.append(f"AI models predict {ts_return:.2f}% return in 1 day")
        total_score += ts_score * weights['time_series']
    
    # Risk adjustment
    risk_score = 0
    if risk_metrics['sharpe_ratio'] > 1:
        risk_score += 0.5
    if risk_metrics['max_drawdown'] > -0.2:  # Less than 20% drawdown
        risk_score += 0.5
    total_score += risk_score * weights['risk']
    
    # Final recommendation
    if total_score > 0.6:
        final_action = 'STRONG BUY'
    elif total_score > 0.2:
        final_action = 'BUY'
    elif total_score < -0.6:
        final_action = 'STRONG SELL'
    elif total_score < -0.2:
        final_action = 'SELL'
    else:
        final_action = 'HOLD'
    
    return {
        'action': final_action,
        'score': round(total_score, 3),
        'confidence': round(0.6 + min(0.35, abs(total_score) * 0.5), 2),
        'key_factors': recommendations[:4],
        'risk_adjusted': True,
        'ai_consensus': 'Bullish' if total_score > 0 else 'Bearish' if total_score < 0 else 'Neutral'
    }

def _generate_pairs_recommendation(pairs_result):
    """Generate recommendation for pairs trading"""
    
    if not pairs_result['is_cointegrated']:
        return {
            'action': 'NO TRADE',
            'reason': 'Stocks are not cointegrated',
            'confidence': 0.1
        }
    
    z_score = pairs_result['current_z_score']
    
    if z_score < -2:
        action = 'OPEN LONG'
        reason = 'Spread is significantly below mean'
    elif z_score > 2:
        action = 'OPEN SHORT'
        reason = 'Spread is significantly above mean'
    elif abs(z_score) < 0.5 and pairs_result['current_signal'] != 0:
        action = 'CLOSE POSITION'
        reason = 'Spread has reverted to mean'
    else:
        action = 'WAIT'
        reason = 'No clear signal'
    
    confidence = min(0.9, 0.3 + abs(z_score) * 0.2)
    
    return {
        'action': action,
        'reason': reason,
        'confidence': round(confidence, 2),
        'z_score': round(z_score, 2),
        'hedge_ratio': round(pairs_result['hedge_ratio'], 4)
    }

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    
    app.run(debug=True, port=5001)  # Using port 5001 for advanced version 