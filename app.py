from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from src.models.stock_predictor import StockPredictor
from src.data_processing.stock_data import StockDataProcessor
from src.utils.technical_indicators import TechnicalAnalysis
from src.utils.recommendations import RecommendationEngine

app = Flask(__name__)
CORS(app)

# Initialize components
stock_predictor = StockPredictor()
data_processor = StockDataProcessor()
tech_analysis = TechnicalAnalysis()
recommendation_engine = RecommendationEngine()

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
    return render_template('index.html')

@app.route('/api/stocks/<sector>')
def get_sector_stocks(sector):
    """Get all stocks for a specific sector with predictions"""
    try:
        if sector not in SECTORS:
            return jsonify({'error': 'Invalid sector'}), 400
        
        stocks_data = []
        for symbol in SECTORS[sector]:
            # Fetch stock data
            stock_info = data_processor.fetch_stock_data(symbol)
            
            # Get technical indicators
            indicators = tech_analysis.calculate_indicators(stock_info['historical_data'])
            
            # Make prediction
            prediction = stock_predictor.predict(stock_info['historical_data'], indicators)
            
            # Get recommendation
            recommendation = recommendation_engine.get_recommendation(
                prediction, indicators, stock_info['current_price']
            )
            
            stocks_data.append({
                'symbol': symbol,
                'name': stock_info['name'],
                'current_price': stock_info['current_price'],
                'change_percent': stock_info['change_percent'],
                'prediction': prediction,
                'recommendation': recommendation,
                'indicators': indicators
            })
        
        return jsonify({
            'sector': sector,
            'stocks': stocks_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>')
def get_stock_details(symbol):
    """Get detailed analysis for a specific stock"""
    try:
        # Fetch comprehensive stock data
        stock_info = data_processor.fetch_stock_data(symbol, period='1y')
        
        # Calculate technical indicators
        indicators = tech_analysis.calculate_indicators(stock_info['historical_data'])
        
        # Make predictions for different timeframes
        predictions = {
            '1_day': stock_predictor.predict(stock_info['historical_data'], indicators, days=1),
            '7_days': stock_predictor.predict(stock_info['historical_data'], indicators, days=7),
            '30_days': stock_predictor.predict(stock_info['historical_data'], indicators, days=30)
        }
        
        # Get recommendation with confidence score
        recommendation = recommendation_engine.get_detailed_recommendation(
            predictions, indicators, stock_info
        )
        
        return jsonify({
            'symbol': symbol,
            'company_info': stock_info,
            'technical_indicators': indicators,
            'predictions': predictions,
            'recommendation': recommendation,
            'chart_data': stock_info['historical_data'].to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors')
def get_all_sectors():
    """Get list of all available sectors"""
    return jsonify({
        'sectors': list(SECTORS.keys()),
        'total_stocks': sum(len(stocks) for stocks in SECTORS.values())
    })

@app.route('/api/market-overview')
def get_market_overview():
    """Get overall market overview with top recommendations"""
    try:
        all_recommendations = []
        
        for sector, symbols in SECTORS.items():
            for symbol in symbols[:3]:  # Top 3 from each sector
                stock_info = data_processor.fetch_stock_data(symbol)
                indicators = tech_analysis.calculate_indicators(stock_info['historical_data'])
                prediction = stock_predictor.predict(stock_info['historical_data'], indicators)
                recommendation = recommendation_engine.get_recommendation(
                    prediction, indicators, stock_info['current_price']
                )
                
                all_recommendations.append({
                    'symbol': symbol,
                    'sector': sector,
                    'recommendation': recommendation,
                    'predicted_return': prediction['return_percentage']
                })
        
        # Sort by predicted return
        buy_recommendations = [r for r in all_recommendations if r['recommendation']['action'] == 'BUY']
        buy_recommendations.sort(key=lambda x: x['predicted_return'], reverse=True)
        
        return jsonify({
            'top_buys': buy_recommendations[:10],
            'market_sentiment': recommendation_engine.calculate_market_sentiment(all_recommendations),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, port=5000) 