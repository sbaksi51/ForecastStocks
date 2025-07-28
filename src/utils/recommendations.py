import numpy as np
import pandas as pd
from datetime import datetime

class RecommendationEngine:
    def __init__(self):
        self.thresholds = {
            'strong_buy': 0.05,  # 5% predicted gain
            'buy': 0.02,         # 2% predicted gain
            'sell': -0.02,       # 2% predicted loss
            'strong_sell': -0.05 # 5% predicted loss
        }
    
    def get_recommendation(self, prediction, indicators, current_price):
        """Generate recommendation based on prediction and indicators"""
        # Base recommendation on predicted return
        predicted_return = prediction['return_percentage'] / 100
        
        # Get technical signal
        technical_signal = indicators.get('signals', {}).get('overall', 'HOLD')
        
        # Combine prediction and technical analysis
        score = self._calculate_recommendation_score(predicted_return, technical_signal, indicators)
        
        # Generate recommendation
        action = self._determine_action(score, predicted_return)
        confidence = self._calculate_confidence(prediction, indicators)
        
        return {
            'action': action,
            'confidence': confidence,
            'score': round(score, 2),
            'target_price': prediction['predicted_price'],
            'potential_return': f"{prediction['return_percentage']:.2f}%",
            'risk_level': self._assess_risk(indicators),
            'timeframe': f"{prediction['prediction_days']} days",
            'reasons': self._generate_reasons(action, prediction, indicators)
        }
    
    def get_detailed_recommendation(self, predictions, indicators, stock_info):
        """Generate detailed recommendation with multiple timeframes"""
        recommendations = {}
        
        for timeframe, prediction in predictions.items():
            recommendations[timeframe] = self.get_recommendation(prediction, indicators, stock_info['current_price'])
        
        # Overall recommendation based on all timeframes
        overall = self._determine_overall_recommendation(recommendations)
        
        # Add additional analysis
        overall['technical_summary'] = self._summarize_technical_indicators(indicators)
        overall['fundamental_metrics'] = self._extract_fundamental_metrics(stock_info)
        overall['risk_reward_ratio'] = self._calculate_risk_reward(predictions, indicators)
        
        return overall
    
    def _calculate_recommendation_score(self, predicted_return, technical_signal, indicators):
        """Calculate overall recommendation score"""
        # Base score from prediction
        prediction_score = predicted_return * 10
        
        # Technical signal score
        tech_score_map = {
            'STRONG BUY': 2,
            'BUY': 1,
            'HOLD': 0,
            'SELL': -1,
            'STRONG SELL': -2
        }
        tech_score = tech_score_map.get(technical_signal, 0)
        
        # Additional factors
        rsi = indicators.get('rsi', {}).get('value', 50)
        rsi_score = 0
        if rsi < 30:
            rsi_score = 1
        elif rsi > 70:
            rsi_score = -1
        
        # Volume trend
        volume_score = 1 if indicators.get('obv', {}).get('trend') == 'bullish' else -1
        
        # Combine scores with weights
        total_score = (prediction_score * 0.4) + (tech_score * 0.3) + (rsi_score * 0.2) + (volume_score * 0.1)
        
        return total_score
    
    def _determine_action(self, score, predicted_return):
        """Determine buy/sell/hold action based on score"""
        if score >= 2 and predicted_return >= self.thresholds['strong_buy']:
            return 'STRONG BUY'
        elif score >= 1 and predicted_return >= self.thresholds['buy']:
            return 'BUY'
        elif score <= -2 and predicted_return <= self.thresholds['strong_sell']:
            return 'STRONG SELL'
        elif score <= -1 and predicted_return <= self.thresholds['sell']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_confidence(self, prediction, indicators):
        """Calculate confidence level of recommendation"""
        base_confidence = prediction.get('confidence', 0.5)
        
        # Adjust based on technical indicators alignment
        signals = indicators.get('signals', {}).get('signals', [])
        if len(signals) > 0:
            positive_signals = sum(1 for s in signals if 'Buy' in s or 'bullish' in s or 'Uptrend' in s)
            negative_signals = sum(1 for s in signals if 'Sell' in s or 'bearish' in s or 'Downtrend' in s)
            
            signal_ratio = (positive_signals - negative_signals) / len(signals)
            confidence_adjustment = signal_ratio * 0.2
        else:
            confidence_adjustment = 0
        
        # Adjust based on volatility
        atr = indicators.get('atr', {}).get('volatility', 'low')
        if atr == 'high':
            confidence_adjustment -= 0.1
        
        final_confidence = max(0.1, min(0.95, base_confidence + confidence_adjustment))
        return round(final_confidence, 2)
    
    def _assess_risk(self, indicators):
        """Assess risk level based on indicators"""
        risk_factors = 0
        
        # Volatility
        if indicators.get('atr', {}).get('volatility') == 'high':
            risk_factors += 2
        
        # RSI extremes
        rsi = indicators.get('rsi', {}).get('value', 50)
        if rsi > 80 or rsi < 20:
            risk_factors += 1
        
        # Bollinger Bands position
        bb_position = indicators.get('bollinger_bands', {}).get('price_position', 'within_bands')
        if bb_position in ['above_upper', 'below_lower']:
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 3:
            return 'HIGH'
        elif risk_factors >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_reasons(self, action, prediction, indicators):
        """Generate reasons for the recommendation"""
        reasons = []
        
        # Prediction-based reasons
        if prediction['return_percentage'] > 0:
            reasons.append(f"Predicted {prediction['return_percentage']:.2f}% gain in {prediction['prediction_days']} days")
        else:
            reasons.append(f"Predicted {abs(prediction['return_percentage']):.2f}% decline in {prediction['prediction_days']} days")
        
        # Technical indicator reasons
        tech_signals = indicators.get('signals', {}).get('signals', [])
        for signal in tech_signals[:3]:  # Top 3 signals
            reasons.append(signal)
        
        # Trend reasons
        if indicators.get('sma_20', {}).get('trend') == 'bullish':
            reasons.append("Price trending above 20-day moving average")
        
        # Support/resistance
        support = indicators.get('support_resistance', {}).get('support', 0)
        resistance = indicators.get('support_resistance', {}).get('resistance', 0)
        current_price = prediction['current_price']
        
        if current_price - support < (resistance - support) * 0.2:
            reasons.append("Price near support level")
        elif resistance - current_price < (resistance - support) * 0.2:
            reasons.append("Price near resistance level")
        
        return reasons[:5]  # Return top 5 reasons
    
    def _determine_overall_recommendation(self, timeframe_recommendations):
        """Determine overall recommendation from multiple timeframes"""
        # Count recommendations
        action_counts = {}
        total_confidence = 0
        
        for tf, rec in timeframe_recommendations.items():
            action = rec['action']
            action_counts[action] = action_counts.get(action, 0) + 1
            total_confidence += rec['confidence']
        
        # Most common action
        overall_action = max(action_counts, key=action_counts.get)
        
        # Average confidence
        avg_confidence = total_confidence / len(timeframe_recommendations)
        
        # Compile overall recommendation
        return {
            'action': overall_action,
            'confidence': round(avg_confidence, 2),
            'timeframe_analysis': timeframe_recommendations,
            'consensus': f"{action_counts.get(overall_action, 0)}/{len(timeframe_recommendations)} timeframes agree"
        }
    
    def _summarize_technical_indicators(self, indicators):
        """Summarize technical indicators"""
        summary = {
            'trend': 'BULLISH' if indicators.get('sma_20', {}).get('trend') == 'bullish' else 'BEARISH',
            'momentum': indicators.get('rsi', {}).get('signal', 'neutral').upper(),
            'volatility': indicators.get('atr', {}).get('volatility', 'low').upper(),
            'volume_trend': indicators.get('obv', {}).get('trend', 'neutral').upper(),
            'signal_strength': indicators.get('signals', {}).get('strength', 0)
        }
        
        return summary
    
    def _extract_fundamental_metrics(self, stock_info):
        """Extract key fundamental metrics"""
        return {
            'pe_ratio': stock_info.get('pe_ratio', 'N/A'),
            'market_cap': self._format_market_cap(stock_info.get('market_cap', 0)),
            'dividend_yield': f"{stock_info.get('dividend_yield', 0) * 100:.2f}%",
            '52_week_position': self._calculate_52_week_position(
                stock_info.get('current_price', 0),
                stock_info.get('52_week_low', 0),
                stock_info.get('52_week_high', 0)
            )
        }
    
    def _format_market_cap(self, market_cap):
        """Format market cap in billions/millions"""
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        else:
            return f"${market_cap:,.0f}"
    
    def _calculate_52_week_position(self, current, low, high):
        """Calculate position within 52-week range"""
        if high == low or high == 0:
            return "N/A"
        
        position = (current - low) / (high - low) * 100
        return f"{position:.1f}%"
    
    def _calculate_risk_reward(self, predictions, indicators):
        """Calculate risk-reward ratio"""
        # Get potential upside from predictions
        max_return = max(p['return_percentage'] for p in predictions.values())
        
        # Estimate downside risk
        support = indicators.get('support_resistance', {}).get('support', 0)
        current_price = next(iter(predictions.values()))['current_price']
        
        if support > 0 and current_price > 0:
            downside_risk = ((current_price - support) / current_price) * 100
        else:
            # Use ATR-based risk estimate
            atr = indicators.get('atr', {}).get('value', 0)
            downside_risk = (atr / current_price * 100) if current_price > 0 else 5
        
        # Calculate ratio
        if downside_risk > 0:
            ratio = max_return / downside_risk
            return f"{ratio:.2f}:1"
        else:
            return "N/A"
    
    def calculate_market_sentiment(self, all_recommendations):
        """Calculate overall market sentiment from multiple recommendations"""
        if not all_recommendations:
            return "NEUTRAL"
        
        buy_count = sum(1 for r in all_recommendations if 'BUY' in r['recommendation']['action'])
        sell_count = sum(1 for r in all_recommendations if 'SELL' in r['recommendation']['action'])
        total = len(all_recommendations)
        
        buy_percentage = (buy_count / total) * 100
        sell_percentage = (sell_count / total) * 100
        
        if buy_percentage > 60:
            sentiment = "BULLISH"
        elif sell_percentage > 60:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'sentiment': sentiment,
            'buy_percentage': round(buy_percentage, 1),
            'sell_percentage': round(sell_percentage, 1),
            'hold_percentage': round(100 - buy_percentage - sell_percentage, 1)
        } 