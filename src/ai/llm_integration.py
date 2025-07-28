"""
LLM Integration for Market Analysis and Sentiment
Supports multiple LLM providers for comprehensive analysis
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# LLM Providers
import openai
import anthropic
import google.generativeai as genai

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# News and Social Media
import requests
from newsapi import NewsApiClient
import tweepy
import praw

# Initialize NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """Multi-LLM analyzer for comprehensive market analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize LLM providers
        self._init_llm_providers()
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize data sources
        self._init_data_sources()
        
    def _init_llm_providers(self):
        """Initialize available LLM providers"""
        self.llm_providers = {}
        
        # OpenAI
        if self.config.get('openai_api_key'):
            openai.api_key = self.config['openai_api_key']
            self.llm_providers['openai'] = True
            
        # Anthropic
        if self.config.get('anthropic_api_key'):
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.config['anthropic_api_key']
            )
            self.llm_providers['anthropic'] = True
            
        # Google AI
        if self.config.get('google_api_key'):
            genai.configure(api_key=self.config['google_api_key'])
            self.llm_providers['google'] = True
            
    def _init_data_sources(self):
        """Initialize news and social media data sources"""
        # News API
        if self.config.get('news_api_key'):
            self.news_api = NewsApiClient(api_key=self.config['news_api_key'])
        else:
            self.news_api = None
            
        # Twitter/X API
        if self.config.get('twitter_bearer_token'):
            self.twitter_client = tweepy.Client(
                bearer_token=self.config['twitter_bearer_token']
            )
        else:
            self.twitter_client = None
            
        # Reddit API
        if self.config.get('reddit_client_id'):
            self.reddit = praw.Reddit(
                client_id=self.config['reddit_client_id'],
                client_secret=self.config['reddit_client_secret'],
                user_agent='StockAnalyzer/1.0'
            )
        else:
            self.reddit = None
    
    def analyze_market_sentiment(self, symbol: str, company_name: str) -> Dict:
        """Comprehensive sentiment analysis using multiple sources"""
        
        # Gather data from multiple sources
        news_data = self._fetch_news(symbol, company_name)
        social_data = self._fetch_social_media(symbol)
        
        # Analyze sentiment from each source
        news_sentiment = self._analyze_news_sentiment(news_data)
        social_sentiment = self._analyze_social_sentiment(social_data)
        
        # Get LLM interpretation
        llm_analysis = self._get_llm_market_analysis(
            symbol, company_name, news_data, social_data
        )
        
        # Combine all sentiments
        combined_sentiment = self._combine_sentiments(
            news_sentiment, social_sentiment, llm_analysis
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'llm_analysis': llm_analysis,
            'combined_sentiment': combined_sentiment,
            'data_sources': {
                'news_articles': len(news_data),
                'social_posts': len(social_data)
            }
        }
    
    def _fetch_news(self, symbol: str, company_name: str) -> List[Dict]:
        """Fetch recent news articles"""
        if not self.news_api:
            return []
        
        try:
            # Search for company news
            articles = self.news_api.get_everything(
                q=f"{company_name} OR {symbol}",
                from_param=(datetime.now() - timedelta(days=7)).isoformat(),
                language='en',
                sort_by='relevancy',
                page_size=20
            )
            
            return articles.get('articles', [])
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _fetch_social_media(self, symbol: str) -> List[Dict]:
        """Fetch social media posts"""
        social_data = []
        
        # Twitter/X
        if self.twitter_client:
            try:
                tweets = self.twitter_client.search_recent_tweets(
                    query=f"${symbol} -is:retweet lang:en",
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics']
                )
                if tweets.data:
                    social_data.extend([
                        {'source': 'twitter', 'text': tweet.text, 'metrics': tweet.public_metrics}
                        for tweet in tweets.data
                    ])
            except Exception as e:
                logger.error(f"Error fetching tweets: {e}")
        
        # Reddit
        if self.reddit:
            try:
                subreddits = ['wallstreetbets', 'stocks', 'investing']
                for subreddit_name in subreddits:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    for post in subreddit.search(symbol, limit=10, time_filter='week'):
                        social_data.append({
                            'source': 'reddit',
                            'text': f"{post.title} {post.selftext}",
                            'score': post.score,
                            'comments': post.num_comments
                        })
            except Exception as e:
                logger.error(f"Error fetching Reddit posts: {e}")
        
        return social_data
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment from news articles"""
        if not news_data:
            return {'score': 0, 'label': 'neutral', 'confidence': 0}
        
        sentiments = []
        for article in news_data:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            
            sentiments.append({
                'vader': vader_scores['compound'],
                'textblob': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        
        # Average sentiments
        avg_sentiment = sum(s['vader'] for s in sentiments) / len(sentiments)
        
        # Determine label
        if avg_sentiment >= 0.1:
            label = 'positive'
        elif avg_sentiment <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': round(avg_sentiment, 3),
            'label': label,
            'confidence': round(1 - sum(s['subjectivity'] for s in sentiments) / len(sentiments), 2),
            'article_count': len(news_data)
        }
    
    def _analyze_social_sentiment(self, social_data: List[Dict]) -> Dict:
        """Analyze sentiment from social media"""
        if not social_data:
            return {'score': 0, 'label': 'neutral', 'confidence': 0}
        
        sentiments = []
        engagement_weights = []
        
        for post in social_data:
            # Analyze sentiment
            scores = self.vader.polarity_scores(post['text'])
            sentiments.append(scores['compound'])
            
            # Weight by engagement
            if post['source'] == 'twitter':
                weight = 1 + (post.get('metrics', {}).get('like_count', 0) / 100)
            elif post['source'] == 'reddit':
                weight = 1 + (post.get('score', 0) / 100)
            else:
                weight = 1
            
            engagement_weights.append(weight)
        
        # Weighted average sentiment
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, engagement_weights)) / sum(engagement_weights)
        
        # Determine label
        if weighted_sentiment >= 0.1:
            label = 'positive'
        elif weighted_sentiment <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': round(weighted_sentiment, 3),
            'label': label,
            'post_count': len(social_data),
            'avg_engagement': round(sum(engagement_weights) / len(engagement_weights), 2)
        }
    
    def _get_llm_market_analysis(self, symbol: str, company_name: str, 
                                news_data: List[Dict], social_data: List[Dict]) -> Dict:
        """Get comprehensive analysis from available LLMs"""
        
        # Prepare context
        news_summary = self._summarize_news(news_data[:5])  # Top 5 articles
        social_summary = self._summarize_social(social_data[:10])  # Top 10 posts
        
        prompt = f"""
        Analyze {company_name} ({symbol}) based on recent market data:
        
        Recent News Summary:
        {news_summary}
        
        Social Media Sentiment:
        {social_summary}
        
        Provide:
        1. Overall market sentiment (Bullish/Bearish/Neutral)
        2. Key factors driving sentiment
        3. Potential risks and catalysts
        4. Short-term outlook (1-7 days)
        5. Trading recommendation with confidence level
        """
        
        # Try available LLMs
        analysis = None
        
        if 'openai' in self.llm_providers:
            analysis = self._query_openai(prompt)
        elif 'anthropic' in self.llm_providers:
            analysis = self._query_anthropic(prompt)
        elif 'google' in self.llm_providers:
            analysis = self._query_google(prompt)
        
        if not analysis:
            # Fallback to rule-based analysis
            analysis = self._fallback_analysis(news_data, social_data)
        
        return analysis
    
    def _query_openai(self, prompt: str) -> Dict:
        """Query OpenAI GPT models"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _query_anthropic(self, prompt: str) -> Dict:
        """Query Anthropic Claude"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            content = response.content[0].text
            return self._parse_llm_response(content)
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
    
    def _query_google(self, prompt: str) -> Dict:
        """Query Google Gemini"""
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return self._parse_llm_response(response.text)
            
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            return None
    
    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response into structured format"""
        # Simple parsing - in production, use more sophisticated parsing
        sentiment = 'neutral'
        if 'bullish' in content.lower():
            sentiment = 'bullish'
        elif 'bearish' in content.lower():
            sentiment = 'bearish'
        
        return {
            'sentiment': sentiment,
            'analysis': content,
            'confidence': 0.75,  # Default confidence
            'source': 'llm'
        }
    
    def _combine_sentiments(self, news_sent: Dict, social_sent: Dict, llm_analysis: Dict) -> Dict:
        """Combine all sentiment sources into final recommendation"""
        
        # Weight different sources
        weights = {
            'news': 0.3,
            'social': 0.2,
            'llm': 0.5
        }
        
        # Convert to numeric scores
        sentiment_scores = {
            'positive': 1, 'bullish': 1,
            'neutral': 0,
            'negative': -1, 'bearish': -1
        }
        
        # Calculate weighted score
        news_score = sentiment_scores.get(news_sent['label'], 0)
        social_score = sentiment_scores.get(social_sent['label'], 0)
        llm_score = sentiment_scores.get(llm_analysis.get('sentiment', 'neutral'), 0)
        
        weighted_score = (
            news_score * weights['news'] +
            social_score * weights['social'] +
            llm_score * weights['llm']
        )
        
        # Determine final sentiment
        if weighted_score > 0.3:
            final_sentiment = 'bullish'
        elif weighted_score < -0.3:
            final_sentiment = 'bearish'
        else:
            final_sentiment = 'neutral'
        
        return {
            'sentiment': final_sentiment,
            'score': round(weighted_score, 3),
            'confidence': round(
                (news_sent.get('confidence', 0.5) * weights['news'] +
                 0.7 * weights['social'] +  # Social media confidence
                 llm_analysis.get('confidence', 0.75) * weights['llm']),
                2
            )
        }
    
    def _summarize_news(self, articles: List[Dict]) -> str:
        """Summarize news articles"""
        if not articles:
            return "No recent news available."
        
        summary_lines = []
        for article in articles:
            title = article.get('title', '')
            source = article.get('source', {}).get('name', 'Unknown')
            summary_lines.append(f"- {title} (Source: {source})")
        
        return "\n".join(summary_lines)
    
    def _summarize_social(self, posts: List[Dict]) -> str:
        """Summarize social media posts"""
        if not posts:
            return "No social media data available."
        
        positive = sum(1 for p in posts if self.vader.polarity_scores(p['text'])['compound'] > 0.1)
        negative = sum(1 for p in posts if self.vader.polarity_scores(p['text'])['compound'] < -0.1)
        neutral = len(posts) - positive - negative
        
        return f"Analyzed {len(posts)} posts: {positive} positive, {negative} negative, {neutral} neutral"
    
    def _fallback_analysis(self, news_data: List[Dict], social_data: List[Dict]) -> Dict:
        """Fallback analysis when LLMs are not available"""
        return {
            'sentiment': 'neutral',
            'analysis': "Analysis based on sentiment scores from news and social media data.",
            'confidence': 0.6,
            'source': 'rule-based'
        } 