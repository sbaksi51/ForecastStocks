# AI Stock Predictor - Advanced Edition

A comprehensive AI-powered stock prediction and analysis platform that combines Large Language Models (LLMs), advanced time-series models, and quantitative trading strategies to provide accurate market predictions and recommendations.

## 🚀 Features

### Core Capabilities
- **Real-time Stock Analysis**: Live data from Yahoo Finance across all major sectors
- **Multi-Model AI Predictions**: Ensemble of LSTM, GRU, Transformer, and CNN-LSTM models
- **LLM-Powered Sentiment Analysis**: Integration with OpenAI, Anthropic, and Google AI
- **Quantitative Trading Strategies**: Mean reversion, momentum, pairs trading, and risk parity
- **Smart Recommendations**: AI-driven BUY/SELL/HOLD signals with confidence scores

### AI & Machine Learning Components

#### 1. **LLM Integration** (`src/ai/llm_integration.py`)
- Multi-provider support (OpenAI GPT-4, Anthropic Claude, Google Gemini)
- News sentiment analysis from multiple sources
- Social media monitoring (Twitter/X, Reddit)
- Earnings call interpretation
- Market narrative generation

#### 2. **Time Series Models** (`src/ai/time_series_models.py`)
- **LSTM Networks**: Long-term pattern recognition
- **GRU Models**: Efficient sequence learning
- **Transformer Architecture**: Attention-based predictions
- **CNN-LSTM Hybrid**: Feature extraction + sequence modeling
- **Facebook Prophet**: Seasonal trend analysis
- **Ensemble Predictions**: Weighted combination of all models

#### 3. **Quantitative Strategies** (`src/ai/quant_models.py`)
- **Mean Reversion**: Statistical arbitrage opportunities
- **Momentum Trading**: Trend-following strategies
- **Pairs Trading**: Cointegration analysis
- **Volatility Analysis**: GARCH models for risk assessment
- **Risk Parity**: Portfolio optimization
- **ML Price Prediction**: Random Forest, Gradient Boosting, SVM ensemble

### Technical Indicators
- RSI, MACD, Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Support/Resistance with Fibonacci levels

## 📋 Installation

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sbaksi51/ForecastStocks.git
   cd ForecastStocks
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

### API Keys Required

For full functionality, you'll need:
- **LLM APIs** (at least one):
  - OpenAI API Key
  - Anthropic API Key
  - Google AI API Key
- **Data Sources**:
  - News API Key (newsapi.org)
  - Twitter/X API Bearer Token
  - Reddit API Credentials

## 🎯 Usage

### Basic Version (No AI)
```bash
python app.py
# Visit http://localhost:5000
```

### Advanced AI Version
```bash
python app_advanced.py
# Visit http://localhost:5001
```

## 📊 API Endpoints

### Basic Endpoints
- `GET /` - Web interface
- `GET /api/sectors` - List all sectors
- `GET /api/stocks/<sector>` - Stocks by sector with AI analysis
- `GET /api/stock/<symbol>` - Detailed stock analysis
- `GET /api/market-overview` - Market sentiment overview

### Advanced AI Endpoints
- `GET /api/stock/<symbol>/advanced` - Comprehensive AI analysis
- `POST /api/portfolio/optimize` - Risk parity portfolio optimization
- `POST /api/pairs-trading` - Analyze pairs trading opportunities
- `POST /api/train-models/<symbol>` - Train custom models for a stock

## 🏗️ Architecture

```
ForecastStocks/
├── app.py                 # Basic Flask app
├── app_advanced.py        # AI-enhanced version
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
├── templates/            
│   └── index.html        # Web interface
├── src/
│   ├── ai/               # AI/ML components
│   │   ├── llm_integration.py    # LLM sentiment analysis
│   │   ├── time_series_models.py # LSTM/GRU/Transformer
│   │   └── quant_models.py       # Quantitative strategies
│   ├── data_processing/
│   │   └── stock_data.py         # Data fetching & caching
│   ├── models/
│   │   └── stock_predictor.py    # Basic prediction models
│   └── utils/
│       ├── technical_indicators.py # Technical analysis
│       └── recommendations.py      # Recommendation engine
└── models/               # Trained model storage
    └── trained/
```

## 🤖 How It Works

### 1. Data Collection
- Real-time stock prices from Yahoo Finance
- News articles from News API
- Social media sentiment from Twitter/Reddit
- Technical indicators calculated on-the-fly

### 2. AI Analysis Pipeline
```
Raw Data → Technical Analysis → LLM Sentiment → Time Series Models → Quant Analysis → Final Recommendation
```

### 3. Prediction Methodology
- **Short-term (1-7 days)**: LSTM/GRU ensemble with technical indicators
- **Medium-term (30 days)**: Transformer models with sentiment analysis
- **Long-term**: Prophet + fundamental analysis

### 4. Recommendation Engine
Combines multiple signals:
- Technical indicators (20%)
- Sentiment analysis (25%)
- Quantitative models (25%)
- Time series predictions (20%)
- Risk assessment (10%)

## 📈 Performance Metrics

The system provides:
- **Accuracy Metrics**: MSE, R², Cross-validation scores
- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Max Drawdown
- **Confidence Intervals**: 68% and 95% prediction bands
- **Feature Importance**: Which factors drive predictions

## ⚠️ Disclaimer

**This is for educational and research purposes only.** Stock market predictions are inherently uncertain. Always:
- Do your own research
- Consult with financial advisors
- Never invest more than you can afford to lose
- Understand that past performance doesn't guarantee future results

## 🔧 Advanced Configuration

### Training Custom Models
```python
# Train models for a specific stock
POST /api/train-models/AAPL
```

### Portfolio Optimization
```python
# Optimize portfolio allocation
POST /api/portfolio/optimize
{
    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
    "target_volatility": 0.15
}
```

### Pairs Trading Analysis
```python
# Analyze pairs trading opportunity
POST /api/pairs-trading
{
    "symbol1": "AAPL",
    "symbol2": "MSFT"
}
```

## 🚀 Future Enhancements

- [ ] Real-time WebSocket updates
- [ ] Automated trading execution
- [ ] Backtesting framework
- [ ] More LLM providers (Llama, Mistral)
- [ ] Options pricing models
- [ ] Crypto asset support
- [ ] Custom strategy builder

## 📝 License

This project is for educational purposes. See LICENSE for details.

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub. 