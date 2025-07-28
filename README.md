# AI Stock Predictor

An intelligent web application that provides AI-powered stock predictions and recommendations across all major market sectors.

## Features

- **Real-time Stock Analysis**: Fetches live data from Yahoo Finance
- **AI-Powered Predictions**: 1-day, 7-day, and 30-day price forecasts
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Smart Recommendations**: BUY/SELL/HOLD signals with confidence scores
- **Sector Coverage**: Technology, Healthcare, Finance, Consumer, Energy, Industrial
- **Modern UI**: Responsive design with interactive charts

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Forecast Stocks"
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python3 app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## API Endpoints

- `GET /` - Web interface
- `GET /api/sectors` - List all sectors
- `GET /api/stocks/<sector>` - Get stocks by sector
- `GET /api/stock/<symbol>` - Get detailed stock analysis
- `GET /api/market-overview` - Market sentiment overview

## Project Structure

```
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main web interface
└── src/               # Source code
    ├── data_processing/   # Data fetching and processing
    ├── models/           # AI prediction models
    └── utils/           # Technical indicators and recommendations
```

## Technologies Used

- **Backend**: Flask, Python
- **Data Source**: Yahoo Finance (yfinance)
- **Machine Learning**: scikit-learn
- **Frontend**: HTML, JavaScript, Chart.js
- **Technical Analysis**: NumPy, Pandas

## License

This project is for educational and demonstration purposes. 