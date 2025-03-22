# Stock Trader MCP Agent

This is an automated stock trading agent that analyzes news sentiment and executes trades based on the analysis. The agent uses paper trading by default for safety.

## Features

- Monitors news for selected stocks using NewsAPI
- Performs simple sentiment analysis on news articles
- Executes trades through Alpaca's paper trading API
- Configurable trading parameters and stock watchlist

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:
```
NEWS_API_KEY=your_newsapi_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
```

You'll need to sign up for:
- [NewsAPI](https://newsapi.org/) for news data
- [Alpaca](https://alpaca.markets/) for trading (paper trading account)

## Usage

Run the agent:
```bash
python stock_trader_agent.py
```

The agent will:
1. Monitor news for the watched stocks (AAPL, GOOGL, MSFT, AMZN, META by default)
2. Analyze sentiment from recent news articles
3. Execute trades when strong sentiment signals are detected
4. Wait for 1 hour before the next analysis cycle

## Configuration

You can modify the following parameters in `stock_trader_agent.py`:
- `max_position_size`: Maximum amount to invest in a single position
- `watched_stocks`: List of stock symbols to monitor
- Sentiment thresholds and analysis parameters

## Safety Features

- Uses paper trading by default
- Implements position size limits
- Requires minimum sentiment threshold for trading
- Basic error handling and logging
