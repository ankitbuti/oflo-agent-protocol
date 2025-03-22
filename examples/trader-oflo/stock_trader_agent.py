import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
import yfinance as yf
from newsapi import NewsApiClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class StockTraderAgent:
    def __init__(self):
        load_dotenv()
        
        # Initialize APIs
        self.news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True  # Use paper trading for safety
        )
        
        # Trading parameters
        self.max_position_size = 1000  # Maximum amount to invest in a single position
        self.watched_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    async def get_stock_news(self, symbol: str) -> List[Dict]:
        """Get recent news articles for a given stock symbol."""
        try:
            company_name = yf.Ticker(symbol).info.get('longName', symbol)
            news = self.news_api.get_everything(
                q=company_name,
                language='en',
                sort_by='publishedAt',
                page_size=5
            )
            return news.get('articles', [])
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []

    def analyze_sentiment(self, articles: List[Dict]) -> float:
        """
        Analyze sentiment from news articles.
        Returns a score between -1 (very negative) and 1 (very positive).
        """
        if not articles:
            return 0
        
        # Simple keyword-based sentiment analysis
        positive_keywords = ['surge', 'jump', 'gain', 'profit', 'growth', 'positive']
        negative_keywords = ['drop', 'fall', 'loss', 'decline', 'negative', 'risk']
        
        total_score = 0
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            text = text.lower()
            
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            article_score = (pos_count - neg_count) / (pos_count + neg_count + 1)  # +1 to avoid division by zero
            total_score += article_score
        
        return total_score / len(articles)

    async def execute_trade(self, symbol: str, sentiment_score: float):
        """Execute a trade based on sentiment analysis."""
        try:
            if abs(sentiment_score) < 0.2:  # Ignore weak signals
                return
            
            side = OrderSide.BUY if sentiment_score > 0 else OrderSide.SELL
            
            # Calculate position size based on sentiment strength
            position_size = int(self.max_position_size * abs(sentiment_score))
            
            # Get current price
            current_price = float(yf.Ticker(symbol).info['regularMarketPrice'])
            qty = max(1, int(position_size / current_price))
            
            # Create and submit order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            print(f"Executed {side.value} order for {qty} shares of {symbol}")
            return order
        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
            return None

    async def run(self):
        """Main loop for the trading agent."""
        while True:
            for symbol in self.watched_stocks:
                # Get and analyze news
                articles = await self.get_stock_news(symbol)
                sentiment = self.analyze_sentiment(articles)
                
                print(f"\nAnalysis for {symbol}:")
                print(f"Sentiment score: {sentiment:.2f}")
                
                # Execute trade if sentiment is strong enough
                if abs(sentiment) >= 0.2:
                    await self.execute_trade(symbol, sentiment)
                
            # Wait for 1 hour before next analysis
            await asyncio.sleep(3600)

if __name__ == "__main__":
    # Create and run the agent
    agent = StockTraderAgent()
    asyncio.run(agent.run())
