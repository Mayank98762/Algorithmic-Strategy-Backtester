# data_feed.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFeed:
    """
    A modular data feed component for fetching and preprocessing historical stock data.
    """
    
    def __init__(self):
        self.data = None
        self.ticker = None
        self.start_date = None
        self.end_date = None
        
    def fetch_data(self, ticker, start_date, end_date):
        """
        Fetch historical data for a given ticker using yfinance.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol (e.g., 'AAPL', 'MSFT', 'RELIANCE.NS')
        start_date : str or datetime
            Start date in 'YYYY-MM-DD' format
        end_date : str or datetime
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        try:
            # Convert to string if datetime objects are passed
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Store parameters
            self.ticker = ticker
            self.start_date = start_date
            self.end_date = end_date
            
            # Show a progress indicator (useful for Streamlit)
            with st.spinner(f'Fetching data for {ticker}...'):
                # Download data from yfinance
                stock = yf.Ticker(ticker)
                self.data = stock.history(start=start_date, end=end_date)
                
                # Check if data is empty
                if self.data.empty:
                    st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
                    return None
                
                # Add additional useful columns
                self.data['Returns'] = self.data['Close'].pct_change()
                self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
                self.data['Day'] = self.data.index.day
                self.data['Month'] = self.data.index.month
                self.data['Year'] = self.data.index.year
                self.data['Day_of_Week'] = self.data.index.dayofweek  # 0=Monday, 6=Sunday
                
                # Add rolling statistics (useful for strategies)
                self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
                self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
                self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
                self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
                
                # Add volatility (20-day rolling standard deviation of returns)
                self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
                
                st.success(f"Successfully fetched {len(self.data)} days of data for {ticker}")
                return self.data
                
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_data_summary(self):
        """
        Generate a summary of the fetched data.
        
        Returns:
        --------
        dict
            Dictionary containing data statistics
        """
        if self.data is None or self.data.empty:
            return None
        
        summary = {
            'ticker': self.ticker,
            'start_date': self.data.index[0].strftime('%Y-%m-%d'),
            'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
            'trading_days': len(self.data),
            'years_of_data': round(len(self.data) / 252, 2),  # Approx trading days per year
            'avg_daily_return': f"{self.data['Returns'].mean()*100:.4f}%",
            'avg_annual_return': f"{(1 + self.data['Returns'].mean())**252 - 1:.4%}",
            'volatility_annual': f"{self.data['Volatility'].iloc[-1]:.2%}",
            'min_price': f"${self.data['Low'].min():.2f}",
            'max_price': f"${self.data['High'].max():.2f}",
            'avg_volume': f"{self.data['Volume'].mean():,.0f}"
        }
        return summary
    
    def get_date_range_options(self):
        """
        Provide predefined date range options for user selection.
        
        Returns:
        --------
        dict
            Dictionary of preset date ranges
        """
        today = datetime.now().date()
        
        presets = {
            '1 Month': (today - timedelta(days=30), today),
            '3 Months': (today - timedelta(days=90), today),
            '6 Months': (today - timedelta(days=180), today),
            '1 Year': (today - timedelta(days=365), today),
            '3 Years': (today - timedelta(days=3*365), today),
            '5 Years': (today - timedelta(days=5*365), today),
            '10 Years': (today - timedelta(days=10*365), today),
            'Max': (datetime(2000, 1, 1).date(), today)  # yfinance has data from ~2000 for most stocks
        }
        return presets
    
    def get_available_tickers(self):
        """
        Return a list of popular tickers for user convenience.
        """
        popular_tickers = {
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Google': 'GOOGL',
            'Amazon': 'AMZN',
            'Tesla': 'TSLA',
            'Meta': 'META',
            'NVIDIA': 'NVDA',
            'JPMorgan': 'JPM',
            'Goldman Sachs': 'GS',
            'Berkshire Hathaway': 'BRK-B',
            'S&P 500 ETF': 'SPY',
            'Nasdaq ETF': 'QQQ',
            'Reliance Industries': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'Infosys': 'INFY.NS'
        }
        return popular_tickers
    
    def validate_ticker(self, ticker):
        """
        Quick validation to check if a ticker exists.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol to validate
            
        Returns:
        --------
        bool
            True if ticker exists, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'regularMarketPrice' in info or 'currentPrice' in info:
                return True
            # Try to fetch a small amount of data as backup validation
            hist = stock.history(period="1d")
            if not hist.empty:
                return True
            return False
        except Exception:
            return False
