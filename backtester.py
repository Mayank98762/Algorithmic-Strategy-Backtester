# backtester.py
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """
    Backtesting engine that simulates trading strategy performance.
    """
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        """
        Parameters:
        -----------
        initial_capital : float
            Starting capital for the backtest
        commission : float
            Trading commission as percentage (e.g., 0.001 = 0.1%)
        slippage : float
            Slippage as percentage (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None
        self.trades = None
        
    def run_backtest(self, data, signals, position_sizing='fixed', fixed_units=100, allocation_percentage=0.95):

        """
        Run the backtest simulation.
     
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with 'Close' column
        signals : pandas.Series
        Trading signals: 1 = Buy, -1 = Sell, 0 = No action
        position_sizing : str
            'fixed' = fixed number of units per trade
            'percentage' = fixed percentage of capital per trade
        fixed_units : int
            Number of units to trade if position_sizing='fixed'
        allocation_percentage : float
            Percentage of capital to allocate per trade (e.g., 0.95 = 95%)
        
        Returns:
        --------
        dict
            Backtest results including portfolio value, returns, trades
        """
    # Create a copy to avoid modifying original
        data = data.copy()
        signals = signals.copy()
    
    # Initialize tracking variables
        portfolio = pd.DataFrame(index=data.index)
        portfolio['Close'] = data['Close']
        portfolio['Signal'] = signals
        portfolio['Position'] = 0
        portfolio['Cash'] = self.initial_capital
        portfolio['Holdings'] = 0
        portfolio['Portfolio_Value'] = self.initial_capital
        portfolio['Returns'] = 0
        portfolio['Trade_Action'] = ''
    
    # Trade tracking
        trades_list = []
        current_position = 0
        entry_price = 0
        units_held = 0

        for i in range(1, len(portfolio)):
            current_price = portfolio["Close"].iloc[i]
            signal = portfolio["Signal"].iloc[i]
            prev_portfolio_value = portfolio["Portfolio_Value"].iloc[i - 1]
            cash = portfolio["Cash"].iloc[i - 1]
        
            holdings = portfolio['Holdings'].iloc[i-1]
        
        # Calculate units to trade based on position sizing
            units_to_trade = 0
            trade_price = current_price * (1 + self.slippage if signal < 0 else 1 - self.slippage)
        
            if signal == 1:  # Buy signal
                if position_sizing == 'fixed':
                # Fixed number of shares per trade
                    units_to_trade = fixed_units
                    cost = units_to_trade * trade_price * (1 + self.commission)
                    if cost <= cash:
                        units_held += units_to_trade
                        cash -= cost
                        trades_list.append({
                            'Date': portfolio.index[i],
                            'Type': 'BUY',
                            'Price': trade_price,
                            'Units': units_to_trade,
                            'Cost': cost,
                            'Cash_After': cash,
                            'Portfolio_Value': prev_portfolio_value
                        })
                        portfolio.loc[portfolio.index[i], 'Trade_Action'] = f'BUY {units_to_trade} units @ ${trade_price:.2f}'
            
                elif position_sizing == 'percentage':
                # Allocate percentage of available cash to this trade
                    if cash > 0:
                        invest_amount = cash * allocation_percentage
                        units_to_trade = int(invest_amount / trade_price)  # Integer units
                        if units_to_trade > 0:
                            cost = units_to_trade * trade_price * (1 + self.commission)
                            units_held += units_to_trade
                            cash -= cost
                            trades_list.append({
                                'Date': portfolio.index[i],
                                'Type': 'BUY',
                                'Price': trade_price,
                                'Units': units_to_trade,
                                'Cost': cost,
                                'Cash_After': cash,
                                'Portfolio_Value': prev_portfolio_value
                            })
                            portfolio.loc[portfolio.index[i], 'Trade_Action'] = f'BUY {units_to_trade} units @ ${trade_price:.2f} ({(allocation_percentage*100):.0f}% of cash)'
                    
            elif signal == -1:  # Sell signal
                if units_held > 0:
                    units_to_trade = units_held  # Sell all holdings
                    proceeds = units_to_trade * trade_price * (1 - self.commission)
                    cash += proceeds
                    units_held = 0
                    trades_list.append({
                        'Date': portfolio.index[i],
                        'Type': 'SELL',
                        'Price': trade_price,
                        'Units': units_to_trade,
                        'Proceeds': proceeds,
                        'Cash_After': cash,
                        'Portfolio_Value': prev_portfolio_value
                    })
                    portfolio.loc[portfolio.index[i], 'Trade_Action'] = f'SELL {units_to_trade} units @ ${trade_price:.2f}'
        
        # Update holdings value
            holdings_value = units_held * current_price
        
        # Update portfolio
            portfolio.loc[portfolio.index[i], 'Position'] = units_held
            portfolio.loc[portfolio.index[i], 'Cash'] = cash
            portfolio.loc[portfolio.index[i], 'Holdings'] = holdings_value
            portfolio.loc[portfolio.index[i], 'Portfolio_Value'] = cash + holdings_value
        
        # Calculate returns
            if i > 0:
                portfolio.loc[portfolio.index[i], 'Returns'] = (
                    portfolio.loc[portfolio.index[i], 'Portfolio_Value'] / 
                    portfolio.loc[portfolio.index[i-1], 'Portfolio_Value'] - 1
                )
    
    # Calculate benchmark returns (Buy and Hold)
        benchmark_returns = data['Close'].pct_change()
        portfolio['Benchmark_Value'] = self.initial_capital * (1 + benchmark_returns).cumprod()
        portfolio['Benchmark_Returns'] = benchmark_returns
    
    # Store results
        self.results = portfolio
        self.trades = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    
    # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio)
    
        return {
            'portfolio': portfolio,
            'trades': self.trades,
            'metrics': metrics
        }
    
    def calculate_metrics(self, portfolio):
        """
        Calculate key performance metrics.
        """
        # Extract returns
        strategy_returns = portfolio['Returns'].dropna()
        benchmark_returns = portfolio['Benchmark_Returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {}
        
        # Total Return
        total_return = portfolio['Portfolio_Value'].iloc[-1] / self.initial_capital - 1
        benchmark_return = portfolio['Benchmark_Value'].iloc[-1] / self.initial_capital - 1
        
        # Annualized Return (assuming 252 trading days)
        trading_days = len(portfolio)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        benchmark_annualized = (1 + benchmark_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = (annualized_return - 0) / volatility if volatility > 0 else 0
        benchmark_sharpe = (benchmark_annualized - 0) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate (if trades exist)
        win_rate = 0
        if len(self.trades) > 0:
            # Pair buys and sells to calculate trade P&L
            trades_df = self.trades.copy()
            if len(trades_df) >= 2:
                profits = []
                for i in range(0, len(trades_df) - 1, 2):
                    if trades_df.iloc[i]['Type'] == 'BUY' and trades_df.iloc[i+1]['Type'] == 'SELL':
                        buy_price = trades_df.iloc[i]['Price']
                        sell_price = trades_df.iloc[i+1]['Price']
                        units = trades_df.iloc[i]['Units']
                        profit = (sell_price - buy_price) * units
                        profits.append(profit)
                
                if profits:
                    winning_trades = sum(1 for p in profits if p > 0)
                    win_rate = winning_trades / len(profits)
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility (Annual)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Number of Trades': len(self.trades),
            'Win Rate': win_rate,
            'Benchmark Return': benchmark_return,
            'Benchmark Annualized': benchmark_annualized,
            'Benchmark Sharpe': benchmark_sharpe
        }