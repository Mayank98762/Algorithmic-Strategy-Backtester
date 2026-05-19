# backtester.py
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    """
    Backtesting engine that simulates trading strategy performance.
    """

    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005, risk_free_rate=0.04):
        """
        Parameters:
        -----------
        initial_capital : float
            Starting capital for the backtest
        commission : float
            Trading commission as percentage (e.g., 0.001 = 0.1%)
        slippage : float
            Slippage as percentage (e.g., 0.0005 = 0.05%)
        risk_free_rate : float
            Annualised risk-free rate used in Sharpe calculation (e.g., 0.04 = 4%).
            Defaulting to 4% reflects a realistic short-term rate environment.
            Using 0% inflates Sharpe ratios and makes them non-comparable to
            industry benchmarks when rates are elevated.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
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
        data = data.copy()
        signals = signals.copy()

        portfolio = pd.DataFrame(index=data.index)
        portfolio['Close'] = data['Close']
        portfolio['Signal'] = signals
        portfolio['Position'] = 0
        portfolio['Cash'] = self.initial_capital
        portfolio['Holdings'] = 0
        portfolio['Portfolio_Value'] = self.initial_capital
        portfolio['Returns'] = 0
        portfolio['Trade_Action'] = ''

        trades_list = []
        units_held = 0

        for i in range(1, len(portfolio)):
            current_price = portfolio["Close"].iloc[i]
            # Use the PREVIOUS bar's signal, not the current bar's.
            # A signal generated at the close of bar i-1 can only realistically
            # be acted on at bar i. Reading signal[i] and trading at close[i]
            # is look-ahead bias — the same bar that generated the signal.
            signal = portfolio["Signal"].iloc[i - 1]
            prev_portfolio_value = portfolio["Portfolio_Value"].iloc[i - 1]
            cash = portfolio["Cash"].iloc[i - 1]

            units_to_trade = 0

            # Slippage always hurts: buy at higher price, sell at lower price
            if signal == 1:
                trade_price = current_price * (1 + self.slippage)
            else:
                trade_price = current_price * (1 - self.slippage)

            if signal == 1:
                if position_sizing == 'fixed':
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
                        portfolio.loc[portfolio.index[i], 'Trade_Action'] = (
                            f'BUY {units_to_trade} units @ ${trade_price:.2f}'
                        )

                elif position_sizing == 'percentage':
                    if cash > 0:
                        invest_amount = cash * allocation_percentage
                        units_to_trade = int(invest_amount / trade_price)
                        if units_to_trade > 0:
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
                                portfolio.loc[portfolio.index[i], 'Trade_Action'] = (
                                    f'BUY {units_to_trade} units @ ${trade_price:.2f} '
                                    f'({(allocation_percentage * 100):.0f}% of cash)'
                                )

            elif signal == -1:
                if units_held > 0:
                    units_to_trade = units_held
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
                    portfolio.loc[portfolio.index[i], 'Trade_Action'] = (
                        f'SELL {units_to_trade} units @ ${trade_price:.2f}'
                    )

            holdings_value = units_held * current_price
            portfolio.loc[portfolio.index[i], 'Position'] = units_held
            portfolio.loc[portfolio.index[i], 'Cash'] = cash
            portfolio.loc[portfolio.index[i], 'Holdings'] = holdings_value
            portfolio.loc[portfolio.index[i], 'Portfolio_Value'] = cash + holdings_value
            portfolio.loc[portfolio.index[i], 'Returns'] = (
                portfolio.loc[portfolio.index[i], 'Portfolio_Value'] /
                portfolio.loc[portfolio.index[i - 1], 'Portfolio_Value'] - 1
            )

        # fillna(0) so cumprod starts cleanly at initial_capital (not NaN)
        benchmark_returns = data['Close'].pct_change().fillna(0)
        portfolio['Benchmark_Value'] = self.initial_capital * (1 + benchmark_returns).cumprod()
        portfolio['Benchmark_Returns'] = benchmark_returns

        self.results = portfolio
        self.trades = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        metrics = self.calculate_metrics(portfolio)

        return {'portfolio': portfolio, 'trades': self.trades, 'metrics': metrics}

    def calculate_metrics(self, portfolio):
        """Calculate key performance metrics."""
        strategy_returns = portfolio['Returns'].dropna()
        benchmark_returns = portfolio['Benchmark_Returns'].dropna()

        if len(strategy_returns) == 0:
            return {}

        total_return = portfolio['Portfolio_Value'].iloc[-1] / self.initial_capital - 1
        benchmark_return = portfolio['Benchmark_Value'].iloc[-1] / self.initial_capital - 1

        trading_days = len(portfolio)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        benchmark_annualized = (1 + benchmark_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

        # Sharpe uses configurable risk_free_rate — not hard-coded 0%
        rf = self.risk_free_rate
        sharpe_ratio = (annualized_return - rf) / volatility if volatility > 0 else 0
        benchmark_sharpe = (benchmark_annualized - rf) / benchmark_volatility if benchmark_volatility > 0 else 0

        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win Rate: match BUY/SELL pairs chronologically; P&L is net of commissions
        # because Cost and Proceeds already embed commission in the execution loop.
        win_rate = 0
        if len(self.trades) > 0:
            trades_df = self.trades.copy()
            buy_rows = trades_df[trades_df['Type'] == 'BUY'].copy()
            sell_rows = trades_df[trades_df['Type'] == 'SELL'].copy()

            if len(sell_rows) > 0 and len(buy_rows) > 0:
                profits = []
                buy_iter = buy_rows.itertuples()
                pending_cost = 0.0
                next_buy = next(buy_iter, None)

                for sell_row in sell_rows.itertuples():
                    while next_buy is not None and next_buy.Date < sell_row.Date:
                        pending_cost += next_buy.Cost
                        next_buy = next(buy_iter, None)
                    if pending_cost > 0:
                        net_profit = sell_row.Proceeds - pending_cost
                        profits.append(net_profit)
                        pending_cost = 0.0

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
