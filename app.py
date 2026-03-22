# app.py (updated with backtesting functionality)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from data_feed import DataFeed
from strategies import (
    MovingAverageCrossover, 
    RSIStrategy, 
    BollingerBandsStrategy,
    MomentumStrategy,
    BuyAndHold
)
from backtester import Backtester
from utils import format_large_number, validate_dates, get_market_hours_status

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading Backtester",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Algorithmic Trading Strategy Backtester")
st.markdown("""
Test trading strategies on historical stock data. Select a strategy, configure parameters, 
and see how it would have performed!
""")

# Initialize the data feed
data_feed = DataFeed()

# Sidebar for data configuration
with st.sidebar:
    st.header("Data Configuration")
    
    # Ticker selection
    st.subheader("1. Select Stock")
    ticker_option = st.radio(
        "Choose input method:",
        ["Select from popular stocks", "Enter custom ticker"]
    )
    
    popular_tickers = data_feed.get_available_tickers()
    
    if ticker_option == "Select from popular stocks":
        company = st.selectbox(
            "Choose a company:",
            list(popular_tickers.keys())
        )
        ticker = popular_tickers[company]
        st.info(f"Selected ticker: **{ticker}**")
    else:
        ticker = st.text_input(
            "Enter stock ticker symbol:",
            value="AAPL",
            help="Examples: AAPL, MSFT, GOOGL, TSLA, RELIANCE.NS for Indian stocks"
        ).upper()
    
    # Date range selection
    st.subheader("2. Select Date Range")
    presets = data_feed.get_date_range_options()
    preset_option = st.selectbox(
        "Quick select:",
        ["Custom"] + list(presets.keys())
    )
    
    if preset_option != "Custom":
        start_date, end_date = presets[preset_option]
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    else:
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=5*365)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=start_date,
                max_value=datetime.now().date()
            )
    
    # Fetch data button
    st.subheader("3. Fetch Data")
    fetch_button = st.button("Fetch Historical Data", type="primary", use_container_width=True)
    
    st.divider()
    
    # Strategy Configuration (only show if data is loaded)
    if 'data' in st.session_state:
        st.header("Strategy Configuration")
        
        # Strategy selection
        st.subheader("4. Select Strategy")
        strategy_options = {
            "Moving Average Crossover": MovingAverageCrossover,
            "RSI Mean Reversion": RSIStrategy,
            "Bollinger Bands": BollingerBandsStrategy,
            "Momentum": MomentumStrategy,
            "Buy and Hold (Benchmark)": BuyAndHold
        }
        
        strategy_name = st.selectbox(
            "Choose a strategy:",
            list(strategy_options.keys())
        )
        
        # Strategy-specific parameters
        st.subheader("5. Configure Strategy Parameters")
        
        if strategy_name == "Moving Average Crossover":
            fast_period = st.slider("Fast MA Period", 5, 100, 20, 5)
            slow_period = st.slider("Slow MA Period", 20, 200, 50, 10)
            strategy = MovingAverageCrossover(fast_period, slow_period)
            
        elif strategy_name == "RSI Mean Reversion":
            rsi_period = st.slider("RSI Period", 5, 30, 14, 1)
            oversold = st.slider("Oversold Threshold", 20, 40, 30, 1)
            overbought = st.slider("Overbought Threshold", 60, 80, 70, 1)
            strategy = RSIStrategy(rsi_period, oversold, overbought)
            
        elif strategy_name == "Bollinger Bands":
            bb_period = st.slider("Period", 10, 50, 20, 5)
            num_std = st.slider("Number of Standard Deviations", 1.0, 3.0, 2.0, 0.5)
            strategy = BollingerBandsStrategy(bb_period, num_std)
            
        elif strategy_name == "Momentum":
            lookback = st.slider("Momentum Lookback (days)", 21, 504, 252, 21)
            buy_threshold = st.slider("Buy Threshold", 0.0, 0.5, 0.2, 0.05, format="%.0f%%")
            sell_threshold = st.slider("Sell Threshold", -0.5, 0.0, -0.1, 0.05, format="%.0f%%")
            strategy = MomentumStrategy(lookback, buy_threshold, sell_threshold)
            
        else:  # Buy and Hold
            strategy = BuyAndHold()
        
        # Backtest parameters
        st.subheader("6. Backtest Settings")
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=10000)
        commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.05) / 100
        slippage = st.slider("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100
        position_sizing = st.selectbox("Position Sizing", ["fixed"], disabled=True)  # More options later
        fixed_units = st.number_input("Units per Trade", min_value=1, value=100, step=10)
        
        # Run backtest button
        st.subheader("7. Run Backtest")
        backtest_button = st.button("Run Backtest", type="primary", use_container_width=True)
        
        st.divider()
        st.caption(f"Market Status: **{get_market_hours_status()}** (NYSE)")

# Main content area
if fetch_button:
    if validate_dates(start_date, end_date):
        with st.spinner(f'Fetching data for {ticker}...'):
            data = data_feed.fetch_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.session_state['data_feed'] = data_feed
            st.rerun()
        else:
            st.error("Failed to fetch data. Please check the ticker symbol and try again.")
    else:
        st.error("Please fix the date range and try again.")

# Backtest execution
if 'data' in st.session_state and 'backtest_button' in locals() and backtest_button:
    data = st.session_state['data']
    
    with st.spinner(f'Running backtest for {strategy_name}...'):
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize backtester
        backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        # Run backtest
        results = backtester.run_backtest(
            data, 
            signals, 
            position_sizing='fixed',
            fixed_units=fixed_units
        )
        
        portfolio = results['portfolio']
        trades = results['trades']
        metrics = results['metrics']
        
        # Store results in session state
        st.session_state['backtest_results'] = results
        st.session_state['strategy_name'] = strategy_name
        st.session_state['signals'] = signals
        st.session_state['strategy'] = strategy

# Display backtest results if available
if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    portfolio = results['portfolio']
    trades = results['trades']
    metrics = results['metrics']
    strategy_name = st.session_state['strategy_name']
    ticker = st.session_state['ticker']
    data = st.session_state['data']
    
    st.header(f"Backtest Results: {strategy_name} on {ticker}")
    
    # Performance Metrics Dashboard
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{metrics.get('Total Return', 0):.2%}",
            delta=f"vs Benchmark: {metrics.get('Benchmark Return', 0):.2%}"
        )
        st.metric("Annualized Return", f"{metrics.get('Annualized Return', 0):.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
        st.metric("Volatility", f"{metrics.get('Volatility (Annual)', 0):.2%}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
        st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.1%}")
    
    with col4:
        st.metric("Number of Trades", metrics.get('Number of Trades', 0))
        st.metric("Initial Capital", f"${initial_capital:,.0f}")
    
    # Portfolio Value Chart
    st.subheader("Portfolio Growth Over Time")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio.index, 
            y=portfolio['Portfolio_Value'], 
            mode='lines', 
            name='Strategy Portfolio',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Benchmark value
    fig.add_trace(
        go.Scatter(
            x=portfolio.index, 
            y=portfolio['Benchmark_Value'], 
            mode='lines', 
            name='Buy & Hold',
            line=dict(color='gray', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add markers for trades
    if not trades.empty:
        buy_trades = trades[trades['Type'] == 'BUY']
        sell_trades = trades[trades['Type'] == 'SELL']
        
        fig.add_trace(
            go.Scatter(
                x=buy_trades['Date'], 
                y=buy_trades['Portfolio_Value'], 
                mode='markers', 
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sell_trades['Date'], 
                y=sell_trades['Portfolio_Value'], 
                mode='markers', 
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # Drawdown chart
    cumulative_returns = portfolio['Portfolio_Value'] / portfolio['Portfolio_Value'].iloc[0]
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, 
            y=drawdown * 100, 
            mode='lines', 
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'Portfolio Growth: Strategy vs Buy & Hold',
        yaxis_title='Portfolio Value ($)',
        yaxis2_title='Drawdown (%)',
        height=600,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Signals Chart
    st.subheader("Trading Signals and Price Action")
    
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig2.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            name='Close Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add strategy-specific indicators
    if 'signals' in st.session_state and hasattr(st.session_state['strategy'], 'rsi'):
        # RSI indicator
        fig2.add_trace(
            go.Scatter(
                x=data.index, 
                y=st.session_state['strategy'].rsi, 
                mode='lines', 
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
    elif 'signals' in st.session_state and hasattr(st.session_state['strategy'], 'upper_band'):
        # Bollinger Bands
        fig2.add_trace(
            go.Scatter(
                x=data.index, 
                y=st.session_state['strategy'].upper_band, 
                mode='lines', 
                name='Upper Band',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(
                x=data.index, 
                y=st.session_state['strategy'].lower_band, 
                mode='lines', 
                name='Lower Band',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty'
            ),
            row=1, col=1
        )
    
    # Add buy/sell markers
    signals = st.session_state['signals']
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    
    fig2.add_trace(
        go.Scatter(
            x=buy_signals, 
            y=data.loc[buy_signals, 'Close'], 
            mode='markers', 
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=sell_signals, 
            y=data.loc[sell_signals, 'Close'], 
            mode='markers', 
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ),
        row=1, col=1
    )
    
    fig2.update_layout(
        title='Price Action with Trading Signals',
        yaxis_title='Price ($)',
        yaxis2_title='Indicator Value',
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Trade Log
    if not trades.empty:
        with st.expander("Detailed Trade Log"):
            st.dataframe(trades, use_container_width=True)
            
            # Download trades as CSV
            csv = trades.to_csv()
            st.download_button(
                label="Download Trade Log as CSV",
                data=csv,
                file_name=f"{ticker}_{strategy_name}_trades.csv",
                mime="text/csv"
            )
    
    # Monthly Returns Heatmap
    st.subheader("Monthly Returns Heatmap")
    
    portfolio_returns = portfolio['Returns'].copy()
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    
    monthly_returns = portfolio_returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    if not monthly_returns.empty:
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })
        
        pivot_returns = monthly_returns_df.pivot(
            index='Year', 
            columns='Month', 
            values='Return'
        )
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_returns.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot_returns.values.round(2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_heatmap.update_layout(
            title='Monthly Returns (%)',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Export full results
    st.subheader("Export Results")
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        portfolio_csv = portfolio.to_csv()
        st.download_button(
            label="Download Portfolio Data (CSV)",
            data=portfolio_csv,
            file_name=f"{ticker}_{strategy_name}_portfolio.csv",
            mime="text/csv"
        )
    
    with col_export2:
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_csv = metrics_df.to_csv()
        st.download_button(
            label="Download Performance Metrics (CSV)",
            data=metrics_csv,
            file_name=f"{ticker}_{strategy_name}_metrics.csv",
            mime="text/csv"
        )

elif 'data' in st.session_state:
    st.info("Data loaded! Use the sidebar to select a strategy and run the backtest.")
    
    # Show data summary
    data = st.session_state['data']
    ticker = st.session_state['ticker']
    
    st.subheader(f"{ticker} - Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trading Days", len(data))
    with col2:
        st.metric("Start Date", data.index[0].strftime('%Y-%m-%d'))
    with col3:
        st.metric("End Date", data.index[-1].strftime('%Y-%m-%d'))
    with col4:
        st.metric("Price Range", f"${data['Low'].min():.2f} - ${data['High'].max():.2f}")
    
    # Quick preview
    st.line_chart(data['Close'])

else:
    st.info("👈 Use the sidebar to select a stock and date range, then click 'Fetch Historical Data' to begin.")

# Footer
st.divider()
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. Not financial advice.")
