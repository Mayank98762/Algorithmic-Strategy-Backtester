# strategies.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name):
        self.name = name
        self.signals = None
        
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy logic.
        
        Returns:
        --------
        pandas.Series
            Signals: 1 = Buy, -1 = Sell, 0 = Hold/No position
        """
        pass
    
    def get_strategy_info(self):
        """
        Return strategy description for display.
        """
        return self.name

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy:
    Buy when fast MA crosses above slow MA
    Sell when fast MA crosses below slow MA
    """
    
    def __init__(self, fast_period=20, slow_period=50):
        super().__init__(f"Moving Average Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data):
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=self.fast_period).mean()
        slow_ma = data['Close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: fast MA crosses above slow MA
        buy_signal = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals[buy_signal] = 1
        
        # Sell signal: fast MA crosses below slow MA
        sell_signal = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals[sell_signal] = -1
        
        self.signals = signals
        return signals

class RSIStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy:
    Buy when RSI < oversold_threshold
    Sell when RSI > overbought_threshold
    """
    
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__(f"RSI Strategy (Period={period}, Oversold={oversold}, Overbought={overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_rsi(self, prices):
        """
        Calculate RSI using Wilder's Smoothed Moving Average (the industry standard).

        Wilder's method seeds the first average with a simple mean over the initial
        `period` bars, then applies exponential smoothing with alpha = 1/period for
        all subsequent bars.  This matches Bloomberg, TradingView, FactSet, and every
        other professional platform — unlike a plain rolling SMA which diverges
        meaningfully and produces non-comparable values.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's EMA: alpha = 1/period, equivalent to com = period - 1
        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data):
        rsi = self.calculate_rsi(data['Close'])
        
        signals = pd.Series(0, index=data.index)
        
        # Buy when RSI crosses above oversold threshold
        buy_signal = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        signals[buy_signal] = 1
        
        # Sell when RSI crosses below overbought threshold
        sell_signal = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
        signals[sell_signal] = -1
        
        self.signals = signals
        self.rsi = rsi
        return signals

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy:
    Buy when price touches lower band
    Sell when price touches upper band
    """
    
    def __init__(self, period=20, num_std=2):
        super().__init__(f"Bollinger Bands (Period={period}, Std={num_std})")
        self.period = period
        self.num_std = num_std
        
    def generate_signals(self, data):
        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=self.period).mean()
        std = data['Close'].rolling(window=self.period).std()
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        signals = pd.Series(0, index=data.index)

        # Event-based signals: fire only on the FIRST bar that touches/crosses the band,
        # not on every subsequent bar while price remains outside.  The old state-based
        # approach (data['Close'] <= lower_band) generated a buy signal for every
        # consecutive bar below the band, causing the backtester to place a new order
        # each bar and rapidly exhaust cash — misrepresenting the strategy's intent.
        buy_signal = (data['Close'] <= lower_band) & (data['Close'].shift(1) > lower_band.shift(1))
        signals[buy_signal] = 1

        sell_signal = (data['Close'] >= upper_band) & (data['Close'].shift(1) < upper_band.shift(1))
        signals[sell_signal] = -1
        
        self.signals = signals
        self.upper_band = upper_band
        self.lower_band = lower_band
        return signals

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy:
    Buy when momentum > threshold, Sell when momentum < threshold
    """
    
    def __init__(self, lookback=252, buy_threshold=0.2, sell_threshold=-0.1):
        super().__init__(f"Momentum Strategy (Lookback={lookback}, Buy>{buy_threshold:.0%}, Sell<{sell_threshold:.0%})")
        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def generate_signals(self, data):
        # Calculate momentum as percentage return over lookback period
        momentum = data['Close'].pct_change(self.lookback)
        
        signals = pd.Series(0, index=data.index)

        # Event-based: signal fires only when momentum CROSSES the threshold,
        # not on every bar where it remains above/below it.  The old state-based
        # approach triggered a new buy order on every single bar momentum stayed
        # elevated, causing the backtester to continuously accumulate units until
        # cash was exhausted — which is not how a momentum strategy is intended to work.
        buy_signal = (momentum > self.buy_threshold) & (momentum.shift(1) <= self.buy_threshold)
        signals[buy_signal] = 1

        sell_signal = (momentum < self.sell_threshold) & (momentum.shift(1) >= self.sell_threshold)
        signals[sell_signal] = -1
        
        self.signals = signals
        return signals

class BuyAndHold(BaseStrategy):
    """
    Buy and Hold benchmark strategy
    """
    
    def __init__(self):
        super().__init__("Buy and Hold (Benchmark)")
        
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        # The backtester loop starts at i=1, so iloc[0] is never evaluated.
        # Place the buy signal at iloc[1] so it is picked up on the first loop iteration.
        if len(signals) > 1:
            signals.iloc[1] = 1
        return signals
