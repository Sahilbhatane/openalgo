"""
Technical Indicators
====================
Calculate common technical indicators for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class OpeningRange:
    """Opening range breakout levels"""
    high: float
    low: float
    range_size: float
    vwap: float
    avg_volume: float
    breakout_buy: float   # Buy trigger (high + buffer)
    breakout_sell: float  # Sell trigger (low - buffer)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures volatility and is used for:
    - Setting stop loss distances
    - Position sizing
    - Volatility-based entries
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period (default: 14)
    
    Returns:
        ATR series
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is smoothed average of TR (Wilder's smoothing)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    reset_daily: bool = True,
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        reset_daily: Reset VWAP at start of each day
    
    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume
    
    if reset_daily and isinstance(close.index, pd.DatetimeIndex):
        # Group by date and cumsum within each day
        cumulative_tp_vol = tp_volume.groupby(tp_volume.index.date).cumsum()
        cumulative_vol = volume.groupby(volume.index.date).cumsum()
    else:
        cumulative_tp_vol = tp_volume.cumsum()
        cumulative_vol = volume.cumsum()
    
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    
    return vwap


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    Args:
        close: Close prices
        period: Lookback period (default: 14)
    
    Returns:
        RSI series (0-100)
    """
    close = pd.Series(close)
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Use Wilder's smoothing (same as EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        close: Close prices
        period: Lookback period
    
    Returns:
        EMA series
    """
    return pd.Series(close).ewm(span=period, adjust=False).mean()


def calculate_sma(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        close: Close prices
        period: Lookback period
    
    Returns:
        SMA series
    """
    return pd.Series(close).rolling(window=period).mean()


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        close: Close prices
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    close = pd.Series(close)
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Close prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    close = pd.Series(close)
    
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate SuperTrend indicator.
    
    SuperTrend is a trend-following indicator that provides
    clear buy/sell signals.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 10)
        multiplier: ATR multiplier (default: 3.0)
    
    Returns:
        Tuple of (supertrend_line, trend_direction)
        trend_direction: 1 for uptrend, -1 for downtrend
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    # Basic upper and lower bands
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)
    
    # Initialize
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    for i in range(1, len(close)):
        # Upper band
        if upper_basic.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
        
        # Lower band
        if lower_basic.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
        
        # Supertrend
        if i == 1:
            direction.iloc[i] = 1
        elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > upper_band.iloc[i] else -1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < lower_band.iloc[i] else 1
        
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return supertrend, direction


def calculate_opening_range(
    df: pd.DataFrame,
    minutes: int = 15,
    buffer_pct: float = 0.001,
) -> Optional[OpeningRange]:
    """
    Calculate opening range for breakout strategy.
    
    Uses first N minutes of trading to define range,
    then calculates breakout levels with buffer.
    
    Args:
        df: OHLCV DataFrame with datetime index
        minutes: Number of minutes for opening range (default: 15)
        buffer_pct: Buffer percentage above/below range (default: 0.1%)
    
    Returns:
        OpeningRange with high, low, VWAP, and breakout levels
        Returns None if insufficient data
    """
    if len(df) == 0:
        return None
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    
    # Get today's data
    today = df.index[0].date()
    today_data = df[df.index.date == today]
    
    if len(today_data) == 0:
        return None
    
    # Market opens at 9:15
    market_open = today_data.index[0]
    range_end = market_open + pd.Timedelta(minutes=minutes)
    
    # Filter to opening range period
    opening_data = today_data[today_data.index <= range_end]
    
    if len(opening_data) < 2:
        return None
    
    # Calculate range
    range_high = opening_data['high'].max()
    range_low = opening_data['low'].min()
    range_size = range_high - range_low
    
    # Calculate VWAP for opening range
    vwap = calculate_vwap(
        opening_data['high'],
        opening_data['low'],
        opening_data['close'],
        opening_data['volume'],
        reset_daily=False
    ).iloc[-1]
    
    # Average volume
    avg_volume = opening_data['volume'].mean()
    
    # Breakout levels with buffer
    buffer = range_size * buffer_pct
    breakout_buy = range_high + buffer
    breakout_sell = range_low - buffer
    
    return OpeningRange(
        high=range_high,
        low=range_low,
        range_size=range_size,
        vwap=vwap,
        avg_volume=avg_volume,
        breakout_buy=breakout_buy,
        breakout_sell=breakout_sell,
    )


def calculate_volume_surge(
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate volume surge ratio.
    
    Compares current volume to rolling average.
    Values > 1 indicate higher than average volume.
    
    Args:
        volume: Volume series
        period: Lookback period for average (default: 20)
    
    Returns:
        Volume surge ratio series
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume.replace(0, np.nan)


def is_bullish_engulfing(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Detect bullish engulfing candlestick pattern.
    
    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
    
    Returns:
        Boolean series (True where pattern detected)
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    
    # Previous candle is bearish
    prev_bearish = prev_close < prev_open
    
    # Current candle is bullish
    curr_bullish = close > open_
    
    # Current body engulfs previous body
    engulfs = (open_ < prev_close) & (close > prev_open)
    
    return prev_bearish & curr_bullish & engulfs


def is_bearish_engulfing(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Detect bearish engulfing candlestick pattern.
    
    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
    
    Returns:
        Boolean series (True where pattern detected)
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    
    # Previous candle is bullish
    prev_bullish = prev_close > prev_open
    
    # Current candle is bearish
    curr_bearish = close < open_
    
    # Current body engulfs previous body
    engulfs = (open_ > prev_close) & (close < prev_open)
    
    return prev_bullish & curr_bearish & engulfs


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_indicators():
    """Run basic tests for indicators"""
    print("Testing indicators...")
    
    # Create sample data
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range(start='2024-01-01 09:15', periods=n, freq='5min')
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close.shift(1).fillna(100)
    volume = pd.Series(np.random.randint(1000, 10000, n), index=dates)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Test ATR
    atr = calculate_atr(high, low, close)
    print(f"ATR (last 5): {atr.tail().tolist()}")
    assert not atr.isna().all(), "ATR should have values"
    
    # Test VWAP
    vwap = calculate_vwap(high, low, close, volume)
    print(f"VWAP (last 5): {vwap.tail().tolist()}")
    assert not vwap.isna().all(), "VWAP should have values"
    
    # Test RSI
    rsi = calculate_rsi(close)
    print(f"RSI (last 5): {rsi.tail().tolist()}")
    assert rsi.dropna().between(0, 100).all(), "RSI should be between 0 and 100"
    
    # Test EMA
    ema = calculate_ema(close, 20)
    print(f"EMA(20) (last 5): {ema.tail().tolist()}")
    
    # Test Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(close)
    assert (upper >= middle).all(), "Upper band should be >= middle"
    assert (middle >= lower).all(), "Middle should be >= lower band"
    print(f"BB upper/middle/lower: {upper.iloc[-1]:.2f}/{middle.iloc[-1]:.2f}/{lower.iloc[-1]:.2f}")
    
    # Test MACD
    macd, signal, hist = calculate_macd(close)
    print(f"MACD: {macd.iloc[-1]:.4f}, Signal: {signal.iloc[-1]:.4f}")
    
    # Test Opening Range
    or_result = calculate_opening_range(df, minutes=15)
    if or_result:
        print(f"Opening Range: High={or_result.high:.2f}, Low={or_result.low:.2f}")
        print(f"  VWAP={or_result.vwap:.2f}, Range={or_result.range_size:.2f}")
        print(f"  Buy trigger={or_result.breakout_buy:.2f}, Sell trigger={or_result.breakout_sell:.2f}")
    
    # Test Volume Surge
    vol_surge = calculate_volume_surge(volume)
    print(f"Volume surge (last 5): {vol_surge.tail().tolist()}")
    
    print("\n✅ All indicator tests passed!")


if __name__ == "__main__":
    _test_indicators()
