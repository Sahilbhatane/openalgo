"""
Performance Metrics
===================
Calculate trading performance metrics for evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    """Complete performance metrics report"""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    
    # Streak
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    def __str__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    PERFORMANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║  RETURNS                                                     ║
║    Total Return:        {self.total_return:>10.2f}%                        ║
║    Annualized Return:   {self.annualized_return:>10.2f}%                        ║
╠══════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                ║
║    Volatility:          {self.volatility:>10.2f}%                        ║
║    Max Drawdown:        {self.max_drawdown:>10.2f}%                        ║
║    Max DD Duration:     {self.max_drawdown_duration:>10} days                     ║
╠══════════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED RETURNS                                       ║
║    Sharpe Ratio:        {self.sharpe_ratio:>10.2f}                          ║
║    Sortino Ratio:       {self.sortino_ratio:>10.2f}                          ║
║    Calmar Ratio:        {self.calmar_ratio:>10.2f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                            ║
║    Total Trades:        {self.total_trades:>10}                          ║
║    Winning Trades:      {self.winning_trades:>10}                          ║
║    Losing Trades:       {self.losing_trades:>10}                          ║
║    Win Rate:            {self.win_rate:>10.1f}%                        ║
╠══════════════════════════════════════════════════════════════╣
║  PROFIT METRICS                                              ║
║    Avg Win:            ₹{self.avg_win:>10.2f}                         ║
║    Avg Loss:           ₹{self.avg_loss:>10.2f}                         ║
║    Profit Factor:       {self.profit_factor:>10.2f}                          ║
║    Avg Trade:          ₹{self.avg_trade:>10.2f}                         ║
║    Largest Win:        ₹{self.largest_win:>10.2f}                         ║
║    Largest Loss:       ₹{self.largest_loss:>10.2f}                         ║
╠══════════════════════════════════════════════════════════════╣
║  STREAKS                                                     ║
║    Max Consecutive Wins:  {self.max_consecutive_wins:>8}                            ║
║    Max Consecutive Losses:{self.max_consecutive_losses:>8}                            ║
╚══════════════════════════════════════════════════════════════╝
"""


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate daily returns from equity curve"""
    return equity_curve.pct_change().dropna()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,  # 6% annual (India)
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Sharpe = (Return - Risk Free Rate) / Volatility
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate (default: 6% for India)
        periods_per_year: Trading periods per year (default: 252)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - daily_rf
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino Ratio.
    
    Like Sharpe, but only penalizes downside volatility.
    Sortino = (Return - Risk Free Rate) / Downside Deviation
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - daily_rf
    
    # Downside returns only
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    sortino = excess_returns.mean() / downside_std
    
    # Annualize
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity_curve: Equity curve series
    
    Returns:
        Tuple of (max_drawdown_percentage, max_duration_days)
    """
    if len(equity_curve) < 2:
        return 0.0, 0
    
    # Running maximum
    running_max = equity_curve.expanding().max()
    
    # Drawdown at each point
    drawdown = (equity_curve - running_max) / running_max * 100
    
    max_dd = drawdown.min()
    
    # Calculate duration
    # Find when we're in drawdown
    in_drawdown = drawdown < 0
    
    # Group consecutive drawdown periods
    drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_groups = drawdown_groups[in_drawdown]
    
    if len(drawdown_groups) == 0:
        return max_dd, 0
    
    # Duration of each drawdown period
    durations = drawdown_groups.groupby(drawdown_groups).size()
    max_duration = durations.max() if len(durations) > 0 else 0
    
    return max_dd, int(max_duration)


def calculate_win_rate(trades: List[float]) -> float:
    """
    Calculate win rate from list of trade P&L.
    
    Args:
        trades: List of trade P&L values
    
    Returns:
        Win rate as percentage
    """
    if len(trades) == 0:
        return 0.0
    
    winning = sum(1 for t in trades if t > 0)
    return (winning / len(trades)) * 100


def calculate_profit_factor(trades: List[float]) -> float:
    """
    Calculate profit factor.
    
    Profit Factor = Gross Profit / Gross Loss
    
    Args:
        trades: List of trade P&L values
    
    Returns:
        Profit factor (>1 is profitable)
    """
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_avg_win_loss(trades: List[float]) -> Tuple[float, float]:
    """
    Calculate average winning and losing trade.
    
    Args:
        trades: List of trade P&L values
    
    Returns:
        Tuple of (avg_win, avg_loss)
    """
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    return avg_win, avg_loss


def calculate_consecutive_wins_losses(trades: List[float]) -> Tuple[int, int]:
    """
    Calculate maximum consecutive wins and losses.
    
    Args:
        trades: List of trade P&L values
    
    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses)
    """
    if len(trades) == 0:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for t in trades:
        if t > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif t < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    return max_wins, max_losses


def calculate_calmar_ratio(
    returns: pd.Series,
    max_drawdown: float,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = Annualized Return / Max Drawdown
    
    Args:
        returns: Daily returns series
        max_drawdown: Maximum drawdown percentage (negative)
        periods_per_year: Trading periods per year
    
    Returns:
        Calmar ratio
    """
    if max_drawdown >= 0:
        return float('inf') if returns.mean() > 0 else 0.0
    
    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    return (annualized_return * 100) / abs(max_drawdown)


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Daily returns series
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized volatility as percentage
    """
    return returns.std() * np.sqrt(periods_per_year) * 100


def generate_performance_report(
    equity_curve: pd.Series,
    trades: List[float],
    periods_per_year: int = 252,
) -> PerformanceReport:
    """
    Generate comprehensive performance report.
    
    Args:
        equity_curve: Daily equity values
        trades: List of individual trade P&L
        periods_per_year: Trading periods per year
    
    Returns:
        PerformanceReport with all metrics
    """
    returns = calculate_returns(equity_curve)
    
    # Basic returns
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    annualized_return = ((1 + returns.mean()) ** periods_per_year - 1) * 100
    
    # Risk metrics
    volatility = calculate_volatility(returns, periods_per_year)
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
    
    # Risk-adjusted
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(returns, max_dd, periods_per_year)
    
    # Trade stats
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t > 0)
    losing_trades = sum(1 for t in trades if t < 0)
    win_rate = calculate_win_rate(trades)
    
    # Profit metrics
    avg_win, avg_loss = calculate_avg_win_loss(trades)
    profit_factor = calculate_profit_factor(trades)
    avg_trade = np.mean(trades) if trades else 0.0
    largest_win = max(trades) if trades else 0.0
    largest_loss = min(trades) if trades else 0.0
    
    # Streaks
    max_wins, max_losses = calculate_consecutive_wins_losses(trades)
    
    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

def _test_metrics():
    """Run basic tests for metrics"""
    print("Testing performance metrics...")
    
    # Create sample equity curve
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    
    # Simulated equity curve with some drawdowns
    returns = np.random.randn(n) * 0.02 + 0.001  # 0.1% average daily return
    equity = pd.Series(100000 * np.cumprod(1 + returns), index=dates)
    
    # Sample trades
    trades = [100, -50, 200, -30, 150, -80, 50, 120, -40, 90]
    
    # Test individual metrics
    returns_series = calculate_returns(equity)
    print(f"Daily returns mean: {returns_series.mean()*100:.4f}%")
    
    sharpe = calculate_sharpe_ratio(returns_series)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    sortino = calculate_sortino_ratio(returns_series)
    print(f"Sortino Ratio: {sortino:.2f}")
    
    max_dd, dd_duration = calculate_max_drawdown(equity)
    print(f"Max Drawdown: {max_dd:.2f}%, Duration: {dd_duration} days")
    
    win_rate = calculate_win_rate(trades)
    print(f"Win Rate: {win_rate:.1f}%")
    
    profit_factor = calculate_profit_factor(trades)
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Generate full report
    report = generate_performance_report(equity, trades)
    print(report)
    
    # Assertions
    assert -100 <= report.max_drawdown <= 0, "Max drawdown should be negative or zero"
    assert 0 <= report.win_rate <= 100, "Win rate should be 0-100%"
    assert report.profit_factor >= 0, "Profit factor should be non-negative"
    
    print("✅ All metrics tests passed!")


if __name__ == "__main__":
    _test_metrics()
