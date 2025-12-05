"""
Performance Tracker
===================
Long-term performance tracking and analysis.

Features:
1. Equity curve management
2. Drawdown tracking
3. Rolling metrics
4. Benchmark comparison
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_sortino_ratio,
)


logger = logging.getLogger("bot.reports.performance")


class PerformanceTracker:
    """
    Tracks long-term trading performance.
    
    Maintains:
    - Daily equity curve
    - Running statistics
    - Peak/trough tracking for drawdowns
    - Rolling metrics (7d, 30d, 90d)
    
    Usage:
        tracker = PerformanceTracker(data_dir=Path("./reports"))
        tracker.record_day(date="2024-01-15", pnl=1500.50, trades=5)
        stats = tracker.get_statistics()
    """
    
    def __init__(
        self,
        data_dir: Path = Path("./bot/reports"),
        initial_capital: float = 100000.0,
    ):
        """
        Initialize performance tracker.
        
        Args:
            data_dir: Directory for data files
            initial_capital: Starting capital
        """
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        
        self.equity_file = data_dir / "equity_curve.json"
        self.stats_file = data_dir / "performance_stats.json"
        
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.equity_curve: List[Dict[str, Any]] = self._load_equity()
        self.current_equity = self._get_current_equity()
        self.peak_equity = max([e['equity'] for e in self.equity_curve], default=initial_capital)
    
    def _load_equity(self) -> List[Dict[str, Any]]:
        """Load equity curve from file"""
        if self.equity_file.exists():
            try:
                with open(self.equity_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load equity curve: {e}")
        return []
    
    def _save_equity(self) -> None:
        """Save equity curve to file"""
        try:
            with open(self.equity_file, 'w') as f:
                json.dump(self.equity_curve, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save equity curve: {e}")
    
    def _get_current_equity(self) -> float:
        """Get current equity from curve or initial capital"""
        if self.equity_curve:
            return self.equity_curve[-1].get('equity', self.initial_capital)
        return self.initial_capital
    
    def record_day(
        self,
        record_date: Optional[str] = None,
        pnl: float = 0.0,
        trades: int = 0,
        winning_trades: int = 0,
        charges: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record a day's performance.
        
        Args:
            record_date: Date string (defaults to today)
            pnl: Net P&L for the day
            trades: Number of trades
            winning_trades: Number of winning trades
            charges: Total charges/brokerage
            
        Returns:
            The recorded entry
        """
        record_date = record_date or date.today().isoformat()
        
        # Calculate new equity
        new_equity = self.current_equity + pnl
        
        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        # Calculate drawdown
        drawdown = (self.peak_equity - new_equity) / self.peak_equity * 100
        
        entry = {
            'date': record_date,
            'equity': new_equity,
            'pnl': pnl,
            'trades': trades,
            'winning_trades': winning_trades,
            'charges': charges,
            'drawdown_pct': drawdown,
            'peak_equity': self.peak_equity,
        }
        
        # Check if we're updating existing entry
        existing_idx = None
        for i, e in enumerate(self.equity_curve):
            if e['date'] == record_date:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.equity_curve[existing_idx] = entry
        else:
            self.equity_curve.append(entry)
        
        self.current_equity = new_equity
        self._save_equity()
        
        logger.info(f"Recorded: {record_date} | P&L: ₹{pnl:+,.2f} | Equity: ₹{new_equity:,.2f}")
        
        return entry
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics.
        
        Returns:
            Dictionary with all performance metrics
        """
        if not self.equity_curve:
            return self._empty_stats()
        
        # Extract data series
        equities = [e['equity'] for e in self.equity_curve]
        pnls = [e['pnl'] for e in self.equity_curve]
        trades = sum(e['trades'] for e in self.equity_curve)
        wins = sum(e['winning_trades'] for e in self.equity_curve)
        
        # Calculate returns
        returns = []
        for i in range(1, len(equities)):
            if equities[i-1] > 0:
                ret = (equities[i] - equities[i-1]) / equities[i-1]
                returns.append(ret)
        
        # Core metrics
        total_pnl = sum(pnls)
        total_return = (self.current_equity - self.initial_capital) / self.initial_capital * 100
        
        # Drawdown
        max_dd = 0.0
        peak = equities[0]
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe ratio (annualized)
        sharpe = 0.0
        if returns:
            import statistics
            if len(returns) > 1:
                avg_ret = statistics.mean(returns)
                std_ret = statistics.stdev(returns)
                if std_ret > 0:
                    sharpe = (avg_ret * 252) / (std_ret * (252 ** 0.5))
        
        # Win rate
        win_rate = wins / trades * 100 if trades > 0 else 0
        
        # Profit factor
        winning_pnl = sum(p for p in pnls if p > 0)
        losing_pnl = abs(sum(p for p in pnls if p < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        # Trading days
        trading_days = len(self.equity_curve)
        profitable_days = sum(1 for e in self.equity_curve if e['pnl'] > 0)
        
        return {
            'current_equity': self.current_equity,
            'initial_capital': self.initial_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'total_trades': trades,
            'winning_trades': wins,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'profitable_days': profitable_days,
            'profitable_days_pct': profitable_days / trading_days * 100 if trading_days > 0 else 0,
            'avg_daily_pnl': total_pnl / trading_days if trading_days > 0 else 0,
            'peak_equity': self.peak_equity,
            'current_drawdown_pct': (self.peak_equity - self.current_equity) / self.peak_equity * 100,
        }
    
    def get_rolling_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate rolling statistics for last N days.
        
        Args:
            days: Number of days to include
            
        Returns:
            Statistics for the period
        """
        if len(self.equity_curve) < 2:
            return self._empty_stats()
        
        # Get last N entries
        recent = self.equity_curve[-days:]
        
        pnls = [e['pnl'] for e in recent]
        trades = sum(e['trades'] for e in recent)
        wins = sum(e['winning_trades'] for e in recent)
        
        return {
            'period_days': len(recent),
            'total_pnl': sum(pnls),
            'avg_daily_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'total_trades': trades,
            'win_rate': wins / trades * 100 if trades > 0 else 0,
            'profitable_days': sum(1 for p in pnls if p > 0),
            'max_daily_pnl': max(pnls) if pnls else 0,
            'min_daily_pnl': min(pnls) if pnls else 0,
        }
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get the full equity curve"""
        return self.equity_curve.copy()
    
    def can_go_live(self, min_profitable_days: int = 20) -> tuple[bool, str]:
        """
        Check if performance meets live trading criteria.
        
        Args:
            min_profitable_days: Minimum profitable days required
            
        Returns:
            (can_go_live, reason)
        """
        stats = self.get_statistics()
        
        # Check minimum trading days
        if stats['trading_days'] < 20:
            return False, f"Not enough trading days: {stats['trading_days']}/20"
        
        # Check profitable days
        if stats['profitable_days'] < min_profitable_days:
            return False, f"Not enough profitable days: {stats['profitable_days']}/{min_profitable_days}"
        
        # Check win rate
        if stats['win_rate'] < 40:
            return False, f"Win rate too low: {stats['win_rate']:.1f}%"
        
        # Check profit factor
        if stats['profit_factor'] < 1.0:
            return False, f"Profit factor < 1: {stats['profit_factor']:.2f}"
        
        # Check drawdown
        if stats['max_drawdown_pct'] > 15:
            return False, f"Max drawdown too high: {stats['max_drawdown_pct']:.1f}%"
        
        # Check total return
        if stats['total_return_pct'] < 0:
            return False, f"Negative total return: {stats['total_return_pct']:.1f}%"
        
        return True, "All criteria met"
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics"""
        return {
            'current_equity': self.initial_capital,
            'initial_capital': self.initial_capital,
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'trading_days': 0,
            'profitable_days': 0,
            'profitable_days_pct': 0.0,
            'avg_daily_pnl': 0.0,
            'peak_equity': self.initial_capital,
            'current_drawdown_pct': 0.0,
        }
