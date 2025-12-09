"""
Trading Mode Management
=======================
Handle PAPER_MODE and LIVE_MODE switching with safety checks.
"""

import os
import json
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "PAPER"  # Simulated trading
    LIVE = "LIVE"    # Real money trading
    
    @classmethod
    def from_string(cls, value: str) -> "TradingMode":
        """Parse mode from string, default to PAPER for safety"""
        value = value.upper().strip()
        if value == "LIVE":
            return cls.LIVE
        return cls.PAPER  # Default to PAPER for any other value


@dataclass
class ModeTransitionResult:
    """Result of a mode transition attempt"""
    success: bool
    message: str
    previous_mode: TradingMode
    current_mode: TradingMode


@dataclass
class PaperTradingStats:
    """Statistics from paper trading phase"""
    start_date: str = ""
    trading_days: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100


class ModeManager:
    """
    Manages trading mode (PAPER/LIVE) with safety checks.
    
    Safety Features:
    - Default to PAPER mode
    - Require minimum paper trading period before LIVE
    - Require minimum win rate before LIVE
    - Log all mode transitions
    - Manual confirmation required for LIVE mode
    """
    
    # Minimum requirements before LIVE mode is allowed
    MIN_PAPER_DAYS: int = 21  # 3 weeks minimum
    MIN_PAPER_TRADES: int = 50  # At least 50 trades
    MIN_WIN_RATE: float = 45.0  # At least 45% win rate
    MIN_SHARPE_RATIO: float = 0.5  # Positive Sharpe ratio
    MAX_DRAWDOWN_ALLOWED: float = -15.0  # Max -15% drawdown
    
    def __init__(self, data_dir: str = "bot_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mode_file = self.data_dir / "mode_state.json"
        self.stats_file = self.data_dir / "paper_stats.json"
        
        # Load current mode (default to PAPER)
        self._current_mode = self._load_mode()
        self._paper_stats = self._load_stats()
    
    @property
    def current_mode(self) -> TradingMode:
        return self._current_mode
    
    @property
    def is_paper(self) -> bool:
        return self._current_mode == TradingMode.PAPER
    
    @property
    def is_live(self) -> bool:
        return self._current_mode == TradingMode.LIVE
    
    def _load_mode(self) -> TradingMode:
        """Load mode from file or env, default to PAPER"""
        # First check env variable
        env_mode = os.getenv("TRADING_MODE", "PAPER")
        
        # Then check file
        if self.mode_file.exists():
            try:
                with open(self.mode_file) as f:
                    data = json.load(f)
                    file_mode = data.get("mode", "PAPER")
                    # Env var takes precedence if explicitly set
                    if os.getenv("TRADING_MODE"):
                        return TradingMode.from_string(env_mode)
                    return TradingMode.from_string(file_mode)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return TradingMode.from_string(env_mode)
    
    def _save_mode(self):
        """Persist current mode to file"""
        data = {
            "mode": self._current_mode.value,
            "last_changed": datetime.now().isoformat(),
            "changed_by": "system"
        }
        with open(self.mode_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_stats(self) -> PaperTradingStats:
        """Load paper trading statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    data = json.load(f)
                    return PaperTradingStats(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return PaperTradingStats()
    
    def save_stats(self, stats: PaperTradingStats):
        """Save paper trading statistics"""
        self._paper_stats = stats
        with open(self.stats_file, "w") as f:
            json.dump(stats.__dict__, f, indent=2)
    
    def get_paper_stats(self) -> PaperTradingStats:
        """Get current paper trading statistics"""
        return self._paper_stats
    
    def check_live_readiness(self) -> tuple[bool, list[str]]:
        """
        Check if system is ready for LIVE mode.
        
        Returns:
            Tuple of (is_ready, list of issues/warnings)
        """
        issues = []
        stats = self._paper_stats
        
        if stats.trading_days < self.MIN_PAPER_DAYS:
            issues.append(
                f"Insufficient paper trading days: {stats.trading_days}/{self.MIN_PAPER_DAYS}"
            )
        
        if stats.total_trades < self.MIN_PAPER_TRADES:
            issues.append(
                f"Insufficient paper trades: {stats.total_trades}/{self.MIN_PAPER_TRADES}"
            )
        
        if stats.win_rate < self.MIN_WIN_RATE:
            issues.append(
                f"Win rate too low: {stats.win_rate:.1f}% (min {self.MIN_WIN_RATE}%)"
            )
        
        if stats.sharpe_ratio < self.MIN_SHARPE_RATIO:
            issues.append(
                f"Sharpe ratio too low: {stats.sharpe_ratio:.2f} (min {self.MIN_SHARPE_RATIO})"
            )
        
        if stats.max_drawdown < self.MAX_DRAWDOWN_ALLOWED:
            issues.append(
                f"Max drawdown too high: {stats.max_drawdown:.1f}% (max {self.MAX_DRAWDOWN_ALLOWED}%)"
            )
        
        is_ready = len(issues) == 0
        return is_ready, issues
    
    def switch_to_paper(self) -> ModeTransitionResult:
        """Switch to PAPER mode (always allowed)"""
        previous = self._current_mode
        self._current_mode = TradingMode.PAPER
        self._save_mode()
        
        return ModeTransitionResult(
            success=True,
            message="Switched to PAPER mode",
            previous_mode=previous,
            current_mode=self._current_mode
        )
    
    def switch_to_live(self, force: bool = False, confirmation: str = "") -> ModeTransitionResult:
        """
        Switch to LIVE mode with safety checks.
        
        Args:
            force: Skip safety checks (DANGEROUS)
            confirmation: Must be "I UNDERSTAND THE RISKS" to proceed
        
        Returns:
            ModeTransitionResult with success status and message
        """
        previous = self._current_mode
        
        # Require explicit confirmation
        if confirmation != "I UNDERSTAND THE RISKS":
            return ModeTransitionResult(
                success=False,
                message="Must provide confirmation='I UNDERSTAND THE RISKS'",
                previous_mode=previous,
                current_mode=self._current_mode
            )
        
        if not force:
            is_ready, issues = self.check_live_readiness()
            if not is_ready:
                return ModeTransitionResult(
                    success=False,
                    message=f"Not ready for LIVE mode:\n" + "\n".join(f"  - {i}" for i in issues),
                    previous_mode=previous,
                    current_mode=self._current_mode
                )
        
        self._current_mode = TradingMode.LIVE
        self._save_mode()
        
        return ModeTransitionResult(
            success=True,
            message="⚠️ SWITCHED TO LIVE MODE - REAL MONEY TRADING ENABLED",
            previous_mode=previous,
            current_mode=self._current_mode
        )
    
    def get_status_report(self) -> str:
        """Get a human-readable status report"""
        stats = self._paper_stats
        is_ready, issues = self.check_live_readiness()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRADING MODE STATUS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Current Mode: {self._current_mode.value:>10}                               ║
╠══════════════════════════════════════════════════════════════╣
║  Paper Trading Statistics:                                   ║
║    Days Traded:    {stats.trading_days:>6} / {self.MIN_PAPER_DAYS} required              ║
║    Total Trades:   {stats.total_trades:>6} / {self.MIN_PAPER_TRADES} required              ║
║    Win Rate:       {stats.win_rate:>6.1f}% / {self.MIN_WIN_RATE}% required             ║
║    Sharpe Ratio:   {stats.sharpe_ratio:>6.2f} / {self.MIN_SHARPE_RATIO} required              ║
║    Max Drawdown:   {stats.max_drawdown:>6.1f}% / {self.MAX_DRAWDOWN_ALLOWED}% max                ║
║    Total P&L:      ₹{stats.total_pnl:>9.2f}                           ║
╠══════════════════════════════════════════════════════════════╣
║  Live Mode Ready: {'✅ YES' if is_ready else '❌ NO ':>10}                               ║
"""
        if issues:
            report += "╠══════════════════════════════════════════════════════════════╣\n"
            report += "║  Issues:                                                     ║\n"
            for issue in issues:
                report += f"║    ⚠️ {issue:<53}║\n"
        
        report += "╚══════════════════════════════════════════════════════════════╝"
        return report
