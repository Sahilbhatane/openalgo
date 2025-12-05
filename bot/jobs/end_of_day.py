"""
End of Day Job
==============
Job that runs after market close (3:35 PM IST) for daily housekeeping.

Tasks:
1. Calculate daily P&L and charges
2. Update strategy metrics
3. Generate daily report
4. Archive logs
5. Prepare for next trading day
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

from ..utils.charges import calculate_total_charges
from ..utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
)


logger = logging.getLogger("bot.jobs.end_of_day")


@dataclass
class TradeRecord:
    """Record of a single trade"""
    symbol: str
    side: str  # BUY or SELL
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    charges: float
    net_pnl: float
    strategy: str = ""
    exit_reason: str = ""


@dataclass
class DailyReport:
    """Complete daily trading report"""
    date: str = ""
    
    # Trade summary
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    gross_pnl: float = 0.0
    total_charges: float = 0.0
    net_pnl: float = 0.0
    
    # Metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Risk
    max_drawdown: float = 0.0
    max_position_value: float = 0.0
    
    # Charge breakdown
    brokerage: float = 0.0
    stt: float = 0.0
    transaction_charges: float = 0.0
    gst: float = 0.0
    stamp_duty: float = 0.0
    
    # Strategy breakdown
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    
    # Trade list
    trades: List[TradeRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'date': self.date,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'gross_pnl': self.gross_pnl,
            'total_charges': self.total_charges,
            'net_pnl': self.net_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'max_drawdown': self.max_drawdown,
            'max_position_value': self.max_position_value,
            'charges': {
                'brokerage': self.brokerage,
                'stt': self.stt,
                'transaction_charges': self.transaction_charges,
                'gst': self.gst,
                'stamp_duty': self.stamp_duty,
            },
            'strategy_pnl': self.strategy_pnl,
            'trades': [
                {
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'gross_pnl': t.gross_pnl,
                    'charges': t.charges,
                    'net_pnl': t.net_pnl,
                    'strategy': t.strategy,
                    'exit_reason': t.exit_reason,
                }
                for t in self.trades
            ],
        }


class EndOfDayJob:
    """
    End of day processing job.
    
    Runs after market close to:
    - Compile daily statistics
    - Calculate true P&L after charges
    - Update historical records
    - Generate reports
    
    Usage:
        job = EndOfDayJob(
            trade_getter=get_today_trades,
            report_dir=Path("./reports"),
        )
        report = job.run()
    """
    
    def __init__(
        self,
        trade_getter: Callable[[], List[Dict[str, Any]]],
        report_dir: Path = Path("./bot/reports/daily"),
        equity_curve_path: Path = Path("./bot/reports/equity_curve.json"),
        on_complete: Optional[Callable[[DailyReport], None]] = None,
    ):
        """
        Initialize end of day job.
        
        Args:
            trade_getter: Function that returns today's trades
                Each trade should have: symbol, side, entry_time, exit_time,
                entry_price, exit_price, quantity, strategy, exit_reason
            report_dir: Directory to save daily reports
            equity_curve_path: Path to equity curve file
            on_complete: Callback when processing is complete
        """
        self.get_trades = trade_getter
        self.report_dir = report_dir
        self.equity_curve_path = equity_curve_path
        self.on_complete = on_complete
        
        # Ensure directories exist
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.equity_curve_path.parent.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> DailyReport:
        """
        Run end of day processing.
        
        Returns:
            DailyReport with all statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING END OF DAY PROCESSING")
        logger.info("=" * 60)
        
        today = date.today().isoformat()
        report = DailyReport(date=today)
        
        # Get today's trades
        try:
            raw_trades = self.get_trades()
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return report
        
        if not raw_trades:
            logger.info("No trades today")
            self._save_report(report)
            return report
        
        logger.info(f"Processing {len(raw_trades)} trades")
        
        # Process each trade
        total_charges = {
            'brokerage': 0.0,
            'stt': 0.0,
            'transaction_charges': 0.0,
            'gst': 0.0,
            'stamp_duty': 0.0,
        }
        
        winning_pnls = []
        losing_pnls = []
        strategy_pnl: Dict[str, float] = {}
        
        for trade_data in raw_trades:
            trade = self._process_trade(trade_data, total_charges)
            report.trades.append(trade)
            
            # Track wins/losses
            if trade.net_pnl > 0:
                report.winning_trades += 1
                winning_pnls.append(trade.net_pnl)
            elif trade.net_pnl < 0:
                report.losing_trades += 1
                losing_pnls.append(trade.net_pnl)
            
            # Aggregate by strategy
            strategy = trade.strategy or "default"
            if strategy not in strategy_pnl:
                strategy_pnl[strategy] = 0.0
            strategy_pnl[strategy] += trade.net_pnl
            
            report.gross_pnl += trade.gross_pnl
            report.net_pnl += trade.net_pnl
        
        # Calculate summary statistics
        report.total_trades = len(report.trades)
        report.total_charges = sum(total_charges.values())
        report.strategy_pnl = strategy_pnl
        
        # Charge breakdown
        report.brokerage = total_charges['brokerage']
        report.stt = total_charges['stt']
        report.transaction_charges = total_charges['transaction_charges']
        report.gst = total_charges['gst']
        report.stamp_duty = total_charges['stamp_duty']
        
        # Win/loss statistics
        if report.total_trades > 0:
            report.win_rate = report.winning_trades / report.total_trades * 100
        
        if winning_pnls:
            report.largest_win = max(winning_pnls)
            report.average_win = sum(winning_pnls) / len(winning_pnls)
        
        if losing_pnls:
            report.largest_loss = min(losing_pnls)  # Most negative
            report.average_loss = sum(losing_pnls) / len(losing_pnls)
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
        if total_losses > 0:
            report.profit_factor = total_wins / total_losses
        
        # Save report
        self._save_report(report)
        
        # Update equity curve
        self._update_equity_curve(report)
        
        # Log summary
        self._log_summary(report)
        
        # Callback
        if self.on_complete:
            try:
                self.on_complete(report)
            except Exception as e:
                logger.error(f"on_complete callback failed: {e}")
        
        return report
    
    def _process_trade(
        self,
        trade_data: Dict[str, Any],
        total_charges: Dict[str, float],
    ) -> TradeRecord:
        """Process a single trade and calculate P&L"""
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        quantity = trade_data.get('quantity', 0)
        side = trade_data.get('side', 'BUY')
        
        # Calculate gross P&L
        if side == 'BUY':
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Calculate charges
        turnover = (entry_price + exit_price) * quantity
        charges_result = calculate_total_charges(
            buy_price=entry_price if side == 'BUY' else exit_price,
            sell_price=exit_price if side == 'BUY' else entry_price,
            quantity=quantity,
            segment='equity_intraday',
        )
        
        charges = charges_result.total_charges
        
        # Aggregate charges
        total_charges['brokerage'] += charges_result.brokerage
        total_charges['stt'] += charges_result.stt
        total_charges['transaction_charges'] += charges_result.exchange_charges
        total_charges['gst'] += charges_result.gst
        total_charges['stamp_duty'] += charges_result.stamp_duty
        
        return TradeRecord(
            symbol=trade_data.get('symbol', 'UNKNOWN'),
            side=side,
            entry_time=trade_data.get('entry_time', ''),
            exit_time=trade_data.get('exit_time', ''),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=gross_pnl,
            charges=charges,
            net_pnl=gross_pnl - charges,
            strategy=trade_data.get('strategy', ''),
            exit_reason=trade_data.get('exit_reason', ''),
        )
    
    def _save_report(self, report: DailyReport) -> None:
        """Save daily report to JSON file"""
        filename = f"report_{report.date}.json"
        filepath = self.report_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _update_equity_curve(self, report: DailyReport) -> None:
        """Update the equity curve file"""
        # Load existing curve
        curve = []
        if self.equity_curve_path.exists():
            try:
                with open(self.equity_curve_path, 'r') as f:
                    curve = json.load(f)
            except Exception:
                curve = []
        
        # Calculate new equity
        previous_equity = curve[-1]['equity'] if curve else 100000  # Start with 1L
        new_equity = previous_equity + report.net_pnl
        
        # Add today's entry
        curve.append({
            'date': report.date,
            'equity': new_equity,
            'daily_pnl': report.net_pnl,
            'trades': report.total_trades,
        })
        
        # Save updated curve
        try:
            with open(self.equity_curve_path, 'w') as f:
                json.dump(curve, f, indent=2)
            logger.info(f"Equity curve updated: ₹{new_equity:,.2f}")
        except Exception as e:
            logger.error(f"Failed to update equity curve: {e}")
    
    def _log_summary(self, report: DailyReport) -> None:
        """Log daily summary"""
        logger.info("=" * 60)
        logger.info("END OF DAY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Date: {report.date}")
        logger.info(f"Total Trades: {report.total_trades}")
        logger.info(f"Win Rate: {report.win_rate:.1f}%")
        logger.info("")
        logger.info("P&L:")
        logger.info(f"  Gross P&L: ₹{report.gross_pnl:+,.2f}")
        logger.info(f"  Charges:   ₹{report.total_charges:,.2f}")
        logger.info(f"  Net P&L:   ₹{report.net_pnl:+,.2f}")
        logger.info("")
        logger.info("Charge Breakdown:")
        logger.info(f"  Brokerage:    ₹{report.brokerage:,.2f}")
        logger.info(f"  STT:          ₹{report.stt:,.2f}")
        logger.info(f"  Transaction:  ₹{report.transaction_charges:,.2f}")
        logger.info(f"  GST:          ₹{report.gst:,.2f}")
        logger.info(f"  Stamp Duty:   ₹{report.stamp_duty:,.2f}")
        logger.info("")
        if report.strategy_pnl:
            logger.info("Strategy P&L:")
            for strategy, pnl in report.strategy_pnl.items():
                logger.info(f"  {strategy}: ₹{pnl:+,.2f}")
        logger.info("=" * 60)
