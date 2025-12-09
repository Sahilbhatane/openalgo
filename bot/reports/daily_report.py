"""
Daily Report Generator
======================
Generates comprehensive daily trading reports.

Features:
1. HTML and JSON report formats
2. Performance metrics
3. Trade breakdown
4. Strategy analysis
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
)


logger = logging.getLogger("bot.reports.daily")


class DailyReportGenerator:
    """
    Generates daily trading reports.
    
    Produces both JSON (for programmatic use) and HTML (for viewing) reports.
    
    Usage:
        generator = DailyReportGenerator(output_dir=Path("./reports"))
        report = generator.generate(trades, date="2024-01-15")
    """
    
    def __init__(
        self,
        output_dir: Path = Path("./bot/reports/daily"),
        template: Optional[str] = None,
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report output
            template: Optional custom HTML template
        """
        self.output_dir = output_dir
        self.template = template or self._default_template()
        
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        trades: List[Dict[str, Any]],
        report_date: Optional[str] = None,
        capital: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Generate daily report.
        
        Args:
            trades: List of trade dictionaries
            report_date: Date for report (defaults to today)
            capital: Starting capital for percentage calculations
            
        Returns:
            Report data dictionary
        """
        report_date = report_date or date.today().isoformat()
        
        # Calculate metrics
        report = self._calculate_metrics(trades, capital)
        report['date'] = report_date
        report['trades'] = trades
        
        # Save JSON report
        self._save_json(report, report_date)
        
        # Generate HTML report
        self._save_html(report, report_date)
        
        logger.info(f"Report generated for {report_date}")
        
        return report
    
    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        capital: float,
    ) -> Dict[str, Any]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'gross_pnl': 0.0,
                'net_pnl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'return_pct': 0.0,
            }
        
        # Separate wins and losses
        pnls = [t.get('net_pnl', t.get('gross_pnl', 0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'gross_pnl': sum(t.get('gross_pnl', 0) for t in trades),
            'net_pnl': total_pnl,
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf'),
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'return_pct': total_pnl / capital * 100,
        }
    
    def _save_json(self, report: Dict[str, Any], report_date: str) -> None:
        """Save report as JSON"""
        filepath = self.output_dir / f"report_{report_date}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.debug(f"JSON report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
    
    def _save_html(self, report: Dict[str, Any], report_date: str) -> None:
        """Save report as HTML"""
        filepath = self.output_dir / f"report_{report_date}.html"
        
        try:
            html = self._render_html(report)
            with open(filepath, 'w') as f:
                f.write(html)
            logger.debug(f"HTML report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
    
    def _render_html(self, report: Dict[str, Any]) -> str:
        """Render report as HTML"""
        trades_html = ""
        for t in report.get('trades', []):
            pnl = t.get('net_pnl', t.get('gross_pnl', 0))
            pnl_class = 'profit' if pnl > 0 else 'loss'
            trades_html += f"""
            <tr>
                <td>{t.get('symbol', 'N/A')}</td>
                <td>{t.get('side', 'N/A')}</td>
                <td>{t.get('quantity', 0)}</td>
                <td>₹{t.get('entry_price', 0):,.2f}</td>
                <td>₹{t.get('exit_price', 0):,.2f}</td>
                <td class="{pnl_class}">₹{pnl:+,.2f}</td>
                <td>{t.get('strategy', 'N/A')}</td>
            </tr>
            """
        
        return self.template.format(
            date=report.get('date', 'N/A'),
            total_trades=report.get('total_trades', 0),
            winning_trades=report.get('winning_trades', 0),
            losing_trades=report.get('losing_trades', 0),
            win_rate=report.get('win_rate', 0),
            gross_pnl=report.get('gross_pnl', 0),
            net_pnl=report.get('net_pnl', 0),
            profit_factor=report.get('profit_factor', 0),
            max_win=report.get('max_win', 0),
            max_loss=report.get('max_loss', 0),
            return_pct=report.get('return_pct', 0),
            trades_html=trades_html,
        )
    
    def _default_template(self) -> str:
        """Default HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Report - {date}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00ff88;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
        }}
        .metric-label {{
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #16213e;
            color: #00ff88;
        }}
        tr:hover {{
            background: #1e2a4a;
        }}
        .profit {{
            color: #00ff88;
        }}
        .loss {{
            color: #ff4444;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Trading Report - {date}</h1>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">₹{net_pnl:+,.2f}</div>
                <div class="metric-label">Net P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{return_pct:+.2f}%</div>
                <div class="metric-label">Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{winning_trades}/{losing_trades}</div>
                <div class="metric-label">Win/Loss</div>
            </div>
        </div>
        
        <h2>Trade Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                    <th>Strategy</th>
                </tr>
            </thead>
            <tbody>
                {trades_html}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
