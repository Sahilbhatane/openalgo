"""
Report Exporter
===============
Export reports to various formats.

Features:
1. CSV export for trades and equity curve
2. Excel export with multiple sheets
3. PDF report generation
"""

import csv
import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("bot.reports.export")


class ReportExporter:
    """
    Export trading data to various formats.
    
    Supported formats:
    - CSV: Trades, equity curve
    - JSON: Full report data
    - HTML: Styled reports
    
    Usage:
        exporter = ReportExporter(output_dir=Path("./exports"))
        exporter.export_trades_csv(trades)
        exporter.export_equity_csv(equity_curve)
    """
    
    def __init__(self, output_dir: Path = Path("./bot/exports")):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for exports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_trades_csv(
        self,
        trades: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export trades to CSV.
        
        Args:
            trades: List of trade dictionaries
            filename: Optional filename (defaults to trades_YYYY-MM-DD.csv)
            
        Returns:
            Path to the exported file
        """
        if not trades:
            logger.warning("No trades to export")
            return Path()
        
        filename = filename or f"trades_{date.today().isoformat()}.csv"
        filepath = self.output_dir / filename
        
        # Determine columns from first trade
        columns = list(trades[0].keys())
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(trades)
            
            logger.info(f"Exported {len(trades)} trades to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export trades: {e}")
            return Path()
    
    def export_equity_csv(
        self,
        equity_curve: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export equity curve to CSV.
        
        Args:
            equity_curve: List of equity entries
            filename: Optional filename
            
        Returns:
            Path to the exported file
        """
        if not equity_curve:
            logger.warning("No equity data to export")
            return Path()
        
        filename = filename or "equity_curve.csv"
        filepath = self.output_dir / filename
        
        columns = ['date', 'equity', 'pnl', 'trades', 'drawdown_pct']
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(equity_curve)
            
            logger.info(f"Exported equity curve to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export equity curve: {e}")
            return Path()
    
    def export_performance_summary(
        self,
        stats: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export performance summary to JSON.
        
        Args:
            stats: Performance statistics dictionary
            filename: Optional filename
            
        Returns:
            Path to the exported file
        """
        filename = filename or f"performance_{date.today().isoformat()}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Exported performance summary to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export performance: {e}")
            return Path()
    
    def export_full_report(
        self,
        report_data: Dict[str, Any],
        format: str = "all",
    ) -> Dict[str, Path]:
        """
        Export complete report in multiple formats.
        
        Args:
            report_data: Complete report data
            format: "csv", "json", "html", or "all"
            
        Returns:
            Dictionary of format -> filepath
        """
        results = {}
        report_date = report_data.get('date', date.today().isoformat())
        
        if format in ("csv", "all"):
            # Export trades
            trades = report_data.get('trades', [])
            if trades:
                path = self.export_trades_csv(
                    trades,
                    f"trades_{report_date}.csv",
                )
                if path:
                    results['trades_csv'] = path
        
        if format in ("json", "all"):
            # Export full JSON
            filepath = self.output_dir / f"report_{report_date}.json"
            try:
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                results['json'] = filepath
            except Exception as e:
                logger.error(f"Failed to export JSON: {e}")
        
        if format in ("html", "all"):
            # Export HTML
            html = self._render_full_html(report_data)
            filepath = self.output_dir / f"report_{report_date}.html"
            try:
                with open(filepath, 'w') as f:
                    f.write(html)
                results['html'] = filepath
            except Exception as e:
                logger.error(f"Failed to export HTML: {e}")
        
        return results
    
    def _render_full_html(self, report_data: Dict[str, Any]) -> str:
        """Render complete HTML report"""
        trades = report_data.get('trades', [])
        
        # Build trades table
        trades_rows = ""
        for t in trades:
            pnl = t.get('net_pnl', t.get('gross_pnl', 0))
            pnl_class = 'positive' if pnl > 0 else 'negative'
            trades_rows += f"""
            <tr>
                <td>{t.get('entry_time', '')[:10]}</td>
                <td>{t.get('symbol', '')}</td>
                <td>{t.get('side', '')}</td>
                <td>{t.get('quantity', 0)}</td>
                <td>₹{t.get('entry_price', 0):,.2f}</td>
                <td>₹{t.get('exit_price', 0):,.2f}</td>
                <td class="{pnl_class}">₹{pnl:+,.2f}</td>
            </tr>
            """
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Report - {report_data.get('date', 'N/A')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2em;
            color: #00ff88;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #00ff88;
        }}
        h2 {{
            font-size: 1.3em;
            color: #4a9eff;
            margin: 30px 0 15px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .card-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
        }}
        .card-label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            overflow: hidden;
        }}
        th {{
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Trading Report</h1>
        <p style="color: #888; margin-bottom: 20px;">Date: {report_data.get('date', 'N/A')}</p>
        
        <div class="summary">
            <div class="card">
                <div class="card-value">{report_data.get('total_trades', 0)}</div>
                <div class="card-label">Total Trades</div>
            </div>
            <div class="card">
                <div class="card-value">{report_data.get('win_rate', 0):.1f}%</div>
                <div class="card-label">Win Rate</div>
            </div>
            <div class="card">
                <div class="card-value">₹{report_data.get('net_pnl', 0):+,.0f}</div>
                <div class="card-label">Net P&L</div>
            </div>
            <div class="card">
                <div class="card-value">{report_data.get('profit_factor', 0):.2f}</div>
                <div class="card-label">Profit Factor</div>
            </div>
        </div>
        
        <h2>Trade Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody>
                {trades_rows or '<tr><td colspan="7" style="text-align: center;">No trades</td></tr>'}
            </tbody>
        </table>
        
        <footer>
            Generated by OpenAlgo Trading Bot | {report_data.get('date', 'N/A')}
        </footer>
    </div>
</body>
</html>
        """
