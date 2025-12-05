# OpenAlgo Intraday Trading Bot

A fully autonomous intraday trading bot built on the OpenAlgo framework for the Indian stock market.

## 🎯 Features

- **Morning Learning** (8:00 AM IST): Analyzes previous day data, scores symbols, creates trading plan
- **Volatility Breakout Strategy**: Opening range breakout with VWAP confirmation
- **ATR-based Stops**: Dynamic stop losses based on market volatility
- **Risk Management**: Daily drawdown limits, position sizing, kill switch
- **Paper Mode**: Safe simulation before live trading
- **Daily Reports**: HTML/JSON reports with P&L, metrics, equity curve

## 📁 Project Structure

```
bot/
├── main.py              # Entry point - TradingBot class
├── core/                # Configuration and constants
│   ├── config.py       # Central config (loads from .env)
│   ├── mode.py         # PAPER/LIVE mode management
│   └── constants.py    # Trading hours, limits
├── strategies/          # Trading strategies
│   ├── signal.py       # Signal classes
│   └── volatility_breakout.py
├── jobs/                # Scheduled tasks
│   ├── scheduler.py    # APScheduler wrapper
│   ├── morning_learning.py  # 8:00 AM pre-market
│   ├── market_open.py       # 9:15 AM
│   ├── square_off.py        # 3:10 PM
│   └── end_of_day.py        # 3:35 PM
├── execution/           # Order management
│   ├── order_manager.py    # Retry, idempotency
│   ├── paper_executor.py   # Paper trading
│   └── live_executor.py    # Live via OpenAlgo API
├── risk/                # Risk management
│   ├── risk_manager.py    # Limits, drawdown
│   ├── position_sizer.py  # Kelly, ATR sizing
│   └── kill_switch.py     # Emergency stop
├── reports/             # Reporting
│   ├── daily_report.py   # HTML reports
│   ├── performance.py    # Equity curve
│   └── export.py         # CSV/JSON export
└── utils/               # Utilities
    ├── charges.py       # Brokerage calculator
    ├── indicators.py    # ATR, RSI, VWAP, etc.
    ├── metrics.py       # Sharpe, drawdown
    └── time_utils.py    # IST handling
```

## 🚀 Quick Start

### 1. Configure Environment

Create/update `.env` in project root:

```env
# OpenAlgo API
OPENALGO_HOST=127.0.0.1
OPENALGO_PORT=5000
OPENALGO_APIKEY=your-api-key

# Broker (AngelOne)
BROKER_API_KEY=your-key
BROKER_API_SECRET=your-secret
BROKER_TOTP_SECRET=your-totp

# Trading
TRADING_CAPITAL=100000
TRADING_MODE=PAPER
```

### 2. Start OpenAlgo Server

```bash
python app.py
```

### 3. Run the Bot (Paper Mode)

```bash
python -m bot.main
```

The bot will:
1. Start in PAPER mode (safe simulation)
2. Schedule all jobs (morning learning, market open, square off, EOD)
3. Log all activities to `bot_data/logs/`
4. Generate reports in `bot_data/reports/`

## ⏰ Daily Schedule

| Time | Job | Description |
|------|-----|-------------|
| 8:00 AM | Morning Learning | Fetch data, analyze, create plan |
| 9:15 AM | Market Open | Verify connection, start watching |
| 9:20 AM | First Trade Window | Opening range established |
| 3:10 PM | Square Off | Close all positions |
| 3:35 PM | EOD Report | Generate daily report |

## 🛡️ Safety Features

### Paper Mode First
- Bot starts in PAPER mode by default
- Simulates all trades with realistic slippage
- Tracks performance over weeks
- Only switch to LIVE after consistent profitability

### Kill Switch
- Automatically triggers on:
  - 5% daily drawdown
  - API connection loss
  - Risk limit breach
- Manual activation available
- Squares off all positions immediately

### Position Limits
- Max 3 concurrent positions
- Max 30% capital in single position
- Max 20 orders per day
- 1% risk per trade

## 📊 Reports

Daily reports are saved in `bot_data/reports/daily/`:

- `report_YYYY-MM-DD.json` - Machine-readable
- `report_YYYY-MM-DD.html` - Human-readable

Reports include:
- Trade list with entry/exit
- P&L with charge breakdown
- Win rate, profit factor
- Equity curve

## ⚠️ Disclaimer

This bot is for educational purposes. Trading involves substantial risk. 

**NEVER:**
- Trade with money you can't afford to lose
- Switch to LIVE mode without weeks of paper testing
- Run without monitoring during initial live phase

## 📝 License

Same license as OpenAlgo project.
