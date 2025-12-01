# Intraday Trading Bot - Project Roadmap

## Project Status: ✅ Foundation Ready

### Completed Components

#### 1. **Core Infrastructure** ✅
- [x] OpenAlgo framework installed (Python 3.14.0)
- [x] All dependencies installed and verified
- [x] Virtual environment (.venv) configured
- [x] Environment variables configured (.env)
- [x] Git pre-commit hooks configured (black, isort, flake8, detect-secrets)
- [x] Docker setup (Dockerfile + docker-compose.yaml)
- [x] Branching strategy documented (BRANCHING_GUIDE.md)

#### 2. **Broker Integration** ✅
- [x] Groww broker adapter created (`broker/groww_adapter.py`)
  - Authentication with API keys
  - Balance & margin management
  - Market data (LTP, candles)
  - Order placement & cancellation
  - Position tracking
  - **Paper mode with slippage & partial fills**
  - Error handling & retries
  - Full docstrings

#### 3. **Trading Charges Module** ✅
- [x] Comprehensive charges calculator (`utils/charges.py`)
  - Brokerage calculation
  - GST on brokerage & transaction charges
  - STT (Securities Transaction Tax)
  - Exchange fees & SEBI charges
  - Stamp duty
  - API subscription amortization
  - Total trade cost analysis
  - Breakeven calculations
- [x] Complete test suite (55 tests) - **ALL PASSING** ✅

---

## 🎯 Phase 1: Core Trading System (Week 1-2)

### Priority Tasks

#### 1.1 Data Management Module
**Location**: `utils/data_manager.py`

```python
Features:
- Fetch historical OHLC data (1m, 5m, 15m, 1d)
- Cache management for historical data
- Real-time data streaming integration
- Data validation & cleaning
- Symbol universe management
```

**Test**: `test/test_data_manager.py`

#### 1.2 Technical Indicators Module
**Location**: `utils/indicators.py`

```python
Indicators Required:
- ATR (Average True Range)
- RSI (Relative Strength Index)
- VWAP (Volume Weighted Average Price)
- Moving Averages (SMA, EMA)
- Bollinger Bands
- Volume Profile
- Momentum indicators
```

**Test**: `test/test_indicators.py`

#### 1.3 News & Sentiment Module
**Location**: `utils/sentiment.py`

```python
Features:
- Fetch news headlines (RSS feeds, APIs)
- Simple lexicon-based sentiment scoring
- News caching & deduplication
- Symbol-to-news mapping
```

**Test**: `test/test_sentiment.py`

---

## 🌅 Phase 2: Morning Learning Job (Week 2)

### 2.1 Morning Learning Service
**Location**: `jobs/morning_learning.py`

**Schedule**: 08:00 IST daily (APScheduler)

**Workflow**:
1. Load user-defined symbol watchlist
2. For each symbol:
   - Fetch OHLC data (1m, 5m, 1d)
   - Calculate ATR, RSI, VWAP gap, momentum
   - Get average volume
   - Fetch last 48h news headlines
   - Compute sentiment score
3. Rank symbols by score: `liquidity × momentum × volatility × sentiment`
4. Generate entry zones, stop loss, targets
5. Calculate position sizing using `charges.py`
6. Output: `data/today_plan.json`

**Test**: `test/test_morning_learning.py`

### 2.2 Symbol Watchlist Manager
**Location**: `config/watchlist.json`

```json
{
  "watchlist": [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN",
    "ICICIBANK", "KOTAKBANK", "BAJFINANCE"
  ],
  "filters": {
    "min_price": 50,
    "max_price": 5000,
    "min_volume": 500000,
    "sectors": ["Banking", "IT", "Auto"]
  }
}
```

---

## 📊 Phase 3: Strategy Engine (Week 3)

### 3.1 Base Strategy Framework
**Location**: `strategies/base_strategy.py`

```python
class BaseStrategy:
    - on_candle(candle_data)
    - generate_signal() -> BUY|SELL|HOLD
    - calculate_position_size()
    - set_stop_loss()
    - set_target()
    - check_exit_conditions()
```

### 3.2 Volatility Breakout Strategy
**Location**: `strategies/volatility_breakout.py`

**Rules**:
- Entry: Price breaks high of first 15 minutes
- Volume Filter: Current volume > 1.5x average volume
- VWAP Filter: Price > VWAP for longs
- Stop Loss: ATR-based (e.g., entry - 2×ATR)
- Target: 1.5R (Risk:Reward = 1:1.5)
- Trailing SL: After 0.75R profit
- Exit: 15:10 IST or SL/TP hit

**Test**: `test/test_volatility_breakout.py`

### 3.3 Strategy Manager
**Location**: `strategies/strategy_manager.py`

```python
Features:
- Load strategy from config
- Execute strategy logic
- Log all signals
- Performance tracking
```

---

## 🛡️ Phase 4: Risk Management (Week 3-4)

### 4.1 Risk Manager
**Location**: `risk/risk_manager.py`

```python
Risk Rules:
- Max exposure per symbol: 10% of capital
- Max daily drawdown: 5%
- Max simultaneous open trades: 5
- Max loss per trade: 2% of capital
- Max daily loss: 5% of capital
- Position sizing based on ATR
```

**Test**: `test/test_risk_manager.py`

### 4.2 Order Manager
**Location**: `risk/order_manager.py`

```python
Features:
- Order validation before execution
- Risk checks
- Idempotency (prevent duplicate orders)
- Retry logic with exponential backoff
- Order state machine
- Emergency stop (kill switch)
```

**Test**: `test/test_order_manager.py`

### 4.3 Kill Switch
**Location**: `risk/kill_switch.py`

**Triggers**:
- Daily loss limit hit
- Manual trigger
- System errors
- Circuit breaker events

**Actions**:
- Square off all positions
- Cancel all pending orders
- Pause trading
- Send alert

---

## 📈 Phase 5: Backtesting & Paper Trading (Week 4-5)

### 5.1 Backtesting Engine
**Location**: `backtest/backtester.py`

```python
Features:
- Walk-forward backtesting
- Realistic slippage modeling
- Partial fills simulation
- Brokerage & charges integration
- Performance metrics:
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Profit Factor
  - Expectancy
  - MAE/MFE
```

**Test**: `test/test_backtester.py`

### 5.2 Paper Trading Mode
**Status**: ✅ Already implemented in `groww_adapter.py`

**Configuration**:
```env
PAPER_MODE=true
PAPER_SLIPPAGE=0.1
PAPER_PARTIAL_FILL_PROB=0.2
PAPER_INITIAL_BALANCE=1000000
```

**Evaluation Period**: 3-4 weeks before live trading

---

## 📊 Phase 6: Reporting & Monitoring (Week 5)

### 6.1 Report Generator
**Location**: `tools/report_generator.py`

**Daily Reports** (Generated at EOD):
- Equity curve chart
- Drawdown chart
- Per-trade P&L table
- Performance metrics:
  - Total P&L
  - Win rate
  - Avg win/loss
  - Profit factor
  - Expectancy
  - MAE/MFE
- Cost breakdown
- Daily summary JSON
- HTML report
- CSV export

**Test**: `test/test_report_generator.py`

### 6.2 Live Monitoring Dashboard
**Location**: `templates/dashboard.html`

**Metrics**:
- Real-time P&L
- Open positions
- Today's trades
- Current exposure
- Risk utilization
- System health

### 6.3 Alert System
**Location**: `utils/alerts.py`

**Channels**:
- Telegram bot integration
- Slack webhooks
- Email alerts

**Alert Types**:
- Trade execution
- Stop loss hit
- Daily loss limit
- System errors
- Kill switch activation

---

## 🤖 Phase 7: ML Enhancement (Optional - Week 6+)

### 7.1 Reinforcement Learning Framework
**Location**: `ml/rl_agent.py`

**IMPORTANT**: Shadow mode only - **NO LIVE EXECUTION**

```python
Features:
- Discrete action space: [BUY_SMALL, BUY_MED, SELL, HOLD]
- State space: Technical indicators + market conditions
- Reward function: Risk-adjusted returns
- Offline training from historical trades
- Decision logging for analysis
```

### 7.2 ML Model Training
**Location**: `ml/train.py`

```python
Process:
1. Load historical trade logs
2. Extract features
3. Train RL agent
4. Validate on hold-out data
5. Export confusion matrix
6. Generate decision quality metrics
```

**Test**: `test/test_rl_agent.py`

---

## 🚀 Phase 8: Production Deployment (Week 7)

### 8.1 Pre-Deployment Checklist

- [ ] Paper trading results reviewed (3-4 weeks data)
- [ ] All tests passing
- [ ] Backtesting results satisfactory
- [ ] Risk limits configured
- [ ] Kill switch tested
- [ ] Monitoring dashboards operational
- [ ] Alert system configured
- [ ] Database backups configured
- [ ] Logging infrastructure ready
- [ ] Manual review process established

### 8.2 Phased Live Rollout

**Phase 8.1: Minimal Capital (Week 7)**
- Allocate ₹50,000 initially
- 1 symbol max
- Monitor for 1 week

**Phase 8.2: Gradual Scale (Week 8)**
- Increase to ₹200,000
- 2-3 symbols
- Monitor for 2 weeks

**Phase 8.3: Full Deployment (Week 10+)**
- Full capital allocation
- All watchlist symbols
- Continuous monitoring

---

## 📁 Project Structure

```
openalgo/
├── broker/
│   ├── groww/                  # Existing Groww broker
│   └── groww_adapter.py        # ✅ Created (Paper trading ready)
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py        # TODO: Base strategy class
│   ├── volatility_breakout.py  # TODO: Main intraday strategy
│   └── strategy_manager.py     # TODO: Strategy orchestration
│
├── jobs/
│   ├── __init__.py
│   └── morning_learning.py     # TODO: 08:00 IST research job
│
├── risk/
│   ├── __init__.py
│   ├── risk_manager.py         # TODO: Risk rules engine
│   ├── order_manager.py        # TODO: Order validation & execution
│   └── kill_switch.py          # TODO: Emergency stop
│
├── utils/
│   ├── charges.py              # ✅ Created (55 tests passing)
│   ├── data_manager.py         # TODO: Historical data fetching
│   ├── indicators.py           # TODO: Technical indicators
│   ├── sentiment.py            # TODO: News sentiment analysis
│   └── alerts.py               # TODO: Telegram/Slack alerts
│
├── backtest/
│   ├── __init__.py
│   └── backtester.py           # TODO: Walk-forward backtesting
│
├── tools/
│   ├── __init__.py
│   └── report_generator.py     # TODO: Daily reports & charts
│
├── ml/                         # Optional ML enhancements
│   ├── __init__.py
│   ├── rl_agent.py            # TODO: Reinforcement learning
│   └── train.py               # TODO: Offline training
│
├── config/
│   ├── watchlist.json         # TODO: Symbol universe
│   ├── strategy_config.json   # TODO: Strategy parameters
│   └── risk_config.json       # TODO: Risk limits
│
├── data/
│   ├── historical/            # Cached OHLC data
│   ├── today_plan.json        # Morning learning output
│   └── trade_log.json         # All trade records
│
├── reports/
│   ├── daily/                 # Daily HTML reports
│   └── summary/               # Monthly summaries
│
├── test/
│   ├── test_charges.py        # ✅ 55 tests passing
│   ├── test_data_manager.py   # TODO
│   ├── test_indicators.py     # TODO
│   ├── test_sentiment.py      # TODO
│   ├── test_morning_learning.py  # TODO
│   ├── test_volatility_breakout.py  # TODO
│   ├── test_risk_manager.py   # TODO
│   ├── test_order_manager.py  # TODO
│   └── test_backtester.py     # TODO
│
├── .env                       # ✅ Configured with PAPER_MODE
├── .pre-commit-config.yaml    # ✅ Created (black, isort, flake8)
├── .gitignore                 # ✅ Updated (logs, reports, .env)
├── BRANCHING_GUIDE.md         # ✅ Created (3-branch model)
├── Dockerfile                 # ✅ Created (Python 3.14.0)
├── docker-compose.yaml        # ✅ Created
└── requirements.txt           # ✅ All dependencies installed
```

---

## 🔒 Safety Guidelines

### PAPER MODE (Current Phase)
- ✅ Enabled in `.env`: `PAPER_MODE=true`
- ✅ Simulates all trades with realistic slippage
- ✅ No real money at risk
- **Mandatory Duration**: 3-4 weeks minimum

### LIVE MODE (Future Phase)
- ⚠️ Requires explicit manual confirmation
- ⚠️ Must pass all paper trading evaluations
- ⚠️ Start with minimal capital (₹50K)
- ⚠️ Gradual scale-up only after proven results
- ⚠️ Kill switch must be tested and operational

### Never Commit
- API keys, secrets, tokens
- Actual trading credentials
- Personal broker account details

---

## 📊 Success Metrics

### Paper Trading Evaluation (3-4 weeks)
- **Min Win Rate**: >45%
- **Max Drawdown**: <5%
- **Profit Factor**: >1.5
- **Sharpe Ratio**: >1.0
- **Consistency**: Positive 60% of days

### Live Trading Goals (After approval)
- **Daily Target**: ₹1,000-₹2,000
- **Max Daily Loss**: ₹5,000 (5% of ₹1L capital)
- **Monthly Target**: ₹30,000-₹40,000
- **Risk:Reward**: Minimum 1:1.5

---

## ⏭️ Next Steps

### Immediate Actions (This Week)
1. ✅ Review this roadmap
2. Create `utils/data_manager.py` for historical data
3. Create `utils/indicators.py` for ATR, RSI, VWAP
4. Create `jobs/morning_learning.py` with 08:00 IST scheduler
5. Test morning learning with sample data

### Development Sequence
```
Week 1: Data + Indicators
  ↓
Week 2: Morning Learning + Strategy Framework
  ↓
Week 3: Risk Management + Order Manager
  ↓
Week 4-5: Backtesting + Paper Trading
  ↓
Week 5: Reporting + Monitoring
  ↓
Week 6: (Optional) ML Enhancement
  ↓
Week 7: Paper Trading Evaluation
  ↓
Week 8+: Gradual Live Rollout (if approved)
```

---

## 📞 Support & Resources

- **OpenAlgo Docs**: https://docs.openalgo.in
- **Groww API Docs**: https://groww.in/trade-api/docs/
- **Project Repo**: openalgo (local)
- **Tests**: Run with `pytest test/ -v`
- **Code Quality**: `pre-commit run --all-files`

---

## 📝 Development Notes

### Coding Standards
- Python 3.11+ (Currently 3.14.0)
- Type hints for all functions
- Docstrings (Google/NumPy style)
- Test coverage >80%
- Pre-commit hooks (black, isort, flake8, detect-secrets)

### Git Workflow
- `main` → Production (protected)
- `staging` → Testing & QA
- `dev/*` → Feature branches
- See `BRANCHING_GUIDE.md` for details

### Testing Strategy
- Unit tests for all modules
- Integration tests for workflows
- Paper trading for strategy validation
- Backtesting for historical performance

---

**Document Version**: 1.0
**Last Updated**: December 1, 2025
**Status**: 🟢 Foundation Complete | 🟡 Development Phase
