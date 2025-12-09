# OpenAlgo Trading Bot - Complete Setup & Operations Guide

## 📋 Table of Contents

1. [Running the Application](#running-the-application)
2. [Navigation & Dashboard](#navigation--dashboard)
3. [Manual Configuration Required](#manual-configuration-required)
4. [Paper Mode Training](#paper-mode-training)
5. [Potential Pitfalls & Failure Points](#potential-pitfalls--failure-points)
6. [Suggestions for Improvement](#suggestions-for-improvement)

---

## 🚀 Running the Application

### Step 1: Prerequisites

```bash
# Navigate to project directory
cd c:\Users\sahil\OneDrive\Desktop\CODE\Projects\Task\openalgo

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if not done)
pip install -r requirements.txt
pip install apscheduler pandas numpy ta
```

### Step 2: Configure Environment

Edit `.env` file with your credentials:

```env
# ===== REQUIRED: Broker Credentials =====
BROKER_API_KEY=91GRmbnA
BROKER_API_SECRET=f43117d1-c5a9-4a53-a36b-83179ba24ca6
BROKER_TOTP_SECRET=6WRI6B32YDZM3TKQTSYFCIL4AM

# ===== REQUIRED: OpenAlgo API Key =====
# Generate this from OpenAlgo dashboard after login
OPENALGO_APIKEY=your-generated-api-key

# ===== TRADING SETTINGS =====
TRADING_CAPITAL=100000
TRADING_MODE=PAPER
MAX_RISK_PER_TRADE=0.01
MAX_DAILY_DRAWDOWN=0.05
```

### Step 3: Start OpenAlgo Server

```bash
# Terminal 1: Start main application
python app.py
```

Server starts at: http://127.0.0.1:5000

### Step 4: First-Time Setup via Browser

1. Open http://127.0.0.1:5000
2. **Login**: Default admin credentials (set during first run)
3. **Broker Login**: Navigate to Broker → Login → Enter credentials
4. **Generate API Key**: Settings → API Keys → Generate New Key
5. **Copy API Key** to `.env` as `OPENALGO_APIKEY`

### Step 5: Start Trading Bot

```bash
# Terminal 2: Start bot (paper mode)
python -m bot.main
```

---

## 🧭 Navigation & Dashboard

### OpenAlgo Web Interface

| URL Path | Purpose |
|----------|---------|
| `/` | Dashboard - Overview of positions, orders |
| `/login` | User authentication |
| `/broker` | Broker connection status |
| `/orders` | Order book and history |
| `/positions` | Current positions |
| `/settings` | API keys, preferences |
| `/analyzer` | Strategy analyzer |
| `/logs` | Application logs |

### Bot Data Locations

```
bot_data/
├── logs/                    # Bot execution logs
│   └── bot_YYYY-MM-DD.log
├── reports/
│   ├── daily/               # Daily trading reports
│   │   └── report_YYYY-MM-DD.html
│   └── equity_curve.json    # Equity tracking
├── today_plan.json          # Morning learning output
└── mode_state.json          # PAPER/LIVE mode state
```

### Checking Bot Status

1. **Logs**: `bot_data/logs/bot_YYYY-MM-DD.log`
2. **Console**: Watch the terminal running `python -m bot.main`
3. **Reports**: Open `bot_data/reports/daily/report_YYYY-MM-DD.html` in browser
4. **Equity Curve**: `bot_data/reports/equity_curve.json`

---

## ⚙️ Manual Configuration Required

### File: `.env` (MUST EDIT)

```env
# Generate fresh API key from AngelOne SmartAPI dashboard
BROKER_API_KEY=your-key
BROKER_API_SECRET=your-secret
BROKER_TOTP_SECRET=your-totp-secret

# Set your trading capital
TRADING_CAPITAL=100000

# Keep as PAPER until ready
TRADING_MODE=PAPER
```

### File: `bot/core/constants.py` (OPTIONAL TUNING)

```python
# Line ~15: Adjust capital
DEFAULT_CAPITAL: Final[float] = 100_000.0  # Change to your capital

# Line ~62-65: Position limits
MAX_POSITIONS: int = 3        # Max concurrent trades
MAX_POSITION_SIZE_PCT: float = 0.30  # Max 30% in one stock

# Line ~75-78: Risk limits  
MAX_DAILY_LOSS: float = 0.05  # 5% daily stop
MAX_SINGLE_TRADE_LOSS: float = 0.01  # 1% per trade
```

### File: `bot/strategies/volatility_breakout.py` (STRATEGY TUNING)

```python
# Line ~30-40: Strategy parameters
OPENING_RANGE_MINUTES: int = 15    # First N minutes for range
ATR_PERIOD: int = 14               # ATR lookback
ATR_MULTIPLIER_SL: float = 1.5     # Stop loss = 1.5 × ATR
RISK_REWARD_RATIO: float = 1.5     # Target = 1.5 × risk
```

### File: `bot/core/config.py` (SYMBOL WATCHLIST)

You need to add your watchlist symbols. Create a file `bot/data/watchlist.py`:

```python
# Symbols to trade (NSE)
WATCHLIST = [
    "RELIANCE",
    "TCS", 
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "BHARTIARTL",
    "HINDUNILVR",
    "ITC",
    "KOTAKBANK",
]
```

---

## 📊 Paper Mode Training

### Duration Recommendations

| Phase | Duration | Purpose |
|-------|----------|---------|
| **Phase 1** | 1 week | Bug detection, logging verification |
| **Phase 2** | 2 weeks | Strategy validation, metric collection |
| **Phase 3** | 1 week | Stress testing, edge cases |
| **Total** | **4 weeks minimum** | Before considering live |

### Daily Training Schedule

The bot runs automatically during market hours:

| Time | Activity |
|------|----------|
| 8:00 AM | Morning learning runs automatically |
| 9:15 AM | Market open - bot starts watching |
| 9:20-3:10 PM | Active trading window |
| 3:10 PM | Square off all positions |
| 3:35 PM | Daily report generated |

**You don't need to manually run anything daily** - just ensure:
1. OpenAlgo server is running
2. Bot process is running
3. Internet connection stable

### Enabling Paper Mode

Paper mode is **enabled by default**. To verify:

```bash
# Check mode state
type bot_data\mode_state.json
```

Should show:
```json
{"mode": "PAPER", ...}
```

To explicitly ensure paper mode:
```env
# In .env file
TRADING_MODE=PAPER
```

### Monitoring Training Progress

```bash
# Daily: Check equity curve
type bot_data\reports\equity_curve.json

# Weekly: Review all reports
dir bot_data\reports\daily\*.html
```

### Graduation Criteria (Paper → Live)

Before switching to LIVE, ensure:
- [ ] 20+ trading days completed
- [ ] Win rate > 40%
- [ ] Profit factor > 1.2
- [ ] Max drawdown < 10%
- [ ] No system crashes
- [ ] All jobs running correctly

---

## ⚠️ Potential Pitfalls & Failure Points

### CRITICAL FAILURES

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **TOTP Expiry** | Login fails daily | Regenerate TOTP secret monthly |
| **API Rate Limits** | Orders rejected | Implement exponential backoff (done) |
| **Network Timeout** | Missed trades | Use retry logic, check connectivity |
| **Session Expiry** | Mid-day disconnect | Auto-relogin (needs implementation) |
| **Kill Switch False Trigger** | Premature exit | Tune drawdown threshold |

### DATA ISSUES

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Stale Quotes** | Wrong entry/exit | Validate data freshness |
| **Missing OHLC** | Strategy fails | Handle gaps gracefully |
| **Split/Bonus** | Wrong signals | Use adjusted prices |
| **Holiday Calendar** | Runs on holidays | Check NSE calendar |

### STRATEGY FAILURES

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Overfitting** | Poor live performance | Monte Carlo validation |
| **Regime Change** | Strategy stops working | Multi-strategy approach |
| **Flash Crash** | Massive loss | Hard stop at 5% daily |
| **Low Liquidity** | Slippage | Filter by volume |
| **Gap Opens** | Stop loss skipped | Use SL-M orders |

### BOT "HALLUCINATIONS"

| Issue | Cause | Fix |
|-------|-------|-----|
| **Duplicate Orders** | No idempotency | Order ID tracking (done) |
| **Wrong Quantity** | Rounding errors | Floor all quantities |
| **Stale Signals** | Old data | Timestamp validation |
| **Phantom Positions** | State mismatch | Sync with broker |

---

## 💡 Suggestions for Improvement

### HIGH PRIORITY

| Suggestion | Benefit | Pitfall |
|------------|---------|---------|
| **Auto-relogin** | Continuous operation | May miss reconnect window |
| **Multi-broker** | Redundancy | Code complexity |
| **Cloud deployment** | 24/7 uptime | Cost, latency |
| **Real-time P&L** | Better monitoring | API overhead |

### MEDIUM PRIORITY

| Suggestion | Benefit | Pitfall |
|------------|---------|---------|
| **ML Signal Filter** | Better entries | Overfitting risk |
| **Dynamic Position Sizing** | Optimal capital use | Complex math |
| **Sector Rotation** | Diversification | More data needed |
| **Options Hedging** | Risk reduction | Cost, complexity |

### LOW PRIORITY (ADVANCED)

| Suggestion | Benefit | Pitfall |
|------------|---------|---------|
| **HFT Optimization** | Speed edge | Latency sensitive |
| **Alternative Data** | Alpha generation | Expensive, noisy |
| **Reinforcement Learning** | Adaptive strategy | Training time, instability |

---

## 💰 Cost Analysis

### Free Tier Options

| Service | Free Limit | Sufficient? |
|---------|------------|-------------|
| **GitHub Actions** | 2000 min/month | ✅ Yes (60 min/day) |
| **AngelOne API** | Unlimited calls | ✅ Yes |
| **Local PC** | Always on | ⚠️ Reliability |

### Minimal Cost Options

| Service | Cost | Benefit |
|---------|------|---------|
| **AWS t3.micro** | ~$8/month | 24/7 reliability |
| **DigitalOcean** | $4/month | Simple setup |
| **Railway.app** | $5/month | GitHub integration |
| **Oracle Cloud** | FREE tier | 24/7, ARM instance |

### Recommended: Oracle Cloud Free Tier

Oracle offers **always-free** ARM instances:
- 4 OCPU, 24GB RAM
- Sufficient for bot + OpenAlgo
- Zero cost

---

## 🔄 Switching to LIVE Mode

**Only after 4+ weeks of profitable paper trading:**

1. Verify graduation criteria met
2. Edit `.env`:
   ```env
   TRADING_MODE=LIVE
   TRADING_CAPITAL=10000  # Start with 10% of intended capital
   ```
3. Restart bot
4. Monitor closely for first week
5. Gradually increase capital

**NEVER skip paper trading phase!**
