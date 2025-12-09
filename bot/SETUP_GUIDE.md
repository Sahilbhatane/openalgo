# OpenAlgo Trading Bot - Complete Setup & Operations Guide

## 📋 Table of Contents

1. [Running the Application](#running-the-application)
2. [Navigation & Dashboard](#navigation--dashboard)
3. [Manual Configuration Required](#manual-configuration-required)
4. [Paper Mode Training](#paper-mode-training)
5. [Potential Pitfalls & Failure Points](#potential-pitfalls--failure-points)
   - [Login & Authentication Pitfalls](#-login--authentication-pitfalls)
   - [API Pitfalls](#-api-pitfalls)
   - [System Pitfalls](#️-system-pitfalls)
   - [Data Pitfalls](#-data-pitfalls)
   - [Trading Pitfalls](#-trading-pitfalls)
6. [Suggestions for Improvement](#suggestions-for-improvement)
7. [Quick Reference](#-quick-reference)

> **Related Guides:**
> - [AUTOMATION_GUIDE.md](./AUTOMATION_GUIDE.md) - Oracle Cloud / GitHub Actions setup
> - [PITFALLS.md](./PITFALLS.md) - Extended pitfall analysis

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

> This section covers ALL known failure modes with severity, probability, detection, and automation solutions.

---

### 🔐 LOGIN & AUTHENTICATION PITFALLS

#### 1. TOTP Token Expiry
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 100% (daily occurrence) |
| **What Happens** | Login fails, all API calls rejected, bot cannot trade |
| **Detection** | API returns 401/403 error, "Invalid TOTP" message |

**Manual Fix:**
```bash
# Login via browser or mobile app daily at 7:30 AM
# Token refreshes for ~8 hours
```

**Automation (Partial - Telegram Reminder):**
```bash
# Add to crontab (7:15 AM IST = 1:45 AM UTC)
45 1 * * 1-5 curl -s "https://api.telegram.org/bot$BOT_TOKEN/sendMessage?chat_id=$CHAT_ID&text=🔔 Login to AngelOne now!"
```

**⚠️ Security Warning**: Fully automated TOTP is a security risk. If your server is compromised, attackers get full account access. Recommended: Manual daily login.

---

#### 2. Session Token Expiry (Mid-Day)
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 5-10% (occasional) |
| **What Happens** | Bot loses connection at random time, orders fail |
| **Detection** | Sudden API errors after hours of working |

**Automation (Session Health Check):**
```python
# Add health check every 30 minutes
def check_session_health():
    try:
        response = broker_api.get_profile()
        if not response.get('success'):
            send_telegram_alert("⚠️ Session may be expiring!")
    except Exception as e:
        send_telegram_alert(f"🔴 Session check failed: {e}")
```

---

#### 3. Wrong Credentials in .env
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 30% (first-time setup) |
| **What Happens** | Bot never starts, login always fails |
| **Detection** | "Invalid credentials" error on startup |

**Automation (Validation Script):**
```bash
# Run before starting bot
python -c "
import os
required = ['BROKER_API_KEY', 'BROKER_API_SECRET', 'BROKER_TOTP_SECRET', 'OPENALGO_APIKEY']
missing = [v for v in required if not os.getenv(v)]
if missing: print(f'🔴 Missing: {missing}'); exit(1)
print('✅ All credentials present')
"
```

---

### 🌐 API PITFALLS

#### 4. API Rate Limiting
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 15% (during high activity) |
| **What Happens** | Orders delayed or rejected, 429 errors |
| **Detection** | HTTP 429 status code |

**Broker Rate Limits:**
| Broker | Requests/Second | Daily Limit |
|--------|-----------------|-------------|
| AngelOne | 10 | 10,000 |
| Zerodha | 10 | Unlimited |
| Fyers | 10 | 10,000 |

**Automation (Throttling + Backoff):**
```python
import time

def rate_limited_call(func, max_per_second=8):
    """Rate limit API calls to stay under limits"""
    min_interval = 1.0 / max_per_second
    time.sleep(min_interval)
    
    for attempt in range(3):
        try:
            return func()
        except RateLimitError:
            wait = 2 ** attempt  # 1, 2, 4 seconds
            time.sleep(wait)
    raise Exception("Rate limit exceeded after retries")
```

---

#### 5. API Timeout
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 5-10% |
| **What Happens** | Order placed but no confirmation, duplicate risk |
| **Detection** | Request takes > 30 seconds |

**Automation (Timeout + Verification):**
```python
async def safe_order(order, timeout=30):
    try:
        return await asyncio.wait_for(place_order(order), timeout)
    except asyncio.TimeoutError:
        # CRITICAL: Check if order was placed before retrying!
        existing = await get_order_book()
        if order.tag in [o.tag for o in existing]:
            return "Order already placed"
        raise
```

---

#### 6. Invalid Symbol/Token
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 10% (expiry days, new listings) |
| **What Happens** | Order rejected with "Invalid instrument" |
| **Detection** | API error on order placement |

**Automation (Daily Symbol Refresh):**
```python
# Add to morning_learning.py
async def refresh_valid_symbols():
    instruments = await broker_api.get_instruments("NSE")
    valid_symbols = {i['symbol']: i['token'] for i in instruments}
    save_to_cache(valid_symbols)
    logger.info(f"Refreshed {len(valid_symbols)} symbols")
```

---

#### 7. Order Rejection (Insufficient Margin)
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 10-20% |
| **What Happens** | Order rejected, position not opened |
| **Detection** | "Insufficient margin" error |

**Automation (Pre-Order Check):**
```python
async def check_margin(order):
    required = await calculate_margin(order)
    available = await get_available_margin()
    
    if required > available * 0.9:  # 10% buffer
        send_telegram_alert(f"⚠️ Low margin! Need ₹{required:,.0f}")
        return False
    return True
```

---

### 🖥️ SYSTEM PITFALLS

#### 8. Server/Process Crash
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 2-5% (per month) |
| **What Happens** | Bot stops, open positions unmanaged |
| **Detection** | No logs, Telegram alerts stop |

**Automation (Systemd Auto-Restart):**
```ini
# /etc/systemd/system/trading-bot.service
[Service]
ExecStart=/home/ubuntu/openalgo/.venv/bin/python -m bot.main
Restart=always
RestartSec=10
```

**Health Check Cron:**
```bash
# Every 5 minutes
*/5 * * * * pgrep -f "bot.main" || systemctl restart trading-bot
```

---

#### 9. Memory Leak
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 20% (over long runs) |
| **What Happens** | Bot slows down, eventually crashes |
| **Detection** | Memory usage increases over days |

**Automation (Daily Restart):**
```bash
# Restart at 4 AM IST daily (market closed)
# Cron: 30 22 * * * systemctl restart trading-bot
```

---

#### 10. Internet Connectivity Loss
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 5-10% |
| **What Happens** | All API calls fail, positions stuck |
| **Detection** | Multiple consecutive timeouts |

**Automation (Connectivity Monitor):**
```python
import socket

def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Alert after 3 consecutive failures
```

---

### 📊 DATA PITFALLS

#### 11. Stale/Delayed Quotes
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 10-15% |
| **What Happens** | Trade at wrong price |
| **Detection** | Data timestamp > 30 seconds old |

**Automation (Freshness Validation):**
```python
def validate_quote(quote, max_age_seconds=30):
    age = (datetime.now() - quote.timestamp).total_seconds()
    if age > max_age_seconds:
        logger.warning(f"Stale quote: {age:.0f}s old")
        return False
    return True
```

---

#### 12. Missing OHLC Data (Gaps)
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 5% |
| **What Happens** | Indicators calculate wrong values |
| **Detection** | Gap in candle timestamps |

**Automation (Gap Detection + Fill):**
```python
def detect_and_fill_gaps(df):
    # Detect gaps > expected interval
    gaps = df.index.to_series().diff() > timedelta(minutes=7)
    if gaps.any():
        logger.warning(f"Found {gaps.sum()} data gaps")
        df = df.resample('5T').interpolate()
    return df
```

---

### 💹 TRADING PITFALLS

#### 13. Slippage Higher Than Expected
| Attribute | Details |
|-----------|---------|
| **Severity** | 🟡 MODERATE |
| **Probability** | 90% (always happens) |
| **What Happens** | Fill price worse than signal price |
| **Detection** | Compare signal vs fill |

**Expected Slippage:**
| Instrument | Normal | Volatile |
|------------|--------|----------|
| NIFTY F&O | 0.02-0.05% | 0.1-0.2% |
| BANKNIFTY | 0.03-0.08% | 0.15-0.3% |

**Automation (Slippage Tracking):**
```python
class SlippageTracker:
    def record(self, signal_price, fill_price, side):
        slippage = abs(fill_price - signal_price) / signal_price * 100
        if slippage > 0.2:
            logger.warning(f"High slippage: {slippage:.2f}%")
```

---

#### 14. Stop Loss Not Executed (Gap)
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 2-5% (gap days) |
| **What Happens** | Price gaps through SL, loss > expected |
| **Detection** | Exit price worse than SL |

**Automation (Multi-Layer SL):**
```python
async def place_layered_stop_loss(position, sl_price):
    # Layer 1: SL-M order (fastest)
    await place_sl_m_order(position, sl_price)
    
    # Layer 2: GTT order (survives session expiry)
    await place_gtt_order(position, sl_price * 0.995)
    
    # Layer 3: Bot-level monitoring
    monitor_price_for_exit(position, sl_price)
```

---

#### 15. Duplicate Orders
| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 CRITICAL |
| **Probability** | 5% (network issues) |
| **What Happens** | Same order placed twice |
| **Detection** | Position 2x expected |

**Automation (Idempotency Keys):**
```python
import hashlib

def generate_order_id(signal):
    unique = f"{signal.symbol}_{signal.side}_{signal.timestamp}"
    return hashlib.md5(unique.encode()).hexdigest()[:12]

def is_duplicate(order_id):
    if order_id in recent_orders:
        return True
    recent_orders[order_id] = datetime.now()
    return False
```

---

### 🤖 AUTOMATION SUMMARY

| Pitfall | Automatable? | How | Status |
|---------|--------------|-----|--------|
| TOTP Expiry | ⚠️ Partial | Telegram reminder | Add cron |
| Session Refresh | ✅ Yes | Health check | Implement |
| Rate Limiting | ✅ Yes | Throttling | ✅ Done |
| API Timeout | ✅ Yes | Retry + verify | ✅ Done |
| Symbol Validation | ✅ Yes | Daily refresh | Implement |
| Margin Check | ✅ Yes | Pre-order | ✅ Done |
| Process Crash | ✅ Yes | Systemd | Add service |
| Memory Leak | ✅ Yes | Daily restart | Add cron |
| Connectivity | ✅ Yes | Monitor | Implement |
| Stale Data | ✅ Yes | Validation | ✅ Done |
| Data Gaps | ✅ Yes | Interpolation | Implement |
| Slippage | ✅ Yes | Tracking | Implement |
| SL Gap Risk | ✅ Yes | Multi-layer SL | Implement |
| Duplicate Orders | ✅ Yes | Idempotency | ✅ Done |

---

### CRITICAL FAILURES (Legacy Reference)

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

See [AUTOMATION_GUIDE.md](./AUTOMATION_GUIDE.md) for complete Oracle Cloud setup.

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

---

## 📖 Quick Reference

### Daily Checklist

```
□ 7:30 AM - Login to AngelOne (refresh session)
□ 7:45 AM - Verify bot is running
□ 9:15 AM - Confirm market_open job triggered
□ 3:30 PM - Check daily report generated
□ Evening  - Review trades and P&L
```

### Common Commands

```bash
# Start OpenAlgo server
python app.py

# Start trading bot
python -m bot.main

# Validate environment
python -c "import os; print([v for v in ['BROKER_API_KEY','BROKER_TOTP_SECRET'] if not os.getenv(v)])"

# Check bot logs
type bot_data\logs\bot_%date%.log

# View today's report
start bot_data\reports\daily\report_%date%.html
```

### Telegram Alert Codes

| Alert | Meaning | Action |
|-------|---------|--------|
| 🟢 | Bot started | None |
| 🔵 | Trade entry | Monitor |
| 🟡 | Warning | Check logs |
| 🔴 | Error/Exit | Investigate |
| ⚠️ | Session expiring | Re-login |
| 💰 | Daily P&L | Review |

### Emergency Procedures

| Situation | Action |
|-----------|--------|
| Bot crashed | `python -m bot.main` |
| Session expired | Login to broker, restart bot |
| Wrong position | Manual close via broker app |
| Internet down | Use mobile hotspot |
| Kill switch triggered | Review reason, reset in bot_data/ |

### File Locations

| File | Purpose |
|------|---------|
| `.env` | Credentials & config |
| `bot_data/logs/` | Execution logs |
| `bot_data/reports/` | Daily reports |
| `bot_data/today_plan.json` | Morning analysis |
| `bot_data/mode_state.json` | PAPER/LIVE state |

---

*Last Updated: December 2025*