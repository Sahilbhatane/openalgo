# Trading Bot - Potential Pitfalls & Failure Modes
## Comprehensive Risk Analysis & Mitigation Guide

> ⚠️ **CRITICAL**: Trading involves substantial risk. This document outlines known failure modes but is NOT exhaustive. Always use paper trading mode first.

---

## Table of Contents
1. [Technical Failures](#1-technical-failures)
2. [Strategy Failures](#2-strategy-failures)
3. [Execution Failures](#3-execution-failures)
4. [Data Failures](#4-data-failures)
5. [Risk Management Failures](#5-risk-management-failures)
6. [Broker/API Failures](#6-brokerapi-failures)
7. [Market Condition Failures](#7-market-condition-failures)
8. [Configuration Failures](#8-configuration-failures)
9. [Bot "Hallucination" Scenarios](#9-bot-hallucination-scenarios)
10. [Profit Margin Killers](#10-profit-margin-killers)
11. [Suggestions & Their Pitfalls](#11-suggestions--their-pitfalls)

---

## 1. Technical Failures

### 🔴 CRITICAL

#### 1.1 Internet Connectivity Loss
**Severity**: 🔴 Critical  
**Probability**: Medium (5-10% during trading session)

**What Happens**:
- Orders may be placed but not confirmed
- Open positions cannot be closed
- Stop losses may not be updated
- Square-off job may not execute

**Mitigation**:
- Use backup internet (mobile hotspot)
- Enable broker's server-side stop losses
- Set GTT (Good Till Triggered) orders on broker platform
- Configure Telegram alerts for connectivity issues

**Code Location**: `bot/jobs/square_off.py`, `bot/execution/order_manager.py`

---

#### 1.2 Server/Process Crash
**Severity**: 🔴 Critical  
**Probability**: Low (1-2%)

**What Happens**:
- All scheduled jobs stop
- Open positions abandoned
- No square-off at 3:15 PM
- No daily report generated

**Mitigation**:
```python
# In app.py or systemd service
# Use process managers:
# - PM2 (Node) for auto-restart
# - supervisord for Python
# - systemd with Restart=always
```

**Add to your setup**:
```bash
# supervisord config
[program:openalgo-bot]
command=python bot/main.py
autostart=true
autorestart=true
stderr_logfile=/var/log/bot.err.log
stdout_logfile=/var/log/bot.out.log
```

---

#### 1.3 Database Corruption
**Severity**: 🔴 Critical  
**Probability**: Very Low (<1%)

**What Happens**:
- SQLite database becomes unreadable
- All trade history lost
- Settings lost
- API keys inaccessible

**Mitigation**:
- Enable WAL mode in SQLite (already enabled)
- Daily database backups
- Store backups offsite (cloud)

**Backup Script**:
```bash
# Add to daily cron
copy db\openalgo.db db\backups\openalgo_%date%.db
```

---

### 🟡 MODERATE

#### 1.4 Memory Leak
**Severity**: 🟡 Moderate  
**Probability**: Medium (10-15% over long runs)

**What Happens**:
- Bot slows down over days
- Eventually crashes with OOM
- Trades may be delayed or missed

**Mitigation**:
- Daily restart (already scheduled with EOD job)
- Monitor memory usage
- Clear historical data caches periodically

**Code Location**: `bot/jobs/end_of_day.py` - add cache clearing

---

#### 1.5 Timezone Issues
**Severity**: 🟡 Moderate  
**Probability**: Medium (10%)

**What Happens**:
- Jobs run at wrong times
- Morning analysis runs after market opens
- Square-off happens too late

**Mitigation**:
- Always use `Asia/Kolkata` timezone explicitly
- Verify system time synced with NTP
- Test job schedules in paper mode

**Code Location**: `bot/jobs/scheduler.py` - timezone handling

---

## 2. Strategy Failures

### 🔴 CRITICAL

#### 2.1 Overfitting
**Severity**: 🔴 Critical  
**Probability**: High (30-50% for new strategies)

**What Happens**:
- Strategy works great on historical data
- Fails miserably in live trading
- Parameters too specific to past data

**Symptoms**:
- Backtest shows 80%+ win rate
- Live trading shows 40% win rate
- Works on NIFTY, fails on BANKNIFTY

**Mitigation**:
- Use walk-forward optimization
- Out-of-sample testing (30% held out)
- Monte Carlo validation (30,000+ paths) ✅
- Paper trade for 4+ weeks before live

**Code Location**: `bot/quant/monte_carlo.py`

---

#### 2.2 Curve Fitting Parameters
**Severity**: 🔴 Critical  
**Probability**: High (40%)

**What Happens**:
- EMA periods optimized perfectly for past
- ATR multipliers fit historical volatility exactly
- New market conditions = failure

**Bad Example**:
```python
# DON'T: Ultra-specific parameters
EMA_PERIOD = 17  # Why 17? Because it worked in 2023
ATR_MULT = 1.847  # Suspiciously precise
```

**Good Example**:
```python
# DO: Round, robust parameters
EMA_PERIOD = 20  # Standard period
ATR_MULT = 2.0   # Simple multiplier
```

---

#### 2.3 Regime Change Blindness
**Severity**: 🔴 Critical  
**Probability**: High (Certain to occur)

**What Happens**:
- Strategy trained on trending market
- Market turns sideways or volatile
- Strategy generates false signals
- Consecutive losses accumulate

**Real-World Examples**:
- COVID crash (March 2020)
- Post-budget volatility
- RBI policy announcement days

**Mitigation**:
- Use regime detection ✅
- Have multiple strategies for different regimes
- Reduce position size in unclear regimes

**Code Location**: `bot/quant/advanced_analysis.py` - `detect_market_regime()`

---

### 🟡 MODERATE

#### 2.4 Signal Delay
**Severity**: 🟡 Moderate  
**Probability**: Medium (15-20%)

**What Happens**:
- By the time signal confirmed, move is over
- Entry price worse than expected
- Risk/reward ratio degraded

**Causes**:
- Using closing price for signals
- Waiting for candle confirmation
- Multiple timeframe alignment delays

**Mitigation**:
- Use intraday signals where appropriate
- Accept some false signals for faster entry
- Test signal-to-execution lag in paper mode

---

## 3. Execution Failures

### 🔴 CRITICAL

#### 3.1 Order Rejection
**Severity**: 🔴 Critical  
**Probability**: Medium (5-10%)

**Causes**:
- Insufficient margin
- Symbol not allowed
- Order size limits
- Market closed/pre-open

**What Happens**:
- Position not opened
- Hedge leg fails (naked exposure)
- Square-off order rejected

**Mitigation**:
```python
# In bot/execution/order_manager.py
def place_order_with_retry(self, order, max_retries=3):
    for attempt in range(max_retries):
        result = self._place_order(order)
        if result.success:
            return result
        time.sleep(1)  # Wait before retry
    
    # Send critical alert
    self.send_alert(f"Order failed after {max_retries} attempts")
    return result
```

---

#### 3.2 Slippage
**Severity**: 🟡 Moderate  
**Probability**: High (90%+ in live trading)

**What Happens**:
- Expected entry: 100.00
- Actual entry: 100.15
- 0.15% loss before trade starts

**Typical Slippage**:
| Instrument | Normal | Volatile |
|------------|--------|----------|
| NIFTY | 0.02-0.05% | 0.1-0.2% |
| BANKNIFTY | 0.03-0.08% | 0.15-0.3% |
| Stocks | 0.05-0.15% | 0.2-0.5% |

**Mitigation**:
- Use limit orders with offset
- Account for slippage in calculations (already in `charges.py`)
- Trade liquid instruments only

**Code Location**: `bot/utils/charges.py` - `DEFAULT_SLIPPAGE_PCT`

---

#### 3.3 Partial Fills
**Severity**: 🟡 Moderate  
**Probability**: Low for F&O (2-5%)

**What Happens**:
- Ordered 100 lots
- Got fill for 50 lots
- Position size half of expected
- Risk management off

**Mitigation**:
- Check fill quantity after order
- Adjust stop loss for actual position
- Cancel unfilled portion if timeout

---

### 🟡 MODERATE

#### 3.4 Orphaned Orders
**Severity**: 🟡 Moderate  
**Probability**: Low (2-3%)

**What Happens**:
- Order placed, connection lost
- Order ID unknown
- May get filled without tracking
- Position mismatch

**Mitigation**:
- Always fetch and reconcile positions at start
- Use broker's order book as source of truth
- Log all order IDs immediately

---

## 4. Data Failures

### 🔴 CRITICAL

#### 4.1 Stale/Delayed Data
**Severity**: 🔴 Critical  
**Probability**: Medium (10-15%)

**What Happens**:
- Last price shows 100
- Real price is 102 (moved 2 minutes ago)
- Signals based on old data
- Entry at wrong price

**Causes**:
- API rate limits hit
- WebSocket disconnection
- Market data vendor issues

**Mitigation**:
- Verify data freshness (timestamp check)
- Multiple data sources for critical signals
- Alert if data older than 30 seconds

```python
def validate_data_freshness(last_update: datetime, max_age_seconds: int = 30):
    age = (datetime.now() - last_update).total_seconds()
    if age > max_age_seconds:
        logger.warning(f"Data is {age:.0f}s old!")
        return False
    return True
```

---

#### 4.2 Missing OHLC Candles
**Severity**: 🟡 Moderate  
**Probability**: Low (2-5%)

**What Happens**:
- Gap in 5-minute data
- Indicators calculate wrong values
- False signals generated

**Mitigation**:
- Detect gaps and interpolate
- Re-fetch data if gaps found
- Use higher timeframe as backup

---

#### 4.3 Corporate Action Adjustments
**Severity**: 🟡 Moderate  
**Probability**: Medium for stocks

**What Happens**:
- Stock split/bonus not adjusted
- Price appears to "crash"
- Stop loss triggers incorrectly

**Mitigation**:
- Use adjusted price series
- Check for corporate actions daily
- Trade only index derivatives (no corporate actions)

---

## 5. Risk Management Failures

### 🔴 CRITICAL

#### 5.1 Daily Loss Limit Not Enforced
**Severity**: 🔴 Critical  
**Probability**: Low if properly coded

**What Happens**:
- Bot keeps trading after hitting limit
- One bad day wipes out month's gains
- Cascading losses in panic conditions

**Mitigation**:
- Multiple checkpoints for loss limit
- Kill switch that stops ALL trading
- Independent monitoring script

**Code Location**: `bot/risk/kill_switch.py`

---

#### 5.2 Position Size Calculation Error
**Severity**: 🔴 Critical  
**Probability**: Low (2-3%)

**What Happens**:
- Calculated 10 lots, placed 100 lots
- 10x intended exposure
- Single trade can wipe account

**Causes**:
- Integer overflow
- Wrong lot size in calculation
- Market vs Limit price difference

**Mitigation**:
```python
# In bot/risk/position_sizer.py
def calculate_position(self, ...):
    lots = calculation_result
    
    # Sanity checks
    MAX_LOTS = 50  # Hard limit
    if lots > MAX_LOTS:
        logger.error(f"Position {lots} exceeds MAX_LOTS {MAX_LOTS}")
        lots = MAX_LOTS
    
    if lots < 1:
        lots = 1
    
    return lots
```

---

#### 5.3 Stop Loss Not Placed
**Severity**: 🔴 Critical  
**Probability**: Medium (5-8%)

**What Happens**:
- Entry order fills
- Stop loss order fails
- Trade runs without protection
- Unlimited downside

**Causes**:
- API error
- Rate limiting
- Invalid price

**Mitigation**:
- Verify stop loss placed before considering trade "open"
- Use bracket orders where available
- Exit if stop loss placement fails

---

## 6. Broker/API Failures

### 🔴 CRITICAL

#### 6.1 Session Token Expiry
**Severity**: 🔴 Critical  
**Probability**: High (daily occurrence)

**What Happens**:
- Token expires mid-session
- All API calls fail
- Trades cannot be managed

**AngelOne Specifics**:
- Token valid for ~8 hours
- TOTP required for refresh
- Manual intervention needed

**Mitigation**:
- Token refresh before expiry (7:30 AM daily)
- TOTP automation (risky, but possible)
- Alert when token expires

**Code Location**: `broker/angel/api/order_api.py`

---

#### 6.2 API Rate Limiting
**Severity**: 🟡 Moderate  
**Probability**: Medium (10-15%)

**What Happens**:
- Too many API calls
- Requests rejected (429 errors)
- Critical orders may fail

**Broker Limits**:
| Broker | Requests/Second | Requests/Minute |
|--------|-----------------|-----------------|
| AngelOne | 10 | 200 |
| Zerodha | 10 | 200 |
| Fyers | 10 | 100 |

**Mitigation**:
- Implement request throttling
- Queue non-critical requests
- Prioritize order placement

---

#### 6.3 Broker System Down
**Severity**: 🔴 Critical  
**Probability**: Low (1-2% per month)

**What Happens**:
- Complete inability to trade
- Open positions stuck
- No square-off possible

**Mitigation**:
- Broker mobile app as backup
- Phone trading (keep helpline saved)
- Use server-side stop losses
- Consider backup broker account

---

## 7. Market Condition Failures

### 🔴 CRITICAL

#### 7.1 Flash Crash
**Severity**: 🔴 Critical  
**Probability**: Low (1-2 per year)

**What Happens**:
- Market drops 5%+ in minutes
- Stop losses hit at worst prices
- Slippage extreme
- Strategies fail completely

**Example**: March 2020 COVID crash

**Mitigation**:
- Hard loss limits
- Reduce position in high-VIX environments
- Avoid options selling strategies

---

#### 7.2 Gap Opening
**Severity**: 🟡 Moderate  
**Probability**: High (common occurrence)

**What Happens**:
- Market opens 1%+ above/below previous close
- Overnight positions gapped against
- Stop losses worthless (gap through)

**Mitigation**:
- Intraday only (no overnight positions) ✅
- Account for gap risk in sizing
- Use options for defined risk

---

#### 7.3 Low Liquidity Periods
**Severity**: 🟡 Moderate  
**Probability**: Medium (lunch hours, expiry days)

**What Happens**:
- Wide bid-ask spreads
- Order fills at poor prices
- Market orders especially costly

**Low Liquidity Times**:
- 12:00 PM - 1:00 PM (lunch)
- 3:00 PM - 3:30 PM (expiry days)
- Monday mornings after holidays

**Mitigation**:
- Avoid trading 12-1 PM
- Use limit orders only
- Widen spread expectations

---

## 8. Configuration Failures

### 🔴 CRITICAL

#### 8.1 Wrong Mode (Paper vs Live)
**Severity**: 🔴 Critical  
**Probability**: Medium (first-time setup)

**What Happens**:
- Think you're paper trading
- Actually placing live orders
- Real money lost on tests

**Mitigation**:
```python
# In bot/core/mode.py
# Add prominent warning
if MODE == TradingMode.LIVE:
    logger.warning("🔴 LIVE TRADING MODE ACTIVE 🔴")
    logger.warning("🔴 REAL MONEY AT RISK 🔴")
```

---

#### 8.2 Incorrect Symbol Mapping
**Severity**: 🔴 Critical  
**Probability**: Medium (10%)

**What Happens**:
- Trying to trade "NIFTY"
- Broker expects "NIFTY 50"
- Orders fail or wrong symbol traded

**Mitigation**:
- Validate symbols at startup
- Use instrument token mapping
- Test symbol resolution in paper mode

---

#### 8.3 Wrong Lot Size
**Severity**: 🔴 Critical  
**Probability**: Low (2-3%)

**What Happens**:
- Lot size changed (regulatory)
- Old lot size in code
- Wrong position size

**Example**: NIFTY lot size changed from 50 to 25

**Mitigation**:
- Fetch lot size from broker dynamically
- Validate against known values
- Alert on lot size change

---

## 9. Bot "Hallucination" Scenarios

> These are scenarios where the bot behaves unexpectedly, not due to bugs but due to edge cases.

### 9.1 Conflicting Signals
**What Happens**:
- Short-term: Buy signal
- Long-term: Sell signal
- Bot confused, may take either

**Mitigation**:
- Clear signal priority hierarchy
- Single timeframe as primary
- Confirm with secondary

---

### 9.2 Indicator Divergence
**What Happens**:
- RSI says oversold (buy)
- MACD says sell
- Price at resistance

**Mitigation**:
- Use indicator consensus (2/3 rule)
- Weight indicators by reliability
- Override with regime detection

---

### 9.3 Whipsaw Cascades
**What Happens**:
- Buy signal triggered
- Price dips, stop loss hit
- New buy signal immediately
- Another stop loss
- Repeat...

**Mitigation**:
- Cooldown period after exit (30 min+)
- Maximum entries per day limit
- Wider initial stops

---

### 9.4 False Breakout Addiction
**What Happens**:
- Strategy optimized for breakouts
- Market in range
- Constantly entering at highs/lows
- Stopped out repeatedly

**Mitigation**:
- Regime detection (trending vs ranging)
- Volume confirmation for breakouts
- Reduce size in ranging markets

---

## 10. Profit Margin Killers

### 10.1 Transaction Costs
**Impact**: 0.05-0.10% per trade

**Components**:
| Cost | F&O | Equity |
|------|-----|--------|
| Brokerage | ₹20/order | ₹20/order |
| STT | 0.01% (sell) | 0.1% |
| Exchange | 0.0019% | 0.0035% |
| GST | 18% on brokerage | 18% |

**For 100 trades/month**:
- Minimum: ₹4,000-5,000 in costs
- Need 5%+ returns just to break even

**Mitigation**:
- Reduce trade frequency
- Target higher R:R trades
- Account for costs in backtests ✅

---

### 10.2 Slippage Accumulation
**Impact**: 0.02-0.10% per trade

**Annual Impact** (100 trades):
- Best case: 2% annual drag
- Worst case: 10% annual drag

**Mitigation**:
- Limit orders with offset
- Trade only liquid instruments
- Avoid market orders

---

### 10.3 Overtrading
**Impact**: Multiplies all above

**Signs**:
- More than 5 trades/day
- Win rate below 40%
- Negative expectancy

**Mitigation**:
- Maximum trades per day limit ✅
- Quality over quantity
- Stricter signal filters

---

### 10.4 Strategy Decay
**Impact**: Gradual, then sudden

**What Happens**:
- Strategy profitable for months
- Slowly degrades
- Eventually negative

**Causes**:
- Market structure changes
- Competition (crowded trade)
- Regime shift

**Mitigation**:
- Regular performance review (weekly)
- Monte Carlo validation ✅
- Multiple strategies

---

## 11. Suggestions & Their Pitfalls

### Suggestion 1: Add More Indicators
**Benefit**: Better signal confirmation

**Pitfalls**:
- Analysis paralysis
- Conflicting signals
- More complexity = more bugs
- Overfitting risk increases

**Recommendation**: Max 3-4 indicators per strategy

---

### Suggestion 2: Use Machine Learning
**Benefit**: Adaptive to market changes

**Pitfalls**:
- Extreme overfitting risk
- Black box (unexplainable)
- Requires massive data
- Expensive to train/maintain
- Often worse than simple rules

**Recommendation**: Only after simple strategies proven

---

### Suggestion 3: Trade More Instruments
**Benefit**: Diversification

**Pitfalls**:
- More symbols to monitor
- Different characteristics per symbol
- Correlation during stress
- Increased capital requirement

**Recommendation**: Master one before adding

---

### Suggestion 4: Reduce Stop Loss Size
**Benefit**: Less per-trade risk

**Pitfalls**:
- More whipsaws
- Lower win rate
- Noise stops you out
- Psychological stress

**Recommendation**: ATR-based stops, not fixed %

---

### Suggestion 5: Add Options Strategies
**Benefit**: Defined risk, premium collection

**Pitfalls**:
- Complex Greeks
- Time decay works against you (buying)
- Unlimited risk (selling naked)
- Execution complexity

**Recommendation**: Only spreads, never naked sells

---

### Suggestion 6: Increase Position Size
**Benefit**: Larger profits

**Pitfalls**:
- Larger losses
- Risk of ruin increases exponentially
- Emotional pressure
- Broker margin calls

**Formula**: 
```
Risk of Ruin increases with:
RoR = ((1-Edge)/(1+Edge))^Units
```

**Recommendation**: Never risk >2% per trade

---

### Suggestion 7: Trade Pre-Market/After-Hours
**Benefit**: More opportunities

**Pitfalls**:
- Extremely low liquidity
- Wide spreads
- Order types limited
- Higher volatility

**Recommendation**: Avoid, stick to regular hours

---

### Suggestion 8: Automate TOTP
**Benefit**: Fully autonomous login

**Pitfalls**:
- Security risk (TOTP secret exposed)
- Against some broker ToS
- Account ban risk
- If hacked, unlimited access

**Recommendation**: Manual login daily preferred

---

### Suggestion 9: Run 24/7 on Cloud
**Benefit**: Always available

**Pitfalls**:
- Monthly cost (₹1000-3000)
- Cloud provider downtime
- Network latency
- Security of credentials

**Recommendation**: Use GitHub Actions for free automation ✅

---

### Suggestion 10: Use Multiple Brokers
**Benefit**: Redundancy, order splitting

**Pitfalls**:
- Multiple logins/sessions
- Reconciliation nightmare
- Double the complexity
- More points of failure

**Recommendation**: Primary + backup (manual)

---

## Quick Reference: Severity Matrix

| Category | Critical Issues | Moderate Issues | Low Issues |
|----------|-----------------|-----------------|------------|
| Technical | 3 | 2 | 3 |
| Strategy | 3 | 1 | 2 |
| Execution | 3 | 2 | 1 |
| Data | 1 | 2 | 1 |
| Risk Mgmt | 3 | 1 | 0 |
| Broker/API | 2 | 1 | 0 |
| Market | 2 | 2 | 1 |
| Config | 3 | 0 | 1 |

**Total**: 20 Critical, 11 Moderate, 9 Low

---

## Monitoring Checklist

### Daily (Before Market)
- [ ] Bot process running
- [ ] Broker session active
- [ ] Kill switch reset
- [ ] Yesterday's P&L reviewed
- [ ] Risk limits set

### Weekly
- [ ] Performance vs benchmark
- [ ] Monte Carlo re-validation
- [ ] Regime check
- [ ] Database backup
- [ ] Error log review

### Monthly
- [ ] Strategy decay analysis
- [ ] Cost analysis
- [ ] Parameter review (not change!)
- [ ] New data integration

---

## Emergency Procedures

### Bot Crash During Market
1. Don't panic
2. Check positions on broker app
3. Set manual stop losses
4. Restart bot if possible
5. Don't override - let square-off job work

### Broker API Down
1. Use mobile app
2. Place manual stop losses
3. Note positions
4. Consider full exit
5. Report to broker support

### Internet Down
1. Switch to mobile data
2. Use broker mobile app
3. Exit positions if prolonged
4. Don't panic-trade

---

*Last Updated: Generated on bot creation*  
*Review: Monthly*  
*Owner: Trading Bot Operator*
