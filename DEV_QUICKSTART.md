# 🚀 Quick Start Guide - Intraday Trading Bot Development

## Current Status
✅ **Foundation Complete** | 🟢 **Ready for Development** | 📊 **All Tests Passing (55/55)**

---

## 📋 Daily Development Workflow

### 1. Start Your Day
```powershell
# Navigate to project
cd c:\Users\sahil\OneDrive\Desktop\CODE\Projects\Task\openalgo

# Activate virtual environment
.venv\Scripts\activate

# Pull latest changes (if working in a team)
git pull origin dev

# Start Flask app (in one terminal)
python app.py

# Access web interface
# Open browser: http://127.0.0.1:5000
```

### 2. Before Coding
```powershell
# Create feature branch
git checkout -b dev/feature/your-feature-name

# Install pre-commit hooks (one-time)
pre-commit install

# Run tests to ensure baseline
python -m pytest test/test_charges.py -v
```

### 3. During Development
```powershell
# Run specific tests as you code
python -m pytest test/test_your_module.py -v -x

# Check code quality
black utils/your_module.py
isort utils/your_module.py
flake8 utils/your_module.py

# Or run all pre-commit checks
pre-commit run --all-files
```

### 4. Before Committing
```powershell
# Run all tests
python -m pytest test/ -v

# Stage changes
git add .

# Commit (pre-commit hooks run automatically)
git commit -m "[Feature] Add data manager module"

# Push to remote
git push origin dev/feature/your-feature-name
```

---

## 🎯 Week 1 Tasks - Data Infrastructure

### Task 1: Data Manager (`utils/data_manager.py`)
**Estimated Time**: 2-3 hours

```python
# Quick template to get started
"""
Data Manager - Historical data fetching and caching
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from broker.groww_adapter import GrowwAdapter

class DataManager:
    def __init__(self):
        self.adapter = GrowwAdapter()
        self.cache_dir = "data/historical"

    def get_historical_data(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1minute",
        days_back: int = 30
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        pass

    def get_realtime_candle(self, symbol: str) -> dict:
        """Get latest candle data"""
        pass
```

**Test File**: `test/test_data_manager.py`
```python
def test_historical_data_fetch():
    dm = DataManager()
    data = dm.get_historical_data("SBIN", "NSE", "1minute", 1)
    assert data is not None
    assert len(data) > 0
```

**Run**:
```powershell
python -m pytest test/test_data_manager.py -v
```

---

### Task 2: Technical Indicators (`utils/indicators.py`)
**Estimated Time**: 3-4 hours

```python
# Quick template
"""
Technical Indicators - ATR, RSI, VWAP, etc.
"""

import pandas as pd
import numpy as np

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
```

**Test File**: `test/test_indicators.py`
```python
import pandas as pd

def test_atr_calculation():
    # Sample data
    df = pd.DataFrame({
        'high': [100, 102, 105, 103, 106],
        'low': [98, 100, 102, 101, 104],
        'close': [99, 101, 104, 102, 105]
    })

    atr = calculate_atr(df, period=3)
    assert atr is not None
    assert len(atr) == len(df)
```

---

### Task 3: Sentiment Analysis (`utils/sentiment.py`)
**Estimated Time**: 2-3 hours

```python
# Quick template
"""
Sentiment Analysis - News & market sentiment
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict

class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = ['bullish', 'surge', 'gain', 'profit', 'growth']
        self.negative_words = ['bearish', 'fall', 'loss', 'decline', 'crash']

    def fetch_news(self, symbol: str, hours_back: int = 48) -> List[Dict]:
        """Fetch news headlines for symbol"""
        # Placeholder - integrate with news API
        return []

    def calculate_sentiment(self, headlines: List[str]) -> float:
        """
        Calculate sentiment score (-1 to +1)
        -1 = Very Bearish
         0 = Neutral
        +1 = Very Bullish
        """
        if not headlines:
            return 0.0

        scores = []
        for headline in headlines:
            headline_lower = headline.lower()
            positive = sum(1 for word in self.positive_words if word in headline_lower)
            negative = sum(1 for word in self.negative_words if word in headline_lower)

            if positive + negative > 0:
                scores.append((positive - negative) / (positive + negative))

        return sum(scores) / len(scores) if scores else 0.0
```

---

## 🌅 Week 2 Priority - Morning Learning Job

### Task: Morning Learning Service (`jobs/morning_learning.py`)
**Estimated Time**: 4-5 hours

```python
# Quick template
"""
Morning Learning Job - Runs at 08:00 IST daily
Analyzes symbols and creates today's trading plan
"""

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import json
from utils.data_manager import DataManager
from utils.indicators import calculate_atr, calculate_rsi, calculate_vwap
from utils.sentiment import SentimentAnalyzer
from utils.charges import total_trade_cost

class MorningLearning:
    def __init__(self, watchlist: List[str]):
        self.watchlist = watchlist
        self.data_manager = DataManager()
        self.sentiment = SentimentAnalyzer()

    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze single symbol"""
        # 1. Fetch data
        df_1d = self.data_manager.get_historical_data(symbol, interval="1day", days_back=30)
        df_5m = self.data_manager.get_historical_data(symbol, interval="5minute", days_back=5)

        # 2. Calculate indicators
        atr = calculate_atr(df_1d).iloc[-1]
        rsi = calculate_rsi(df_1d).iloc[-1]
        vwap = calculate_vwap(df_5m).iloc[-1]

        # 3. Get sentiment
        news = self.sentiment.fetch_news(symbol)
        sentiment_score = self.sentiment.calculate_sentiment([n['headline'] for n in news])

        # 4. Calculate score
        liquidity_score = df_1d['volume'].mean() / 1000000  # Normalize
        momentum_score = (df_1d['close'].iloc[-1] - df_1d['close'].iloc[-20]) / df_1d['close'].iloc[-20]
        volatility_score = atr / df_1d['close'].iloc[-1]

        total_score = liquidity_score * momentum_score * volatility_score * (1 + sentiment_score)

        return {
            "symbol": symbol,
            "score": total_score,
            "atr": atr,
            "rsi": rsi,
            "vwap": vwap,
            "sentiment": sentiment_score,
            "entry_zone": df_1d['close'].iloc[-1],
            "stop_loss": df_1d['close'].iloc[-1] - (2 * atr),
            "target": df_1d['close'].iloc[-1] + (3 * atr)
        }

    def run_analysis(self):
        """Main job function - runs at 08:00"""
        results = []

        for symbol in self.watchlist:
            try:
                analysis = self.analyze_symbol(symbol)
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Save to file
        with open('data/today_plan.json', 'w') as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "symbols": results[:10]  # Top 10 symbols
            }, f, indent=2)

        print(f"Morning analysis complete: {len(results)} symbols analyzed")
        return results

# Scheduler setup
def start_morning_job():
    scheduler = BackgroundScheduler()
    ml = MorningLearning(["SBIN", "INFY", "TCS", "RELIANCE"])

    # Run at 08:00 IST every day
    scheduler.add_job(
        ml.run_analysis,
        'cron',
        hour=8,
        minute=0,
        timezone='Asia/Kolkata'
    )

    scheduler.start()
```

---

## 🧪 Testing Best Practices

### 1. Run Tests Frequently
```powershell
# Run all tests
python -m pytest test/ -v

# Run specific test file
python -m pytest test/test_charges.py -v

# Run tests with coverage
python -m pytest test/ --cov=utils --cov=strategies -v

# Stop on first failure
python -m pytest test/ -x
```

### 2. Write Tests First (TDD)
```python
# Write test first
def test_calculate_position_size():
    capital = 100000
    risk_percent = 2.0
    entry = 500
    stop_loss = 480

    qty = calculate_position_size(capital, risk_percent, entry, stop_loss)

    # Should risk max ₹2000 (2% of ₹100k)
    risk_amount = (entry - stop_loss) * qty
    assert risk_amount <= 2000

# Then implement function
def calculate_position_size(capital, risk_percent, entry, stop_loss):
    risk_amount = capital * (risk_percent / 100)
    risk_per_share = entry - stop_loss
    return int(risk_amount / risk_per_share)
```

---

## 📊 Quick Testing Commands

```powershell
# Test charges module (should show 55 passing)
python -m pytest test/test_charges.py -v

# Run example
python utils/charges.py

# Test broker adapter
python broker/groww_adapter.py

# Check for Python errors without running
python -m py_compile utils/your_module.py
```

---

## 🎯 Development Milestones

### Week 1: Data Layer ✅ (Target: Complete by Dec 7)
- [ ] `utils/data_manager.py` + tests
- [ ] `utils/indicators.py` + tests
- [ ] `utils/sentiment.py` + tests
- [ ] Integration test with real data

### Week 2: Morning Job ✅ (Target: Complete by Dec 14)
- [ ] `jobs/morning_learning.py`
- [ ] `config/watchlist.json`
- [ ] Test with sample symbols
- [ ] Verify `today_plan.json` output

### Week 3: Strategy Engine ✅ (Target: Complete by Dec 21)
- [ ] `strategies/base_strategy.py`
- [ ] `strategies/volatility_breakout.py`
- [ ] Paper trading integration
- [ ] Backtest on historical data

### Week 4-5: Risk & Reporting ✅ (Target: Complete by Jan 4)
- [ ] `risk/risk_manager.py`
- [ ] `risk/order_manager.py`
- [ ] `risk/kill_switch.py`
- [ ] `tools/report_generator.py`

---

## 🔥 Pro Tips

### 1. Use Jupyter for Exploration
```powershell
# Install if needed
pip install jupyter

# Start notebook
jupyter notebook

# Test indicators visually
import pandas as pd
from utils.indicators import calculate_atr, calculate_rsi
# ... plot charts
```

### 2. Log Everything
```python
from utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Starting morning analysis")
logger.error(f"Failed to fetch data for {symbol}: {e}")
```

### 3. Use Type Hints
```python
from typing import List, Dict, Optional
import pandas as pd

def get_data(
    symbol: str,
    interval: str = "1minute",
    days_back: int = 30
) -> pd.DataFrame:
    """Fetch historical data"""
    pass
```

---

## 📞 Need Help?

### Documentation
- **OpenAlgo Docs**: https://docs.openalgo.in
- **API Reference**: https://docs.openalgo.in/api-documentation
- **Groww API**: https://groww.in/trade-api/docs/

### Project Files
- `SETUP_COMPLETE.md` - What's been done
- `INTRADAY_BOT_ROADMAP.md` - Complete project plan
- `BRANCHING_GUIDE.md` - Git workflow
- This file - Daily development guide

### Quick Commands Reference
```powershell
# Activate venv
.venv\Scripts\activate

# Run app
python app.py

# Run tests
python -m pytest test/test_charges.py -v

# Code quality
pre-commit run --all-files

# Git workflow
git checkout -b dev/feature/name
git add .
git commit -m "message"
git push origin dev/feature/name
```

---

## 🎉 You're Ready!

Everything is set up and tested. Now just follow the roadmap week by week, write tests, and build incrementally.

**Remember**:
- ✅ Start with paper trading (PAPER_MODE=true)
- ✅ Test everything thoroughly
- ✅ Run pre-commit hooks before committing
- ✅ Review code before pushing
- ✅ 3-4 week paper trading evaluation required

**Let's build something great! 🚀📈**
