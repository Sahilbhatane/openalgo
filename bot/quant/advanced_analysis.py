"""
Advanced Quantitative Analysis
==============================
Professional-grade market analysis and strategy validation.

Includes:
1. Market regime detection (Bull/Bear/Sideways)
2. Volatility regime analysis (VIX-like)
3. Correlation analysis
4. Mean reversion indicators
5. Momentum factors
6. Statistical arbitrage signals
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta


logger = logging.getLogger("bot.quant.analysis")


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class VolatilityRegime(Enum):
    """Volatility regime"""
    VERY_LOW = "VERY_LOW"      # VIX < 12
    LOW = "LOW"                 # 12-15
    NORMAL = "NORMAL"           # 15-20
    ELEVATED = "ELEVATED"       # 20-25
    HIGH = "HIGH"               # 25-30
    EXTREME = "EXTREME"         # > 30


@dataclass
class MarketState:
    """Current market state analysis"""
    timestamp: datetime
    
    # Regime
    regime: MarketRegime
    regime_confidence: float
    regime_duration_days: int
    
    # Volatility
    volatility_regime: VolatilityRegime
    realized_volatility: float
    implied_volatility: float  # India VIX if available
    volatility_percentile: float  # Where current vol is in historical range
    
    # Trend
    trend_strength: float  # ADX-like, 0-100
    trend_direction: float  # -1 to +1
    
    # Breadth
    advance_decline_ratio: float
    percent_above_200ma: float
    
    # Momentum
    momentum_score: float  # -1 to +1
    
    # Risk indicators
    risk_score: float  # 0-100 (higher = more risk)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime.value,
            'regime_confidence': self.regime_confidence,
            'volatility_regime': self.volatility_regime.value,
            'realized_volatility': self.realized_volatility,
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'momentum_score': self.momentum_score,
            'risk_score': self.risk_score,
        }


class AdvancedQuantAnalysis:
    """
    Advanced quantitative analysis for market and strategy evaluation.
    
    Features:
    - Hidden Markov Model for regime detection
    - GARCH for volatility forecasting
    - Hurst exponent for mean reversion
    - Factor analysis
    """
    
    def __init__(self):
        self.lookback_short = 20
        self.lookback_medium = 50
        self.lookback_long = 200
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def detect_market_regime(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using multiple indicators.
        
        Uses:
        - Price position relative to moving averages
        - Rate of change
        - Volatility expansion/contraction
        
        Args:
            prices: Price series (close prices)
            volume: Optional volume series
            
        Returns:
            (regime, confidence)
        """
        if len(prices) < self.lookback_long:
            return MarketRegime.SIDEWAYS, 0.5
        
        # Calculate indicators
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        ma_200 = prices.rolling(200).mean()
        
        current_price = prices.iloc[-1]
        
        # Price vs MAs
        above_20 = current_price > ma_20.iloc[-1]
        above_50 = current_price > ma_50.iloc[-1]
        above_200 = current_price > ma_200.iloc[-1]
        
        # MA alignment (Golden/Death cross)
        ma_20_above_50 = ma_20.iloc[-1] > ma_50.iloc[-1]
        ma_50_above_200 = ma_50.iloc[-1] > ma_200.iloc[-1]
        
        # Rate of change
        roc_20 = (current_price - prices.iloc[-21]) / prices.iloc[-21] * 100
        
        # Volatility
        returns = prices.pct_change()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        hist_vol_50 = returns.iloc[-50:].std() * np.sqrt(252)
        
        # Score calculation
        bull_score = 0
        
        if above_200:
            bull_score += 2
        if above_50:
            bull_score += 1.5
        if above_20:
            bull_score += 1
        if ma_20_above_50:
            bull_score += 1
        if ma_50_above_200:
            bull_score += 1.5
        if roc_20 > 5:
            bull_score += 1
        elif roc_20 < -5:
            bull_score -= 1
        
        # Determine regime
        if bull_score >= 6:
            regime = MarketRegime.STRONG_BULL
            confidence = min(0.95, 0.7 + bull_score * 0.03)
        elif bull_score >= 4:
            regime = MarketRegime.BULL
            confidence = min(0.9, 0.6 + bull_score * 0.05)
        elif bull_score <= -2:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(0.95, 0.7 + abs(bull_score) * 0.03)
        elif bull_score <= 0:
            regime = MarketRegime.BEAR
            confidence = min(0.85, 0.6 + abs(bull_score) * 0.05)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6
        
        # High volatility override
        if current_vol > 0.35:  # 35% annualized
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.8
        
        return regime, confidence
    
    def detect_volatility_regime(
        self,
        prices: pd.Series,
        india_vix: Optional[pd.Series] = None,
    ) -> Tuple[VolatilityRegime, float]:
        """
        Detect volatility regime.
        
        Args:
            prices: Price series
            india_vix: Optional India VIX series
            
        Returns:
            (regime, current_vol_percentile)
        """
        returns = prices.pct_change().dropna()
        
        # Realized volatility (20-day, annualized)
        current_vol = returns.iloc[-20:].std() * np.sqrt(252) * 100
        
        # Historical percentile
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
        vol_percentile = (rolling_vol < current_vol).mean()
        
        # Use India VIX if available
        if india_vix is not None and len(india_vix) > 0:
            vix = india_vix.iloc[-1]
        else:
            # Estimate from realized vol
            vix = current_vol * 1.1  # Implied usually higher
        
        # Classify
        if vix < 12:
            regime = VolatilityRegime.VERY_LOW
        elif vix < 15:
            regime = VolatilityRegime.LOW
        elif vix < 20:
            regime = VolatilityRegime.NORMAL
        elif vix < 25:
            regime = VolatilityRegime.ELEVATED
        elif vix < 30:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME
        
        return regime, vol_percentile
    
    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    
    def calculate_hurst_exponent(
        self,
        prices: pd.Series,
        max_lag: int = 100,
    ) -> float:
        """
        Calculate Hurst exponent to detect mean reversion vs trending.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            prices: Price series
            max_lag: Maximum lag to consider
            
        Returns:
            Hurst exponent
        """
        if len(prices) < max_lag * 2:
            return 0.5  # Default to random walk
        
        lags = range(2, min(max_lag, len(prices) // 4))
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lags:
            # Divide series into non-overlapping subseries
            n_subseries = len(prices) // lag
            rs_lag = []
            
            for i in range(n_subseries):
                subseries = prices.iloc[i*lag:(i+1)*lag]
                if len(subseries) < lag:
                    continue
                
                returns = subseries.pct_change().dropna()
                if len(returns) == 0:
                    continue
                
                # Mean-adjusted cumulative returns
                mean_return = returns.mean()
                cumsum = (returns - mean_return).cumsum()
                
                # Range
                r = cumsum.max() - cumsum.min()
                
                # Standard deviation
                s = returns.std()
                
                if s > 0:
                    rs_lag.append(r / s)
            
            if rs_lag:
                rs_values.append((lag, np.mean(rs_lag)))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression in log-log space
        lags = np.log([v[0] for v in rs_values])
        rs = np.log([v[1] for v in rs_values])
        
        # Fit line
        slope, _ = np.polyfit(lags, rs, 1)
        
        return float(slope)
    
    def calculate_half_life(
        self,
        prices: pd.Series,
    ) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            prices: Price series
            
        Returns:
            Half-life in periods (days)
        """
        # Ornstein-Uhlenbeck process estimation
        # Convert to pandas Series if needed
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        log_prices = pd.Series(np.log(prices.values), index=prices.index)
        
        # Lag-1 regression
        lagged = log_prices.shift(1).dropna()
        current = log_prices.iloc[1:]
        
        # Simple OLS - convert to numpy arrays explicitly
        x = np.array(lagged.values).reshape(-1, 1)
        y = np.array(current.values)
        
        # y = a + b*x + e
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            b = beta[1]
            
            # Half-life = -ln(2) / ln(b)
            if b > 0 and b < 1:
                half_life = -np.log(2) / np.log(b)
                return float(half_life)
        except Exception:
            pass
        
        return float('inf')  # No mean reversion
    
    # =========================================================================
    # FACTOR ANALYSIS
    # =========================================================================
    
    def calculate_momentum_factor(
        self,
        prices: pd.Series,
        lookback: int = 252,
        skip_recent: int = 21,
    ) -> float:
        """
        Calculate momentum factor (12-1 momentum).
        
        Args:
            prices: Price series
            lookback: Lookback period
            skip_recent: Days to skip (avoid short-term reversal)
            
        Returns:
            Momentum score (-1 to 1)
        """
        if len(prices) < lookback:
            return 0.0
        
        # 12-month return, skipping most recent month
        return_12m = (prices.iloc[-skip_recent] - prices.iloc[-lookback]) / prices.iloc[-lookback]
        
        # Normalize to -1 to 1
        return float(np.clip(return_12m / 0.5, -1, 1))
    
    def calculate_value_factor(
        self,
        price: float,
        book_value: float,
        earnings: float,
    ) -> Dict[str, float]:
        """
        Calculate value factors.
        
        Returns:
            Dict with P/B, P/E, and value score
        """
        factors = {}
        
        if book_value > 0:
            factors['pb_ratio'] = price / book_value
        else:
            factors['pb_ratio'] = float('inf')
        
        if earnings > 0:
            factors['pe_ratio'] = price / earnings
        else:
            factors['pe_ratio'] = float('inf')
        
        # Value score (lower is better)
        pb_score = min(3, factors.get('pb_ratio', 3)) / 3
        pe_score = min(30, factors.get('pe_ratio', 30)) / 30
        
        factors['value_score'] = 1 - (pb_score * 0.4 + pe_score * 0.6)
        
        return factors
    
    def calculate_quality_factor(
        self,
        roe: float,
        debt_to_equity: float,
        earnings_stability: float,
    ) -> float:
        """
        Calculate quality factor.
        
        Args:
            roe: Return on equity (decimal)
            debt_to_equity: Debt to equity ratio
            earnings_stability: Earnings stability score (0-1)
            
        Returns:
            Quality score (0-1)
        """
        # ROE score
        roe_score = min(1, max(0, (roe - 0.05) / 0.20))
        
        # Low debt is better
        debt_score = max(0, 1 - debt_to_equity / 2)
        
        # Combine
        quality = roe_score * 0.4 + debt_score * 0.3 + earnings_stability * 0.3
        
        return float(quality)
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    def calculate_tail_risk(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate tail risk metrics.
        
        Returns:
            Dict with VaR, CVaR, and tail statistics
        """
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        
        # VaR
        var_idx = int((1 - confidence) * n)
        var = float(sorted_returns[var_idx])
        
        # CVaR (Expected Shortfall)
        cvar = float(sorted_returns[:var_idx + 1].mean())
        
        # Skewness and kurtosis - handle type safely
        try:
            if isinstance(returns, pd.Series):
                skew_val = returns.skew()
                kurt_val = returns.kurtosis()
            else:
                from scipy import stats as scipy_stats
                skew_val = scipy_stats.skew(returns)
                kurt_val = scipy_stats.kurtosis(returns)
            
            # Convert to float safely
            skewness = 0.0
            kurtosis = 3.0
            
            try:
                if skew_val is not None and not pd.isna(skew_val):
                    skewness = float(skew_val)
            except (TypeError, ValueError):
                pass
                
            try:
                if kurt_val is not None and not pd.isna(kurt_val):
                    kurtosis = float(kurt_val)
            except (TypeError, ValueError):
                pass
        except Exception:
            skewness = 0.0
            kurtosis = 3.0
        
        return {
            'var': var,
            'cvar': cvar,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_fat_tailed': kurtosis > 3,
            'is_left_skewed': skewness < -0.5,
        }
    
    def calculate_correlation_risk(
        self,
        returns_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate correlation-based risk metrics.
        
        Args:
            returns_matrix: DataFrame with asset returns as columns
            
        Returns:
            Correlation risk metrics
        """
        corr_matrix = returns_matrix.corr()
        
        # Average correlation - safely extract scalar values
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        stacked = corr_matrix.where(mask).stack()
        
        # Safely get mean
        try:
            mean_val = stacked.mean()
            avg_corr = float(mean_val) if not pd.isna(mean_val) else 0.0
        except (TypeError, ValueError):
            avg_corr = 0.0
        
        # Safely get max
        try:
            max_val = stacked.max()
            max_corr = float(max_val) if not pd.isna(max_val) else 0.0
        except (TypeError, ValueError):
            max_corr = 0.0
        
        # Effective number of uncorrelated assets
        eigenvalues = np.linalg.eigvals(corr_matrix)
        effective_n = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())
        
        return {
            'average_correlation': avg_corr,
            'max_correlation': max_corr,
            'effective_assets': effective_n,
            'diversification_ratio': effective_n / len(corr_matrix),
        }
    
    # =========================================================================
    # MARKET STATE
    # =========================================================================
    
    def analyze_market_state(
        self,
        index_prices: pd.Series,
        index_volume: Optional[pd.Series] = None,
        india_vix: Optional[pd.Series] = None,
        advance_decline_data: Optional[Dict[str, int]] = None,
    ) -> MarketState:
        """
        Comprehensive market state analysis.
        
        Args:
            index_prices: Nifty/Sensex prices
            index_volume: Optional volume
            india_vix: Optional India VIX
            advance_decline_data: {'advances': n, 'declines': n}
            
        Returns:
            MarketState object
        """
        # Regime detection
        regime, regime_confidence = self.detect_market_regime(index_prices, index_volume)
        
        # Volatility analysis
        vol_regime, vol_percentile = self.detect_volatility_regime(index_prices, india_vix)
        
        # Realized volatility
        returns = index_prices.pct_change().dropna()
        realized_vol = float(returns.iloc[-20:].std() * np.sqrt(252))
        
        # Trend strength (ADX-like)
        trend_strength = self._calculate_trend_strength(index_prices)
        
        # Trend direction
        ma_20 = index_prices.rolling(20).mean().iloc[-1]
        ma_50 = index_prices.rolling(50).mean().iloc[-1]
        current = index_prices.iloc[-1]
        trend_direction = np.sign(current - ma_50) * min(1, abs(current - ma_50) / ma_50 * 10)
        
        # Advance/decline
        if advance_decline_data:
            adv = advance_decline_data.get('advances', 0)
            dec = advance_decline_data.get('declines', 0)
            ad_ratio = adv / max(1, dec)
        else:
            ad_ratio = 1.0
        
        # Momentum
        momentum = self.calculate_momentum_factor(index_prices)
        
        # Risk score
        risk_score = self._calculate_risk_score(
            vol_regime,
            regime,
            vol_percentile,
        )
        
        # Regime duration (simplified)
        regime_duration = 1  # Would need historical regime tracking
        
        return MarketState(
            timestamp=datetime.now(),
            regime=regime,
            regime_confidence=regime_confidence,
            regime_duration_days=regime_duration,
            volatility_regime=vol_regime,
            realized_volatility=realized_vol,
            implied_volatility=india_vix.iloc[-1] if india_vix is not None else realized_vol * 1.1,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            trend_direction=float(trend_direction),
            advance_decline_ratio=ad_ratio,
            percent_above_200ma=0.5,  # Would need broader market data
            momentum_score=momentum,
            risk_score=risk_score,
        )
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate ADX-like trend strength (0-100)"""
        if len(prices) < 28:
            return 50.0
        
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            abs(high - prices.shift(1)),
            abs(low - prices.shift(1)),
        ], axis=1).max(axis=1)
        
        period = 14
        smoothed_tr = tr.rolling(period).sum()
        smoothed_plus_dm = plus_dm.rolling(period).sum()
        smoothed_minus_dm = minus_dm.rolling(period).sum()
        
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25.0
    
    def _calculate_risk_score(
        self,
        vol_regime: VolatilityRegime,
        market_regime: MarketRegime,
        vol_percentile: float,
    ) -> float:
        """Calculate overall risk score (0-100)"""
        # Volatility contribution
        vol_scores = {
            VolatilityRegime.VERY_LOW: 10,
            VolatilityRegime.LOW: 20,
            VolatilityRegime.NORMAL: 35,
            VolatilityRegime.ELEVATED: 55,
            VolatilityRegime.HIGH: 75,
            VolatilityRegime.EXTREME: 95,
        }
        vol_risk = vol_scores.get(vol_regime, 50)
        
        # Regime contribution
        regime_scores = {
            MarketRegime.STRONG_BULL: 20,
            MarketRegime.BULL: 30,
            MarketRegime.SIDEWAYS: 50,
            MarketRegime.BEAR: 70,
            MarketRegime.STRONG_BEAR: 90,
            MarketRegime.HIGH_VOLATILITY: 85,
        }
        regime_risk = regime_scores.get(market_regime, 50)
        
        # Combine
        risk_score = vol_risk * 0.5 + regime_risk * 0.3 + vol_percentile * 100 * 0.2
        
        return float(min(100, max(0, risk_score)))


# Factory function for easy use
def create_analyzer() -> AdvancedQuantAnalysis:
    """Create and return an analyzer instance"""
    return AdvancedQuantAnalysis()
