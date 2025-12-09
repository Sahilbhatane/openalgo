"""
Monte Carlo Simulation for Strategy Validation
===============================================
Run 30,000+ simulations to validate strategy robustness.

Features:
1. Trade sequence randomization (bootstrap)
2. Return distribution analysis
3. Drawdown probability estimation
4. Risk of ruin calculation
5. Confidence intervals for performance metrics
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


logger = logging.getLogger("bot.quant.monte_carlo")


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 30000      # Number of paths to simulate
    n_periods: int = 252            # Trading days to simulate (1 year)
    confidence_level: float = 0.95  # Confidence interval
    initial_capital: float = 100000.0
    risk_free_rate: float = 0.06    # 6% annual risk-free rate (India)
    parallel_workers: int = -1      # -1 = use all CPUs


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    # Basic statistics
    n_simulations: int = 0
    n_periods: int = 0
    
    # Terminal wealth distribution
    mean_terminal_wealth: float = 0.0
    median_terminal_wealth: float = 0.0
    std_terminal_wealth: float = 0.0
    
    # Returns
    mean_return: float = 0.0
    median_return: float = 0.0
    
    # Percentiles
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    
    # Risk metrics
    probability_of_loss: float = 0.0
    probability_of_ruin: float = 0.0  # Equity drops below 50%
    max_drawdown_mean: float = 0.0
    max_drawdown_95th: float = 0.0
    
    # Value at Risk
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    
    # Sharpe ratio distribution
    sharpe_mean: float = 0.0
    sharpe_5th: float = 0.0
    sharpe_95th: float = 0.0
    
    # Confidence intervals
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    # Kelly criterion estimate
    optimal_kelly: float = 0.0
    
    # Raw data (for further analysis)
    terminal_values: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)"""
        return {
            'n_simulations': self.n_simulations,
            'n_periods': self.n_periods,
            'mean_terminal_wealth': self.mean_terminal_wealth,
            'median_terminal_wealth': self.median_terminal_wealth,
            'std_terminal_wealth': self.std_terminal_wealth,
            'mean_return': self.mean_return,
            'median_return': self.median_return,
            'percentile_5': self.percentile_5,
            'percentile_25': self.percentile_25,
            'percentile_75': self.percentile_75,
            'percentile_95': self.percentile_95,
            'probability_of_loss': self.probability_of_loss,
            'probability_of_ruin': self.probability_of_ruin,
            'max_drawdown_mean': self.max_drawdown_mean,
            'max_drawdown_95th': self.max_drawdown_95th,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'sharpe_mean': self.sharpe_mean,
            'sharpe_5th': self.sharpe_5th,
            'sharpe_95th': self.sharpe_95th,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'optimal_kelly': self.optimal_kelly,
        }
    
    def summary(self) -> str:
        """Generate summary report"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║            MONTE CARLO SIMULATION RESULTS                    ║
╠══════════════════════════════════════════════════════════════╣
║  Simulations: {self.n_simulations:>10,}                                   ║
║  Periods:     {self.n_periods:>10} days                               ║
╠══════════════════════════════════════════════════════════════╣
║  TERMINAL WEALTH                                             ║
║    Mean:      ₹{self.mean_terminal_wealth:>12,.2f}                         ║
║    Median:    ₹{self.median_terminal_wealth:>12,.2f}                         ║
║    Std Dev:   ₹{self.std_terminal_wealth:>12,.2f}                         ║
╠══════════════════════════════════════════════════════════════╣
║  RETURN DISTRIBUTION                                         ║
║    Mean:      {self.mean_return*100:>10.2f}%                              ║
║    5th pctl:  {self.percentile_5*100:>10.2f}%                              ║
║    95th pctl: {self.percentile_95*100:>10.2f}%                              ║
╠══════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                ║
║    Prob Loss:     {self.probability_of_loss*100:>8.2f}%                          ║
║    Prob Ruin:     {self.probability_of_ruin*100:>8.2f}%                          ║
║    VaR (95%):     ₹{abs(self.var_95):>10,.2f}                          ║
║    CVaR (95%):    ₹{abs(self.cvar_95):>10,.2f}                          ║
║    Max DD (mean): {self.max_drawdown_mean*100:>8.2f}%                          ║
║    Max DD (95th): {self.max_drawdown_95th*100:>8.2f}%                          ║
╠══════════════════════════════════════════════════════════════╣
║  SHARPE RATIO                                                ║
║    Mean:      {self.sharpe_mean:>10.3f}                                  ║
║    5th pctl:  {self.sharpe_5th:>10.3f}                                  ║
║    95th pctl: {self.sharpe_95th:>10.3f}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  KELLY CRITERION                                             ║
║    Optimal f: {self.optimal_kelly*100:>10.2f}%                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategy validation.
    
    Uses bootstrap resampling of historical trade returns to
    generate thousands of possible equity curves.
    
    Usage:
        simulator = MonteCarloSimulator(config)
        result = simulator.run(trade_returns)
        print(result.summary())
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or MonteCarloConfig()
        
        # Determine number of workers
        if self.config.parallel_workers < 0:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = max(1, self.config.parallel_workers)
    
    def run(
        self,
        trade_returns: List[float],
        trade_probabilities: Optional[List[float]] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            trade_returns: List of individual trade returns (as decimals)
                Example: [0.02, -0.01, 0.03, -0.005, ...]
            trade_probabilities: Optional weights for sampling
            
        Returns:
            MonteCarloResult with all statistics
        """
        if len(trade_returns) < 10:
            raise ValueError("Need at least 10 trades for meaningful simulation")
        
        logger.info(f"Starting Monte Carlo simulation: {self.config.n_simulations:,} paths")
        start_time = datetime.now()
        
        returns = np.array(trade_returns, dtype=np.float64)
        n_sims = self.config.n_simulations
        n_periods = self.config.n_periods
        initial = self.config.initial_capital
        
        # Estimate trades per day from average
        avg_trades_per_day = max(1, len(returns) / 252)  # Assume 1 year of data
        trades_per_sim = int(n_periods * avg_trades_per_day)
        
        # Run simulations in parallel
        terminal_values = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)
        
        # Batch processing for efficiency
        batch_size = 1000
        n_batches = (n_sims + batch_size - 1) // batch_size
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_sims)
            actual_batch_size = end_idx - start_idx
            
            # Bootstrap sample trade returns
            sampled_returns = np.random.choice(
                returns,
                size=(actual_batch_size, trades_per_sim),
                replace=True,
            )
            
            # Calculate equity curves
            cumulative_returns = np.cumprod(1 + sampled_returns, axis=1)
            equity_curves = initial * cumulative_returns
            
            # Terminal values
            terminal_values[start_idx:end_idx] = equity_curves[:, -1]
            
            # Max drawdowns
            for i in range(actual_batch_size):
                max_drawdowns[start_idx + i] = self._calculate_max_drawdown(
                    equity_curves[i]
                )
            
            # Sharpe ratios
            daily_returns = np.diff(equity_curves, axis=1) / equity_curves[:, :-1]
            mean_daily = np.mean(daily_returns, axis=1)
            std_daily = np.std(daily_returns, axis=1)
            
            # Avoid division by zero
            valid_std = std_daily > 0
            sharpe_ratios[start_idx:end_idx][valid_std] = (
                mean_daily[valid_std] / std_daily[valid_std] * np.sqrt(252)
            )
            
            if (batch + 1) % 10 == 0:
                logger.debug(f"Completed batch {batch + 1}/{n_batches}")
        
        # Calculate statistics
        result = self._calculate_statistics(
            terminal_values,
            max_drawdowns,
            sharpe_ratios,
            returns,
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Monte Carlo complete: {elapsed:.2f}s")
        
        return result
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return float(np.max(drawdown))
    
    def _calculate_statistics(
        self,
        terminal_values: np.ndarray,
        max_drawdowns: np.ndarray,
        sharpe_ratios: np.ndarray,
        original_returns: np.ndarray,
    ) -> MonteCarloResult:
        """Calculate all statistics from simulation results"""
        initial = self.config.initial_capital
        
        # Returns
        returns = (terminal_values - initial) / initial
        
        # Basic statistics
        result = MonteCarloResult(
            n_simulations=self.config.n_simulations,
            n_periods=self.config.n_periods,
            mean_terminal_wealth=float(np.mean(terminal_values)),
            median_terminal_wealth=float(np.median(terminal_values)),
            std_terminal_wealth=float(np.std(terminal_values)),
            mean_return=float(np.mean(returns)),
            median_return=float(np.median(returns)),
        )
        
        # Percentiles
        result.percentile_5 = float(np.percentile(returns, 5))
        result.percentile_25 = float(np.percentile(returns, 25))
        result.percentile_75 = float(np.percentile(returns, 75))
        result.percentile_95 = float(np.percentile(returns, 95))
        
        # Risk metrics
        result.probability_of_loss = float(np.mean(returns < 0))
        result.probability_of_ruin = float(np.mean(terminal_values < initial * 0.5))
        result.max_drawdown_mean = float(np.mean(max_drawdowns))
        result.max_drawdown_95th = float(np.percentile(max_drawdowns, 95))
        
        # Value at Risk
        result.var_95 = float(np.percentile(returns, 5) * initial)
        result.var_99 = float(np.percentile(returns, 1) * initial)
        
        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            result.cvar_95 = float(np.mean(tail_returns) * initial)
        
        # Sharpe ratio distribution
        valid_sharpe = sharpe_ratios[np.isfinite(sharpe_ratios)]
        if len(valid_sharpe) > 0:
            result.sharpe_mean = float(np.mean(valid_sharpe))
            result.sharpe_5th = float(np.percentile(valid_sharpe, 5))
            result.sharpe_95th = float(np.percentile(valid_sharpe, 95))
        
        # Confidence interval
        std_error = np.std(returns) / np.sqrt(len(returns))
        z_score = 1.96  # 95% confidence
        result.ci_lower = float(np.mean(returns) - z_score * std_error)
        result.ci_upper = float(np.mean(returns) + z_score * std_error)
        
        # Kelly criterion
        win_rate = np.mean(original_returns > 0)
        if win_rate > 0 and win_rate < 1:
            avg_win = np.mean(original_returns[original_returns > 0])
            avg_loss = abs(np.mean(original_returns[original_returns < 0]))
            if avg_loss > 0:
                result.optimal_kelly = float(
                    win_rate - (1 - win_rate) / (avg_win / avg_loss)
                )
                result.optimal_kelly = max(0, min(1, result.optimal_kelly))
        
        # Store raw data for plotting
        result.terminal_values = terminal_values.tolist()
        result.max_drawdowns = max_drawdowns.tolist()
        
        return result
    
    def run_with_regime(
        self,
        bull_returns: List[float],
        bear_returns: List[float],
        sideways_returns: List[float],
        regime_probabilities: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> MonteCarloResult:
        """
        Run simulation with market regime consideration.
        
        Args:
            bull_returns: Returns during bull market
            bear_returns: Returns during bear market
            sideways_returns: Returns during sideways market
            regime_probabilities: (P(bull), P(bear), P(sideways))
            
        Returns:
            MonteCarloResult
        """
        # Combine returns with regime-based sampling
        all_returns = []
        
        for _ in range(self.config.n_simulations):
            regime = np.random.choice(
                ['bull', 'bear', 'sideways'],
                p=regime_probabilities,
            )
            
            if regime == 'bull' and bull_returns:
                all_returns.extend(np.random.choice(
                    bull_returns,
                    size=self.config.n_periods // 3,
                ))
            elif regime == 'bear' and bear_returns:
                all_returns.extend(np.random.choice(
                    bear_returns,
                    size=self.config.n_periods // 3,
                ))
            else:
                all_returns.extend(np.random.choice(
                    sideways_returns,
                    size=self.config.n_periods // 3,
                ))
        
        return self.run(all_returns)
    
    def save_results(
        self,
        result: MonteCarloResult,
        filepath: Path,
    ) -> None:
        """Save results to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def run_strategy_validation(
    trade_returns: List[float],
    n_simulations: int = 30000,
) -> MonteCarloResult:
    """
    Convenience function to validate a strategy.
    
    Args:
        trade_returns: List of trade returns (decimals)
        n_simulations: Number of simulations (default 30000)
        
    Returns:
        MonteCarloResult
    """
    config = MonteCarloConfig(n_simulations=n_simulations)
    simulator = MonteCarloSimulator(config)
    return simulator.run(trade_returns)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Generate sample trade returns (realistic distribution)
    np.random.seed(42)
    
    # Win rate: 45%, Avg win: 2%, Avg loss: 1.5%
    n_trades = 500
    win_rate = 0.45
    
    n_wins = int(n_trades * win_rate)
    n_losses = n_trades - n_wins
    
    wins = np.random.uniform(0.01, 0.04, n_wins)  # 1-4% wins
    losses = np.random.uniform(-0.025, -0.005, n_losses)  # 0.5-2.5% losses
    
    trade_returns = np.concatenate([wins, losses])
    np.random.shuffle(trade_returns)
    
    print("Running Monte Carlo simulation with 30,000 paths...")
    start = time.time()
    
    result = run_strategy_validation(trade_returns.tolist(), n_simulations=30000)
    
    print(f"\nCompleted in {time.time() - start:.2f} seconds")
    print(result.summary())
