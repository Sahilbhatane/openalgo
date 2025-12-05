"""
Central Configuration
=====================
Load all configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from .mode import TradingMode, ModeManager
from .constants import (
    DEFAULT_CAPITAL,
    DEFAULT_MAX_RISK_PER_TRADE,
    DEFAULT_MAX_DAILY_DRAWDOWN,
    Limits,
    AngelOneLimits,
    BrokerageCharges,
)


# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class BrokerConfig:
    """Broker API configuration"""
    api_key: str = ""
    api_secret: str = ""
    totp_secret: str = ""
    client_id: str = ""
    password: str = ""
    
    # Separate API keys for different services
    historical_api_key: str = ""
    historical_api_secret: str = ""
    market_api_key: str = ""
    market_api_secret: str = ""
    
    @classmethod
    def from_env(cls) -> "BrokerConfig":
        return cls(
            api_key=os.getenv("BROKER_API_KEY", ""),
            api_secret=os.getenv("BROKER_API_SECRET", ""),
            totp_secret=os.getenv("BROKER_TOTP_SECRET", ""),
            client_id=os.getenv("BROKER_CLIENT_ID", ""),
            password=os.getenv("BROKER_PASSWORD", ""),
            historical_api_key=os.getenv("HISTORICAL_API_KEY", ""),
            historical_api_secret=os.getenv("HISTORICAL_API_SECRET", ""),
            market_api_key=os.getenv("BROKER_API_KEY_MARKET", ""),
            market_api_secret=os.getenv("BROKER_API_SECRET_MARKET", ""),
        )
    
    def validate(self) -> tuple[bool, list[str]]:
        """Check if required credentials are present"""
        issues = []
        if not self.api_key:
            issues.append("BROKER_API_KEY not set")
        if not self.api_secret:
            issues.append("BROKER_API_SECRET not set")
        if not self.totp_secret:
            issues.append("BROKER_TOTP_SECRET not set")
        return len(issues) == 0, issues


@dataclass
class OpenAlgoConfig:
    """OpenAlgo connection configuration"""
    host: str = "127.0.0.1"
    port: int = 5000
    api_key: str = ""
    websocket_port: int = 8765
    
    @classmethod
    def from_env(cls) -> "OpenAlgoConfig":
        return cls(
            host=os.getenv("OPENALGO_HOST", "127.0.0.1"),
            port=int(os.getenv("OPENALGO_PORT", "5000")),
            api_key=os.getenv("OPENALGO_APIKEY", ""),
            websocket_port=int(os.getenv("OPENALGO_WS_PORT", "8765")),
        )
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def websocket_url(self) -> str:
        return f"ws://{self.host}:{self.websocket_port}"


@dataclass  
class TradingConfig:
    """Trading parameters configuration"""
    capital: float = DEFAULT_CAPITAL
    max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE
    max_daily_drawdown: float = DEFAULT_MAX_DAILY_DRAWDOWN
    max_positions: int = Limits.MAX_POSITIONS
    max_orders_per_day: int = Limits.MAX_ORDERS_PER_DAY
    
    # Strategy parameters
    opening_range_minutes: int = 15
    atr_multiplier_sl: float = 1.5
    risk_reward_ratio: float = 1.5
    volume_surge_multiplier: float = 1.5
    
    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            capital=float(os.getenv("TRADING_CAPITAL", str(DEFAULT_CAPITAL))),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", str(DEFAULT_MAX_RISK_PER_TRADE))),
            max_daily_drawdown=float(os.getenv("MAX_DAILY_DRAWDOWN", str(DEFAULT_MAX_DAILY_DRAWDOWN))),
            max_positions=int(os.getenv("MAX_POSITIONS", str(Limits.MAX_POSITIONS))),
            max_orders_per_day=int(os.getenv("MAX_ORDERS_PER_DAY", str(Limits.MAX_ORDERS_PER_DAY))),
            opening_range_minutes=int(os.getenv("OPENING_RANGE_MINUTES", "15")),
            atr_multiplier_sl=float(os.getenv("ATR_MULTIPLIER_SL", "1.5")),
            risk_reward_ratio=float(os.getenv("RISK_REWARD_RATIO", "1.5")),
            volume_surge_multiplier=float(os.getenv("VOLUME_SURGE_MULTIPLIER", "1.5")),
        )


@dataclass
class Config:
    """
    Central configuration object.
    
    Usage:
        config = Config.load()
        print(config.trading.capital)
        print(config.is_paper_mode)
    """
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    openalgo: OpenAlgoConfig = field(default_factory=OpenAlgoConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mode_manager: ModeManager = field(default_factory=ModeManager)
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "bot_data")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "bot_data" / "logs")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "bot_data" / "reports")
    
    @classmethod
    def load(cls, data_dir: Optional[str] = None) -> "Config":
        """Load configuration from environment"""
        data_path = Path(data_dir) if data_dir else PROJECT_ROOT / "bot_data"
        
        config = cls(
            broker=BrokerConfig.from_env(),
            openalgo=OpenAlgoConfig.from_env(),
            trading=TradingConfig.from_env(),
            mode_manager=ModeManager(str(data_path)),
            data_dir=data_path,
            logs_dir=data_path / "logs",
            reports_dir=data_path / "reports",
        )
        
        # Ensure directories exist
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        
        return config
    
    @property
    def is_paper_mode(self) -> bool:
        return self.mode_manager.is_paper
    
    @property
    def is_live_mode(self) -> bool:
        return self.mode_manager.is_live
    
    @property
    def trading_mode(self) -> TradingMode:
        return self.mode_manager.current_mode
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate all configuration"""
        issues = []
        
        broker_valid, broker_issues = self.broker.validate()
        if not broker_valid:
            issues.extend(broker_issues)
        
        if self.trading.capital <= 0:
            issues.append("TRADING_CAPITAL must be positive")
        
        if not (0 < self.trading.max_risk_per_trade <= 0.10):
            issues.append("MAX_RISK_PER_TRADE should be between 0 and 10%")
        
        return len(issues) == 0, issues
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    BOT CONFIGURATION                         ║
╠══════════════════════════════════════════════════════════════╣
║  Trading Mode:    {self.trading_mode.value:>10}                               ║
║  Capital:         ₹{self.trading.capital:>12,.2f}                         ║
║  Risk per Trade:  {self.trading.max_risk_per_trade*100:>10.2f}%                             ║
║  Max Daily DD:    {self.trading.max_daily_drawdown*100:>10.2f}%                             ║
║  Max Positions:   {self.trading.max_positions:>10}                               ║
╠══════════════════════════════════════════════════════════════╣
║  OpenAlgo URL:    {self.openalgo.base_url:<40}║
║  WebSocket URL:   {self.openalgo.websocket_url:<40}║
╠══════════════════════════════════════════════════════════════╣
║  Broker API:      {'✅ Configured' if self.broker.api_key else '❌ Missing':<40}║
║  TOTP Secret:     {'✅ Configured' if self.broker.totp_secret else '❌ Missing':<40}║
║  Historical API:  {'✅ Configured' if self.broker.historical_api_key else '❌ Missing':<40}║
║  Market Feed API: {'✅ Configured' if self.broker.market_api_key else '❌ Missing':<40}║
╚══════════════════════════════════════════════════════════════╝
"""


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config
