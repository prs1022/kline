import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    name: str
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit: bool = True

class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"  # sqlite, mysql, tdengine
    host: str = "localhost"
    port: int = 3306
    username: str = ""
    password: str = ""
    database: str = "kline"

class BacktestConfig(BaseModel):
    """Backtest configuration."""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005

class LogConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: bool = True
    console: bool = True

class Config(BaseModel):
    """Main configuration."""
    exchange: ExchangeConfig
    database: DatabaseConfig
    backtest: BacktestConfig
    log: LogConfig = LogConfig()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        exchange_config = ExchangeConfig(
            name=os.getenv("EXCHANGE_NAME", "binance"),
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
            testnet=os.getenv("EXCHANGE_TESTNET", "True").lower() == "true",
        )
        
        database_config = DatabaseConfig(
            type=os.getenv("DB_TYPE", "sqlite"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            username=os.getenv("DB_USERNAME", ""),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_DATABASE", "kline"),
        )
        
        backtest_config = BacktestConfig(
            start_date=os.getenv("BACKTEST_START_DATE", "2023-01-01"),
            end_date=os.getenv("BACKTEST_END_DATE", "2023-12-31"),
            initial_capital=float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000.0")),
            fee_rate=float(os.getenv("BACKTEST_FEE_RATE", "0.001")),
            slippage=float(os.getenv("BACKTEST_SLIPPAGE", "0.0005")),
        )
        
        log_config = LogConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "True").lower() == "true",
            console=os.getenv("LOG_CONSOLE", "True").lower() == "true",
        )
        
        return cls(
            exchange=exchange_config,
            database=database_config,
            backtest=backtest_config,
            log=log_config,
        ) 