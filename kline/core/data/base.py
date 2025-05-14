from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import pandas as pd
import polars as pl
from datetime import datetime

from kline.utils.logger import get_logger

log = get_logger("data")

class TimeFrame(str, Enum):
    """时间周期枚举"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class DataType(str, Enum):
    """数据类型枚举"""
    TICKER = "ticker"      # 最新价格
    KLINE = "kline"        # K线数据
    DEPTH = "depth"        # 订单薄深度
    TRADES = "trades"      # 成交记录


class DataSource(ABC):
    """数据源基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"data.{name}")
    
    @abstractmethod
    async def fetch_kline(
        self, 
        symbol: str, 
        timeframe: Union[TimeFrame, str], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            包含K线数据的DataFrame
        """
        pass
    
    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取最新行情
        
        Args:
            symbol: 交易对符号
            
        Returns:
            最新行情数据
        """
        pass
    
    @abstractmethod
    async def fetch_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        获取订单薄深度
        
        Args:
            symbol: 交易对符号
            limit: 深度数量
            
        Returns:
            订单薄数据
        """
        pass
    
    @abstractmethod
    async def subscribe(self, symbol: str, data_type: DataType, callback):
        """
        订阅实时数据
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
            callback: 回调函数
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbol: str, data_type: DataType):
        """
        取消订阅
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
        """
        pass


class DataCleaner(ABC):
    """数据清洗基类"""
    
    def __init__(self):
        self.logger = get_logger("data.cleaner")
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        pass


class DataRepository(ABC):
    """数据仓库基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"data.repo.{name}")
    
    @abstractmethod
    async def save(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame):
        """
        保存数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            data: 数据
        """
        pass
    
    @abstractmethod
    async def load(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            查询到的数据
        """
        pass


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self, sources: List[DataSource] = None, repositories: List[DataRepository] = None):
        self.sources = sources or []
        self.repositories = repositories or []
        self.logger = get_logger("data.aggregator")
        
    def add_source(self, source: DataSource):
        """
        添加数据源
        
        Args:
            source: 数据源
        """
        self.sources.append(source)
        
    def add_repository(self, repository: DataRepository):
        """
        添加数据仓库
        
        Args:
            repository: 数据仓库
        """
        self.repositories.append(repository)
        
    async def fetch_data(
        self, 
        symbol: str, 
        timeframe: Union[TimeFrame, str], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取数据，优先从仓库读取，仓库没有再从数据源获取
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            source_name: 指定数据源名称
            
        Returns:
            数据
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
            
        # 首先尝试从仓库加载
        for repo in self.repositories:
            try:
                data = await repo.load(symbol, timeframe, start_time, end_time)
                if not data.empty:
                    self.logger.info(f"Loaded data from repository {repo.name}")
                    return data
            except Exception as e:
                self.logger.error(f"Failed to load data from repository {repo.name}: {e}")
        
        # 如果仓库加载失败，从数据源获取
        sources = [source for source in self.sources if source.name == source_name] if source_name else self.sources
        
        for source in sources:
            try:
                data = await source.fetch_kline(symbol, timeframe, start_time, end_time)
                if not data.empty:
                    self.logger.info(f"Fetched data from source {source.name}")
                    
                    # 保存到仓库
                    for repo in self.repositories:
                        try:
                            await repo.save(symbol, timeframe, data)
                            self.logger.info(f"Saved data to repository {repo.name}")
                        except Exception as e:
                            self.logger.error(f"Failed to save data to repository {repo.name}: {e}")
                    
                    return data
            except Exception as e:
                self.logger.error(f"Failed to fetch data from source {source.name}: {e}")
        
        self.logger.warning(f"No data found for {symbol} {timeframe}")
        return pd.DataFrame() 