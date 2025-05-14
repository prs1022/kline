import os
import pandas as pd
from typing import Optional
from datetime import datetime
from pathlib import Path

from kline.core.data.base import DataRepository, TimeFrame
from kline.config.config import BASE_DIR

class CSVRepository(DataRepository):
    """基于CSV文件的数据仓库"""
    
    def __init__(self, base_dir: Optional[str] = None):
        super().__init__("csv")
        
        # 设置基础目录
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = BASE_DIR / "data" / "csv"
            
        # 创建目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"CSV repository initialized at {self.base_dir}")
        
    def _get_file_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        """
        获取文件路径
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            
        Returns:
            文件路径
        """
        # 格式化符号，将/替换为_
        formatted_symbol = symbol.replace("/", "_")
        
        # 获取时间周期值
        if isinstance(timeframe, TimeFrame):
            timeframe_value = timeframe.value
        else:
            timeframe_value = timeframe
            
        # 构建文件名
        filename = f"{formatted_symbol}_{timeframe_value}.csv"
        
        # 创建符号目录
        symbol_dir = self.base_dir / formatted_symbol
        symbol_dir.mkdir(exist_ok=True)
        
        return symbol_dir / filename
        
    async def save(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame):
        """
        保存数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            data: 数据
        """
        if data.empty:
            self.logger.warning(f"Empty data for {symbol} {timeframe}, skip saving")
            return
            
        file_path = self._get_file_path(symbol, timeframe)
        
        # 确保索引是时间戳
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data.set_index("timestamp", inplace=True)
                
        # 确保索引是时间戳类型，不是则尝试转换
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime: {e}")
                return
                
        # 检查文件是否存在
        if file_path.exists():
            # 读取已有数据
            existing_data = pd.read_csv(file_path)
            existing_data["timestamp"] = pd.to_datetime(existing_data["timestamp"])
            existing_data.set_index("timestamp", inplace=True)
            
            # 合并数据
            merged_data = pd.concat([existing_data, data])
            
            # 删除重复数据
            merged_data = merged_data[~merged_data.index.duplicated(keep="last")]
            
            # 按时间排序
            merged_data.sort_index(inplace=True)
            
            # 保存合并后的数据
            merged_data.to_csv(file_path, index=True, index_label="timestamp")
            self.logger.info(f"Updated data for {symbol} {timeframe} at {file_path}")
        else:
            # 保存新数据
            data.to_csv(file_path, index=True, index_label="timestamp")
            self.logger.info(f"Saved data for {symbol} {timeframe} at {file_path}")
            
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
        file_path = self._get_file_path(symbol, timeframe)
        
        if not file_path.exists():
            self.logger.warning(f"File {file_path} does not exist")
            return pd.DataFrame()
            
        try:
            # 读取CSV文件
            data = pd.read_csv(file_path)
            
            # 转换时间戳列
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
            
            # 按时间过滤数据
            if start_time:
                data = data[data.index >= pd.Timestamp(start_time)]
                
            if end_time:
                data = data[data.index <= pd.Timestamp(end_time)]
                
            # 限制返回的数据量
            if limit:
                data = data.tail(limit)
                
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            return pd.DataFrame() 