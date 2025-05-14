import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone

from kline.core.data.base import DataCleaner
from kline.utils.logger import get_logger

class DefaultDataCleaner(DataCleaner):
    """默认数据清洗器"""
    
    def __init__(self, fill_missing=True, remove_outliers=True, normalize_timezone=True):
        """
        初始化清洗器
        
        Args:
            fill_missing: 是否填充缺失值
            remove_outliers: 是否移除异常值
            normalize_timezone: 是否标准化时区
        """
        super().__init__()
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.normalize_timezone = normalize_timezone
        
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        if data.empty:
            return data
            
        # 复制数据，避免修改原始数据
        cleaned_data = data.copy()
        
        # 检查是否为OHLCV数据
        is_ohlcv = all(col in cleaned_data.columns for col in ["open", "high", "low", "close", "volume"])
        
        # 移除重复索引
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep="last")]
        
        # 标准化时区
        if self.normalize_timezone and isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data.index = cleaned_data.index.tz_localize(timezone.utc) if cleaned_data.index.tz is None else cleaned_data.index.tz_convert(timezone.utc)
            
        # 按时间排序
        cleaned_data.sort_index(inplace=True)
        
        # 移除异常值
        if self.remove_outliers and is_ohlcv:
            cleaned_data = self._remove_outliers(cleaned_data)
            
        # 填充缺失值
        if self.fill_missing:
            cleaned_data = self._fill_missing_values(cleaned_data, is_ohlcv)
            
        # 检查和修复OHLCV数据的一致性
        if is_ohlcv:
            cleaned_data = self._fix_ohlcv_consistency(cleaned_data)
            
        return cleaned_data
        
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 复制数据
        cleaned_data = data.copy()
        
        # 计算价格变化百分比
        cleaned_data["price_change_pct"] = cleaned_data["close"].pct_change().abs()
        
        # 使用3倍标准差作为阈值
        mean = cleaned_data["price_change_pct"].mean()
        std = cleaned_data["price_change_pct"].std()
        threshold = mean + 3 * std
        
        # 找出异常值
        outliers = cleaned_data["price_change_pct"] > threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            self.logger.info(f"Found {outlier_count} outliers (threshold: {threshold:.4f})")
            
            # 移除异常值
            cleaned_data = cleaned_data[~outliers]
            
        # 移除临时列
        cleaned_data.drop("price_change_pct", axis=1, inplace=True)
        
        return cleaned_data
        
    def _fill_missing_values(self, data: pd.DataFrame, is_ohlcv: bool) -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            data: 原始数据
            is_ohlcv: 是否为OHLCV数据
            
        Returns:
            处理后的数据
        """
        # 复制数据
        filled_data = data.copy()
        
        # 检查是否有缺失值
        missing_count = filled_data.isnull().sum().sum()
        
        if missing_count > 0:
            self.logger.info(f"Found {missing_count} missing values")
            
            if is_ohlcv:
                # OHLCV数据的特殊处理
                
                # 缺失的开盘价使用前一个收盘价
                filled_data["open"] = filled_data["open"].fillna(filled_data["close"].shift(1))
                
                # 缺失的收盘价使用开盘价
                filled_data["close"] = filled_data["close"].fillna(filled_data["open"])
                
                # 缺失的最高价使用开盘价和收盘价的最大值
                filled_data["high"] = filled_data["high"].fillna(filled_data[["open", "close"]].max(axis=1))
                
                # 缺失的最低价使用开盘价和收盘价的最小值
                filled_data["low"] = filled_data["low"].fillna(filled_data[["open", "close"]].min(axis=1))
                
                # 缺失的成交量使用0
                filled_data["volume"] = filled_data["volume"].fillna(0)
            else:
                # 使用前向填充
                filled_data = filled_data.fillna(method="ffill")
                
                # 仍然缺失的值使用后向填充
                filled_data = filled_data.fillna(method="bfill")
                
        return filled_data
        
    def _fix_ohlcv_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        修复OHLCV数据的一致性
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 复制数据
        fixed_data = data.copy()
        
        # 确保 high >= open, close, low 且 low <= open, close
        inconsistent_high = (fixed_data["high"] < fixed_data[["open", "close"]].max(axis=1))
        inconsistent_low = (fixed_data["low"] > fixed_data[["open", "close"]].min(axis=1))
        
        inconsistent_count = inconsistent_high.sum() + inconsistent_low.sum()
        
        if inconsistent_count > 0:
            self.logger.info(f"Found {inconsistent_count} inconsistent OHLC values")
            
            # 修复最高价
            fixed_data.loc[inconsistent_high, "high"] = fixed_data.loc[inconsistent_high, ["open", "close", "high"]].max(axis=1)
            
            # 修复最低价
            fixed_data.loc[inconsistent_low, "low"] = fixed_data.loc[inconsistent_low, ["open", "close", "low"]].min(axis=1)
            
        # 确保成交量非负
        negative_volume = (fixed_data["volume"] < 0)
        
        if negative_volume.sum() > 0:
            self.logger.info(f"Found {negative_volume.sum()} negative volume values")
            fixed_data.loc[negative_volume, "volume"] = 0
            
        return fixed_data 