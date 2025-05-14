import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from kline.core.strategy.base import Strategy, Signal, OrderSide
from kline.utils.logger import get_logger

def calculate_ma(data: pd.DataFrame, period: int, price_col: str = "close") -> pd.Series:
    """
    计算移动平均线
    
    Args:
        data: 数据
        period: 周期
        price_col: 价格列名
        
    Returns:
        移动平均线
    """
    return data[price_col].rolling(window=period).mean()

class TripleMAStrategy(Strategy):
    """三重移动平均线交叉策略"""
    
    def __init__(
        self,
        fast_period: int = 5,
        medium_period: int = 10,
        slow_period: int = 20,
        name: str = "TripleMA"
    ):
        """
        初始化策略
        
        Args:
            fast_period: 快线周期
            medium_period: 中线周期
            slow_period: 慢线周期
            name: 策略名称
        """
        super().__init__(name)
        
        # 设置参数
        self.set_params({
            "fast_period": fast_period,
            "medium_period": medium_period,
            "slow_period": slow_period,
        })
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标
        
        Args:
            data: 原始数据
            
        Returns:
            添加了指标的数据
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 计算三条移动平均线
        df[f"ma_fast"] = calculate_ma(df, self.get_param("fast_period"))
        df[f"ma_medium"] = calculate_ma(df, self.get_param("medium_period"))
        df[f"ma_slow"] = calculate_ma(df, self.get_param("slow_period"))
        
        # 计算快线与中线的差值
        df["fast_medium_diff"] = df["ma_fast"] - df["ma_medium"]
        
        # 计算中线与慢线的差值
        df["medium_slow_diff"] = df["ma_medium"] - df["ma_slow"]
        
        # 判断快线是否上穿中线
        df["fast_cross_medium"] = (df["fast_medium_diff"] > 0) & (df["fast_medium_diff"].shift(1) <= 0)
        
        # 判断快线是否下穿中线
        df["fast_cross_medium_down"] = (df["fast_medium_diff"] < 0) & (df["fast_medium_diff"].shift(1) >= 0)
        
        # 判断中线是否上穿慢线
        df["medium_cross_slow"] = (df["medium_slow_diff"] > 0) & (df["medium_slow_diff"].shift(1) <= 0)
        
        # 判断中线是否下穿慢线
        df["medium_cross_slow_down"] = (df["medium_slow_diff"] < 0) & (df["medium_slow_diff"].shift(1) >= 0)
        
        # 强信号：快线和中线同时上穿慢线
        df["strong_buy"] = df["fast_cross_medium"] & df["medium_cross_slow"]
        
        # 强信号：快线和中线同时下穿慢线
        df["strong_sell"] = df["fast_cross_medium_down"] & df["medium_cross_slow_down"]
        
        # 计算ATR用于止损设置
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成信号
        
        Args:
            data: 包含指标的数据
            
        Returns:
            信号列表
        """
        signals = []
        
        # 获取符号信息
        if "symbol" in data.columns:
            symbol = data["symbol"].iloc[0]
        else:
            symbol = "Unknown"
            
        # 遍历数据
        for i, row in data.iterrows():
            # 跳过前面的数据，直到所有指标都计算出来
            if pd.isna(row["ma_slow"]):
                continue
                
            # 买入信号
            if row["strong_buy"]:
                # 计算止损和止盈
                price = row["close"]
                stop_loss = price - row["atr"] * 2
                take_profit = price + row["atr"] * 3
                
                # 创建信号
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    time=i,
                    strength=1.5,  # 强信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "fast_ma": row["ma_fast"],
                        "medium_ma": row["ma_medium"],
                        "slow_ma": row["ma_slow"],
                        "atr": row["atr"],
                    }
                )
                signals.append(signal)
                
            # 买入信号（弱）
            elif row["fast_cross_medium"] and row["medium_slow_diff"] > 0:
                # 快线上穿中线，且中线在慢线上方
                price = row["close"]
                stop_loss = price - row["atr"] * 1.5
                take_profit = price + row["atr"] * 2
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    time=i,
                    strength=1.0,  # 一般信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "fast_ma": row["ma_fast"],
                        "medium_ma": row["ma_medium"],
                        "slow_ma": row["ma_slow"],
                        "atr": row["atr"],
                    }
                )
                signals.append(signal)
                
            # 卖出信号
            elif row["strong_sell"]:
                price = row["close"]
                stop_loss = price + row["atr"] * 2
                take_profit = price - row["atr"] * 3
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.5,  # 强信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "fast_ma": row["ma_fast"],
                        "medium_ma": row["ma_medium"],
                        "slow_ma": row["ma_slow"],
                        "atr": row["atr"],
                    }
                )
                signals.append(signal)
                
            # 卖出信号（弱）
            elif row["fast_cross_medium_down"] and row["medium_slow_diff"] < 0:
                # 快线下穿中线，且中线在慢线下方
                price = row["close"]
                stop_loss = price + row["atr"] * 1.5
                take_profit = price - row["atr"] * 2
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.0,  # 一般信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "fast_ma": row["ma_fast"],
                        "medium_ma": row["ma_medium"],
                        "slow_ma": row["ma_slow"],
                        "atr": row["atr"],
                    }
                )
                signals.append(signal)
                
        return signals 