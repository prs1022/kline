import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from kline.core.strategy.base import Strategy, Signal, OrderSide
from kline.utils.logger import get_logger

def calculate_bollinger_bands(
    data: pd.DataFrame, 
    period: int = 20, 
    std_dev: float = 2.0, 
    price_col: str = "close"
) -> pd.DataFrame:
    """
    计算布林带指标
    
    Args:
        data: 数据
        period: 周期
        std_dev: 标准差倍数
        price_col: 价格列名
        
    Returns:
        包含布林带的DataFrame
    """
    # 计算移动平均线
    ma = data[price_col].rolling(window=period).mean()
    
    # 计算标准差
    std = data[price_col].rolling(window=period).std()
    
    # 计算上轨和下轨
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    
    # 构建结果
    result = pd.DataFrame({
        "bb_middle": ma,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": (upper - lower) / ma,  # 计算带宽
        "bb_pct_b": (data[price_col] - lower) / (upper - lower),  # 计算百分比B
    })
    
    return result

class BollingerBandsStrategy(Strategy):
    """布林带策略"""
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        bounce_threshold: float = 0.05,
        squeeze_threshold: float = 0.1,
        breakout_threshold: float = 0.02,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        name: str = "BollingerBands"
    ):
        """
        初始化策略
        
        Args:
            bb_period: 布林带周期
            bb_std_dev: 布林带标准差倍数
            bounce_threshold: 反弹阈值
            squeeze_threshold: 挤压阈值
            breakout_threshold: 突破阈值
            rsi_period: RSI周期
            rsi_oversold: RSI超卖阈值
            rsi_overbought: RSI超买阈值
            name: 策略名称
        """
        super().__init__(name)
        
        # 设置参数
        self.set_params({
            "bb_period": bb_period,
            "bb_std_dev": bb_std_dev,
            "bounce_threshold": bounce_threshold,
            "squeeze_threshold": squeeze_threshold,
            "breakout_threshold": breakout_threshold,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
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
        
        # 计算布林带
        bb_period = self.get_param("bb_period")
        bb_std_dev = self.get_param("bb_std_dev")
        
        bb = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std_dev)
        
        # 合并布林带数据
        df = pd.concat([df, bb], axis=1)
        
        # 计算RSI
        rsi_period = self.get_param("rsi_period")
        delta = df["close"].diff()
        
        # 分离上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # 计算相对强弱
        rs = avg_gain / avg_loss
        
        # 计算RSI
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # 计算布林带相关信号
        
        # 检查价格是否接近上轨
        df["near_upper"] = df["close"] > df["bb_upper"] * (1 - self.get_param("bounce_threshold"))
        
        # 检查价格是否接近下轨
        df["near_lower"] = df["close"] < df["bb_lower"] * (1 + self.get_param("bounce_threshold"))
        
        # 检查价格是否站上/站下中轨
        df["above_middle"] = df["close"] > df["bb_middle"]
        df["below_middle"] = df["close"] < df["bb_middle"]
        
        # 检查布林带宽度
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(window=bb_period).mean() * self.get_param("squeeze_threshold")
        
        # 检查价格是否突破上轨
        df["break_upper"] = (df["close"] > df["bb_upper"]) & (df["close"].shift(1) <= df["bb_upper"].shift(1))
        
        # 检查价格是否突破下轨
        df["break_lower"] = (df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))
        
        # 检查价格是否从上轨回落
        df["fall_from_upper"] = (df["close"] < df["bb_upper"] * (1 - self.get_param("breakout_threshold"))) & (df["close"].shift(1) >= df["bb_upper"].shift(1))
        
        # 检查价格是否从下轨反弹
        df["bounce_from_lower"] = (df["close"] > df["bb_lower"] * (1 + self.get_param("breakout_threshold"))) & (df["close"].shift(1) <= df["bb_lower"].shift(1))
        
        # 计算RSI超买超卖
        df["oversold"] = df["rsi"] < self.get_param("rsi_oversold")
        df["overbought"] = df["rsi"] > self.get_param("rsi_overbought")
        
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
            if pd.isna(row["bb_middle"]) or pd.isna(row["rsi"]) or pd.isna(row["atr"]):
                continue
                
            # 买入信号1：价格从下轨反弹 + RSI超卖
            if row["bounce_from_lower"] and row["oversold"]:
                price = row["close"]
                stop_loss = price - row["atr"] * 1.5
                take_profit = row["bb_middle"]  # 以中轨为目标
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    time=i,
                    strength=1.5,  # 强信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "bb_lower": row["bb_lower"],
                        "bb_middle": row["bb_middle"],
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "bounce_from_lower_oversold"
                    }
                )
                signals.append(signal)
                
            # 买入信号2：布林带挤压 + 价格突破中轨向上
            elif row["bb_squeeze"] and row["above_middle"] and not row["above_middle"].shift(1):
                price = row["close"]
                stop_loss = row["bb_lower"]
                take_profit = price + (price - row["bb_lower"]) * 1.5  # 风险回报比1.5倍
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    time=i,
                    strength=1.2,  # 中强度信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "bb_squeeze": True,
                        "bb_middle": row["bb_middle"],
                        "bb_width": row["bb_width"],
                        "atr": row["atr"],
                        "reason": "squeeze_break_middle_up"
                    }
                )
                signals.append(signal)
                
            # 卖出信号1：价格从上轨回落 + RSI超买
            elif row["fall_from_upper"] and row["overbought"]:
                price = row["close"]
                stop_loss = price + row["atr"] * 1.5
                take_profit = row["bb_middle"]  # 以中轨为目标
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.5,  # 强信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "bb_upper": row["bb_upper"],
                        "bb_middle": row["bb_middle"],
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "fall_from_upper_overbought"
                    }
                )
                signals.append(signal)
                
            # 卖出信号2：布林带挤压 + 价格突破中轨向下
            elif row["bb_squeeze"] and row["below_middle"] and not row["below_middle"].shift(1):
                price = row["close"]
                stop_loss = row["bb_upper"]
                take_profit = price - (row["bb_upper"] - price) * 1.5  # 风险回报比1.5倍
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.2,  # 中强度信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "bb_squeeze": True,
                        "bb_middle": row["bb_middle"],
                        "bb_width": row["bb_width"],
                        "atr": row["atr"],
                        "reason": "squeeze_break_middle_down"
                    }
                )
                signals.append(signal)
                
        return signals 