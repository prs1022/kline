import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from kline.core.strategy.base import Strategy, Signal, OrderSide
from kline.utils.logger import get_logger

def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
    """
    计算RSI指标
    
    Args:
        data: 数据
        period: 周期
        price_col: 价格列名
        
    Returns:
        RSI指标
    """
    delta = data[price_col].diff()
    
    # 分离上涨和下跌
    gain = delta.copy()
    loss = delta.copy()
    
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # 计算平均上涨和下跌
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # 计算相对强弱
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def find_peaks(data: pd.Series, window: int = 5) -> pd.Series:
    """
    查找序列中的峰值
    
    Args:
        data: 数据序列
        window: 窗口大小
        
    Returns:
        峰值标记序列，1表示峰值，-1表示谷值，0表示其他
    """
    # 初始化结果序列
    peaks = pd.Series(0, index=data.index)
    
    # 查找峰值和谷值
    for i in range(window, len(data) - window):
        # 获取当前窗口
        left_window = data.iloc[i-window:i]
        right_window = data.iloc[i+1:i+window+1]
        current_point = data.iloc[i]
        
        # 判断是否为峰值
        if current_point > left_window.max() and current_point > right_window.max():
            peaks.iloc[i] = 1
            
        # 判断是否为谷值
        elif current_point < left_window.min() and current_point < right_window.min():
            peaks.iloc[i] = -1
            
    return peaks

def check_divergence(
    price: pd.Series, 
    indicator: pd.Series, 
    price_peaks: pd.Series, 
    indicator_peaks: pd.Series,
    lookback: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    检查背离
    
    Args:
        price: 价格序列
        indicator: 指标序列
        price_peaks: 价格峰值序列
        indicator_peaks: 指标峰值序列
        lookback: 回溯窗口大小
        
    Returns:
        (看涨背离序列, 看跌背离序列)
    """
    bullish_div = pd.Series(False, index=price.index)
    bearish_div = pd.Series(False, index=price.index)
    
    # 遍历数据
    for i in range(lookback, len(price)):
        # 如果当前是价格的谷值
        if price_peaks.iloc[i] == -1:
            # 在过去的lookback周期内查找另一个谷值
            for j in range(i - lookback, i):
                if price_peaks.iloc[j] == -1:
                    # 如果找到了，比较价格和指标
                    if price.iloc[i] < price.iloc[j] and indicator.iloc[i] > indicator.iloc[j]:
                        # 价格创新低，但指标没有 -> 看涨背离
                        bullish_div.iloc[i] = True
                        break
                        
        # 如果当前是价格的峰值
        elif price_peaks.iloc[i] == 1:
            # 在过去的lookback周期内查找另一个峰值
            for j in range(i - lookback, i):
                if price_peaks.iloc[j] == 1:
                    # 如果找到了，比较价格和指标
                    if price.iloc[i] > price.iloc[j] and indicator.iloc[i] < indicator.iloc[j]:
                        # 价格创新高，但指标没有 -> 看跌背离
                        bearish_div.iloc[i] = True
                        break
                        
    return bullish_div, bearish_div

class RSIDivergenceStrategy(Strategy):
    """RSI背离策略"""
    
    def __init__(
        self,
        rsi_period: int = 14,
        peak_window: int = 5,
        lookback: int = 10,
        oversold: int = 30,
        overbought: int = 70,
        name: str = "RSIDivergence"
    ):
        """
        初始化策略
        
        Args:
            rsi_period: RSI周期
            peak_window: 峰值查找窗口
            lookback: 背离查找回溯窗口
            oversold: 超卖阈值
            overbought: 超买阈值
            name: 策略名称
        """
        super().__init__(name)
        
        # 设置参数
        self.set_params({
            "rsi_period": rsi_period,
            "peak_window": peak_window,
            "lookback": lookback,
            "oversold": oversold,
            "overbought": overbought,
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
        
        # 计算RSI
        rsi_period = self.get_param("rsi_period")
        df["rsi"] = calculate_rsi(df, period=rsi_period)
        
        # 查找价格和RSI的峰值谷值
        peak_window = self.get_param("peak_window")
        df["price_peaks"] = find_peaks(df["close"], window=peak_window)
        df["rsi_peaks"] = find_peaks(df["rsi"], window=peak_window)
        
        # 检查背离
        lookback = self.get_param("lookback")
        bullish_div, bearish_div = check_divergence(
            df["close"], 
            df["rsi"], 
            df["price_peaks"], 
            df["rsi_peaks"],
            lookback=lookback
        )
        
        df["bullish_divergence"] = bullish_div
        df["bearish_divergence"] = bearish_div
        
        # 检查RSI超买超卖
        oversold = self.get_param("oversold")
        overbought = self.get_param("overbought")
        
        df["oversold"] = df["rsi"] < oversold
        df["overbought"] = df["rsi"] > overbought
        
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
            if pd.isna(row["rsi"]) or pd.isna(row["atr"]):
                continue
                
            # 看涨信号：RSI看涨背离 + RSI超卖
            if row["bullish_divergence"] and row["oversold"]:
                # 计算止损和止盈
                price = row["close"]
                stop_loss = price - row["atr"] * 1.5
                take_profit = price + row["atr"] * 2.5
                
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
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "bullish_divergence_oversold"
                    }
                )
                signals.append(signal)
                
            # 看涨信号（弱）：仅RSI看涨背离
            elif row["bullish_divergence"]:
                price = row["close"]
                stop_loss = price - row["atr"] * 1.2
                take_profit = price + row["atr"] * 2.0
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    time=i,
                    strength=1.0,  # 一般信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "bullish_divergence"
                    }
                )
                signals.append(signal)
                
            # 看跌信号：RSI看跌背离 + RSI超买
            elif row["bearish_divergence"] and row["overbought"]:
                price = row["close"]
                stop_loss = price + row["atr"] * 1.5
                take_profit = price - row["atr"] * 2.5
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.5,  # 强信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "bearish_divergence_overbought"
                    }
                )
                signals.append(signal)
                
            # 看跌信号（弱）：仅RSI看跌背离
            elif row["bearish_divergence"]:
                price = row["close"]
                stop_loss = price + row["atr"] * 1.2
                take_profit = price - row["atr"] * 2.0
                
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    time=i,
                    strength=1.0,  # 一般信号
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "rsi": row["rsi"],
                        "atr": row["atr"],
                        "reason": "bearish_divergence"
                    }
                )
                signals.append(signal)
                
        return signals 