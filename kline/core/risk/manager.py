import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
from enum import Enum

from kline.core.strategy.base import Position, OrderSide
from kline.utils.logger import get_logger

class RiskLevel(str, Enum):
    """风险级别"""
    LOW = "low"        # 低风险
    MEDIUM = "medium"  # 中等风险
    HIGH = "high"      # 高风险
    EXTREME = "extreme"  # 极端风险


class StopLossType(str, Enum):
    """止损类型"""
    FIXED = "fixed"              # 固定价格止损
    PERCENTAGE = "percentage"    # 百分比止损
    ATR = "atr"                  # ATR倍数止损
    CHANDELIER = "chandelier"    # 吊灯止损
    TRAILING = "trailing"        # 追踪止损


class RiskManager:
    """风险管理器"""
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 最大仓位比例
        max_positions: int = 5,           # 最大持仓数量
        max_risk_per_trade: float = 0.02,  # 每笔交易最大风险（占总资金比例）
        max_daily_loss: float = 0.05,      # 每日最大亏损（占总资金比例）
        max_drawdown: float = 0.2,         # 最大回撤（占总资金比例）
        risk_level: RiskLevel = RiskLevel.MEDIUM,  # 风险级别
    ):
        """
        初始化风险管理器
        
        Args:
            max_position_size: 最大仓位比例
            max_positions: 最大持仓数量
            max_risk_per_trade: 每笔交易最大风险
            max_daily_loss: 每日最大亏损
            max_drawdown: 最大回撤
            risk_level: 风险级别
        """
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.risk_level = risk_level
        self.logger = get_logger("risk_manager")
        
        # 当前状态
        self.current_positions: List[Position] = []
        self.current_capital = 0.0
        self.initial_capital = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_capital = 0.0
        self.current_drawdown = 0.0
        
        # 风险调整因子
        self.risk_factors = {
            RiskLevel.LOW: 0.5,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.EXTREME: 2.0,
        }
        
    def initialize(self, capital: float):
        """
        初始化资金
        
        Args:
            capital: 初始资金
        """
        self.current_capital = capital
        self.initial_capital = capital
        self.max_capital = capital
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_positions = []
        self.current_drawdown = 0.0
        
    def calculate_position_size(
        self, 
        price: float, 
        stop_loss: float, 
        entry_capital: Optional[float] = None
    ) -> float:
        """
        计算仓位大小
        
        Args:
            price: 入场价格
            stop_loss: 止损价格
            entry_capital: 入场资金，如果为None则使用当前资金
            
        Returns:
            仓位大小（数量）
        """
        if entry_capital is None:
            entry_capital = self.current_capital
            
        # 计算风险金额
        risk_amount = entry_capital * self.max_risk_per_trade * self.risk_factors[self.risk_level]
        
        # 计算每单位的风险
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit == 0:
            self.logger.warning("Risk per unit is zero, using default position size")
            return entry_capital * self.max_position_size / price
            
        # 计算仓位大小
        position_size = risk_amount / risk_per_unit
        
        # 限制最大仓位
        max_position_size = entry_capital * self.max_position_size / price
        position_size = min(position_size, max_position_size)
        
        return position_size
        
    def can_open_position(self, price: float, stop_loss: float) -> bool:
        """
        检查是否可以开仓
        
        Args:
            price: 入场价格
            stop_loss: 止损价格
            
        Returns:
            是否可以开仓
        """
        # 检查当前持仓数量
        if len(self.current_positions) >= self.max_positions:
            self.logger.info(f"Max positions reached: {len(self.current_positions)}/{self.max_positions}")
            return False
            
        # 检查日亏损是否超过限制
        if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
            self.logger.info(f"Daily loss limit reached: {self.daily_pnl}")
            return False
            
        # 检查回撤是否超过限制
        if self.current_drawdown > self.max_drawdown:
            self.logger.info(f"Max drawdown reached: {self.current_drawdown}")
            return False
            
        return True
        
    def add_position(self, position: Position):
        """
        添加仓位
        
        Args:
            position: 仓位
        """
        # 添加到当前仓位列表
        self.current_positions.append(position)
        
        # 计算仓位价值
        position_value = position.entry_price * position.amount
        
        # 更新资金
        self.current_capital -= position_value
        
        self.logger.info(
            f"Added {position.side.value} position at {position.entry_price}, "
            f"size: {position.amount}, capital left: {self.current_capital}"
        )
        
    def remove_position(self, position: Position, exit_price: float):
        """
        移除仓位
        
        Args:
            position: 仓位
            exit_price: 平仓价格
        """
        if position not in self.current_positions:
            self.logger.warning(f"Position not found: {position.symbol} {position.side.value}")
            return
            
        # 计算仓位盈亏
        pnl = position.unrealized_pnl(exit_price)
        
        # 更新日盈亏
        self.daily_pnl += pnl
        
        # 更新资金
        position_value = exit_price * position.amount
        self.current_capital += position_value
        
        # 更新最大资金
        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital
            
        # 更新回撤
        self.current_drawdown = (self.max_capital - self.current_capital) / self.max_capital
        
        # 从当前仓位列表中移除
        self.current_positions.remove(position)
        
        self.logger.info(
            f"Removed {position.side.value} position at {exit_price}, "
            f"PnL: {pnl:.2f}, capital: {self.current_capital}"
        )
        
    def calculate_stop_loss(
        self,
        price: float,
        side: OrderSide,
        stop_type: StopLossType = StopLossType.PERCENTAGE,
        percentage: float = 0.02,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        highest_lowest_n: int = 20,
        data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        计算止损价格
        
        Args:
            price: 入场价格
            side: 仓位方向
            stop_type: 止损类型
            percentage: 百分比止损比例
            atr: ATR值
            atr_multiplier: ATR乘数
            highest_lowest_n: 吊灯止损计算的最高/最低价周期
            data: 历史数据
            
        Returns:
            止损价格
        """
        if stop_type == StopLossType.FIXED:
            # 这个通常由策略决定，这里仅作为示例
            return price * (1 - 0.05) if side == OrderSide.BUY else price * (1 + 0.05)
            
        elif stop_type == StopLossType.PERCENTAGE:
            # 百分比止损
            if side == OrderSide.BUY:
                return price * (1 - percentage)
            else:  # SELL
                return price * (1 + percentage)
                
        elif stop_type == StopLossType.ATR:
            # ATR倍数止损
            if atr is None:
                self.logger.warning("ATR not provided, using percentage stop loss")
                return self.calculate_stop_loss(price, side, StopLossType.PERCENTAGE, percentage)
                
            if side == OrderSide.BUY:
                return price - atr * atr_multiplier
            else:  # SELL
                return price + atr * atr_multiplier
                
        elif stop_type == StopLossType.CHANDELIER:
            # 吊灯止损
            if data is None or len(data) < highest_lowest_n:
                self.logger.warning("Data not provided or insufficient, using percentage stop loss")
                return self.calculate_stop_loss(price, side, StopLossType.PERCENTAGE, percentage)
                
            if atr is None:
                # 计算ATR
                high = data["high"].iloc[-highest_lowest_n:]
                low = data["low"].iloc[-highest_lowest_n:]
                close = data["close"].iloc[-highest_lowest_n:].shift(1)
                
                tr1 = high - low
                tr2 = abs(high - close)
                tr3 = abs(low - close)
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.mean()
                
            if side == OrderSide.BUY:
                # 买入吊灯止损：最高价减去ATR的倍数
                highest = data["high"].iloc[-highest_lowest_n:].max()
                return highest - atr * atr_multiplier
            else:  # SELL
                # 卖出吊灯止损：最低价加上ATR的倍数
                lowest = data["low"].iloc[-highest_lowest_n:].min()
                return lowest + atr * atr_multiplier
                
        elif stop_type == StopLossType.TRAILING:
            # 追踪止损
            # 追踪止损需要持续更新，这里只是初始值
            return self.calculate_stop_loss(price, side, StopLossType.PERCENTAGE, percentage)
            
        else:
            self.logger.warning(f"Unknown stop loss type: {stop_type}, using percentage stop loss")
            return self.calculate_stop_loss(price, side, StopLossType.PERCENTAGE, percentage)
            
    def update_trailing_stop(
        self,
        position: Position,
        current_price: float,
        activation_percentage: float = 0.01,
        trail_percentage: float = 0.02
    ) -> float:
        """
        更新追踪止损
        
        Args:
            position: 仓位
            current_price: 当前价格
            activation_percentage: 触发百分比
            trail_percentage: 追踪百分比
            
        Returns:
            新的止损价格
        """
        # 如果没有止损价格，创建一个
        if position.stop_loss is None:
            if position.side == OrderSide.BUY:
                position.stop_loss = position.entry_price * (1 - trail_percentage)
            else:  # SELL
                position.stop_loss = position.entry_price * (1 + trail_percentage)
                
        # 计算当前盈利百分比
        if position.side == OrderSide.BUY:
            profit_percentage = (current_price / position.entry_price - 1)
            
            # 如果盈利超过激活百分比，更新止损
            if profit_percentage > activation_percentage:
                new_stop = current_price * (1 - trail_percentage)
                
                # 仅当新止损高于旧止损时更新
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(
                        f"Updated trailing stop for BUY position: {position.stop_loss} "
                        f"(profit: {profit_percentage:.2%})"
                    )
        else:  # SELL
            profit_percentage = (position.entry_price / current_price - 1)
            
            # 如果盈利超过激活百分比，更新止损
            if profit_percentage > activation_percentage:
                new_stop = current_price * (1 + trail_percentage)
                
                # 仅当新止损低于旧止损时更新
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(
                        f"Updated trailing stop for SELL position: {position.stop_loss} "
                        f"(profit: {profit_percentage:.2%})"
                    )
                    
        return position.stop_loss
        
    def kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
        max_position_size: Optional[float] = None
    ) -> float:
        """
        凯利公式计算最优仓位
        
        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            max_position_size: 最大仓位比例
            
        Returns:
            最优仓位比例
        """
        # 计算凯利比例
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # 限制最大仓位
        if max_position_size is None:
            max_position_size = self.max_position_size
            
        # 一般会使用半凯利或更保守的比例
        kelly = kelly * 0.5  # 使用半凯利
        
        # 如果凯利为负，返回0
        if kelly <= 0:
            return 0
            
        # 限制最大仓位
        return min(kelly, max_position_size)
        
    def reset_daily_stats(self):
        """重置每日统计"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            状态字典
        """
        return {
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "current_positions": len(self.current_positions),
            "max_positions": self.max_positions,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "max_capital": self.max_capital,
            "current_drawdown": self.current_drawdown,
            "risk_level": self.risk_level.value,
        } 