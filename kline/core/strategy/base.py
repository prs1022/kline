from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from kline.utils.logger import get_logger
from kline.core.data.base import TimeFrame

class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    TAKE_PROFIT = "take_profit"  # 止盈单


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"    # 买入
    SELL = "sell"  # 卖出


class Position:
    """仓位"""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        entry_time: datetime,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """
        初始化仓位
        
        Args:
            symbol: 交易对符号
            side: 仓位方向
            entry_price: 入场价格
            entry_time: 入场时间
            amount: 数量
            stop_loss: 止损价格
            take_profit: 止盈价格
        """
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.amount = amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.exit_price = None
        self.exit_time = None
        self.is_closed = False
        self.pnl = 0.0
        self.pnl_percentage = 0.0
    
    def close(self, exit_price: float, exit_time: datetime):
        """
        平仓
        
        Args:
            exit_price: 平仓价格
            exit_time: 平仓时间
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.is_closed = True
        
        # 计算盈亏
        if self.side == OrderSide.BUY:
            self.pnl = (exit_price - self.entry_price) * self.amount
            self.pnl_percentage = (exit_price / self.entry_price - 1) * 100
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.amount
            self.pnl_percentage = (self.entry_price / exit_price - 1) * 100
    
    def update_stop_loss(self, price: float):
        """
        更新止损价格
        
        Args:
            price: 新的止损价格
        """
        self.stop_loss = price
    
    def update_take_profit(self, price: float):
        """
        更新止盈价格
        
        Args:
            price: 新的止盈价格
        """
        self.take_profit = price
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止损
        """
        if self.stop_loss is None:
            return False
            
        if self.side == OrderSide.BUY:
            return current_price <= self.stop_loss
        else:  # SELL
            return current_price >= self.stop_loss
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        检查是否触发止盈
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止盈
        """
        if self.take_profit is None:
            return False
            
        if self.side == OrderSide.BUY:
            return current_price >= self.take_profit
        else:  # SELL
            return current_price <= self.take_profit
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        计算未实现盈亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            未实现盈亏
        """
        if self.is_closed:
            return self.pnl
            
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) * self.amount
        else:  # SELL
            return (self.entry_price - current_price) * self.amount
    
    def unrealized_pnl_percentage(self, current_price: float) -> float:
        """
        计算未实现盈亏百分比
        
        Args:
            current_price: 当前价格
            
        Returns:
            未实现盈亏百分比
        """
        if self.is_closed:
            return self.pnl_percentage
            
        if self.side == OrderSide.BUY:
            return (current_price / self.entry_price - 1) * 100
        else:  # SELL
            return (self.entry_price / current_price - 1) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "amount": self.amount,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "is_closed": self.is_closed,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
        }


class Order:
    """订单"""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        price: Optional[float] = None,
        amount: float = 0.0,
        stop_price: Optional[float] = None,
    ):
        """
        初始化订单
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            type: 订单类型
            price: 价格
            amount: 数量
            stop_price: 触发价格
        """
        self.symbol = symbol
        self.side = side
        self.type = type
        self.price = price
        self.amount = amount
        self.stop_price = stop_price
        self.id = None
        self.status = "created"
        self.created_at = datetime.now()
        self.filled_at = None
        self.filled_price = None
        self.filled_amount = 0.0
    
    def fill(self, price: float, amount: float, time: datetime):
        """
        成交
        
        Args:
            price: 成交价格
            amount: 成交数量
            time: 成交时间
        """
        self.filled_price = price
        self.filled_amount = amount
        self.filled_at = time
        self.status = "filled"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "price": self.price,
            "amount": self.amount,
            "stop_price": self.stop_price,
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_amount": self.filled_amount,
        }


class Signal:
    """交易信号"""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        time: datetime,
        strength: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化信号
        
        Args:
            symbol: 交易对符号
            side: 信号方向
            price: 信号价格
            time: 信号时间
            strength: 信号强度
            stop_loss: 止损价格
            take_profit: 止盈价格
            metadata: 元数据
        """
        self.symbol = symbol
        self.side = side
        self.price = price
        self.time = time
        self.strength = strength
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "price": self.price,
            "time": self.time.isoformat(),
            "strength": self.strength,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


class StrategyState(str, Enum):
    """策略状态"""
    INITIALIZED = "initialized"  # 初始化
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 暂停
    STOPPED = "stopped"          # 停止


class Strategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str):
        """
        初始化策略
        
        Args:
            name: 策略名称
        """
        self.name = name
        self.logger = get_logger(f"strategy.{name}")
        self.state = StrategyState.INITIALIZED
        self.positions: List[Position] = []
        self.signals: List[Signal] = []
        self.params: Dict[str, Any] = {}
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标
        
        Args:
            data: 原始数据
            
        Returns:
            添加了指标的数据
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成信号
        
        Args:
            data: 包含指标的数据
            
        Returns:
            信号列表
        """
        pass
    
    def add_signal(self, signal: Signal):
        """
        添加信号
        
        Args:
            signal: 信号
        """
        self.signals.append(signal)
        self.logger.info(f"Signal: {signal.side.value} {signal.symbol} at {signal.price}")
    
    def open_position(self, position: Position):
        """
        开仓
        
        Args:
            position: 仓位
        """
        self.positions.append(position)
        self.logger.info(
            f"Open position: {position.side.value} {position.symbol} "
            f"at {position.entry_price}, amount: {position.amount}"
        )
    
    def close_position(self, position: Position, exit_price: float, exit_time: datetime):
        """
        平仓
        
        Args:
            position: 仓位
            exit_price: 平仓价格
            exit_time: 平仓时间
        """
        position.close(exit_price, exit_time)
        self.logger.info(
            f"Close position: {position.side.value} {position.symbol} "
            f"at {position.exit_price}, PnL: {position.pnl:.2f} ({position.pnl_percentage:.2f}%)"
        )
    
    def get_active_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        获取活跃仓位
        
        Args:
            symbol: 交易对符号
            
        Returns:
            活跃仓位列表
        """
        active_positions = [p for p in self.positions if not p.is_closed]
        
        if symbol:
            active_positions = [p for p in active_positions if p.symbol == symbol]
            
        return active_positions
    
    def get_closed_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        获取已平仓位
        
        Args:
            symbol: 交易对符号
            
        Returns:
            已平仓位列表
        """
        closed_positions = [p for p in self.positions if p.is_closed]
        
        if symbol:
            closed_positions = [p for p in closed_positions if p.symbol == symbol]
            
        return closed_positions
    
    def update_stop_loss(self, position: Position, price: float):
        """
        更新止损价格
        
        Args:
            position: 仓位
            price: 新的止损价格
        """
        old_price = position.stop_loss
        position.update_stop_loss(price)
        self.logger.info(
            f"Update stop loss: {position.symbol} from {old_price} to {price}"
        )
    
    def update_take_profit(self, position: Position, price: float):
        """
        更新止盈价格
        
        Args:
            position: 仓位
            price: 新的止盈价格
        """
        old_price = position.take_profit
        position.update_take_profit(price)
        self.logger.info(
            f"Update take profit: {position.symbol} from {old_price} to {price}"
        )
    
    def set_param(self, name: str, value: Any):
        """
        设置参数
        
        Args:
            name: 参数名
            value: 参数值
        """
        self.params[name] = value
        self.logger.info(f"Set parameter: {name} = {value}")
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """
        获取参数
        
        Args:
            name: 参数名
            default: 默认值
            
        Returns:
            参数值
        """
        return self.params.get(name, default)
    
    def set_params(self, params: Dict[str, Any]):
        """
        设置多个参数
        
        Args:
            params: 参数字典
        """
        for name, value in params.items():
            self.set_param(name, value)
    
    def start(self):
        """启动策略"""
        self.state = StrategyState.RUNNING
        self.logger.info(f"Strategy {self.name} started")
    
    def pause(self):
        """暂停策略"""
        self.state = StrategyState.PAUSED
        self.logger.info(f"Strategy {self.name} paused")
    
    def resume(self):
        """恢复策略"""
        self.state = StrategyState.RUNNING
        self.logger.info(f"Strategy {self.name} resumed")
    
    def stop(self):
        """停止策略"""
        self.state = StrategyState.STOPPED
        self.logger.info(f"Strategy {self.name} stopped")
    
    def is_running(self) -> bool:
        """
        是否运行中
        
        Returns:
            是否运行中
        """
        return self.state == StrategyState.RUNNING
    
    def reset(self):
        """重置策略"""
        self.positions = []
        self.signals = []
        self.state = StrategyState.INITIALIZED
        self.logger.info(f"Strategy {self.name} reset")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计信息
        
        Returns:
            统计信息
        """
        closed_positions = self.get_closed_positions()
        
        if not closed_positions:
            return {
                "total_trades": 0,
                "win_trades": 0,
                "loss_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }
            
        # 计算基本统计信息
        total_trades = len(closed_positions)
        win_trades = sum(1 for p in closed_positions if p.pnl > 0)
        loss_trades = sum(1 for p in closed_positions if p.pnl <= 0)
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(p.pnl for p in closed_positions)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        win_pnls = [p.pnl for p in closed_positions if p.pnl > 0]
        loss_pnls = [p.pnl for p in closed_positions if p.pnl <= 0]
        
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        
        total_win = sum(win_pnls)
        total_loss = abs(sum(loss_pnls))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf') if total_win > 0 else 0.0
        
        # 计算最大回撤
        # 按照时间排序
        sorted_positions = sorted(closed_positions, key=lambda p: p.exit_time)
        
        # 计算累计盈亏曲线
        cumulative_pnl = np.cumsum([p.pnl for p in sorted_positions])
        
        # 计算最大回撤
        max_drawdown = 0.0
        peak = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            else:
                drawdown = peak - pnl
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "params": self.params,
            "positions": [p.to_dict() for p in self.positions],
            "signals": [s.to_dict() for s in self.signals],
            "statistics": self.calculate_statistics(),
        } 