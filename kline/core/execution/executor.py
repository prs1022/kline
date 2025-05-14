import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import asyncio
import time
import uuid
import json
from enum import Enum
from abc import ABC, abstractmethod

from kline.core.strategy.base import Order, OrderType, OrderSide, Position
from kline.utils.logger import get_logger

class ExecutionMode(str, Enum):
    """执行模式"""
    LIVE = "live"          # 实盘模式
    PAPER = "paper"        # 模拟盘模式
    SANDBOX = "sandbox"    # 沙盒模式（测试环境）


class OrderStatus(str, Enum):
    """订单状态"""
    CREATED = "created"    # 创建
    SUBMITTED = "submitted"  # 已提交
    PARTIAL = "partial"    # 部分成交
    FILLED = "filled"      # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 被拒绝
    EXPIRED = "expired"    # 已过期


class ExecutionError(Exception):
    """执行错误"""
    pass


class OrderExecutor(ABC):
    """订单执行器基类"""
    
    def __init__(self, name: str, mode: ExecutionMode = ExecutionMode.PAPER):
        """
        初始化执行器
        
        Args:
            name: 执行器名称
            mode: 执行模式
        """
        self.name = name
        self.mode = mode
        self.logger = get_logger(f"executor.{name}")
        
        # 跟踪订单和仓位
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        
        # 回调函数
        self.order_callbacks: Dict[str, List[Callable]] = {}
        self.position_callbacks: Dict[str, List[Callable]] = {}
        
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        提交订单
        
        Args:
            order: 订单
            
        Returns:
            订单ID
        """
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功取消
        """
        pass
        
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            订单状态
        """
        pass
        
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取仓位
        
        Args:
            symbol: 交易对符号
            
        Returns:
            仓位
        """
        pass
        
    def register_order_callback(self, callback: Callable[[Order], None], event: str = "all"):
        """
        注册订单回调函数
        
        Args:
            callback: 回调函数
            event: 触发事件
        """
        if event not in self.order_callbacks:
            self.order_callbacks[event] = []
            
        self.order_callbacks[event].append(callback)
        
    def register_position_callback(self, callback: Callable[[Position], None], event: str = "all"):
        """
        注册仓位回调函数
        
        Args:
            callback: 回调函数
            event: 触发事件
        """
        if event not in self.position_callbacks:
            self.position_callbacks[event] = []
            
        self.position_callbacks[event].append(callback)
        
    def trigger_order_callbacks(self, order: Order, event: str):
        """
        触发订单回调函数
        
        Args:
            order: 订单
            event: 触发事件
        """
        # 触发特定事件的回调
        if event in self.order_callbacks:
            for callback in self.order_callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.error(f"Error in order callback: {e}")
                    
        # 触发所有事件的回调
        if "all" in self.order_callbacks:
            for callback in self.order_callbacks["all"]:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.error(f"Error in order callback: {e}")
                    
    def trigger_position_callbacks(self, position: Position, event: str):
        """
        触发仓位回调函数
        
        Args:
            position: 仓位
            event: 触发事件
        """
        # 触发特定事件的回调
        if event in self.position_callbacks:
            for callback in self.position_callbacks[event]:
                try:
                    callback(position)
                except Exception as e:
                    self.logger.error(f"Error in position callback: {e}")
                    
        # 触发所有事件的回调
        if "all" in self.position_callbacks:
            for callback in self.position_callbacks["all"]:
                try:
                    callback(position)
                except Exception as e:
                    self.logger.error(f"Error in position callback: {e}")


class PaperExecutor(OrderExecutor):
    """模拟盘执行器"""
    
    def __init__(
        self,
        name: str = "paper",
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        delay_ms: int = 500,
    ):
        """
        初始化模拟盘执行器
        
        Args:
            name: 执行器名称
            initial_balance: 初始余额
            commission: 手续费率
            slippage: 滑点率
            delay_ms: 延迟（毫秒）
        """
        super().__init__(name, ExecutionMode.PAPER)
        
        self.balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.delay_ms = delay_ms
        
        # 价格数据
        self.tickers: Dict[str, Dict[str, Any]] = {}
        
    def update_ticker(self, symbol: str, price: float, timestamp: datetime = None):
        """
        更新最新价格
        
        Args:
            symbol: 交易对符号
            price: 价格
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        self.tickers[symbol] = {
            "price": price,
            "timestamp": timestamp,
        }
        
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取最新价格
        
        Args:
            symbol: 交易对符号
            
        Returns:
            价格数据
        """
        return self.tickers.get(symbol)
        
    async def submit_order(self, order: Order) -> str:
        """
        提交订单
        
        Args:
            order: 订单
            
        Returns:
            订单ID
        """
        # 生成订单ID
        order_id = str(uuid.uuid4())
        order.id = order_id
        
        # 更新订单状态
        order.status = OrderStatus.SUBMITTED
        
        # 记录订单
        self.orders[order_id] = order
        
        # 触发回调
        self.trigger_order_callbacks(order, "submitted")
        
        # 模拟延迟
        await asyncio.sleep(self.delay_ms / 1000)
        
        # 获取最新价格
        ticker = self.get_ticker(order.symbol)
        
        if ticker is None:
            order.status = OrderStatus.REJECTED
            self.trigger_order_callbacks(order, "rejected")
            raise ExecutionError(f"No ticker data for {order.symbol}")
            
        # 计算成交价格（加上滑点）
        if order.side == OrderSide.BUY:
            execution_price = ticker["price"] * (1 + self.slippage)
        else:  # SELL
            execution_price = ticker["price"] * (1 - self.slippage)
            
        # 检查限价单的价格
        if order.type == OrderType.LIMIT:
            if (order.side == OrderSide.BUY and execution_price > order.price) or \
               (order.side == OrderSide.SELL and execution_price < order.price):
                self.logger.info(f"Limit order {order_id} not executed: {execution_price} vs {order.price}")
                return order_id
                
        # 计算成交金额
        trade_value = order.amount * execution_price
        
        # 计算手续费
        commission_value = trade_value * self.commission
        
        # 检查余额是否足够
        if order.side == OrderSide.BUY and trade_value + commission_value > self.balance:
            order.status = OrderStatus.REJECTED
            self.trigger_order_callbacks(order, "rejected")
            self.logger.warning(f"Insufficient balance: {self.balance}, required: {trade_value + commission_value}")
            return order_id
            
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_price = execution_price
        order.filled_amount = order.amount
        order.filled_at = datetime.now()
        
        # 更新余额
        if order.side == OrderSide.BUY:
            self.balance -= (trade_value + commission_value)
        else:  # SELL
            self.balance += (trade_value - commission_value)
            
        # 更新仓位
        position_key = f"{order.symbol}_{order.side.value}"
        
        if order.side == OrderSide.BUY:
            # 创建或更新买入仓位
            if position_key not in self.positions:
                position = Position(
                    symbol=order.symbol,
                    side=order.side,
                    entry_price=execution_price,
                    entry_time=order.filled_at,
                    amount=order.amount,
                )
                self.positions[position_key] = position
                self.trigger_position_callbacks(position, "open")
            else:
                # 更新现有仓位
                position = self.positions[position_key]
                # 计算新的平均入场价格
                total_value = position.entry_price * position.amount + execution_price * order.amount
                total_amount = position.amount + order.amount
                new_entry_price = total_value / total_amount
                
                position.entry_price = new_entry_price
                position.amount = total_amount
                self.trigger_position_callbacks(position, "update")
                
        else:  # SELL
            # 检查是否有买入仓位需要平仓
            buy_position_key = f"{order.symbol}_{OrderSide.BUY.value}"
            
            if buy_position_key in self.positions:
                buy_position = self.positions[buy_position_key]
                
                if buy_position.amount <= order.amount:
                    # 平掉整个仓位
                    buy_position.close(execution_price, order.filled_at)
                    del self.positions[buy_position_key]
                    self.trigger_position_callbacks(buy_position, "close")
                else:
                    # 部分平仓
                    # 创建一个新的平仓记录
                    closed_position = Position(
                        symbol=buy_position.symbol,
                        side=buy_position.side,
                        entry_price=buy_position.entry_price,
                        entry_time=buy_position.entry_time,
                        amount=order.amount,
                    )
                    closed_position.close(execution_price, order.filled_at)
                    self.trigger_position_callbacks(closed_position, "close")
                    
                    # 更新原始仓位
                    buy_position.amount -= order.amount
                    self.trigger_position_callbacks(buy_position, "update")
            else:
                # 创建卖出仓位
                position = Position(
                    symbol=order.symbol,
                    side=order.side,
                    entry_price=execution_price,
                    entry_time=order.filled_at,
                    amount=order.amount,
                )
                self.positions[position_key] = position
                self.trigger_position_callbacks(position, "open")
                
        # 触发订单回调
        self.trigger_order_callbacks(order, "filled")
        
        self.logger.info(
            f"Order {order_id} executed: {order.side.value} {order.amount} {order.symbol} at {execution_price}, "
            f"balance: {self.balance}"
        )
        
        return order_id
        
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功取消
        """
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
            return False
            
        # 更新订单状态
        order.status = OrderStatus.CANCELLED
        
        # 触发回调
        self.trigger_order_callbacks(order, "cancelled")
        
        self.logger.info(f"Order {order_id} cancelled")
        
        return True
        
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        获取订单状态
        
        Args:
            order_id: 订单ID
            
        Returns:
            订单状态
        """
        if order_id not in self.orders:
            raise ExecutionError(f"Order {order_id} not found")
            
        return self.orders[order_id].status
        
    async def get_position(self, symbol: str) -> Dict[str, Position]:
        """
        获取仓位
        
        Args:
            symbol: 交易对符号
            
        Returns:
            仓位字典
        """
        result = {}
        
        for key, position in self.positions.items():
            if position.symbol == symbol:
                result[position.side.value] = position
                
        return result
        
    def get_balance(self) -> float:
        """
        获取余额
        
        Returns:
            余额
        """
        return self.balance
        
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float
    ) -> str:
        """
        创建市价单
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            amount: 数量
            
        Returns:
            订单ID
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            amount=amount,
        )
        
        return await self.submit_order(order)
        
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        amount: float
    ) -> str:
        """
        创建限价单
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            price: 价格
            amount: 数量
            
        Returns:
            订单ID
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            price=price,
            amount=amount,
        )
        
        return await self.submit_order(order)
        
    async def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        amount: float
    ) -> str:
        """
        创建止损单
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            stop_price: 触发价格
            amount: 数量
            
        Returns:
            订单ID
        """
        order = Order(
            symbol=symbol,
            side=side,
            type=OrderType.STOP,
            stop_price=stop_price,
            amount=amount,
        )
        
        return await self.submit_order(order) 