import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
from copy import deepcopy
import uuid
import json
from pathlib import Path

from kline.core.strategy.base import (
    Strategy,
    Signal,
    Position,
    Order,
    OrderType,
    OrderSide,
    StrategyState
)
from kline.core.data.base import TimeFrame
from kline.utils.logger import get_logger

class BacktestResult:
    """回测结果"""
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        initial_capital: float,
        signals: List[Signal],
        positions: List[Position],
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
        trades_df: pd.DataFrame,
        statistics: Dict[str, Any]
    ):
        """
        初始化回测结果
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对符号
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            initial_capital: 初始资金
            signals: 信号列表
            positions: 仓位列表
            equity_curve: 权益曲线
            drawdown_curve: 回撤曲线
            trades_df: 交易记录
            statistics: 统计数据
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.initial_capital = initial_capital
        self.signals = signals
        self.positions = positions
        self.equity_curve = equity_curve
        self.drawdown_curve = drawdown_curve
        self.trades_df = trades_df
        self.statistics = statistics
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value if isinstance(self.timeframe, TimeFrame) else self.timeframe,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_capital": self.initial_capital,
            "signals": [s.to_dict() for s in self.signals],
            "positions": [p.to_dict() for p in self.positions],
            "statistics": self.statistics,
        }
        
    def save(self, file_path: str):
        """
        保存结果到文件
        
        Args:
            file_path: 文件路径
        """
        result_dict = self.to_dict()
        
        # 序列化为JSON
        with open(file_path, "w") as f:
            json.dump(result_dict, f, indent=2)
            
        # 保存交易记录
        if not self.trades_df.empty:
            trades_file = Path(file_path).with_suffix(".csv")
            self.trades_df.to_csv(trades_file)
            
        # 保存权益曲线
        equity_file = Path(file_path).with_name(f"{Path(file_path).stem}_equity.csv")
        pd.DataFrame({
            "equity": self.equity_curve,
            "drawdown": self.drawdown_curve
        }).to_csv(equity_file)


class Backtest:
    """回测引擎"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.1,
        symbol: str = "BTC/USDT",
        timeframe: Union[TimeFrame, str] = TimeFrame.HOUR_1
    ):
        """
        初始化回测引擎
        
        Args:
            data: 历史数据
            strategy: 策略
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
            position_size: 仓位大小（占总资金的比例）
            symbol: 交易对符号
            timeframe: 时间周期
        """
        self.data = data.copy()
        self.strategy = deepcopy(strategy)  # 使用深拷贝避免修改原始策略
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.symbol = symbol
        
        if isinstance(timeframe, str):
            self.timeframe = TimeFrame(timeframe)
        else:
            self.timeframe = timeframe
            
        self.logger = get_logger(f"backtest.{strategy.name}")
        
        # 添加symbol列
        if "symbol" not in self.data.columns:
            self.data["symbol"] = self.symbol
            
        # 回测状态
        self.current_capital = initial_capital
        self.current_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.signals: List[Signal] = []
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        
    def calculate_position_size(self, price: float) -> float:
        """
        计算仓位大小
        
        Args:
            price: 当前价格
            
        Returns:
            仓位大小（数量）
        """
        position_value = self.current_capital * self.position_size
        return position_value / price
        
    def apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        应用滑点
        
        Args:
            price: 价格
            side: 交易方向
            
        Returns:
            应用滑点后的价格
        """
        if side == OrderSide.BUY:
            return price * (1 + self.slippage)
        else:  # SELL
            return price * (1 - self.slippage)
            
    def apply_commission(self, value: float) -> float:
        """
        应用手续费
        
        Args:
            value: 交易金额
            
        Returns:
            手续费金额
        """
        return value * self.commission
        
    def open_position(self, signal: Signal, timestamp: datetime) -> Optional[Position]:
        """
        开仓
        
        Args:
            signal: 信号
            timestamp: 时间戳
            
        Returns:
            开仓的仓位，如果开仓失败则返回None
        """
        # 计算实际价格（应用滑点）
        execution_price = self.apply_slippage(signal.price, signal.side)
        
        # 计算仓位大小
        position_size = self.calculate_position_size(execution_price)
        
        # 计算交易金额
        trade_value = position_size * execution_price
        
        # 计算手续费
        commission_value = self.apply_commission(trade_value)
        
        # 检查是否有足够的资金
        if trade_value + commission_value > self.current_capital:
            self.logger.warning(f"Insufficient capital: {self.current_capital}, required: {trade_value + commission_value}")
            return None
            
        # 创建仓位
        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=execution_price,
            entry_time=timestamp,
            amount=position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # 更新资金
        self.current_capital -= (trade_value + commission_value)
        
        # 添加到当前仓位列表
        self.current_positions.append(position)
        
        self.logger.info(
            f"Opened {signal.side.value} position at {execution_price}, "
            f"size: {position_size}, capital left: {self.current_capital}"
        )
        
        return position
        
    def close_position(self, position: Position, price: float, timestamp: datetime) -> bool:
        """
        平仓
        
        Args:
            position: 仓位
            price: 平仓价格
            timestamp: 时间戳
            
        Returns:
            是否成功平仓
        """
        if position.is_closed:
            return False
            
        # 计算实际价格（应用滑点）
        execution_price = self.apply_slippage(price, OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY)
        
        # 计算交易金额
        trade_value = position.amount * execution_price
        
        # 计算手续费
        commission_value = self.apply_commission(trade_value)
        
        # 平仓
        position.close(execution_price, timestamp)
        
        # 更新资金
        self.current_capital += (trade_value - commission_value)
        
        # 从当前仓位列表中移除，添加到已平仓列表
        self.current_positions.remove(position)
        self.closed_positions.append(position)
        
        self.logger.info(
            f"Closed {position.side.value} position at {execution_price}, "
            f"PnL: {position.pnl:.2f} ({position.pnl_percentage:.2f}%), capital: {self.current_capital}"
        )
        
        return True
        
    def check_stop_loss_take_profit(self, position: Position, price: float, timestamp: datetime) -> bool:
        """
        检查止损止盈
        
        Args:
            position: 仓位
            price: 当前价格
            timestamp: 时间戳
            
        Returns:
            是否触发止损止盈
        """
        # 检查止损
        if position.check_stop_loss(price):
            self.logger.info(f"Stop loss triggered at {price} for position entered at {position.entry_price}")
            return self.close_position(position, price, timestamp)
            
        # 检查止盈
        if position.check_take_profit(price):
            self.logger.info(f"Take profit triggered at {price} for position entered at {position.entry_price}")
            return self.close_position(position, price, timestamp)
            
        return False
        
    def update_equity_curve(self, timestamp: datetime):
        """
        更新权益曲线
        
        Args:
            timestamp: 时间戳
        """
        # 计算当前权益（资金 + 未平仓位的价值）
        current_equity = self.current_capital
        
        # 添加到权益曲线
        self.equity_curve[timestamp] = current_equity
        
        # 计算回撤
        peak = self.equity_curve.cummax()
        drawdown = (peak - self.equity_curve) / peak * 100
        self.drawdown_curve[timestamp] = drawdown.iloc[-1]
        
    def get_trades_df(self) -> pd.DataFrame:
        """
        获取交易记录DataFrame
        
        Returns:
            交易记录DataFrame
        """
        trades = []
        
        for position in self.closed_positions:
            trade = {
                "symbol": position.symbol,
                "side": position.side.value,
                "entry_time": position.entry_time,
                "entry_price": position.entry_price,
                "exit_time": position.exit_time,
                "exit_price": position.exit_price,
                "amount": position.amount,
                "pnl": position.pnl,
                "pnl_percentage": position.pnl_percentage,
                "duration": (position.exit_time - position.entry_time).total_seconds() / 60,  # 持仓时间（分钟）
            }
            trades.append(trade)
            
        if not trades:
            return pd.DataFrame()
            
        return pd.DataFrame(trades)
        
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计数据
        
        Returns:
            统计数据
        """
        if not self.closed_positions:
            return {
                "total_trades": 0,
                "win_trades": 0,
                "loss_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_percentage": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "final_capital": self.initial_capital,
                "return": 0,
            }
            
        trades_df = self.get_trades_df()
        
        # 基本统计
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df["pnl"] > 0])
        loss_trades = total_trades - win_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df["pnl"].sum()
        average_pnl = trades_df["pnl"].mean()
        
        # 风险指标
        if not self.drawdown_curve.empty:
            max_drawdown = self.drawdown_curve.max()
        else:
            max_drawdown = 0
            
        # 最终资本和收益率
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # 胜率和盈亏比
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if win_trades > 0 else 0
        avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if loss_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 计算夏普比率
        if not self.equity_curve.empty:
            daily_returns = self.equity_curve.pct_change().dropna()
            if len(daily_returns) > 1:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)  # 假设252个交易日
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        return {
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_pnl": average_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_capital": final_capital,
            "return": total_return,
        }
        
    def run(self) -> BacktestResult:
        """
        运行回测
        
        Returns:
            回测结果
        """
        self.logger.info(f"Starting backtest for {self.strategy.name} on {self.symbol} {self.timeframe.value}")
        
        # 重置状态
        self.current_capital = self.initial_capital
        self.current_positions = []
        self.closed_positions = []
        self.signals = []
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        
        # 计算指标
        data_with_indicators = self.strategy.calculate_indicators(self.data)
        
        # 生成信号
        self.signals = self.strategy.generate_signals(data_with_indicators)
        
        # 按时间排序信号
        self.signals.sort(key=lambda x: x.time)
        
        # 创建信号索引，用于快速查找
        signal_dict = {}
        for signal in self.signals:
            if signal.time not in signal_dict:
                signal_dict[signal.time] = []
            signal_dict[signal.time].append(signal)
            
        # 遍历数据
        for timestamp, row in data_with_indicators.iterrows():
            # 当前价格
            current_price = row["close"]
            
            # 检查现有仓位的止损止盈
            for position in list(self.current_positions):  # 使用副本，因为可能在循环中修改
                self.check_stop_loss_take_profit(position, current_price, timestamp)
                
            # 处理当前时间的信号
            if timestamp in signal_dict:
                for signal in signal_dict[timestamp]:
                    # 检查是否应该开仓
                    if (signal.side == OrderSide.BUY and self.strategy.name.lower().find("short") == -1) or \
                       (signal.side == OrderSide.SELL and self.strategy.name.lower().find("long") == -1):
                        self.open_position(signal, timestamp)
                        
            # 更新权益曲线
            self.update_equity_curve(timestamp)
            
        # 强制平掉所有未平仓位
        last_price = data_with_indicators["close"].iloc[-1]
        last_timestamp = data_with_indicators.index[-1]
        
        for position in list(self.current_positions):
            self.close_position(position, last_price, last_timestamp)
            
        # 计算统计数据
        statistics = self.calculate_statistics()
        
        # 创建回测结果
        result = BacktestResult(
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.data.index[0],
            end_time=self.data.index[-1],
            initial_capital=self.initial_capital,
            signals=self.signals,
            positions=self.closed_positions,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
            trades_df=self.get_trades_df(),
            statistics=statistics
        )
        
        self.logger.info(f"Backtest completed with {len(self.signals)} signals and {len(self.closed_positions)} trades")
        self.logger.info(f"Final capital: {statistics['final_capital']:.2f}, Return: {statistics['return']:.2f}%")
        
        return result 