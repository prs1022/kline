import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import asyncio
import random

from kline.core.data.base import DataSource, DataType, TimeFrame
from kline.utils.logger import get_logger

class MockDataSource(DataSource):
    """模拟数据源，用于测试和开发"""
    
    def __init__(self, seed=None):
        """
        初始化模拟数据源
        
        Args:
            seed: 随机种子，用于生成可重复的数据
        """
        super().__init__("mock")
        self.seed = seed
        self.random_gen = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        
        # 模拟的回调函数
        self.callbacks = {}
        
        # 模拟的订阅状态
        self.subscriptions = {}
        
        # 生成的数据缓存
        self.data_cache = {}
        
    def _get_timeframe_seconds(self, timeframe: Union[TimeFrame, str]) -> int:
        """
        获取时间周期的秒数
        
        Args:
            timeframe: 时间周期
            
        Returns:
            秒数
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
            
        # 映射到秒数
        timeframe_map = {
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.MINUTE_30: 1800,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.HOUR_4: 14400,
            TimeFrame.DAY_1: 86400,
            TimeFrame.WEEK_1: 604800,
        }
        
        return timeframe_map[timeframe]
        
    def _generate_price_series(self, 
                              start_price: float, 
                              periods: int, 
                              volatility: float = 0.01,
                              drift: float = 0.0001) -> np.ndarray:
        """
        生成价格序列
        
        Args:
            start_price: 起始价格
            periods: 周期数
            volatility: 波动率
            drift: 漂移率
            
        Returns:
            价格序列
        """
        # 使用几何布朗运动生成价格
        returns = self.np_random.normal(loc=drift, scale=volatility, size=periods)
        
        # 计算价格
        prices = start_price * np.exp(np.cumsum(returns))
        
        return prices
        
    def _generate_ohlcv(self, 
                       start_price: float, 
                       periods: int, 
                       start_time: datetime,
                       timeframe: Union[TimeFrame, str],
                       volatility: float = 0.01,
                       volume_mean: float = 100.0,
                       volume_std: float = 30.0) -> pd.DataFrame:
        """
        生成OHLCV数据
        
        Args:
            start_price: 起始价格
            periods: 周期数
            start_time: 开始时间
            timeframe: 时间周期
            volatility: 波动率
            volume_mean: 成交量均值
            volume_std: 成交量标准差
            
        Returns:
            OHLCV数据
        """
        # 生成收盘价序列
        closes = self._generate_price_series(start_price, periods, volatility)
        
        # 生成开盘价、最高价、最低价
        opens = np.empty_like(closes)
        highs = np.empty_like(closes)
        lows = np.empty_like(closes)
        
        # 第一个开盘价等于起始价格
        opens[0] = start_price
        
        # 之后的开盘价等于上一个收盘价加上一个小的随机波动
        opens[1:] = closes[:-1] * (1 + self.np_random.normal(0, volatility/3, periods-1))
        
        # 生成最高价和最低价
        for i in range(periods):
            price_range = max(closes[i], opens[i]) * volatility * 2
            high_range = price_range * self.np_random.uniform(0.2, 1.0)
            low_range = price_range * self.np_random.uniform(0.2, 1.0)
            
            highs[i] = max(closes[i], opens[i]) + high_range
            lows[i] = min(closes[i], opens[i]) - low_range
            
        # 生成成交量
        volumes = self.np_random.normal(volume_mean, volume_std, periods)
        volumes = np.maximum(volumes, 0)  # 确保成交量非负
        
        # 生成时间戳
        timeframe_seconds = self._get_timeframe_seconds(timeframe)
        timestamps = [start_time + timedelta(seconds=i*timeframe_seconds) for i in range(periods)]
        
        # 创建DataFrame
        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        }, index=pd.DatetimeIndex(timestamps, name="timestamp"))
        
        return df
        
    def _get_symbol_base_price(self, symbol: str) -> float:
        """
        根据交易对获取基础价格
        
        Args:
            symbol: 交易对符号
            
        Returns:
            基础价格
        """
        # 格式化符号
        if "/" in symbol:
            base, quote = symbol.split("/")
        else:
            for quote in ["USDT", "USD", "BTC", "ETH"]:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    break
            else:
                base = symbol
                quote = "USDT"
                
        # 常见加密货币的基础价格映射
        base_prices = {
            "BTC": 30000.0,
            "ETH": 2000.0,
            "BNB": 300.0,
            "XRP": 0.5,
            "ADA": 0.3,
            "DOGE": 0.1,
            "SOL": 100.0,
            "DOT": 5.0,
        }
        
        # 如果交易对基础币种在列表中，使用对应价格，否则使用随机价格
        if base.upper() in base_prices:
            return base_prices[base.upper()]
        else:
            # 使用符号的哈希值生成一个伪随机价格
            symbol_hash = hash(symbol) % 10000
            self.random_gen.seed(symbol_hash)
            
            # 生成不同数量级的价格
            price_magnitude = self.random_gen.choice([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            base_price = self.random_gen.uniform(0.1, 10.0) * price_magnitude
            
            return base_price
    
    async def fetch_kline(
        self, 
        symbol: str, 
        timeframe: Union[TimeFrame, str], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
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
        # 生成缓存键
        cache_key = f"{symbol}_{timeframe}"
        
        # 如果缓存中没有该交易对的数据，生成新数据
        if cache_key not in self.data_cache:
            # 获取基础价格
            base_price = self._get_symbol_base_price(symbol)
            
            # 为了模拟真实情况，我们生成从过去一年开始的数据
            now = datetime.now()
            history_start = now - timedelta(days=365)
            
            # 计算需要生成的周期数
            tf_seconds = self._get_timeframe_seconds(timeframe)
            periods = int(365 * 24 * 3600 / tf_seconds)
            
            # 限制生成的数据量，避免内存问题
            periods = min(periods, 10000)
            
            # 生成数据
            df = self._generate_ohlcv(
                start_price=base_price,
                periods=periods,
                start_time=history_start,
                timeframe=timeframe
            )
            
            # 缓存数据
            self.data_cache[cache_key] = df
            
        # 获取缓存数据
        df = self.data_cache[cache_key].copy()
        
        # 根据开始和结束时间过滤数据
        if start_time:
            df = df[df.index >= pd.Timestamp(start_time)]
            
        if end_time:
            df = df[df.index <= pd.Timestamp(end_time)]
            
        # 限制返回的数据量
        if limit and len(df) > limit:
            df = df.tail(limit)
            
        # 模拟网络延迟
        await asyncio.sleep(0.1)
        
        return df
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取最新行情
        
        Args:
            symbol: 交易对符号
            
        Returns:
            最新行情数据
        """
        # 获取最新K线数据
        df = await self.fetch_kline(symbol, TimeFrame.MINUTE_1, limit=1)
        
        if df.empty:
            return {}
            
        # 构造ticker
        last_row = df.iloc[-1]
        
        ticker = {
            "symbol": symbol,
            "timestamp": int(df.index[-1].timestamp() * 1000),
            "datetime": df.index[-1].isoformat(),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "bid": float(last_row["close"] * 0.999),
            "ask": float(last_row["close"] * 1.001),
            "last": float(last_row["close"]),
            "close": float(last_row["close"]),
            "open": float(last_row["open"]),
            "volume": float(last_row["volume"]),
            "change": float(last_row["close"] - last_row["open"]),
            "percentage": float((last_row["close"] / last_row["open"] - 1) * 100),
        }
        
        return ticker
    
    async def fetch_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        获取订单薄深度
        
        Args:
            symbol: 交易对符号
            limit: 深度数量
            
        Returns:
            订单薄数据
        """
        # 获取最新价格
        ticker = await self.fetch_ticker(symbol)
        
        if not ticker:
            return {"bids": [], "asks": []}
            
        last_price = ticker["last"]
        
        # 模拟深度数据
        bids = []
        asks = []
        
        # 生成买单
        for i in range(limit):
            price = last_price * (1 - 0.0001 * (i + 1) * (1 + self.random_gen.random() * 0.5))
            amount = self.random_gen.uniform(0.1, 10.0) * (1 + 0.1 * self.random_gen.random())
            bids.append([price, amount])
            
        # 生成卖单
        for i in range(limit):
            price = last_price * (1 + 0.0001 * (i + 1) * (1 + self.random_gen.random() * 0.5))
            amount = self.random_gen.uniform(0.1, 10.0) * (1 + 0.1 * self.random_gen.random())
            asks.append([price, amount])
            
        # 排序
        bids = sorted(bids, key=lambda x: x[0], reverse=True)
        asks = sorted(asks, key=lambda x: x[0])
        
        depth = {
            "bids": bids,
            "asks": asks,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "datetime": datetime.now().isoformat(),
            "symbol": symbol,
        }
        
        return depth
    
    async def subscribe(self, symbol: str, data_type: DataType, callback: Callable):
        """
        订阅实时数据
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
            callback: 回调函数
        """
        # 生成订阅键
        sub_key = f"{symbol}_{data_type.value}"
        
        # 注册回调
        if sub_key not in self.callbacks:
            self.callbacks[sub_key] = []
            
        self.callbacks[sub_key].append(callback)
        
        # 标记为已订阅
        self.subscriptions[sub_key] = True
        
        self.logger.info(f"Subscribed to {sub_key}")
        
        # 启动模拟数据推送任务
        asyncio.create_task(self._simulate_data_push(symbol, data_type))
    
    async def unsubscribe(self, symbol: str, data_type: DataType):
        """
        取消订阅
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
        """
        # 生成订阅键
        sub_key = f"{symbol}_{data_type.value}"
        
        # 移除回调
        self.callbacks.pop(sub_key, None)
        
        # 标记为未订阅
        self.subscriptions[sub_key] = False
        
        self.logger.info(f"Unsubscribed from {sub_key}")
    
    async def _simulate_data_push(self, symbol: str, data_type: DataType):
        """
        模拟数据推送
        
        Args:
            symbol: 交易对符号
            data_type: 数据类型
        """
        sub_key = f"{symbol}_{data_type.value}"
        
        # 检查是否有回调
        if sub_key not in self.callbacks:
            return
            
        # 根据数据类型确定推送间隔
        if data_type == DataType.KLINE:
            interval = 60  # 每分钟推送一次K线
        elif data_type == DataType.TICKER:
            interval = 1  # 每秒推送一次行情
        elif data_type == DataType.DEPTH:
            interval = 1  # 每秒推送一次深度
        elif data_type == DataType.TRADES:
            interval = 0.5  # 每0.5秒推送一次成交
        else:
            interval = 5  # 默认5秒
            
        # 持续推送，直到取消订阅
        while self.subscriptions.get(sub_key, False):
            try:
                # 根据数据类型获取不同的数据
                if data_type == DataType.KLINE:
                    data = await self.fetch_kline(symbol, TimeFrame.MINUTE_1, limit=1)
                    data = {
                        "stream": sub_key,
                        "data": data.iloc[-1].to_dict(),
                        "timestamp": int(data.index[-1].timestamp() * 1000),
                    }
                elif data_type == DataType.TICKER:
                    data = await self.fetch_ticker(symbol)
                    data = {
                        "stream": sub_key,
                        "data": data,
                    }
                elif data_type == DataType.DEPTH:
                    data = await self.fetch_depth(symbol)
                    data = {
                        "stream": sub_key,
                        "data": data,
                    }
                elif data_type == DataType.TRADES:
                    ticker = await self.fetch_ticker(symbol)
                    price = ticker.get("last", 0)
                    amount = self.random_gen.uniform(0.001, 1.0)
                    side = "buy" if self.random_gen.random() > 0.5 else "sell"
                    data = {
                        "stream": sub_key,
                        "data": {
                            "symbol": symbol,
                            "id": int(datetime.now().timestamp() * 1000),
                            "price": price,
                            "amount": amount,
                            "cost": price * amount,
                            "side": side,
                            "timestamp": int(datetime.now().timestamp() * 1000),
                            "datetime": datetime.now().isoformat(),
                        }
                    }
                else:
                    data = {
                        "stream": sub_key,
                        "data": {},
                    }
                    
                # 调用回调函数
                for callback in self.callbacks.get(sub_key, []):
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}")
                        
                # 等待下一次推送
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error simulating data push: {e}")
                await asyncio.sleep(1)  # 出错时等待一秒再重试 