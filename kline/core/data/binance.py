import asyncio
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
import time
import json
import websocket
from threading import Thread

from kline.core.data.base import DataSource, DataType, TimeFrame
from kline.utils.logger import get_logger

class BinanceDataSource(DataSource):
    """Binance数据源实现"""
    
    KLINE_COLUMNS = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    
    def __init__(self, api_key="", api_secret="", testnet=True):
        super().__init__("binance")
        
        # 创建CCXT交易所实例
        self.exchange_params = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        
        if testnet:
            self.exchange_params["urls"] = {
                "api": {
                    "rest": "https://testnet.binance.vision/api",
                }
            }
        
        self.exchange = ccxt.binance(self.exchange_params)
        
        # WebSocket连接
        self.ws_base_url = "wss://testnet.binance.vision/ws" if testnet else "wss://stream.binance.com:9443/ws"
        self.ws_connections = {}
        self.ws_callbacks = {}
        self.ws_thread = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def close(self):
        """关闭所有连接"""
        if self.exchange:
            await self.exchange.close()
        
        # 关闭WebSocket连接
        for key, ws in self.ws_connections.items():
            ws.close()
    
    def _get_timeframe_ms(self, timeframe: Union[TimeFrame, str]) -> int:
        """
        获取时间周期的毫秒数
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
            
        # 时间周期映射到毫秒
        timeframe_map = {
            TimeFrame.MINUTE_1: 60 * 1000,
            TimeFrame.MINUTE_5: 5 * 60 * 1000,
            TimeFrame.MINUTE_15: 15 * 60 * 1000,
            TimeFrame.MINUTE_30: 30 * 60 * 1000,
            TimeFrame.HOUR_1: 60 * 60 * 1000,
            TimeFrame.HOUR_4: 4 * 60 * 60 * 1000,
            TimeFrame.DAY_1: 24 * 60 * 60 * 1000,
            TimeFrame.WEEK_1: 7 * 24 * 60 * 60 * 1000,
        }
        
        return timeframe_map[timeframe]
    
    def _format_symbol(self, symbol: str) -> str:
        """
        格式化交易对符号
        """
        # 如果包含/，转换为ccxt格式
        if "/" in symbol:
            return symbol
            
        # 如果是小写，转换为大写
        if symbol.islower():
            symbol = symbol.upper()
            
        # 如果不包含/，添加/
        if "/" not in symbol:
            # 常见的基础货币对
            if symbol.endswith(("USDT", "BUSD", "USDC", "USD", "BTC", "ETH")):
                for quote in ["USDT", "BUSD", "USDC", "USD", "BTC", "ETH"]:
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        return f"{base}/{quote}"
            
            # 默认添加USDT
            return f"{symbol}/USDT"
            
        return symbol
        
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
        """
        symbol = self._format_symbol(symbol)
        
        if isinstance(timeframe, TimeFrame):
            timeframe = timeframe.value
            
        params = {}
        
        # 添加开始和结束时间参数
        if start_time:
            params["since"] = int(start_time.timestamp() * 1000)
            
        # 由于CCXT的限制，我们需要分批获取数据
        all_candles = []
        
        # 计算批次数量
        batch_size = min(limit, 1000)  # Binance单次最多返回1000条
        current_start = start_time
        
        # 如果没有指定结束时间，使用当前时间
        if not end_time:
            end_time = datetime.now()
            
        # 计算时间间隔
        timeframe_ms = self._get_timeframe_ms(timeframe)
        max_records = (end_time - current_start).total_seconds() * 1000 // timeframe_ms
        
        # 如果记录数量超过限制，分批获取
        while current_start and current_start < end_time and (limit is None or len(all_candles) < limit):
            try:
                self.logger.debug(f"Fetching {symbol} {timeframe} kline from {current_start} to {end_time}")
                
                params["since"] = int(current_start.timestamp() * 1000)
                params["limit"] = batch_size
                
                # 获取K线数据
                candles = await self.exchange.fetch_ohlcv(symbol, timeframe, params=params)
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                # 更新开始时间为最后一条记录的时间加一个周期
                last_candle_time = candles[-1][0]
                current_start = datetime.fromtimestamp(last_candle_time / 1000) + timedelta(milliseconds=timeframe_ms)
                
                # 如果获取的数据不足一批，说明已经获取完所有数据
                if len(candles) < batch_size:
                    break
                    
                # 避免频繁请求触发限流
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch kline data: {e}")
                break
                
        # 限制返回的数据量
        if limit and len(all_candles) > limit:
            all_candles = all_candles[:limit]
            
        # 转换为DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
            
        return pd.DataFrame()
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取最新行情
        """
        symbol = self._format_symbol(symbol)
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker: {e}")
            return {}
    
    async def fetch_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        获取订单薄深度
        """
        symbol = self._format_symbol(symbol)
        
        try:
            depth = await self.exchange.fetch_order_book(symbol, limit=limit)
            return depth
        except Exception as e:
            self.logger.error(f"Failed to fetch depth: {e}")
            return {"bids": [], "asks": []}
    
    def _start_ws_thread(self):
        """
        启动WebSocket线程
        """
        if self.ws_thread is None or not self.ws_thread.is_alive():
            self.ws_thread = Thread(target=self._run_ws_thread, daemon=True)
            self.ws_thread.start()
    
    def _run_ws_thread(self):
        """
        运行WebSocket线程
        """
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_forever()
    
    def _get_stream_name(self, symbol: str, data_type: DataType) -> str:
        """
        获取流名称
        """
        symbol = symbol.lower().replace("/", "")
        
        if data_type == DataType.KLINE:
            return f"{symbol}@kline_1m"
        elif data_type == DataType.TICKER:
            return f"{symbol}@ticker"
        elif data_type == DataType.DEPTH:
            return f"{symbol}@depth20"
        elif data_type == DataType.TRADES:
            return f"{symbol}@trade"
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _on_ws_message(self, ws, message):
        """
        WebSocket消息回调
        """
        try:
            data = json.loads(message)
            stream = data.get("stream")
            
            if stream and stream in self.ws_callbacks:
                for callback in self.ws_callbacks[stream]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in callback for stream {stream}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_ws_error(self, ws, error):
        """
        WebSocket错误回调
        """
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """
        WebSocket关闭回调
        """
        self.logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
    
    def _on_ws_open(self, ws):
        """
        WebSocket打开回调
        """
        self.logger.info("WebSocket connection opened")
    
    async def subscribe(self, symbol: str, data_type: DataType, callback: Callable):
        """
        订阅实时数据
        """
        symbol = self._format_symbol(symbol)
        symbol_ws = symbol.replace("/", "").lower()
        
        stream_name = self._get_stream_name(symbol, data_type)
        
        # 启动WebSocket线程
        self._start_ws_thread()
        
        # 检查是否已经建立连接
        ws_key = f"{symbol_ws}_{data_type.value}"
        
        if ws_key not in self.ws_connections:
            # 创建WebSocket连接
            url = f"{self.ws_base_url}/{stream_name}"
            
            ws = websocket.WebSocketApp(
                url,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
                on_open=self._on_ws_open
            )
            
            self.ws_connections[ws_key] = ws
            
            # 在单独的线程中运行WebSocket
            Thread(target=ws.run_forever, daemon=True).start()
            
            # 等待WebSocket连接建立
            await asyncio.sleep(1)
            
        # 注册回调
        if stream_name not in self.ws_callbacks:
            self.ws_callbacks[stream_name] = []
            
        self.ws_callbacks[stream_name].append(callback)
        
        self.logger.info(f"Subscribed to {stream_name}")
    
    async def unsubscribe(self, symbol: str, data_type: DataType):
        """
        取消订阅
        """
        symbol = self._format_symbol(symbol)
        symbol_ws = symbol.replace("/", "").lower()
        
        ws_key = f"{symbol_ws}_{data_type.value}"
        stream_name = self._get_stream_name(symbol, data_type)
        
        # 移除回调
        if stream_name in self.ws_callbacks:
            self.ws_callbacks.pop(stream_name, None)
            
        # 关闭WebSocket连接
        if ws_key in self.ws_connections:
            ws = self.ws_connections.pop(ws_key)
            ws.close()
            
        self.logger.info(f"Unsubscribed from {stream_name}") 