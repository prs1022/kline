import asyncio
import pandas as pd
from datetime import datetime, timedelta

from kline.core.data import MockDataSource
from kline.core.strategy import TripleMAStrategy
from kline.core.backtest import Backtest


async def main():
    # 创建模拟数据源
    data_source = MockDataSource(seed=42)
    
    # 获取模拟数据
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # 获取过去30天的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"获取数据：{symbol} {timeframe} from {start_time} to {end_time}")
    data = await data_source.fetch_kline(symbol, timeframe, start_time, end_time)
    
    # 创建策略
    strategy = TripleMAStrategy(fast_period=5, medium_period=10, slow_period=20)
    
    # 创建回测引擎
    backtest = Backtest(
        data=data,
        strategy=strategy,
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        position_size=0.1,
        symbol=symbol,
        timeframe=timeframe
    )
    
    # 运行回测
    print("开始回测...")
    result = backtest.run()
    
    # 打印回测结果
    print("\n===== 回测结果 =====")
    print(f"策略：{result.strategy_name}")
    print(f"交易对：{result.symbol}")
    print(f"周期：{result.timeframe}")
    print(f"时间范围：{result.start_time} - {result.end_time}")
    print(f"初始资金：{result.initial_capital}")
    print(f"最终资金：{result.statistics['final_capital']}")
    print(f"总收益率：{result.statistics['return']:.2f}%")
    print(f"总交易次数：{result.statistics['total_trades']}")
    print(f"胜率：{result.statistics['win_rate']*100:.2f}%")
    print(f"最大回撤：{result.statistics['max_drawdown']:.2f}%")
    
    # 保存回测结果
    result.save("backtest_result.json")
    print("回测结果已保存到backtest_result.json")


if __name__ == "__main__":
    asyncio.run(main()) 