import asyncio
import pandas as pd
from datetime import datetime, timedelta

from kline.core.data import MockDataSource
from kline.core.strategy import TripleMAStrategy
from kline.utils.optimizer import StrategyOptimizer


async def main():
    # 创建模拟数据源
    data_source = MockDataSource(seed=42)
    
    # 获取模拟数据
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # 获取过去60天的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(days=60)
    
    print(f"获取数据：{symbol} {timeframe} from {start_time} to {end_time}")
    data = await data_source.fetch_kline(symbol, timeframe, start_time, end_time)
    
    # 设置默认参数
    default_params = {
        "name": "TripleMA_Optimized"
    }
    
    # 设置参数空间
    param_space = {
        "fast_period": {
            "type": "int",
            "low": 3,
            "high": 20,
            "step": 1
        },
        "medium_period": {
            "type": "int",
            "low": 10,
            "high": 50,
            "step": 2
        },
        "slow_period": {
            "type": "int",
            "low": 30,
            "high": 100,
            "step": 5
        }
    }
    
    # 创建优化器
    optimizer = StrategyOptimizer(
        strategy_class=TripleMAStrategy,
        default_params=default_params,
        param_space=param_space,
        data=data,
        metric="return",  # 优化目标：收益率
        maximize=True,    # 最大化目标
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        position_size=0.1,
        symbol=symbol,
        timeframe=timeframe,
        n_trials=30,      # 优化次数
        n_jobs=1          # 并行任务数
    )
    
    # 运行优化
    print("开始参数优化...")
    best_params, best_result = optimizer.optimize()
    
    # 打印优化结果
    print("\n===== 优化结果 =====")
    print(f"最佳参数: {best_params}")
    print(f"最佳收益率: {best_result.statistics['return']:.2f}%")
    
    # 保存优化结果
    optimizer.save_results("optimization_result.json")
    print("优化结果已保存到optimization_result.json")


if __name__ == "__main__":
    asyncio.run(main()) 