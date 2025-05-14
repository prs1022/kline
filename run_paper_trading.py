import asyncio
import time
from datetime import datetime, timedelta

from kline.core.data import MockDataSource, TimeFrame
from kline.core.strategy import TripleMAStrategy
from kline.core.execution import PaperExecutor
from kline.core.risk import RiskManager, RiskLevel


async def position_callback(position):
    """仓位变化回调"""
    if position.exit_time:
        print(f"平仓: {position.symbol} {position.side.value} @ {position.exit_price:.2f}, "
              f"盈亏: {position.pnl:.2f} ({position.pnl_percentage:.2f}%)")
    else:
        print(f"开仓: {position.symbol} {position.side.value} @ {position.entry_price:.2f}, "
              f"数量: {position.amount:.6f}")


async def main():
    # 设置初始资金和交易对
    initial_balance = 10000.0
    symbol = "BTC/USDT"
    timeframe = TimeFrame.MINUTE_5
    
    # 创建模拟数据源
    data_source = MockDataSource(seed=42)
    
    # 创建策略
    strategy = TripleMAStrategy(fast_period=5, medium_period=10, slow_period=20)
    
    # 创建风险管理器
    risk_manager = RiskManager(
        max_position_size=0.2,
        max_positions=3,
        max_risk_per_trade=0.02,
        max_daily_loss=0.05,
        max_drawdown=0.15,
        risk_level=RiskLevel.MEDIUM
    )
    risk_manager.initialize(initial_balance)
    
    # 创建模拟执行器
    executor = PaperExecutor(
        name="paper_trader",
        initial_balance=initial_balance,
        commission=0.001,
        slippage=0.0005
    )
    
    # 注册回调函数
    executor.register_position_callback(position_callback)
    
    # 初始化策略
    strategy.start()
    
    print(f"=== 启动模拟交易 ===")
    print(f"交易对: {symbol}")
    print(f"初始资金: {initial_balance}")
    print(f"策略: {strategy.name}")
    print(f"风险级别: {risk_manager.risk_level}")
    
    # 模拟交易循环
    try:
        # 用于记录上次处理的时间
        last_process_time = None
        
        # 运行100个周期或按Ctrl+C停止
        for i in range(100):
            # 获取当前时间
            now = datetime.now()
            
            # 计算数据开始时间和结束时间
            end_time = now
            start_time = end_time - timedelta(hours=24)  # 获取过去24小时数据
            
            # 获取历史K线数据
            data = await data_source.fetch_kline(symbol, timeframe, start_time, end_time)
            
            # 更新最新价格
            latest_price = data["close"].iloc[-1]
            timestamp = data.index[-1]
            
            # 更新执行器中的价格
            executor.update_ticker(symbol, latest_price, timestamp)
            
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 当前价格: {latest_price:.2f}")
            print(f"当前资金: {executor.get_balance():.2f}")
            
            # 计算指标
            data_with_indicators = strategy.calculate_indicators(data)
            
            # 生成信号
            signals = strategy.generate_signals(data_with_indicators)
            
            # 处理信号
            for signal in signals:
                # 只处理最新的信号
                if last_process_time is None or signal.time > last_process_time:
                    print(f"信号: {signal.side.value} {signal.symbol} @ {signal.price:.2f}")
                    
                    # 检查是否可以开仓
                    if risk_manager.can_open_position(signal.price, signal.stop_loss):
                        # 计算仓位大小
                        position_size = risk_manager.calculate_position_size(signal.price, signal.stop_loss)
                        
                        # 创建订单
                        if signal.side.value == "buy":
                            await executor.create_market_order(symbol, signal.side, position_size)
                        else:
                            await executor.create_market_order(symbol, signal.side, position_size)
                    else:
                        print("风险控制阻止了此次交易")
            
            # 更新最后处理时间
            last_process_time = timestamp
            
            # 每个周期之间暂停3秒，模拟实时交易
            await asyncio.sleep(3)
            
    except KeyboardInterrupt:
        print("\n用户中断，停止模拟交易")
    finally:
        # 结束时输出摘要信息
        print("\n=== 模拟交易摘要 ===")
        print(f"最终资金: {executor.get_balance():.2f}")
        print(f"收益率: {(executor.get_balance() / initial_balance - 1) * 100:.2f}%")
        
        # 关闭资源
        await executor.close()
        await data_source.close()
        strategy.stop()


if __name__ == "__main__":
    asyncio.run(main()) 