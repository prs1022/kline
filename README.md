# KLine - 模块化比特币量化交易系统

一个高性能、可扩展的比特币量化交易系统，整合多源数据、AI策略和风险控制。

## 核心功能
- 多源数据聚合（Coinbase/Binance API、历史数据仓库）
- 策略工厂（经典策略库、AI策略生成器、参数优化）
- 事件驱动型回测引擎
- 风险控制系统
- 交易执行沙箱

## 技术栈
- Python 3.11+（asyncio异步架构）
- 数据处理：Polars + TA-Lib
- AI框架：PyTorch Lightning + Optuna
- 可视化：Plotly Dash + Grafana
- 部署：Docker + Kubernetes

## 项目结构
```
kline/
├── core/                 # 核心模块
│   ├── data/             # 数据聚合层
│   ├── strategy/         # 策略工厂
│   ├── backtest/         # 回测引擎
│   ├── risk/             # 风险控制
│   └── execution/        # 执行沙箱
├── models/               # AI模型
├── utils/                # 工具函数
├── config/               # 配置文件
├── tests/                # 测试用例
├── notebooks/            # 实验笔记本
└── docs/                 # 文档
```

## 安装
```bash
pip install -r requirements.txt
```

## 快速开始
```python
from kline.core.backtest import Backtest
from kline.core.strategy import TripleMACrossover
from kline.core.data import BinanceDataSource

# 初始化数据源
data_source = BinanceDataSource(symbol="BTC/USDT", timeframe="1h")

# 创建策略
strategy = TripleMACrossover(fast=5, medium=10, slow=20)

# 运行回测
backtest = Backtest(data_source, strategy)
results = backtest.run()
```

## 开发路线图
1. MVP版本开发（使用Mock数据）
2. 回测引擎开发
3. 策略工厂实现
4. 数据聚合层整合
5. 风险控制系统
6. 执行沙箱

## 贡献指南
请参阅[CONTRIBUTING.md](CONTRIBUTING.md)

## 许可证
MIT License 