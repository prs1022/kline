import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import asyncio
from copy import deepcopy
import json
import concurrent.futures
from pathlib import Path

from kline.core.strategy.base import Strategy
from kline.core.backtest.engine import Backtest, BacktestResult
from kline.utils.logger import get_logger

class StrategyOptimizer:
    """策略参数优化器"""
    
    def __init__(
        self,
        strategy_class: type,
        default_params: Dict[str, Any],
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "return",
        maximize: bool = True,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.1,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        n_trials: int = 100,
        n_jobs: int = 1,
    ):
        """
        初始化优化器
        
        Args:
            strategy_class: 策略类
            default_params: 默认参数
            param_space: 参数空间
            data: 回测数据
            metric: 优化指标
            maximize: 是否最大化指标
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
            position_size: 仓位大小
            symbol: 交易对符号
            timeframe: 时间周期
            n_trials: 优化次数
            n_jobs: 并行任务数
        """
        self.strategy_class = strategy_class
        self.default_params = default_params
        self.param_space = param_space
        self.data = data.copy()
        self.metric = metric
        self.maximize = maximize
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        self.logger = get_logger("optimizer")
        self.study = None
        self.best_params = None
        self.best_result = None
        
        # 支持的指标
        self.supported_metrics = [
            "return", "total_pnl", "win_rate", "profit_factor", 
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "expectancy"
        ]
        
        # 检查指标是否有效
        if metric not in self.supported_metrics:
            raise ValueError(f"Invalid metric: {metric}. Supported metrics: {self.supported_metrics}")
            
    def _create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        创建试验参数
        
        Args:
            trial: 试验对象
            
        Returns:
            参数字典
        """
        params = deepcopy(self.default_params)
        
        for param_name, param_info in self.param_space.items():
            param_type = param_info["type"]
            
            if param_type == "int":
                low = param_info.get("low", 1)
                high = param_info.get("high", 100)
                step = param_info.get("step", 1)
                value = trial.suggest_int(param_name, low, high, step)
                
            elif param_type == "float":
                low = param_info.get("low", 0.0)
                high = param_info.get("high", 1.0)
                log = param_info.get("log", False)
                value = trial.suggest_float(param_name, low, high, log=log)
                
            elif param_type == "categorical":
                choices = param_info.get("choices", [])
                value = trial.suggest_categorical(param_name, choices)
                
            else:
                continue
                
            params[param_name] = value
            
        return params
        
    def _run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            params: 参数字典
            
        Returns:
            回测结果
        """
        # 创建策略实例
        strategy = self.strategy_class(**params)
        
        # 创建回测实例
        backtest = Backtest(
            data=self.data,
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            position_size=self.position_size,
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # 运行回测
        result = backtest.run()
        
        return result.statistics
        
    def _objective(self, trial: optuna.Trial) -> float:
        """
        优化目标函数
        
        Args:
            trial: 试验对象
            
        Returns:
            优化指标值
        """
        # 生成参数
        params = self._create_trial_params(trial)
        
        # 运行回测
        statistics = self._run_backtest(params)
        
        # 获取指标值
        value = statistics.get(self.metric, 0.0)
        
        # 处理最大化/最小化
        if self.metric == "max_drawdown":
            # 对于最大回撤，我们希望最小化它，因此取负值
            return -value if self.maximize else value
            
        return value if self.maximize else -value
        
    def optimize(self) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        运行优化
        
        Returns:
            最优参数和回测结果
        """
        self.logger.info(f"Starting optimization with {self.n_trials} trials")
        
        # 创建优化研究
        direction = optuna.study.StudyDirection.MAXIMIZE if self.maximize else optuna.study.StudyDirection.MINIMIZE
        self.study = optuna.create_study(direction=direction)
        
        # 运行优化
        self.study.optimize(self._objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # 获取最优参数
        self.best_params = {**self.default_params, **self.study.best_params}
        
        # 用最优参数运行回测
        strategy = self.strategy_class(**self.best_params)
        
        backtest = Backtest(
            data=self.data,
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            position_size=self.position_size,
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        self.best_result = backtest.run()
        
        self.logger.info(f"Optimization completed with best {self.metric}: {self.study.best_value}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.best_result
        
    def get_trials_df(self) -> pd.DataFrame:
        """
        获取试验结果DataFrame
        
        Returns:
            试验结果DataFrame
        """
        if self.study is None:
            return pd.DataFrame()
            
        trials = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = trial.params.copy()
                params["value"] = trial.value
                trials.append(params)
                
        return pd.DataFrame(trials)
        
    def save_results(self, file_path: str):
        """
        保存优化结果
        
        Args:
            file_path: 文件路径
        """
        if self.study is None or self.best_params is None or self.best_result is None:
            self.logger.warning("No optimization results to save")
            return
            
        # 创建结果字典
        result_dict = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "metric": self.metric,
            "maximize": self.maximize,
            "n_trials": self.n_trials,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "initial_capital": self.initial_capital,
            "commission": self.commission,
            "slippage": self.slippage,
            "position_size": self.position_size,
            "strategy": self.strategy_class.__name__,
            "statistics": self.best_result.statistics,
        }
        
        # 创建目录
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        with open(file_path, "w") as f:
            json.dump(result_dict, f, indent=2)
            
        # 保存试验结果
        trials_df = self.get_trials_df()
        if not trials_df.empty:
            trials_file = Path(file_path).with_name(f"{Path(file_path).stem}_trials.csv")
            trials_df.to_csv(trials_file, index=False)
            
        # 保存回测结果
        if self.best_result is not None:
            result_file = Path(file_path).with_name(f"{Path(file_path).stem}_backtest.json")
            self.best_result.save(str(result_file))
            
        self.logger.info(f"Optimization results saved to {file_path}")
        
    def plot_optimization_history(self, file_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            file_path: 文件路径，如果不为None，则保存图像
        """
        if self.study is None:
            self.logger.warning("No optimization results to plot")
            return
            
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # 绘制优化历史
            fig1 = plot_optimization_history(self.study)
            
            # 绘制参数重要性
            fig2 = plot_param_importances(self.study)
            
            if file_path is not None:
                # 保存图像
                history_file = Path(file_path).with_name(f"{Path(file_path).stem}_history.png")
                importance_file = Path(file_path).with_name(f"{Path(file_path).stem}_importance.png")
                
                fig1.write_image(str(history_file))
                fig2.write_image(str(importance_file))
                
            return fig1, fig2
            
        except ImportError:
            self.logger.warning("matplotlib or plotly is not installed, cannot plot")
            return None 