import os
import sys
from pathlib import Path
from loguru import logger

from kline.config.config import BASE_DIR


def setup_logger(log_level="INFO", log_file=True, log_console=True):
    """
    设置日志记录器
    
    Args:
        log_level: 日志级别
        log_file: 是否记录到文件
        log_console: 是否输出到控制台
    """
    # 首先移除默认处理程序
    logger.remove()
    
    # 创建日志目录
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 添加控制台处理程序
    if log_console:
        logger.add(sys.stderr, level=log_level)
    
    # 添加文件处理程序
    if log_file:
        log_file_path = log_dir / "kline_{time:YYYY-MM-DD}.log"
        logger.add(
            log_file_path,
            rotation="00:00",  # 每天 00:00 创建一个新文件
            retention="7 days",  # 保留7天的日志
            level=log_level,
            compression="zip",  # 压缩旧的日志文件
            enqueue=True,  # 多进程安全
        )
    
    # 设置异常捕获
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="ERROR",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    
    return logger


# 导出日志实例
log = setup_logger()


def get_logger(name=None):
    """获取命名日志器"""
    return logger.bind(name=name) 