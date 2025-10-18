"""
Logging Configuration
统一的日志配置系统
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None
) -> None:
    """
    配置全局日志系统
    
    Args:
        log_file: 日志文件路径（None则只输出到控制台）
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string: 自定义格式字符串
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的handlers
    root_logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称（通常使用 __name__）
        
    Returns:
        配置好的logger实例
    """
    return logging.getLogger(name)