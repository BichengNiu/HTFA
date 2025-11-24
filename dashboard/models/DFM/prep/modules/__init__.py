"""
数据准备模块包

这个包包含了数据准备过程中的各个模块化组件：
- config_constants: 配置和常量
- format_detection: 格式检测
- data_loader: 数据加载
- data_aligner: 数据对齐
- data_cleaner: 数据清理
- detrend_processor: 去趋势处理
"""

from dashboard.models.DFM.prep.modules.detrend_processor import (
    DetrendProcessor,
    DetrendError,
    InsufficientDataError,
    RegressionFailedError
)

__version__ = "2.0.0"
__author__ = "Data Preparation Module"

__all__ = [
    'DetrendProcessor',
    'DetrendError',
    'InsufficientDataError',
    'RegressionFailedError'
]
