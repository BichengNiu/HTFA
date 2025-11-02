"""
数据准备模块包

这个包包含了数据准备过程中的各个模块化组件：
- config_constants: 配置和常量
- format_detection: 格式检测
- mapping_manager: 映射管理
- stationarity_processor: 平稳性处理
- data_loader: 数据加载
- data_aligner: 数据对齐
- data_cleaner: 数据清理
- main_data_processor: 主数据处理器
"""

__version__ = "1.0.0"
__author__ = "Data Preparation Module"

# 导入主要接口
from dashboard.models.DFM.prep.modules.main_data_processor import prepare_data, prepare_data_from_dataframe

__all__ = [
    'prepare_data',
    'prepare_data_from_dataframe'
]
