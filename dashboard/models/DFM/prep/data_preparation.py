"""
DFM数据准备模块 - UI接口
这个模块为UI提供统一的数据准备接口，调用重构后的模块化数据处理系统。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import tempfile
import os

# 导入重构后的模块
from dashboard.models.DFM.prep.modules.main_data_processor import prepare_data as process_data_internal
from dashboard.models.DFM.prep.modules.mapping_manager import load_mappings

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(
    excel_path: str,
    target_freq: str = "W-FRI",
    target_sheet_name: str = "工业增加值同比增速_月度_同花顺",
    target_variable_name: str = "规模以上工业增加值:当月同比",
    consecutive_nan_threshold: int = 10,
    data_start_date: str = "2010-01-31",
    data_end_date: str = "2025-07-03",
    reference_sheet_name: str = "指标体系",
    reference_column_name: str = "指标名称"
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
    """
    执行DFM数据准备流程

    参数:
        excel_path: Excel文件路径
        target_freq: 目标频率
        target_sheet_name: 目标工作表名称
        target_variable_name: 目标变量名称
        consecutive_nan_threshold: 连续NaN阈值
        data_start_date: 开始日期
        data_end_date: 结束日期
        reference_sheet_name: 指标映射表名称
        reference_column_name: 参考列名称

    返回:
        Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
            - 最终对齐的周度数据 (DataFrame)
            - 变量到行业的映射 (Dict)
            - 合并的转换日志 (Dict)
            - 详细的移除日志 (List[Dict])
    """
    
    try:
        logger.info("开始DFM数据准备流程...")
        
        # 处理文件路径
        if hasattr(excel_path, 'read'):
            # 如果是文件对象，保存到临时文件
            excel_path.seek(0)  # 重置文件指针到开始位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(excel_path.read())
                excel_path = tmp_file.name
                logger.info(f"已将文件对象保存到临时文件: {excel_path}")
        
        # 执行数据处理
        logger.info(f"调用数据处理函数...")
        result = process_data_internal(
            excel_path=excel_path,
            target_freq=target_freq,
            target_sheet_name=target_sheet_name,
            target_variable_name=target_variable_name,
            consecutive_nan_threshold=consecutive_nan_threshold,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            reference_sheet_name=reference_sheet_name,
            reference_column_name=reference_column_name
        )

        # 解包返回结果
        processed_data, variable_mapping, transform_log, removal_log = result

        # 检查是否处理成功
        if processed_data is None:
            raise ValueError("数据处理失败，返回了None")

        logger.info(f"数据准备完成! 数据形状: {processed_data.shape}")

        # 直接返回与原始函数相同的格式
        return processed_data, variable_mapping, transform_log, removal_log
        
    except Exception as e:
        logger.error(f"数据准备过程中发生错误: {str(e)}")
        return None, None, None, None


# 导出的函数
__all__ = [
    'prepare_data',
    'load_mappings'
]
