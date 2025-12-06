# -*- coding: utf-8 -*-
"""
DFM数据准备并行处理模块

提供数据准备各环节的并行处理能力
"""

from dashboard.models.DFM.prep.parallel.adf_processor import (
    parallel_adf_check,
    serial_adf_check,
)
from dashboard.models.DFM.prep.parallel.missing_detector import (
    parallel_detect_consecutive_nans,
    serial_detect_consecutive_nans,
)
from dashboard.models.DFM.prep.parallel.frequency_processor import (
    parallel_process_frequencies,
    serial_process_frequencies,
)
from dashboard.models.DFM.prep.parallel.sheet_reader import (
    parallel_read_sheets,
    serial_read_sheets,
)

__all__ = [
    # ADF检验
    'parallel_adf_check',
    'serial_adf_check',
    # 缺失值检测
    'parallel_detect_consecutive_nans',
    'serial_detect_consecutive_nans',
    # 频率处理
    'parallel_process_frequencies',
    'serial_process_frequencies',
    # Sheet读取
    'parallel_read_sheets',
    'serial_read_sheets',
]
