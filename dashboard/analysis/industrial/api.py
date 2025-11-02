# -*- coding: utf-8 -*-
"""
工业分析模块 - 统一API接口

提供工业数据分析的前后端分离接口
"""

import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

# 使用动态导入避免循环依赖
# from dashboard.analysis.industrial.industrial_analysis import analyze_industrial_value as _analyze_industrial
# from dashboard.analysis.industrial.enterprise_analysis import analyze_enterprise_profit as _analyze_enterprise

logger = logging.getLogger(__name__)


def analyze_industrial_value(
    uploaded_file: Any,
    time_range: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    工业增加值分析

    Args:
        uploaded_file: 上传的Excel文件
        time_range: 时间范围 (start_date, end_date)

    Returns:
        dict: {
            'status': str,
            'message': str,
            'charts': Dict[str, Any],
            'tables': Dict[str, pd.DataFrame],
            'metadata': {...}
        }
    """
    try:
        logger.info("开始工业增加值分析...")

        # 动态导入
        from dashboard.analysis.industrial.industrial_analysis import industrial_analysis_main
        result = industrial_analysis_main(uploaded_file, time_range)

        if result is None:
            return {
                'status': 'error',
                'message': '分析失败',
                'charts': None,
                'tables': None,
                'metadata': None
            }

        return {
            'status': 'success',
            'message': '工业增加值分析完成',
            'charts': result.get('charts', {}),
            'tables': result.get('tables', {}),
            'metadata': result.get('metadata', {})
        }

    except Exception as e:
        logger.error(f"工业增加值分析失败: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'分析失败: {str(e)}',
            'charts': None,
            'tables': None,
            'metadata': None
        }


def analyze_enterprise_profit(
    uploaded_file: Any,
    time_range: Optional[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    工业企业利润分析

    Args:
        uploaded_file: 上传的Excel文件
        time_range: 时间范围

    Returns:
        dict: {
            'status': str,
            'message': str,
            'charts': Dict,
            'tables': Dict,
            'metadata': Dict
        }
    """
    try:
        logger.info("开始工业企业利润分析...")

        # 动态导入
        from dashboard.analysis.industrial.enterprise_analysis import enterprise_analysis_main
        result = enterprise_analysis_main(uploaded_file, time_range)

        if result is None:
            return {
                'status': 'error',
                'message': '分析失败',
                'charts': None,
                'tables': None,
                'metadata': None
            }

        return {
            'status': 'success',
            'message': '工业企业利润分析完成',
            'charts': result.get('charts', {}),
            'tables': result.get('tables', {}),
            'metadata': result.get('metadata', {})
        }

    except Exception as e:
        logger.error(f"工业企业利润分析失败: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'分析失败: {str(e)}',
            'charts': None,
            'tables': None,
            'metadata': None
        }


__all__ = [
    'analyze_industrial_value',
    'analyze_enterprise_profit'
]
