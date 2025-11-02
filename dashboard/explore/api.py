# -*- coding: utf-8 -*-
"""
数据探索模块 - 统一API接口

提供时间序列分析功能的前后端分离接口
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging

# 注释掉具体导入，使用动态导入避免循环依赖
# from dashboard.explore.metrics.correlation import calculate_correlation as _calc_corr
# from dashboard.explore.metrics.dtw import calculate_dtw_distance as _calc_dtw
# from dashboard.explore.analysis.stationarity import test_stationarity as _test_stat
# from dashboard.explore.analysis.lead_lag import analyze_lead_lag as _analyze_ll

logger = logging.getLogger(__name__)


def calculate_correlation(
    data: pd.DataFrame,
    method: str = 'pearson',
    selected_columns: Optional[list] = None
) -> Dict[str, Any]:
    """
    计算相关性

    Args:
        data: 输入数据
        method: 相关性方法 ('pearson', 'spearman', 'kendall')
        selected_columns: 选择的列

    Returns:
        dict: {
            'status': str,
            'message': str,
            'correlation_matrix': pd.DataFrame,
            'metadata': {...}
        }
    """
    try:
        if selected_columns:
            data = data[selected_columns]

        # 使用pandas内置相关性计算
        corr_matrix = data.corr(method=method)

        return {
            'status': 'success',
            'message': f'成功计算 {method} 相关性',
            'correlation_matrix': corr_matrix,
            'metadata': {
                'method': method,
                'variables': len(corr_matrix)
            }
        }
    except Exception as e:
        logger.error(f"相关性计算失败: {e}")
        return {
            'status': 'error',
            'message': f'计算失败: {str(e)}',
            'correlation_matrix': None,
            'metadata': None
        }


def calculate_dtw(
    series1: pd.Series,
    series2: pd.Series
) -> Dict[str, Any]:
    """
    计算DTW距离

    Args:
        series1: 时间序列1
        series2: 时间序列2

    Returns:
        dict: {
            'status': str,
            'message': str,
            'distance': float,
            'metadata': {...}
        }
    """
    try:
        distance = _calc_dtw(series1, series2)

        return {
            'status': 'success',
            'message': f'DTW距离: {distance:.4f}',
            'distance': distance,
            'metadata': {
                'series1_length': len(series1),
                'series2_length': len(series2)
            }
        }
    except Exception as e:
        logger.error(f"DTW计算失败: {e}")
        return {
            'status': 'error',
            'message': f'计算失败: {str(e)}',
            'distance': None,
            'metadata': None
        }


def test_stationarity(
    series: pd.Series,
    test_type: str = 'adf'
) -> Dict[str, Any]:
    """
    平稳性检验

    Args:
        series: 时间序列
        test_type: 检验类型 ('adf', 'kpss')

    Returns:
        dict: {
            'status': str,
            'message': str,
            'is_stationary': bool,
            'test_statistic': float,
            'p_value': float
        }
    """
    try:
        result = _test_stat(series, test_type=test_type)

        return {
            'status': 'success',
            'message': '平稳性检验完成',
            'is_stationary': result['is_stationary'],
            'test_statistic': result['test_statistic'],
            'p_value': result['p_value']
        }
    except Exception as e:
        logger.error(f"平稳性检验失败: {e}")
        return {
            'status': 'error',
            'message': f'检验失败: {str(e)}',
            'is_stationary': None,
            'test_statistic': None,
            'p_value': None
        }


def analyze_lead_lag(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 12
) -> Dict[str, Any]:
    """
    领先滞后分析

    Args:
        series1: 时间序列1
        series2: 时间序列2
        max_lag: 最大滞后期

    Returns:
        dict: {
            'status': str,
            'message': str,
            'best_lag': int,
            'correlations': pd.Series
        }
    """
    try:
        result = _analyze_ll(series1, series2, max_lag=max_lag)

        return {
            'status': 'success',
            'message': f'最佳滞后期: {result["best_lag"]}',
            'best_lag': result['best_lag'],
            'correlations': result['correlations']
        }
    except Exception as e:
        logger.error(f"领先滞后分析失败: {e}")
        return {
            'status': 'error',
            'message': f'分析失败: {str(e)}',
            'best_lag': None,
            'correlations': None
        }


__all__ = [
    'calculate_correlation',
    'calculate_dtw',
    'test_stationarity',
    'analyze_lead_lag'
]
