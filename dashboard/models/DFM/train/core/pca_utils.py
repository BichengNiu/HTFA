# -*- coding: utf-8 -*-
"""
PCA工具模块

提供因子数选择相关的PCA分析功能
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Callable
from sklearn.decomposition import PCA
from dashboard.models.DFM.train.utils.logger import get_logger

logger = get_logger(__name__)


def select_num_factors(
    data: pd.DataFrame,
    selected_vars: List[str],
    method: str,
    fixed_k: int,
    pca_threshold: float = 0.9,
    elbow_threshold: float = 0.1,
    progress_callback: Optional[Callable] = None
) -> Tuple[int, Optional[Dict]]:
    """
    因子数选择

    支持三种方法：
    1. fixed: 固定因子数
    2. cumulative: PCA累积方差贡献率
    3. elbow: Elbow方法（边际方差阈值）

    Args:
        data: 完整数据 (DataFrame)
        selected_vars: 选中的变量列表
        method: 选择方法 ('fixed', 'cumulative', 'elbow')
        fixed_k: 固定因子数（method='fixed'时使用）
        pca_threshold: PCA累积方差阈值（method='cumulative'时使用，默认0.9）
        elbow_threshold: Elbow边际方差阈值（method='elbow'时使用，默认0.1）
        progress_callback: 进度回调函数

    Returns:
        (k_factors, pca_analysis):
            - k_factors: 选定的因子数
            - pca_analysis: PCA分析结果字典（method='fixed'时为None）

    Raises:
        ValueError: 如果method不在支持的方法中

    Examples:
        >>> # 固定因子数
        >>> k, _ = select_num_factors(data, vars, 'fixed', fixed_k=3)

        >>> # PCA累积方差
        >>> k, analysis = select_num_factors(
        ...     data, vars, 'cumulative',
        ...     fixed_k=None, pca_threshold=0.85
        ... )
        >>> print(f"选定因子数: {k}, 累积方差: {analysis['cumsum_variance'][k-1]:.1%}")
    """
    logger.info("=" * 60)
    logger.info("因子数选择")
    logger.info("=" * 60)

    if progress_callback:
        progress_callback("[SELECTION] 开始因子数选择")

    # 方法1: 固定因子数
    if method == 'fixed':
        k = fixed_k
        logger.info(f"使用固定因子数: k={k}")
        if progress_callback:
            progress_callback(f"[SELECTION] 使用固定因子数: k={k}")
        return k, None

    # 方法2/3: 基于PCA分析
    logger.info(f"执行PCA分析 (method={method})...")

    # 准备数据（填充NaN）
    data_for_pca = data[selected_vars].fillna(0)

    # PCA分析
    pca = PCA()
    pca.fit(data_for_pca)

    explained_variance = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance)

    pca_analysis = {
        'explained_variance': explained_variance,
        'cumsum_variance': cumsum_variance,
        'eigenvalues': pca.explained_variance_
    }

    # 方法2: 累积方差贡献率
    if method == 'cumulative':
        k = np.argmax(cumsum_variance >= pca_threshold) + 1
        logger.info(
            f"PCA累积方差方法: 阈值={pca_threshold:.1%}, k={k}, "
            f"累积方差={cumsum_variance[k-1]:.1%}"
        )
        if progress_callback:
            progress_callback(
                f"[SELECTION] PCA选择因子数: k={k} "
                f"(累积方差={cumsum_variance[k-1]:.1%})"
            )

    # 方法3: Elbow方法
    elif method == 'elbow':
        marginal_variance = np.diff(explained_variance)
        k = np.argmax(marginal_variance < elbow_threshold) + 1
        logger.info(f"Elbow方法: 阈值={elbow_threshold:.1%}, k={k}")
        if progress_callback:
            progress_callback(f"[SELECTION] Elbow选择因子数: k={k}")

    else:
        raise ValueError(f"未知的因子选择方法: {method}")

    # 确保k在合理范围内
    k = max(1, min(k, len(selected_vars) - 1))
    logger.info(f"因子数选择完成: k={k}")

    return k, pca_analysis


__all__ = ['select_num_factors']
