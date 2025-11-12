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
    train_end: Optional[str] = None,
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
        train_end: 训练集结束日期（用于标准化参数计算，避免数据泄露）
        progress_callback: 进度回调函数

    Returns:
        (k_factors, pca_analysis):
            - k_factors: 选定的因子数
            - pca_analysis: PCA分析结果字典（包含explained_variance, cumsum_variance, eigenvalues）

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

    # 步骤1: 数据标准化（避免数据泄露，仅使用训练集计算标准化参数）
    data_subset = data[selected_vars].copy()

    if train_end is not None:
        # 使用训练集数据计算标准化参数
        try:
            train_data = data_subset.loc[:train_end]
            global_mean = train_data.mean(axis=0)
            global_std = train_data.std(axis=0)
            logger.info(f"使用训练集数据计算标准化参数 (截止到 {train_end}, {len(train_data)} 样本)")
        except (KeyError, IndexError) as e:
            logger.warning(f"无法提取训练集进行标准化: {e}，回退到全数据集")
            global_mean = data_subset.mean(axis=0)
            global_std = data_subset.std(axis=0)
    else:
        # 如果没有提供train_end，使用全部数据
        global_mean = data_subset.mean(axis=0)
        global_std = data_subset.std(axis=0)
        logger.warning("未提供train_end参数，使用全部数据计算标准化参数（可能导致数据泄露）")

    # 处理标准差为0的列（避免除零错误）
    zero_std_cols = global_std[global_std == 0].index.tolist()
    if zero_std_cols:
        logger.warning(f"以下列标准差为0，将被移除: {zero_std_cols}")
        data_subset = data_subset.drop(columns=zero_std_cols)
        global_mean = global_mean.drop(labels=zero_std_cols)
        global_std = global_std.drop(labels=zero_std_cols)

    # 应用标准化
    data_standardized = (data_subset - global_mean) / global_std
    logger.info(f"数据标准化完成. Shape: {data_standardized.shape}")

    # 步骤2: 填充NaN
    data_for_pca = data_standardized.fillna(0)

    # 步骤3: 执行PCA分析（所有方法都需要PCA结果用于UI显示）
    logger.info(f"执行PCA分析 (method={method})...")
    pca = PCA()
    pca.fit(data_for_pca)

    explained_variance = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance)

    pca_analysis = {
        'explained_variance': explained_variance,
        'cumsum_variance': cumsum_variance,
        'eigenvalues': pca.explained_variance_
    }

    # 获取实际可用的主成分数量
    max_components = len(explained_variance)
    logger.info(f"实际可提取的主成分数量: {max_components}")

    # 方法1: 固定因子数
    if method == 'fixed':
        k = fixed_k
        # 验证fixed_k是否超过实际可用的主成分数量
        if k > max_components:
            logger.warning(
                f"固定因子数k={k}超过实际可提取的主成分数量{max_components}, "
                f"自动调整为k={max_components}"
            )
            k = max_components

        logger.info(f"使用固定因子数: k={k}")
        if k > 0:
            logger.info(f"对应累积方差: {cumsum_variance[k-1]:.1%}")
        return k, pca_analysis

    # 方法2: 累积方差贡献率
    if method == 'cumulative':
        k = np.argmax(cumsum_variance >= pca_threshold) + 1
        logger.info(
            f"PCA累积方差方法: 阈值={pca_threshold:.1%}, k={k}, "
            f"累积方差={cumsum_variance[k-1]:.1%}"
        )

    # 方法3: Elbow方法
    elif method == 'elbow':
        marginal_variance = np.diff(explained_variance)
        k = np.argmax(marginal_variance < elbow_threshold) + 1
        logger.info(f"Elbow方法: 阈值={elbow_threshold:.1%}, k={k}")

    else:
        raise ValueError(f"未知的因子选择方法: {method}")

    # 确保k在合理范围内
    k = max(1, min(k, len(selected_vars) - 1))
    logger.info(f"因子数选择完成: k={k}")

    return k, pca_analysis


__all__ = ['select_num_factors']
