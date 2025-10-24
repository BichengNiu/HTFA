# -*- coding: utf-8 -*-
"""
预计算引擎模块

提供数据预计算功能以优化训练性能：
- 预计算标准化参数（均值、标准差）
- 预计算协方差矩阵
- 缓存预计算结果
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PrecomputedContext:
    """预计算上下文数据类

    存储预计算的统计量和矩阵，供训练流程复用
    """
    # 标准化统计量: {变量名: (均值, 标准差)}
    standardization_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # 协方差矩阵
    covariance_matrix: Optional[np.ndarray] = None

    # 相关矩阵
    correlation_matrix: Optional[np.ndarray] = None

    # 数据形状 (样本数, 变量数)
    data_shape: Tuple[int, int] = (0, 0)

    # 变量名称列表（保持顺序）
    variable_names: list = field(default_factory=list)


class PrecomputeEngine:
    """预计算引擎

    负责预计算训练过程中重复使用的统计量和矩阵，
    减少重复计算，提升训练效率。
    """

    def __init__(self):
        """初始化预计算引擎"""
        self.logger = logger

    def precompute_standardization_stats(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        """预计算标准化统计量

        Args:
            data: 输入数据

        Returns:
            Dict: {变量名: (均值, 标准差)}
        """
        stats = {}

        for col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()

            # 处理异常情况
            if pd.isna(mean_val):
                mean_val = 0.0
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
                self.logger.warning(
                    f"变量 {col} 标准差为0或NaN，设为1.0"
                )

            stats[col] = (float(mean_val), float(std_val))

        self.logger.info(f"预计算标准化统计量完成，变量数: {len(stats)}")

        return stats

    def precompute_covariance(
        self,
        data: pd.DataFrame,
        standardized: bool = False
    ) -> np.ndarray:
        """预计算协方差矩阵

        Args:
            data: 输入数据
            standardized: 数据是否已标准化（若是则计算相关矩阵）

        Returns:
            np.ndarray: 协方差矩阵 (n_vars, n_vars)
        """
        try:
            # 移除含NaN的行
            clean_data = data.dropna()

            if len(clean_data) < 2:
                self.logger.warning("有效样本数不足，返回零矩阵")
                n_vars = len(data.columns)
                return np.zeros((n_vars, n_vars))

            # 计算协方差矩阵
            cov_matrix = clean_data.cov().values

            # 确保矩阵对称
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

            matrix_type = "相关矩阵" if standardized else "协方差矩阵"
            self.logger.info(
                f"预计算{matrix_type}完成，形状: {cov_matrix.shape}, "
                f"有效样本数: {len(clean_data)}"
            )

            return cov_matrix

        except Exception as e:
            self.logger.error(f"预计算协方差矩阵失败: {e}")
            n_vars = len(data.columns)
            return np.zeros((n_vars, n_vars))

    def precompute_correlation(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """预计算相关系数矩阵

        Args:
            data: 输入数据

        Returns:
            np.ndarray: 相关系数矩阵 (n_vars, n_vars)
        """
        try:
            # 移除含NaN的行
            clean_data = data.dropna()

            if len(clean_data) < 2:
                self.logger.warning("有效样本数不足，返回单位矩阵")
                n_vars = len(data.columns)
                return np.eye(n_vars)

            # 计算相关系数矩阵
            corr_matrix = clean_data.corr().values

            # 确保矩阵对称且对角线为1
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix, 1.0)

            self.logger.info(
                f"预计算相关矩阵完成，形状: {corr_matrix.shape}, "
                f"有效样本数: {len(clean_data)}"
            )

            return corr_matrix

        except Exception as e:
            self.logger.error(f"预计算相关矩阵失败: {e}")
            n_vars = len(data.columns)
            return np.eye(n_vars)

    def precompute_all(
        self,
        data: pd.DataFrame,
        compute_covariance: bool = True,
        compute_correlation: bool = True
    ) -> PrecomputedContext:
        """预计算所有统计量

        Args:
            data: 输入数据
            compute_covariance: 是否计算协方差矩阵
            compute_correlation: 是否计算相关系数矩阵

        Returns:
            PrecomputedContext: 预计算上下文对象
        """
        self.logger.info(f"开始预计算，数据形状: {data.shape}")

        # 预计算标准化统计量
        stats = self.precompute_standardization_stats(data)

        # 预计算协方差矩阵
        cov_matrix = None
        if compute_covariance:
            cov_matrix = self.precompute_covariance(data, standardized=False)

        # 预计算相关系数矩阵
        corr_matrix = None
        if compute_correlation:
            corr_matrix = self.precompute_correlation(data)

        # 构建上下文对象
        context = PrecomputedContext(
            standardization_stats=stats,
            covariance_matrix=cov_matrix,
            correlation_matrix=corr_matrix,
            data_shape=data.shape,
            variable_names=list(data.columns)
        )

        self.logger.info("预计算完成")

        return context

    def validate_context(
        self,
        context: PrecomputedContext,
        data: pd.DataFrame
    ) -> bool:
        """验证预计算上下文是否适用于给定数据

        Args:
            context: 预计算上下文
            data: 待验证的数据

        Returns:
            bool: 是否有效
        """
        # 检查变量名称是否匹配
        if list(data.columns) != context.variable_names:
            self.logger.warning("变量名称不匹配")
            return False

        # 检查数据形状是否兼容（至少变量数相同）
        if data.shape[1] != context.data_shape[1]:
            self.logger.warning("变量数不匹配")
            return False

        # 检查统计量是否存在
        if not context.standardization_stats:
            self.logger.warning("标准化统计量为空")
            return False

        self.logger.info("预计算上下文验证通过")
        return True

    def apply_standardization(
        self,
        data: pd.DataFrame,
        stats: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """使用预计算的统计量标准化数据

        Args:
            data: 原始数据
            stats: 预计算的统计量 {变量名: (均值, 标准差)}

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        standardized = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col not in stats:
                self.logger.warning(f"变量 {col} 无预计算统计量，跳过标准化")
                standardized[col] = data[col]
                continue

            mean_val, std_val = stats[col]
            standardized[col] = (data[col] - mean_val) / std_val

        return standardized
