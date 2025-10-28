# -*- coding: utf-8 -*-
"""
统一数据模型定义

整合所有train模块使用的数据类，确保类型一致性和可维护性
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ==================== 评估指标相关 ====================

@dataclass
class EvaluationMetrics:
    """评估指标"""
    is_rmse: float = np.inf
    oos_rmse: float = np.inf
    is_mae: float = np.inf
    oos_mae: float = np.inf
    is_hit_rate: float = -np.inf
    oos_hit_rate: float = -np.inf


@dataclass
class MetricsResult:
    """详细评估指标结果"""
    is_rmse: float
    is_mae: float
    is_hit_rate: float
    oos_rmse: float
    oos_mae: float
    oos_hit_rate: float
    aligned_data: Optional[pd.DataFrame] = None


# ==================== DFM模型相关 ====================

@dataclass
class DFMModelResult:
    """DFM模型完整结果（统一版）

    整合了原DFMResults和DFMModelResult的功能，
    提供统一的数据模型，避免重复和转换开销。

    字段命名说明：
    - A: 状态转移矩阵
    - Q: 状态噪声协方差
    - H: 观测矩阵（因子载荷）
    - R: 观测噪声协方差
    """
    # EM估计参数（统一命名：A/Q/H/R）
    A: np.ndarray = None  # 状态转移矩阵
    Q: np.ndarray = None  # 状态噪声协方差
    H: np.ndarray = None  # 观测矩阵（因子载荷）
    R: np.ndarray = None  # 观测噪声协方差

    # 卡尔曼滤波结果
    factors: np.ndarray = None  # 因子时间序列（滤波）
    factors_smooth: np.ndarray = None  # 平滑因子
    kalman_gains_history: Optional[List[np.ndarray]] = None  # 卡尔曼增益历史（用于新闻分解）

    # 预测结果
    forecast_is: np.ndarray = None  # 样本内预测
    forecast_oos: np.ndarray = None  # 样本外预测

    # 训练信息
    converged: bool = False
    iterations: int = 0
    log_likelihood: float = -np.inf


# ==================== 卡尔曼滤波相关 ====================

@dataclass
class KalmanFilterResult:
    """卡尔曼滤波结果"""
    x_filtered: np.ndarray      # 滤波状态估计
    P_filtered: np.ndarray      # 滤波协方差
    x_predicted: np.ndarray     # 预测状态估计
    P_predicted: np.ndarray     # 预测协方差
    loglikelihood: float        # 对数似然
    innovation: np.ndarray      # 新息序列
    kalman_gains_history: Optional[List[np.ndarray]] = None  # 卡尔曼增益历史（每个时刻的K_t矩阵）


@dataclass
class KalmanSmootherResult:
    """卡尔曼平滑结果"""
    x_smoothed: np.ndarray      # 平滑状态估计
    P_smoothed: np.ndarray      # 平滑协方差
    P_lag_smoothed: np.ndarray  # 滞后协方差


# ==================== 变量选择相关 ====================

@dataclass
class SelectionResult:
    """变量选择结果"""
    selected_variables: List[str]  # 最终选中的变量列表
    selection_history: List[Dict]  # 选择历史记录
    final_score: Tuple[float, float]  # 最终得分 (HR, -RMSE)
    total_evaluations: int  # 总评估次数
    svd_error_count: int  # SVD错误次数


# ==================== 训练结果相关 ====================

@dataclass
class TrainingResult:
    """训练结果"""
    # 变量选择结果
    selected_variables: List[str] = field(default_factory=list)
    selection_history: List[Dict] = field(default_factory=list)

    # 因子数选择结果
    k_factors: int = 0
    factor_selection_method: str = 'fixed'
    pca_analysis: Optional[Dict] = None

    # 模型结果
    model_result: Optional[DFMModelResult] = None

    # 评估指标
    metrics: Optional[EvaluationMetrics] = None

    # 训练统计
    total_evaluations: int = 0
    svd_error_count: int = 0
    training_time: float = 0.0

    # 导出文件路径
    export_files: Optional[Dict[str, str]] = None

    # 数据统计
    target_mean_original: float = 0.0
    target_std_original: float = 1.0

    # 输出路径
    output_dir: Optional[str] = None

    @classmethod
    def build(
        cls,
        selected_variables: List[str],
        selection_history: List[Dict],
        k_factors: int,
        factor_selection_method: str,
        pca_analysis: Optional[Dict],
        model_result: DFMModelResult,
        metrics: EvaluationMetrics,
        total_evaluations: int,
        svd_error_count: int,
        training_time: float,
        output_dir: Optional[str] = None
    ) -> 'TrainingResult':
        """
        构建训练结果对象

        Args:
            selected_variables: 选定的变量列表
            selection_history: 变量选择历史
            k_factors: 因子数
            factor_selection_method: 因子选择方法
            pca_analysis: PCA分析结果
            model_result: 模型结果
            metrics: 评估指标
            total_evaluations: 总评估次数
            svd_error_count: SVD错误次数
            training_time: 训练时间（秒）
            output_dir: 输出目录

        Returns:
            TrainingResult: 训练结果对象
        """
        return cls(
            selected_variables=selected_variables,
            selection_history=selection_history,
            k_factors=k_factors,
            factor_selection_method=factor_selection_method,
            pca_analysis=pca_analysis,
            model_result=model_result,
            metrics=metrics,
            total_evaluations=total_evaluations,
            svd_error_count=svd_error_count,
            training_time=training_time,
            output_dir=output_dir
        )


# ==================== 导出所有模型 ====================

__all__ = [
    # 评估指标
    'EvaluationMetrics',
    'MetricsResult',

    # DFM模型（已合并DFMResults到DFMModelResult）
    'DFMModelResult',

    # 卡尔曼滤波
    'KalmanFilterResult',
    'KalmanSmootherResult',

    # 变量选择
    'SelectionResult',

    # 训练结果
    'TrainingResult',
]
