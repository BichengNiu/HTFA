# -*- coding: utf-8 -*-
"""
DFM训练器 - 合并评估器和流水线逻辑(方案B精简架构)

包含:
- ModelEvaluator: 模型评估器(内部类)
- DFMTrainer: 主训练器(包含两阶段训练流程)

参考:
- dashboard/DFM/train_model/dfm_core.py (评估逻辑)
- dashboard/DFM/train_model/tune_dfm.py (训练流程)
"""

import os
import sys
import time
import random
import logging
import multiprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

@dataclass
class EvaluationMetrics:
    """评估指标"""
    is_rmse: float = np.inf
    oos_rmse: float = np.inf
    is_hit_rate: float = -np.inf
    oos_hit_rate: float = -np.inf
    is_correlation: float = -np.inf
    oos_correlation: float = -np.inf

    def to_tuple(self) -> Tuple:
        """转换为9元组格式(兼容train_model)"""
        return (
            self.is_rmse,
            self.oos_rmse,
            None,  # placeholder
            None,  # placeholder
            self.is_hit_rate,
            self.oos_hit_rate,
            False,  # is_svd_error
            None,  # placeholder
            None   # placeholder
        )


@dataclass
class DFMModelResult:
    """DFM模型结果"""
    # EM估计参数
    A: np.ndarray = None  # 状态转移矩阵
    Q: np.ndarray = None  # 状态噪声协方差
    H: np.ndarray = None  # 观测矩阵(因子载荷)
    R: np.ndarray = None  # 观测噪声协方差

    # 卡尔曼滤波结果
    factors: np.ndarray = None  # 因子时间序列
    factors_smooth: np.ndarray = None  # 平滑因子

    # 预测结果
    forecast_is: np.ndarray = None  # 样本内预测
    forecast_oos: np.ndarray = None  # 样本外预测

    # 其他信息
    converged: bool = False
    iterations: int = 0
    log_likelihood: float = -np.inf


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

    # 数据统计
    target_mean_original: float = 0.0
    target_std_original: float = 1.0

    # 输出路径
    output_dir: Optional[str] = None


# ==================== ModelEvaluator (内部类) ====================

class ModelEvaluator:
    """
    模型评估器(内部类)

    功能:
    - calculate_rmse: RMSE计算
    - calculate_hit_rate: 命中率计算
    - calculate_correlation: 相关系数计算
    - evaluate: 完整评估流程
    """

    def __init__(self):
        """初始化评估器"""
        pass

    def calculate_rmse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        计算RMSE

        Args:
            predictions: 预测值
            actuals: 实际值

        Returns:
            RMSE值
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return np.inf

        if len(predictions) != len(actuals):
            logger.warning(f"预测值和实际值长度不一致: {len(predictions)} vs {len(actuals)}")
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if not valid_mask.any():
            return np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]

        mse = np.mean((predictions_clean - actuals_clean) ** 2)
        rmse = np.sqrt(mse)

        return float(rmse)

    def calculate_hit_rate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        previous_values: np.ndarray
    ) -> float:
        """
        计算命中率(方向预测准确率)

        Args:
            predictions: 预测值
            actuals: 实际值
            previous_values: 前一期值

        Returns:
            命中率(0-100)
        """
        if len(predictions) == 0 or len(actuals) == 0 or len(previous_values) == 0:
            return -np.inf

        # 确保长度一致
        min_len = min(len(predictions), len(actuals), len(previous_values))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        previous_values = previous_values[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isnan(previous_values))
        if not valid_mask.any():
            return -np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]
        previous_clean = previous_values[valid_mask]

        # 计算方向
        pred_direction = np.sign(predictions_clean - previous_clean)
        actual_direction = np.sign(actuals_clean - previous_clean)

        # 计算命中率
        hits = (pred_direction == actual_direction).sum()
        total = len(pred_direction)

        if total == 0:
            return -np.inf

        hit_rate = (hits / total) * 100.0

        return float(hit_rate)

    def calculate_correlation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        计算相关系数

        Args:
            predictions: 预测值
            actuals: 实际值

        Returns:
            相关系数
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return -np.inf

        # 确保长度一致
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if not valid_mask.any():
            return -np.inf

        predictions_clean = predictions[valid_mask]
        actuals_clean = actuals[valid_mask]

        if len(predictions_clean) < 2:
            return -np.inf

        # 计算相关系数
        corr = np.corrcoef(predictions_clean, actuals_clean)[0, 1]

        return float(corr) if np.isfinite(corr) else -np.inf

    def evaluate(
        self,
        model_result: DFMModelResult,
        target_data: pd.Series,
        train_end_date: str,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None
    ) -> EvaluationMetrics:
        """
        完整评估流程

        Args:
            model_result: 模型结果
            target_data: 目标变量数据
            train_end_date: 训练期结束日期
            validation_start: 验证期开始日期
            validation_end: 验证期结束日期

        Returns:
            EvaluationMetrics对象
        """
        metrics = EvaluationMetrics()

        try:
            # 分割样本内和样本外
            train_data = target_data.loc[:train_end_date]

            if validation_start and validation_end:
                val_data = target_data.loc[validation_start:validation_end]
            else:
                # 如果没有指定验证期,使用训练期之后的所有数据
                val_data = target_data.loc[train_end_date:]
                if len(val_data) > 0:
                    val_data = val_data.iloc[1:]  # 排除训练期最后一天

            # 样本内评估
            if model_result.forecast_is is not None and len(train_data) > 0:
                # 对齐长度
                min_len = min(len(model_result.forecast_is), len(train_data))
                forecast_is = model_result.forecast_is[:min_len]
                actual_is = train_data.values[:min_len]

                # RMSE
                metrics.is_rmse = self.calculate_rmse(forecast_is, actual_is)

                # 相关系数
                metrics.is_correlation = self.calculate_correlation(forecast_is, actual_is)

                # 命中率(需要前一期值)
                if len(actual_is) > 1:
                    previous_is = np.concatenate([[np.nan], actual_is[:-1]])
                    metrics.is_hit_rate = self.calculate_hit_rate(
                        forecast_is[1:], actual_is[1:], previous_is[1:]
                    )

            # 样本外评估
            if model_result.forecast_oos is not None and len(val_data) > 0:
                # 对齐长度
                min_len = min(len(model_result.forecast_oos), len(val_data))
                forecast_oos = model_result.forecast_oos[:min_len]
                actual_oos = val_data.values[:min_len]

                # RMSE
                metrics.oos_rmse = self.calculate_rmse(forecast_oos, actual_oos)

                # 相关系数
                metrics.oos_correlation = self.calculate_correlation(forecast_oos, actual_oos)

                # 命中率(需要前一期值)
                if len(actual_oos) > 1:
                    # 使用训练期最后一个值作为第一个previous值
                    if len(train_data) > 0:
                        first_previous = train_data.values[-1]
                        previous_oos = np.concatenate([[first_previous], actual_oos[:-1]])
                    else:
                        previous_oos = np.concatenate([[np.nan], actual_oos[:-1]])

                    metrics.oos_hit_rate = self.calculate_hit_rate(
                        forecast_oos, actual_oos, previous_oos
                    )

        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()

        return metrics


# ==================== DFMTrainer (主训练器) ====================

class DFMTrainer:
    """
    DFM主训练器(合并pipeline逻辑)

    两阶段训练流程:
    1. 阶段1: 变量选择(固定k=块数)
    2. 阶段2: 因子数选择(PCA/Elbow/Fixed)
    3. 最终训练: 使用选定变量和因子数训练模型

    环境初始化:
    - 多线程BLAS配置
    - 随机种子设置
    - 静默模式控制
    """

    def __init__(self, config: dict):
        """
        初始化训练器

        Args:
            config: 训练配置字典(临时使用dict,后续改为TrainingConfig对象)
        """
        self.config = config
        self.evaluator = ModelEvaluator()

        # 环境初始化
        self._init_environment()

        # 训练状态
        self.total_evaluations = 0
        self.svd_error_count = 0

    def _init_environment(self):
        """
        环境初始化和可重现性控制

        配置:
        - 多线程BLAS: 使用所有CPU核心
        - 随机种子: 42
        - 静默模式: 通过环境变量控制
        """
        # 1. 多线程BLAS配置
        cpu_count = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

        logger.info(f"配置多线程BLAS: {cpu_count}个线程")

        # 2. 随机种子设置
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)

        logger.info(f"设置随机种子: {SEED}")

        # 3. 静默模式(可选)
        silent_mode = os.getenv('DFM_SILENT_WARNINGS', 'false').lower() == 'true'
        if silent_mode:
            logger.info("静默模式已启用")

    def train(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> TrainingResult:
        """
        完整两阶段训练流程

        Args:
            progress_callback: 进度回调函数,签名为 (message: str) -> None

        Returns:
            TrainingResult对象
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("开始DFM模型训练")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback("[TRAIN] 开始DFM模型训练")

        try:
            # TODO: 实现完整训练流程
            # 这是简化版本,展示基本结构

            result = TrainingResult(
                training_time=time.time() - start_time
            )

            logger.info("训练完成!")
            if progress_callback:
                progress_callback("[TRAIN] 训练完成!")

            return result

        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()

            if progress_callback:
                progress_callback(f"[ERROR] 训练失败: {e}")

            raise


# ==================== 导出 ====================

__all__ = [
    'ModelEvaluator',
    'DFMTrainer',
    'EvaluationMetrics',
    'DFMModelResult',
    'TrainingResult',
]
