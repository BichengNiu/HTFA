# -*- coding: utf-8 -*-
"""
模型评估器模块

协调DFM模型的评估流程
参考: dashboard/DFM/train_model/dfm_core.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from dashboard.DFM.train_ref.core.factor_model import fit_dfm
from dashboard.DFM.train_ref.core.estimator import estimate_target_loading
from dashboard.DFM.train_ref.evaluation.validator import DataValidator
from dashboard.DFM.train_ref.evaluation.metrics import calculate_metrics, MetricsResult
from dashboard.DFM.train_ref.utils.logger import get_logger
from dashboard.DFM.train_ref.utils.data_utils import apply_seasonal_mask


logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """评估结果"""
    metrics: MetricsResult
    nowcast: pd.Series
    loadings_df: pd.DataFrame
    success: bool
    error_message: str = ""


class DFMEvaluator:
    """DFM模型评估器

    完整的评估流程：
    1. 数据验证
    2. 季节性掩码
    3. DFM模型拟合
    4. Nowcast计算
    5. 指标评估
    """

    def __init__(
        self,
        apply_seasonal_mask_flag: bool = True,
        seasonal_months: List[int] = None
    ):
        """
        Args:
            apply_seasonal_mask_flag: 是否应用季节性掩码
            seasonal_months: 要掩码的月份
        """
        self.apply_seasonal_mask_flag = apply_seasonal_mask_flag
        self.seasonal_months = seasonal_months or [1, 2]
        self.validator = DataValidator()

    def evaluate(
        self,
        data: pd.DataFrame,
        target_variable: str,
        predictor_variables: List[str],
        k_factors: int,
        train_end: str,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None,
        max_lags: int = 1,
        max_iter: int = 30
    ) -> EvaluationResult:
        """评估DFM模型

        Args:
            data: 完整数据（已标准化）
            target_variable: 目标变量名
            predictor_variables: 预测变量列表
            k_factors: 因子数量
            train_end: 训练期结束日期
            validation_start: 验证期开始日期
            validation_end: 验证期结束日期
            max_lags: 最大滞后阶数
            max_iter: 最大迭代次数

        Returns:
            EvaluationResult: 评估结果
        """
        logger.info(f"开始评估: 变量数={len(predictor_variables)}, k={k_factors}")

        data_eval = data.copy()

        validation_result = self.validator.validate_data(
            data_eval, target_variable, predictor_variables, k_factors
        )

        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            logger.error(f"数据验证失败: {error_msg}")
            return EvaluationResult(
                metrics=self._get_failed_metrics(),
                nowcast=pd.Series(),
                loadings_df=pd.DataFrame(),
                success=False,
                error_message=error_msg
            )

        all_variables = predictor_variables + [target_variable]
        data_subset = data_eval[all_variables].copy()

        if validation_end:
            try:
                data_subset = data_subset.loc[:validation_end]
            except KeyError:
                logger.warning(f"验证结束日期{validation_end}无效，使用全部数据")

        predictor_data = data_subset[predictor_variables].copy()
        target_data = data_subset[target_variable].copy()

        if self.apply_seasonal_mask_flag:
            target_data_masked = self._apply_seasonal_mask(
                target_data, self.seasonal_months
            )
        else:
            target_data_masked = target_data

        try:
            dfm_result = fit_dfm(
                data=predictor_data,
                n_factors=k_factors,
                max_lags=max_lags,
                max_iter=max_iter,
                train_end=train_end
            )

            logger.info(f"DFM拟合成功: {dfm_result.n_iter}次迭代")

        except Exception as e:
            error_msg = f"DFM拟合失败: {str(e)}"
            logger.error(error_msg)
            return EvaluationResult(
                metrics=self._get_failed_metrics(),
                nowcast=pd.Series(),
                loadings_df=pd.DataFrame(),
                success=False,
                error_message=error_msg
            )

        try:
            target_loading = estimate_target_loading(
                target=target_data_masked,
                factors=dfm_result.factors,
                train_end=train_end
            )

            nowcast_standardized = dfm_result.factors.values @ target_loading

            nowcast_series = pd.Series(
                nowcast_standardized,
                index=dfm_result.factors.index,
                name='Nowcast'
            )

            logger.info(f"Nowcast计算完成: {len(nowcast_series)}个值")

        except Exception as e:
            error_msg = f"Nowcast计算失败: {str(e)}"
            logger.error(error_msg)
            return EvaluationResult(
                metrics=self._get_failed_metrics(),
                nowcast=pd.Series(),
                loadings_df=pd.DataFrame(),
                success=False,
                error_message=error_msg
            )

        loadings_df = pd.DataFrame(
            dfm_result.loadings,
            index=predictor_variables,
            columns=[f'Factor{i+1}' for i in range(k_factors)]
        )

        target_row = pd.DataFrame(
            [target_loading],
            index=[target_variable],
            columns=loadings_df.columns
        )
        loadings_df = pd.concat([loadings_df, target_row])

        try:
            metrics_result = calculate_metrics(
                nowcast=nowcast_series,
                target=target_data,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                apply_lag_alignment=True
            )

            logger.info(
                f"评估完成: OOS RMSE={metrics_result.oos_rmse:.4f}, "
                f"HR={metrics_result.oos_hit_rate:.2%}"
            )

        except Exception as e:
            error_msg = f"指标计算失败: {str(e)}"
            logger.error(error_msg)
            return EvaluationResult(
                metrics=self._get_failed_metrics(),
                nowcast=nowcast_series,
                loadings_df=loadings_df,
                success=False,
                error_message=error_msg
            )

        return EvaluationResult(
            metrics=metrics_result,
            nowcast=nowcast_series,
            loadings_df=loadings_df,
            success=True
        )

    def _apply_seasonal_mask(
        self,
        target_data: pd.Series,
        months_to_mask: List[int]
    ) -> pd.Series:
        """应用季节性掩码

        Args:
            target_data: 目标变量数据
            months_to_mask: 要掩码的月份列表

        Returns:
            pd.Series: 掩码后的数据
        """
        masked_data = target_data.copy()
        mask = masked_data.index.month.isin(months_to_mask)
        masked_count = mask.sum()
        masked_data.loc[mask] = np.nan

        logger.info(f"季节性掩码应用: {months_to_mask}月, 掩码数量: {masked_count}")

        return masked_data

    @staticmethod
    def _get_failed_metrics() -> MetricsResult:
        """获取失败时的指标"""
        return MetricsResult(
            is_rmse=np.inf,
            is_mae=np.inf,
            is_hit_rate=-np.inf,
            oos_rmse=np.inf,
            oos_mae=np.inf,
            oos_hit_rate=-np.inf,
            aligned_data=None
        )


def evaluate_model(
    data: pd.DataFrame,
    target_variable: str,
    predictor_variables: List[str],
    k_factors: int,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None,
    max_lags: int = 1,
    max_iter: int = 30
) -> Tuple[float, float, float, float, float, float, pd.DataFrame]:
    """评估模型的函数接口

    Args:
        data: 完整数据
        target_variable: 目标变量
        predictor_variables: 预测变量列表
        k_factors: 因子数量
        train_end: 训练期结束日期
        validation_start: 验证期开始日期
        validation_end: 验证期结束日期
        max_lags: 最大滞后阶数
        max_iter: 最大迭代次数

    Returns:
        Tuple: (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, loadings_df)
    """
    evaluator = DFMEvaluator()

    result = evaluator.evaluate(
        data=data,
        target_variable=target_variable,
        predictor_variables=predictor_variables,
        k_factors=k_factors,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        max_lags=max_lags,
        max_iter=max_iter
    )

    if not result.success:
        return (
            np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf,
            pd.DataFrame()
        )

    return (
        result.metrics.is_rmse,
        result.metrics.oos_rmse,
        result.metrics.is_mae,
        result.metrics.oos_mae,
        result.metrics.is_hit_rate,
        result.metrics.oos_hit_rate,
        result.loadings_df
    )
