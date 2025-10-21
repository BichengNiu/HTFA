# -*- coding: utf-8 -*-
"""
数据验证模块

验证输入数据和参数的有效性
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataValidator:
    """数据验证器

    验证DFM训练所需的数据质量和参数有效性
    """

    @staticmethod
    def validate_data(
        data: pd.DataFrame,
        target_variable: str,
        predictor_variables: List[str],
        k_factors: int,
        min_observations: Optional[int] = None
    ) -> ValidationResult:
        """验证数据有效性

        Args:
            data: 输入数据
            target_variable: 目标变量
            predictor_variables: 预测变量列表
            k_factors: 因子数量
            min_observations: 最小观测数

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        if data is None or data.empty:
            errors.append("数据为空")
            return ValidationResult(False, errors, warnings)

        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("数据索引必须是DatetimeIndex")

        if target_variable not in data.columns:
            errors.append(f"目标变量'{target_variable}'不在数据中")

        missing_predictors = [v for v in predictor_variables if v not in data.columns]
        if missing_predictors:
            errors.append(f"预测变量缺失: {missing_predictors[:5]}")

        if len(predictor_variables) < k_factors:
            errors.append(
                f"预测变量数({len(predictor_variables)}) < 因子数({k_factors})"
            )

        if min_observations is None:
            min_observations = k_factors * 2

        if data.shape[0] < min_observations:
            errors.append(
                f"观测数({data.shape[0]}) < 最小要求({min_observations})"
            )

        nan_ratio = data.isna().sum() / len(data)
        high_nan_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
        if high_nan_cols:
            warnings.append(
                f"{len(high_nan_cols)}个变量的缺失值>50%: {high_nan_cols[:3]}"
            )

        zero_var_cols = data.columns[data.std() < 1e-10].tolist()
        if zero_var_cols:
            warnings.append(f"{len(zero_var_cols)}个变量方差为0: {zero_var_cols[:3]}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("数据验证通过")
        else:
            logger.error(f"数据验证失败: {errors}")

        if warnings:
            logger.warning(f"数据警告: {warnings}")

        return ValidationResult(is_valid, errors, warnings)

    @staticmethod
    def validate_date_ranges(
        data: pd.DataFrame,
        train_end: Optional[str] = None,
        validation_start: Optional[str] = None,
        validation_end: Optional[str] = None
    ) -> ValidationResult:
        """验证日期范围

        Args:
            data: 输入数据
            train_end: 训练结束日期
            validation_start: 验证开始日期
            validation_end: 验证结束日期

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        data_start = data.index.min()
        data_end = data.index.max()

        if train_end:
            try:
                train_end_dt = pd.to_datetime(train_end)
                if train_end_dt < data_start or train_end_dt > data_end:
                    errors.append(
                        f"训练结束日期{train_end}超出数据范围"
                        f"({data_start} 到 {data_end})"
                    )
            except Exception as e:
                errors.append(f"训练结束日期格式错误: {e}")

        if validation_start:
            try:
                val_start_dt = pd.to_datetime(validation_start)
                if val_start_dt < data_start or val_start_dt > data_end:
                    warnings.append(
                        f"验证开始日期{validation_start}可能超出数据范围"
                    )
            except Exception as e:
                errors.append(f"验证开始日期格式错误: {e}")

        if validation_end:
            try:
                val_end_dt = pd.to_datetime(validation_end)
                if val_end_dt < data_start or val_end_dt > data_end:
                    warnings.append(
                        f"验证结束日期{validation_end}可能超出数据范围"
                    )
            except Exception as e:
                errors.append(f"验证结束日期格式错误: {e}")

        if train_end and validation_start:
            if pd.to_datetime(train_end) >= pd.to_datetime(validation_start):
                warnings.append("训练期和验证期可能存在重叠")

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, errors, warnings)


def validate_data(
    data: pd.DataFrame,
    target_variable: str,
    predictor_variables: List[str],
    k_factors: int
) -> Tuple[bool, str]:
    """验证数据的函数接口

    Args:
        data: 输入数据
        target_variable: 目标变量
        predictor_variables: 预测变量列表
        k_factors: 因子数量

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    validator = DataValidator()
    result = validator.validate_data(
        data, target_variable, predictor_variables, k_factors
    )

    if result.is_valid:
        return True, ""
    else:
        error_msg = "; ".join(result.errors)
        return False, error_msg
