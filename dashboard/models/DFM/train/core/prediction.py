# -*- coding: utf-8 -*-
"""
目标变量预测模块

通过因子回归生成目标变量的预测值
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dashboard.models.DFM.train.core.models import DFMModelResult
from dashboard.models.DFM.train.utils.logger import get_logger

logger = get_logger(__name__)


def generate_target_forecast(
    model_result: DFMModelResult,
    target_data: pd.Series,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> DFMModelResult:
    """
    生成目标变量的预测值

    通过将目标变量回归到因子上，得到目标变量的因子载荷，
    然后使用因子和载荷进行预测。

    Args:
        model_result: DFM模型结果
        target_data: 目标变量数据
        train_end: 训练结束日期
        validation_start: 验证开始日期
        validation_end: 验证结束日期
        progress_callback: 进度回调函数

    Returns:
        更新了forecast_is和forecast_oos的DFMModelResult
    """
    try:
        if progress_callback:
            progress_callback("[PREDICTION] 生成目标变量预测...")

        # 提取因子（时间 x 因子数）
        factors = model_result.factors.T  # (n_time, n_factors)

        # 对齐目标变量和因子的时间索引
        common_index = target_data.index[:factors.shape[0]]
        y = target_data.loc[common_index].values
        X = factors[:len(common_index), :]

        # 移除NaN值（同时移除y和X中对应的行）
        y_isnan = np.isnan(y)
        X_has_nan = np.any(np.isnan(X), axis=1)
        valid_mask = ~(y_isnan | X_has_nan)

        if not valid_mask.any():
            logger.error("没有找到有效的数据点用于回归")
            raise ValueError("目标变量和因子都没有有效的对齐数据点")

        y_clean = y[valid_mask]
        X_clean = X[valid_mask, :]

        # 回归：y = X @ beta + epsilon
        # 使用最小二乘法估计beta（目标变量的因子载荷）
        from numpy.linalg import lstsq
        beta, residuals, rank, s = lstsq(X_clean, y_clean, rcond=None)

        logger.debug(f"目标变量回归完成: beta shape={beta.shape}, rank={rank}")

        # 生成完整预测（包含NaN位置）
        forecast_full = X @ beta

        # 分割样本内和样本外
        train_end_date = pd.to_datetime(train_end)
        train_data_filtered = target_data[:train_end_date]
        train_end_idx = len(train_data_filtered) - 1

        # 样本内预测
        forecast_is = forecast_full[:train_end_idx + 1]

        # 样本外预测
        if validation_start and validation_end:
            val_start_date = pd.to_datetime(validation_start)
            val_end_date = pd.to_datetime(validation_end)

            val_data_filtered = target_data[val_start_date:val_end_date]
            if len(val_data_filtered) > 0:
                val_start_idx = target_data.index.get_loc(val_data_filtered.index[0])
                val_end_idx = target_data.index.get_loc(val_data_filtered.index[-1])

                if val_start_idx < len(forecast_full) and val_end_idx < len(forecast_full):
                    forecast_oos = forecast_full[val_start_idx:val_end_idx + 1]
                else:
                    forecast_oos = None
            else:
                forecast_oos = None
        else:
            forecast_oos = forecast_full[train_end_idx + 1:] if train_end_idx + 1 < len(forecast_full) else None

        # 更新model_result
        model_result.forecast_is = forecast_is
        model_result.forecast_oos = forecast_oos if forecast_oos is not None and len(forecast_oos) > 0 else None

        logger.info(
            f"预测生成完成: IS={len(forecast_is) if forecast_is is not None else 0}, "
            f"OOS={len(forecast_oos) if forecast_oos is not None else 0}"
        )

    except Exception as e:
        logger.error(f"生成目标变量预测时出错: {e}")
        import traceback
        traceback.print_exc()
        # 保持forecast为None

    return model_result


__all__ = ['generate_target_forecast']
