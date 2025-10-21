# -*- coding: utf-8 -*-
"""
指标计算模块

计算模型评估指标
参考: dashboard/DFM/train_model/analysis_utils.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """评估指标结果"""
    is_rmse: float
    is_mae: float
    is_hit_rate: float
    oos_rmse: float
    oos_mae: float
    oos_hit_rate: float
    aligned_data: Optional[pd.DataFrame] = None


def calculate_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """计算方向命中率

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        float: 命中率 (0-1)
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan

    true_direction = np.sign(y_true.diff())
    pred_direction = np.sign(y_pred.diff())

    valid_mask = ~(true_direction.isna() | pred_direction.isna())

    if valid_mask.sum() == 0:
        return np.nan

    hits = (true_direction[valid_mask] == pred_direction[valid_mask]).sum()
    total = valid_mask.sum()

    hit_rate = hits / total if total > 0 else np.nan

    return hit_rate


def align_target_with_lag(
    nowcast: pd.Series,
    target: pd.Series,
    max_lag: int = 3
) -> Tuple[pd.Series, pd.Series, int]:
    """对齐nowcast和目标变量，考虑可能的滞后

    Args:
        nowcast: nowcast序列
        target: 目标变量序列
        max_lag: 最大滞后月数

    Returns:
        Tuple[pd.Series, pd.Series, int]: (对齐的nowcast, 对齐的目标, 最优滞后)
    """
    best_corr = -np.inf
    best_lag = 0
    best_nowcast = nowcast
    best_target = target

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            aligned_nowcast = nowcast
            aligned_target = target
        elif lag > 0:
            aligned_nowcast = nowcast.shift(lag)
            aligned_target = target
        else:
            aligned_nowcast = nowcast
            aligned_target = target.shift(-lag)

        common_idx = aligned_nowcast.index.intersection(aligned_target.index)
        if len(common_idx) == 0:
            continue

        nc = aligned_nowcast.loc[common_idx]
        tg = aligned_target.loc[common_idx]

        valid_mask = ~(nc.isna() | tg.isna())
        if valid_mask.sum() < 10:
            continue

        corr = nc[valid_mask].corr(tg[valid_mask])

        if corr > best_corr:
            best_corr = corr
            best_lag = lag
            best_nowcast = aligned_nowcast
            best_target = aligned_target

    logger.debug(f"最优滞后: {best_lag}个月, 相关系数: {best_corr:.4f}")

    return best_nowcast, best_target, best_lag


def calculate_metrics(
    nowcast: pd.Series,
    target: pd.Series,
    train_end: str,
    validation_start: Optional[str] = None,
    validation_end: Optional[str] = None,
    apply_lag_alignment: bool = True
) -> MetricsResult:
    """计算评估指标

    Args:
        nowcast: nowcast预测序列
        target: 目标变量真实值
        train_end: 训练期结束日期
        validation_start: 验证期开始日期
        validation_end: 验证期结束日期
        apply_lag_alignment: 是否应用滞后对齐

    Returns:
        MetricsResult: 评估指标结果
    """
    if apply_lag_alignment:
        nowcast_aligned, target_aligned, best_lag = align_target_with_lag(
            nowcast, target
        )
    else:
        nowcast_aligned = nowcast
        target_aligned = target
        best_lag = 0

    common_idx = nowcast_aligned.index.intersection(target_aligned.index)

    if len(common_idx) == 0:
        logger.error("nowcast和target没有共同的时间索引")
        return MetricsResult(
            np.inf, np.inf, -np.inf, np.inf, np.inf, -np.inf, None
        )

    aligned_df = pd.DataFrame({
        'Nowcast': nowcast_aligned.loc[common_idx],
        'Target': target_aligned.loc[common_idx]
    })

    aligned_df = aligned_df.dropna()

    if len(aligned_df) == 0:
        logger.error("没有有效的对齐数据")
        return MetricsResult(
            np.inf, np.inf, -np.inf, np.inf, np.inf, -np.inf, None
        )

    try:
        train_data = aligned_df.loc[:train_end]
    except KeyError:
        logger.warning(f"训练结束日期{train_end}不在对齐数据中，使用全部数据")
        train_data = aligned_df

    if validation_start and validation_end:
        try:
            oos_data = aligned_df.loc[validation_start:validation_end]
        except KeyError:
            logger.warning("验证期日期不在对齐数据中，使用训练期之后的数据")
            oos_data = aligned_df.loc[train_end:]
    else:
        oos_data = aligned_df.loc[train_end:]

    if len(train_data) < 2:
        logger.warning(f"训练期数据不足: {len(train_data)}")
        is_rmse = is_mae = np.inf
        is_hit_rate = -np.inf
    else:
        is_rmse = np.sqrt(mean_squared_error(
            train_data['Target'], train_data['Nowcast']
        ))
        is_mae = mean_absolute_error(
            train_data['Target'], train_data['Nowcast']
        )
        is_hit_rate = calculate_hit_rate(
            train_data['Target'], train_data['Nowcast']
        )

    if len(oos_data) < 2:
        logger.warning(f"样本外数据不足: {len(oos_data)}")
        oos_rmse = oos_mae = np.inf
        oos_hit_rate = -np.inf
    else:
        oos_rmse = np.sqrt(mean_squared_error(
            oos_data['Target'], oos_data['Nowcast']
        ))
        oos_mae = mean_absolute_error(
            oos_data['Target'], oos_data['Nowcast']
        )
        oos_hit_rate = calculate_hit_rate(
            oos_data['Target'], oos_data['Nowcast']
        )

    logger.info(
        f"指标计算完成 - IS: RMSE={is_rmse:.4f}, MAE={is_mae:.4f}, HR={is_hit_rate:.2%} | "
        f"OOS: RMSE={oos_rmse:.4f}, MAE={oos_mae:.4f}, HR={oos_hit_rate:.2%}"
    )

    return MetricsResult(
        is_rmse=is_rmse,
        is_mae=is_mae,
        is_hit_rate=is_hit_rate,
        oos_rmse=oos_rmse,
        oos_mae=oos_mae,
        oos_hit_rate=oos_hit_rate,
        aligned_data=aligned_df
    )
