# -*- coding: utf-8 -*-
"""
指标计算模块

计算模型评估指标
参考: dashboard/DFM/train_model/analysis_utils.py
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dashboard.models.DFM.train.utils.logger import get_logger
from dashboard.models.DFM.train.core.models import MetricsResult


logger = get_logger(__name__)


def calculate_rmse(y_true, y_pred) -> float:
    """
    统一的RMSE计算函数

    Args:
        y_true: 真实值（array-like或Series）
        y_pred: 预测值（array-like或Series）

    Returns:
        float: RMSE值，无有效数据时返回np.inf
    """
    # 转换为numpy数组
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not valid_mask.any():
        logger.warning("[RMSE] 无有效数据，返回np.inf")
        return np.inf

    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    # 使用sklearn计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

    return float(rmse)


def calculate_correlation(y_true, y_pred) -> float:
    """
    计算相关系数

    Args:
        y_true: 真实值（array-like或Series）
        y_pred: 预测值（array-like或Series）

    Returns:
        float: 相关系数，无有效数据时返回-np.inf
    """
    # 转换为numpy数组
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid_mask.sum() < 2:
        logger.warning("[Correlation] 有效数据不足2个，返回-np.inf")
        return -np.inf

    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    try:
        corr_matrix = np.corrcoef(y_true_clean, y_pred_clean)
        corr = float(corr_matrix[0, 1])
        return corr if not np.isnan(corr) else -np.inf
    except Exception as e:
        logger.warning(f"[Correlation] 计算失败: {e}")
        return -np.inf


def calculate_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """计算方向命中率（与train_model/analysis_utils.py:650-673完全一致）

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        float: 命中率百分比 (0-100)，失败返回np.nan
    """
    logger.debug(f"[Hit Rate] 输入数据: y_true长度={len(y_true)}, y_pred长度={len(y_pred)}")

    if len(y_true) < 2 or len(y_pred) < 2:
        logger.warning(f"[Hit Rate] 数据不足: y_true={len(y_true)}, y_pred={len(y_pred)}, 需要至少2个样本")
        return np.nan

    # 1. 先diff再dropna（匹配老代码逻辑）
    true_diff = y_true.diff().dropna()
    pred_diff = y_pred.diff().dropna()

    logger.debug(f"[Hit Rate] diff后数据: true_diff长度={len(true_diff)}, pred_diff长度={len(pred_diff)}")

    # 2. 对齐索引（匹配老代码逻辑）
    common_idx = true_diff.index.intersection(pred_diff.index)

    logger.debug(f"[Hit Rate] 索引对齐: common_idx长度={len(common_idx)}")

    if len(common_idx) == 0:
        logger.warning("[Hit Rate] 对齐后无共同索引，无法计算命中率")
        return np.nan

    # 3. 计算方向
    true_direction = np.sign(true_diff.loc[common_idx])
    pred_direction = np.sign(pred_diff.loc[common_idx])

    # 4. 统计命中
    hits = (true_direction == pred_direction).sum()
    total = len(common_idx)

    # 返回百分比（0-100），与老代码train_model/analysis_utils.py:673一致
    hit_rate = (hits / total) * 100.0 if total > 0 else np.nan
    logger.debug(f"[Hit Rate] 计算完成: 命中={hits}/{total} = {hit_rate:.2f}%")

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
    logger.info(f"[Metrics] 接收日期参数 - train_end={train_end}, validation_start={validation_start}, validation_end={validation_end}")

    # 转换日期字符串为Timestamp对象以确保正确索引
    train_end_ts = pd.to_datetime(train_end) if isinstance(train_end, str) else train_end
    validation_start_ts = pd.to_datetime(validation_start) if isinstance(validation_start, str) and validation_start else validation_start
    validation_end_ts = pd.to_datetime(validation_end) if isinstance(validation_end, str) and validation_end else validation_end

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
        logger.error("nowcast和target没有共同的时间索引，无法计算指标")
        return MetricsResult(
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None
        )

    aligned_df = pd.DataFrame({
        'Nowcast': nowcast_aligned.loc[common_idx],
        'Target': target_aligned.loc[common_idx]
    })

    aligned_df = aligned_df.dropna()

    if len(aligned_df) == 0:
        logger.error("没有有效的对齐数据，无法计算指标")
        return MetricsResult(
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None
        )

    try:
        train_data = aligned_df.loc[:train_end_ts]
    except KeyError:
        logger.warning(f"训练结束日期{train_end}不在对齐数据中，使用全部数据")
        train_data = aligned_df

    if validation_start_ts and validation_end_ts:
        try:
            oos_data = aligned_df.loc[validation_start_ts:validation_end_ts]
        except KeyError:
            logger.warning("验证期日期不在对齐数据中，使用训练期之后的数据")
            oos_data = aligned_df.loc[train_end_ts:]
    else:
        oos_data = aligned_df.loc[train_end_ts:]

    # 记录数据切分结果
    logger.info(f"[Metrics] 数据切分完成 - 训练期样本数: {len(train_data)} (截止 {train_end}), "
                f"验证期样本数: {len(oos_data)} ({validation_start} 至 {validation_end})")

    if len(train_data) < 2:
        logger.warning(f"训练期数据不足({len(train_data)}条)，无法计算样本内指标")
        is_rmse = is_mae = np.nan
        is_hit_rate = np.nan
    else:
        is_rmse = calculate_rmse(
            train_data['Target'], train_data['Nowcast']
        )
        is_mae = mean_absolute_error(
            train_data['Target'], train_data['Nowcast']
        )
        is_hit_rate = calculate_hit_rate(
            train_data['Target'], train_data['Nowcast']
        )

    if len(oos_data) < 2:
        logger.warning(f"样本外数据不足({len(oos_data)}条)，无法计算样本外指标")
        oos_rmse = oos_mae = np.nan
        oos_hit_rate = np.nan
    else:
        oos_rmse = calculate_rmse(
            oos_data['Target'], oos_data['Nowcast']
        )
        oos_mae = mean_absolute_error(
            oos_data['Target'], oos_data['Nowcast']
        )
        oos_hit_rate = calculate_hit_rate(
            oos_data['Target'], oos_data['Nowcast']
        )

    logger.info(
        f"指标计算完成 - IS: RMSE={is_rmse:.4f}, MAE={is_mae:.4f}, HR={is_hit_rate:.2f}% | "
        f"OOS: RMSE={oos_rmse:.4f}, MAE={oos_mae:.4f}, HR={oos_hit_rate:.2f}%"
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


def calculate_combined_score(
    is_rmse: float,
    oos_rmse: float,
    is_hit_rate: float,
    oos_hit_rate: float
) -> Tuple[float, float]:
    """
    计算组合得分（用于变量选择）

    评分标准：先优化Hit Rate，再优化RMSE

    Args:
        is_rmse: 样本内RMSE
        oos_rmse: 样本外RMSE
        is_hit_rate: 样本内命中率
        oos_hit_rate: 样本外命中率

    Returns:
        Tuple[float, float]: (combined_hit_rate, -combined_rmse)
            - 第一个元素：平均命中率（越大越好）
            - 第二个元素：负平均RMSE（越大越好）
    """
    # 计算平均RMSE
    finite_rmses = [r for r in [is_rmse, oos_rmse] if np.isfinite(r)]
    combined_rmse = np.mean(finite_rmses) if finite_rmses else np.inf

    # 计算平均命中率
    finite_hrs = [hr for hr in [is_hit_rate, oos_hit_rate] if np.isfinite(hr)]
    combined_hr = np.mean(finite_hrs) if finite_hrs else np.nan

    return (combined_hr, -combined_rmse)
