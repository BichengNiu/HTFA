# -*- coding: utf-8 -*-
"""
分析工具函数

提供各种指标计算和数据分析功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dashboard.DFM.train_ref.utils.logger import get_logger


logger = get_logger(__name__)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """计算RMSE

    Args:
        actual: 实际值
        predicted: 预测值

    Returns:
        RMSE值
    """
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not valid_mask.any():
        return np.nan

    errors = actual[valid_mask] - predicted[valid_mask]
    return np.sqrt(np.mean(errors ** 2))


def calculate_hit_rate(
    actual: np.ndarray,
    predicted: np.ndarray,
    lag: int = 1
) -> float:
    """计算命中率（方向一致性）

    Args:
        actual: 实际值
        predicted: 预测值
        lag: 滞后期数（用于计算变化方向）

    Returns:
        命中率（百分比）
    """
    if len(actual) < lag + 1:
        return np.nan

    # 计算变化方向
    actual_change = np.diff(actual, n=lag)
    predicted_change = np.diff(predicted, n=lag)

    valid_mask = ~(np.isnan(actual_change) | np.isnan(predicted_change))
    if not valid_mask.any():
        return np.nan

    # 方向一致性
    hits = (actual_change[valid_mask] * predicted_change[valid_mask]) > 0
    return np.mean(hits) * 100


def calculate_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """计算相关系数

    Args:
        actual: 实际值
        predicted: 预测值

    Returns:
        相关系数
    """
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    if valid_mask.sum() < 2:
        return np.nan

    return np.corrcoef(actual[valid_mask], predicted[valid_mask])[0, 1]


def calculate_metrics_with_lagged_target(
    factors: pd.DataFrame,
    target: pd.Series,
    target_loading: np.ndarray,
    train_end: str,
    validation_start: str,
    validation_end: str
) -> Dict[str, float]:
    """计算带滞后目标的指标

    Args:
        factors: 因子时间序列
        target: 目标变量
        target_loading: 目标变量载荷
        train_end: 训练集结束日期
        validation_start: 验证集开始日期
        validation_end: 验证集结束日期

    Returns:
        指标字典 {'is_rmse', 'oos_rmse', 'is_hit_rate', 'oos_hit_rate', ...}
    """
    # 计算预测值
    forecast = (factors.values @ target_loading).flatten()
    forecast_series = pd.Series(forecast, index=factors.index)

    # 对齐目标变量
    aligned_target = target.reindex(factors.index)

    # 样本内指标
    train_mask = (factors.index <= train_end)
    is_actual = aligned_target[train_mask].values
    is_pred = forecast_series[train_mask].values

    is_rmse = calculate_rmse(is_actual, is_pred)
    is_hit_rate = calculate_hit_rate(is_actual, is_pred)
    is_corr = calculate_correlation(is_actual, is_pred)

    # 样本外指标
    val_mask = (factors.index >= validation_start) & (factors.index <= validation_end)
    oos_actual = aligned_target[val_mask].values
    oos_pred = forecast_series[val_mask].values

    oos_rmse = calculate_rmse(oos_actual, oos_pred)
    oos_hit_rate = calculate_hit_rate(oos_actual, oos_pred)
    oos_corr = calculate_correlation(oos_actual, oos_pred)

    return {
        'is_rmse': is_rmse,
        'oos_rmse': oos_rmse,
        'is_hit_rate': is_hit_rate,
        'oos_hit_rate': oos_hit_rate,
        'is_correlation': is_corr,
        'oos_correlation': oos_corr
    }


def calculate_factor_contributions(
    factors: pd.DataFrame,
    loadings: np.ndarray,
    target_loading: np.ndarray
) -> pd.DataFrame:
    """计算因子贡献度

    Args:
        factors: 因子时间序列 (n_time, n_factors)
        loadings: 载荷矩阵 (n_obs, n_factors)
        target_loading: 目标变量载荷 (n_factors,)

    Returns:
        贡献度DataFrame，包含各因子对目标变量的贡献
    """
    n_factors = factors.shape[1]

    # 计算各因子贡献
    contributions = pd.DataFrame(index=factors.index)

    for i in range(n_factors):
        factor_contribution = factors.iloc[:, i].values * target_loading[i]
        contributions[f'Factor{i+1}_contribution'] = factor_contribution

    # 总预测值
    contributions['Total_forecast'] = contributions.sum(axis=1)

    return contributions


def calculate_individual_variable_r2(
    observables: pd.DataFrame,
    factors: pd.DataFrame,
    loadings: np.ndarray
) -> pd.Series:
    """计算个体变量R²

    Args:
        observables: 观测变量
        factors: 因子
        loadings: 载荷矩阵

    Returns:
        各变量的R²
    """
    r2_dict = {}

    for i, var_name in enumerate(observables.columns):
        y = observables[var_name].values
        y_pred = (factors.values @ loadings[i, :]).flatten()

        valid_mask = ~(np.isnan(y) | np.isnan(y_pred))
        if valid_mask.sum() < 2:
            r2_dict[var_name] = np.nan
            continue

        y_valid = y[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        ss_res = np.sum((y_valid - y_pred_valid) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        r2_dict[var_name] = r2

    return pd.Series(r2_dict)


def calculate_industry_r2(
    observables: pd.DataFrame,
    factors: pd.DataFrame,
    loadings: np.ndarray,
    variable_industry_map: Optional[Dict[str, str]] = None
) -> pd.Series:
    """计算行业聚合R²

    Args:
        observables: 观测变量
        factors: 因子
        loadings: 载荷矩阵
        variable_industry_map: 变量到行业的映射

    Returns:
        各行业的R²
    """
    if variable_industry_map is None:
        logger.warning("未提供variable_industry_map，跳过行业R²计算")
        return pd.Series()

    # 按行业分组计算R²
    industry_r2 = {}

    for industry in set(variable_industry_map.values()):
        # 找到该行业的变量
        industry_vars = [
            var for var, ind in variable_industry_map.items()
            if ind == industry and var in observables.columns
        ]

        if not industry_vars:
            continue

        # 计算该行业变量的平均R²
        var_r2 = []
        for var in industry_vars:
            i = observables.columns.get_loc(var)
            y = observables[var].values
            y_pred = (factors.values @ loadings[i, :]).flatten()

            valid_mask = ~(np.isnan(y) | np.isnan(y_pred))
            if valid_mask.sum() < 2:
                continue

            y_valid = y[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            ss_res = np.sum((y_valid - y_pred_valid) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            if not np.isnan(r2):
                var_r2.append(r2)

        if var_r2:
            industry_r2[industry] = np.mean(var_r2)

    return pd.Series(industry_r2)


def calculate_pca_variance(
    data: pd.DataFrame,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算PCA方差贡献

    Args:
        data: 输入数据
        n_components: 主成分数量

    Returns:
        (explained_variance, explained_variance_ratio, cumulative_variance_ratio)
    """
    from sklearn.decomposition import PCA

    # 标准化数据
    data_std = (data - data.mean()) / data.std()
    data_std = data_std.fillna(0)

    if n_components is None:
        n_components = min(data.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_std)

    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    return explained_variance, explained_variance_ratio, cumulative_variance_ratio


def calculate_monthly_friday_metrics(
    forecast_series: pd.Series,
    actual_series: pd.Series,
    freq: str = 'W-FRI'
) -> pd.DataFrame:
    """计算月度周五指标

    对于周度数据，计算每个月最后一个周五的指标

    Args:
        forecast_series: 预测序列
        actual_series: 实际序列
        freq: 频率（默认周五）

    Returns:
        月度指标DataFrame
    """
    if freq != 'W-FRI':
        logger.warning(f"非周五数据（{freq}），直接返回")
        return pd.DataFrame()

    # 对齐两个序列
    aligned = pd.DataFrame({
        'forecast': forecast_series,
        'actual': actual_series
    }).dropna()

    if aligned.empty:
        return pd.DataFrame()

    # 按月分组，取最后一个值
    monthly = aligned.resample('ME').last()

    # 计算月度指标
    monthly['error'] = monthly['actual'] - monthly['forecast']
    monthly['abs_error'] = np.abs(monthly['error'])
    monthly['squared_error'] = monthly['error'] ** 2

    return monthly
