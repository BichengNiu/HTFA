# -*- coding: utf-8 -*-
"""
参数估计模块

实现DFM模型的参数估计算法
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from dashboard.models.DFM.train.utils.logger import get_logger


logger = get_logger(__name__)


def estimate_loadings(
    observables: Union[pd.DataFrame, pd.Series],
    factors: pd.DataFrame,
    train_end: Optional[str] = None,
    use_train_only: bool = False
) -> np.ndarray:
    """统一的因子载荷估计（支持DataFrame和Series）

    使用OLS回归估计观测变量对因子的载荷

    Args:
        observables: 观测变量（DataFrame或Series）
        factors: 共同因子 (n_time, n_factors)
        train_end: 训练集结束日期（避免信息泄漏）
        use_train_only: 是否仅使用训练期数据

    Returns:
        np.ndarray: 载荷矩阵或向量
            - DataFrame输入: (n_obs, n_factors)
            - Series输入: (n_factors,)
    """
    # 处理训练期截取
    factors_data = factors.copy()
    obs_data = observables.copy()

    if use_train_only and train_end:
        try:
            factors_data = factors_data.loc[:train_end]
            if isinstance(obs_data, pd.Series):
                obs_data = obs_data.loc[:train_end]
            else:
                obs_data = obs_data.loc[:train_end]
            logger.debug(f"使用训练期数据估计载荷: {len(factors_data)}个样本")
        except KeyError:
            logger.warning(f"训练期结束日期{train_end}无效，使用全部数据")

    n_factors = factors_data.shape[1]

    # 情况1：单个变量（Series）
    if isinstance(obs_data, pd.Series):
        valid_idx = ~(obs_data.isna() | factors_data.isna().any(axis=1))

        if valid_idx.sum() < n_factors:
            raise ValueError(
                f"有效样本数({valid_idx.sum()}) < 因子数({n_factors})"
            )

        y_valid = obs_data[valid_idx].values
        X_valid = factors_data[valid_idx].values

        # 使用sklearn LinearRegression（更快）
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_valid, y_valid)

        r2 = reg.score(X_valid, y_valid)
        logger.debug(f"单变量载荷估计完成, R² = {r2:.4f}")

        return reg.coef_

    # 情况2：多个变量（DataFrame）
    n_obs = obs_data.shape[1]
    Lambda = np.full((n_obs, n_factors), np.nan)

    for i in range(n_obs):
        y_i = obs_data.iloc[:, i]
        valid_idx = y_i.notna() & factors_data.notna().all(axis=1)

        y_i_valid = y_i[valid_idx]
        F_valid = factors_data.loc[valid_idx]

        # 需要足够样本进行回归
        if len(y_i_valid) > n_factors:
            try:
                # 使用statsmodels.OLS（匹配原实现）
                ols_model = sm.OLS(y_i_valid, F_valid)
                ols_results = ols_model.fit()
                Lambda[i, :] = ols_results.params.values
            except Exception as e:
                logger.warning(f"变量{obs_data.columns[i]}: OLS失败 - {e}")
                # Lambda[i, :] 保持NaN

    return Lambda


def estimate_transition_matrix(
    factors: np.ndarray,
    max_lags: int = 1
) -> np.ndarray:
    """估计状态转移矩阵（匹配老代码_calculate_prediction_matrix）

    使用最小二乘法估计：A = (F_t' F_{t-1})(F_{t-1}' F_{t-1})^-1

    Args:
        factors: 因子序列 (n_time, n_factors)
        max_lags: 最大滞后阶数

    Returns:
        np.ndarray: 转移矩阵 A (n_states, n_states)
            其中 n_states = n_factors * max_lags
    """
    import scipy.linalg

    n_time, n_factors = factors.shape

    try:
        if max_lags == 1:
            # 匹配老代码：A = (F_t' F_{t-1})(F_{t-1}' F_{t-1} + epsilon*I)^-1
            F_t = factors[1:, :]      # Shape (n_time-1, n_factors)
            F_tm1 = factors[:-1, :]   # Shape (n_time-1, n_factors)

            Ft_Ftm1 = F_t.T @ F_tm1     # (n_factors, n_factors)
            Ftm1_Ftm1 = F_tm1.T @ F_tm1  # (n_factors, n_factors)

            # 添加小的正则化项确保数值稳定性（匹配老代码）
            A = scipy.linalg.solve(
                (Ftm1_Ftm1 + np.eye(n_factors) * 1e-7).T,
                Ft_Ftm1.T,
                assume_a='pos'
            ).T
        else:
            # 对于max_lags > 1，使用statsmodels VAR
            from statsmodels.tsa.api import VAR
            var_model = VAR(factors)
            var_result = var_model.fit(maxlags=max_lags, ic=None, trend='n')
            coef_matrices = var_result.params.T

            n_states = n_factors * max_lags
            A = np.zeros((n_states, n_states))
            A[:n_factors, :] = coef_matrices.reshape(n_factors, -1)
            if max_lags > 1:
                A[n_factors:, :-n_factors] = np.eye(n_factors * (max_lags - 1))

    except Exception as e:
        logger.warning(f"A矩阵估计失败: {e}，使用单位矩阵")
        n_states = n_factors * max_lags
        A = np.eye(n_states) * 0.95

    return A


def estimate_covariance_matrices(
    smoothed_result,
    observables: pd.DataFrame,
    Lambda: np.ndarray,
    n_factors: int,
    A: np.ndarray = None,
    n_shocks: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """估计协方差矩阵Q和R以及冲击矩阵B（匹配老代码_calculate_shock_matrix）

    Args:
        smoothed_result: 卡尔曼平滑结果
        observables: 观测变量
        Lambda: 载荷矩阵
        n_factors: 因子数量
        A: 状态转移矩阵（用于Q矩阵计算）
        n_shocks: 冲击数量（用于B矩阵计算）

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (B, Q, R)
    """
    n_time = observables.shape[0]
    n_obs = observables.shape[1]

    x_smooth = smoothed_result.x_smoothed

    # Q矩阵和B矩阵计算（完全匹配老代码_calculate_shock_matrix）
    # Sigma = E[F_t F_t'] - A E[F_{t-1} F_{t-1}'] A'
    F = x_smooth[:n_factors, :].T  # (n_time, n_factors)

    # Calculate F_{t-1}' F_{t-1}
    F_tm1 = F[:-1, :]  # Shape (n_time-1, n_factors)
    temp = F_tm1.T @ F_tm1  # Shape (n_factors, n_factors)

    # Calculate F_t' F_t
    F_t = F[1:, :]  # Shape (n_time-1, n_factors)
    term1 = F_t.T @ F_t  # Shape (n_factors, n_factors)
    term1 = term1 / (n_time - 1)

    # Calculate Sigma = E[F_t F_t'] - A E[F_{t-1} F_{t-1}'] A'
    if A is not None:
        term2 = A @ (temp / (n_time - 1)) @ A.T
        Sigma = term1 - term2
    else:
        # 如果没有传入A，使用简单估计
        Sigma = term1

    # 计算B矩阵和Q矩阵（匹配老代码_calculate_shock_matrix line 114-145）
    if n_shocks is not None and A is not None:
        try:
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
            # 将负特征值替换为小正数
            min_eig_val = 1e-7
            eigenvalues_corrected = np.maximum(eigenvalues, min_eig_val)

            # 使用修正后的特征值重构Sigma
            Sigma_corrected = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T

            # 计算B矩阵：选择最大的n_shocks个特征值
            sorted_indices = np.argsort(eigenvalues_corrected)[::-1]
            evalues_selected = eigenvalues_corrected[sorted_indices[:n_shocks]]
            M = eigenvectors[:, sorted_indices[:n_shocks]]

            # B = M * sqrt(diag(selected eigenvalues))
            B = M @ np.diag(np.sqrt(evalues_selected))

            # 使用修正后的Sigma作为Q
            Q = Sigma_corrected

        except np.linalg.LinAlgError as e:
            logger.warning(f"Sigma特征值分解失败: {e}. 使用fallback值")
            Q = np.eye(n_factors) * 1e-6
            B = np.zeros((n_factors, n_shocks))
            min_dim_fallback = min(n_factors, n_shocks)
            B[:min_dim_fallback, :min_dim_fallback] = np.eye(min_dim_fallback) * np.sqrt(1e-6)
    else:
        # 如果没有n_shocks，只计算Q矩阵
        Q = _ensure_positive_definite(Sigma, epsilon=1e-7)
        B = np.eye(n_factors) * 0.1  # 默认B矩阵

    # 计算残差和R矩阵（匹配老代码EMstep的实现）
    # 中心化数据：y_np
    # 重构数据：Lambda @ f.T
    Z = observables.values  # (n_time, n_obs) 中心化数据
    predicted_Z = (Lambda @ x_smooth[:n_factors, :]).T  # (n_time, n_obs)
    residuals = Z - predicted_Z  # (n_time, n_obs)

    # R矩阵：残差的方差（每个变量的方差）
    R_diag = np.nanvar(residuals, axis=0, ddof=0)  # (n_obs,), ddof=0避免自由度问题

    # 处理NaN和Inf：替换为默认值
    R_diag = np.where(np.isfinite(R_diag), R_diag, 1.0)  # NaN/Inf用1.0替换
    R_diag = np.maximum(R_diag, 1e-7)  # 确保正定性

    R = np.diag(R_diag)

    return B, Q, R


def _ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """确保矩阵正定（完全匹配老代码_calculate_shock_matrix的实现）

    Args:
        matrix: 输入矩阵
        epsilon: 最小特征值

    Returns:
        np.ndarray: 正定矩阵
    """
    # 不对称化，完全匹配老代码行为
    # 老代码line 115-121没有对称化步骤

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, epsilon)

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
