# -*- coding: utf-8 -*-
"""
端到端追踪新老DFM的每一步
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    n_time = 100
    n_obs = 10
    dates = pd.date_range('2015-01-01', periods=n_time, freq='ME')
    data = pd.DataFrame(
        np.random.randn(n_time, n_obs),
        index=dates,
        columns=[f'var{i}' for i in range(n_obs)]
    )
    return data


def run_new_dfm_one_iter():
    """运行新代码的一次完整迭代"""
    print("="*80)
    print("运行新代码DFMModel一次迭代（手动实现）")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    # ===== 步骤1: 数据预处理 =====
    means = data.mean(skipna=True).values
    stds = data.std(skipna=True).values
    stds = np.where(stds > 0, stds, 1.0)

    obs_centered = data - means
    Z_standardized = (obs_centered / stds).fillna(0).values

    print(f"\n步骤1: 数据预处理")
    print(f"  obs_centered.shape: {obs_centered.shape}")
    print(f"  obs_centered[0, :3]: {obs_centered.iloc[0, :3].values}")

    # ===== 步骤2: PCA初始化 =====
    U, s, Vh = np.linalg.svd(Z_standardized, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_df = pd.DataFrame(
        factors_init,
        index=data.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    print(f"\n步骤2: PCA初始化")
    print(f"  factors_init[0]: {factors_init[0]}")

    # ===== 步骤3: 计算Lambda =====
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings
    Lambda = estimate_loadings(obs_centered, factors_df)

    print(f"\n步骤3: 计算Lambda")
    print(f"  Lambda[:2, :] =\n{Lambda[:2, :]}")

    # ===== 步骤4: VAR估计A和Q =====
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_df.dropna())
    var_results = var_model.fit(max_lags)
    A = var_results.coefs[0]
    Q = np.cov(var_results.resid, rowvar=False)
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    print(f"\n步骤4: VAR估计A和Q")
    print(f"  A对角线: {np.diag(A)}")
    print(f"  Q对角线: {np.diag(Q)}")

    # ===== 步骤5: 计算R矩阵 =====
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = Z_standardized - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (stds ** 2)
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    print(f"\n步骤5: 计算R矩阵")
    print(f"  R对角线[:5]: {np.diag(R)[:5]}")

    # ===== 步骤6: Kalman滤波和平滑 =====
    from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKF

    Z = obs_centered.values.T  # (n_obs, n_time)
    U = np.zeros((n_factors, data.shape[0]))
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors) * 0.1

    kf = NewKF(A, B, Lambda, Q, R, x0, P0)
    filter_result = kf.filter(Z, U)
    smoother_result = kf.smooth(filter_result)

    factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T

    print(f"\n步骤6: Kalman滤波和平滑")
    print(f"  平滑因子shape: {factors_smoothed.shape}")
    print(f"  平滑因子[0]: {factors_smoothed[0]}")
    print(f"  平滑因子[:3, 0]: {factors_smoothed[:3, 0]}")

    return factors_smoothed


def run_old_dfm_one_iter():
    """运行老代码的一次完整迭代"""
    print("\n" + "="*80)
    print("运行老代码DFM_EMalgo一次迭代")
    print("="*80)

    from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo

    data = create_test_data()
    n_factors = 3

    result = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=1,
        max_lags=1
    )

    print(f"\n平滑因子shape: {result.x_sm.shape}")
    print(f"  平滑因子.iloc[0]: {result.x_sm.iloc[0].values}")
    print(f"  平滑因子[:3, 0]: {result.x_sm.iloc[:3, 0].values}")

    return result.x_sm.values


if __name__ == "__main__":
    factors_new = run_new_dfm_one_iter()
    factors_old = run_old_dfm_one_iter()

    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    diff = np.abs(factors_new - factors_old)
    print(f"\n因子差异:")
    print(f"  最大差异: {np.max(diff):.15f}")
    print(f"  平均差异: {np.mean(diff):.15f}")
    print(f"  前3行最大差异: {np.max(diff[:3]):.15f}")

    if np.max(diff) < 1e-10:
        print(f"\n[通过] 完全一致!")
    elif np.max(diff) < 0.01:
        print(f"\n[接近] 结果非常接近")
    else:
        print(f"\n[失败] 存在差异")
