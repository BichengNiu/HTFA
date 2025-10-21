# -*- coding: utf-8 -*-
"""
详细对比第一次EM迭代的E步和M步中间结果
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


def test_first_em_detailed():
    """详细对比第一次EM迭代"""
    print("="*80)
    print("详细对比第一次EM迭代的E步和M步")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    # ========== 共同初始化 ==========
    print("\n步骤1: 数据预处理和PCA初始化")
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    # PCA
    U, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_init_df = pd.DataFrame(
        factors_init,
        index=z.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda_init = calculate_factor_loadings(obs_centered, factors_init_df)

    # VAR初始化A和Q
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(max_lags)
    A_init = var_results.coefs[0]
    Q_init = np.cov(var_results.resid, rowvar=False)
    Q_init = np.diag(np.maximum(np.diag(Q_init), 1e-6))

    # R矩阵
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R_init = np.diag(R_diag)

    # 其他参数
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)

    print(f"  初始Lambda范围: [{Lambda_init.min():.6f}, {Lambda_init.max():.6f}]")
    print(f"  初始A对角线: {np.diag(A_init)}")
    print(f"  初始Q对角线: {np.diag(Q_init)}")

    # ========== 老代码第一次EM迭代 ==========
    print("\n" + "="*80)
    print("老代码第一次EM迭代")
    print("="*80)

    from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKF, FIS, EMstep

    # E步
    print("\nE步: Kalman滤波和平滑")
    error_df = pd.DataFrame(
        np.zeros((len(data), n_factors)),
        columns=[f'shock{i+1}' for i in range(n_factors)],
        index=data.index
    )

    kf_old = OldKF(
        Z=obs_centered,
        U=error_df,
        A=A_init,
        B=B,
        H=Lambda_init,
        state_names=[f'Factor{i+1}' for i in range(n_factors)],
        x0=x0,
        P0=P0,
        Q=Q_init,
        R=R_init
    )
    fis_old = FIS(kf_old)

    print(f"  平滑因子shape: {fis_old.x_sm.shape}")
    print(f"  平滑因子前3行:\n{fis_old.x_sm.iloc[:3]}")
    print(f"  平滑因子范围: [{fis_old.x_sm.values.min():.6f}, {fis_old.x_sm.values.max():.6f}]")

    # M步
    print("\nM步: 参数更新")
    em_old = EMstep(fis_old, n_factors)

    print(f"  更新后Lambda shape: {em_old.Lambda.shape}")
    print(f"  更新后Lambda范围: [{em_old.Lambda.min():.6f}, {em_old.Lambda.max():.6f}]")
    print(f"  更新后Lambda前3行:\n{em_old.Lambda[:3]}")
    print(f"  更新后A:\n{em_old.A}")
    print(f"  更新后Q对角线: {np.diag(em_old.Q)}")
    print(f"  更新后R对角线前5个: {np.diag(em_old.R)[:5]}")

    # ========== 新代码第一次EM迭代 ==========
    print("\n" + "="*80)
    print("新代码第一次EM迭代")
    print("="*80)

    from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKF
    from dashboard.DFM.train_ref.core.estimator import (
        estimate_loadings,
        estimate_transition_matrix,
        estimate_covariance_matrices
    )

    # E步
    print("\nE步: Kalman滤波和平滑")
    Z = obs_centered.values.T  # (n_obs, n_time)
    U = np.zeros((n_factors, len(data)))

    kf_new = NewKF(A_init, B, Lambda_init, Q_init, R_init, x0, P0)
    filter_result = kf_new.filter(Z, U)
    smoother_result = kf_new.smooth(filter_result)

    factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
    factors_df_smoothed = pd.DataFrame(
        factors_smoothed,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    print(f"  平滑因子shape: {factors_smoothed.shape}")
    print(f"  平滑因子前3行:\n{factors_smoothed[:3]}")
    print(f"  平滑因子范围: [{factors_smoothed.min():.6f}, {factors_smoothed.max():.6f}]")

    # 对比E步结果
    print("\n对比E步平滑因子:")
    factors_e_diff = np.abs(factors_smoothed - fis_old.x_sm.values)
    print(f"  最大差异: {np.max(factors_e_diff):.15f}")
    print(f"  平均差异: {np.mean(factors_e_diff):.15f}")

    # M步
    print("\nM步: 参数更新")
    Lambda_new = estimate_loadings(obs_centered, factors_df_smoothed)

    # 使用estimate_transition_matrix（完全匹配factor_model.py和老代码）
    A_new = estimate_transition_matrix(factors_smoothed, max_lags)

    Q_new, R_new = estimate_covariance_matrices(
        smoother_result,
        obs_centered,
        Lambda_new,
        n_factors,
        A_new
    )

    print(f"  更新后Lambda shape: {Lambda_new.shape}")
    print(f"  更新后Lambda范围: [{Lambda_new.min():.6f}, {Lambda_new.max():.6f}]")
    print(f"  更新后Lambda前3行:\n{Lambda_new[:3]}")
    print(f"  更新后A:\n{A_new}")
    print(f"  更新后Q对角线: {np.diag(Q_new)}")
    print(f"  更新后R对角线前5个: {np.diag(R_new)[:5]}")

    # ========== 对比M步结果 ==========
    print("\n" + "="*80)
    print("对比M步参数更新")
    print("="*80)

    Lambda_diff = np.abs(Lambda_new - em_old.Lambda)
    A_diff = np.abs(A_new - em_old.A)
    Q_diff = np.abs(Q_new - em_old.Q)
    R_diff = np.abs(R_new - em_old.R)

    print(f"\nLambda差异:")
    print(f"  最大差异: {np.max(Lambda_diff):.15f}")
    print(f"  平均差异: {np.mean(Lambda_diff):.15f}")
    if np.max(Lambda_diff) > 1e-10:
        print(f"  差异最大的元素位置: {np.unravel_index(np.argmax(Lambda_diff), Lambda_diff.shape)}")
        idx = np.unravel_index(np.argmax(Lambda_diff), Lambda_diff.shape)
        print(f"    老代码值: {em_old.Lambda[idx]:.15f}")
        print(f"    新代码值: {Lambda_new[idx]:.15f}")

    print(f"\nA矩阵差异:")
    print(f"  最大差异: {np.max(A_diff):.15f}")
    print(f"  平均差异: {np.mean(A_diff):.15f}")

    print(f"\nQ矩阵差异:")
    print(f"  最大差异: {np.max(Q_diff):.15f}")
    print(f"  平均差异: {np.mean(Q_diff):.15f}")

    print(f"\nR矩阵差异:")
    print(f"  最大差异: {np.max(R_diff):.15f}")
    print(f"  平均差异: {np.mean(R_diff):.15f}")

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)

    all_close = (
        np.max(factors_e_diff) < 1e-10 and
        np.max(Lambda_diff) < 1e-10 and
        np.max(A_diff) < 1e-10 and
        np.max(Q_diff) < 1e-10 and
        np.max(R_diff) < 1e-10
    )

    if all_close:
        print("\n[通过] 第一次EM迭代完全一致!")
    else:
        print("\n[失败] 仍有差异，需要进一步调查:")
        if np.max(factors_e_diff) >= 1e-10:
            print(f"  - E步平滑因子差异: {np.max(factors_e_diff):.15f}")
        if np.max(Lambda_diff) >= 1e-10:
            print(f"  - Lambda差异: {np.max(Lambda_diff):.15f}")
        if np.max(A_diff) >= 1e-10:
            print(f"  - A矩阵差异: {np.max(A_diff):.15f}")
        if np.max(Q_diff) >= 1e-10:
            print(f"  - Q矩阵差异: {np.max(Q_diff):.15f}")
        if np.max(R_diff) >= 1e-10:
            print(f"  - R矩阵差异: {np.max(R_diff):.15f}")


if __name__ == "__main__":
    test_first_em_detailed()
