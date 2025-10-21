# -*- coding: utf-8 -*-
"""
精确对比新老代码进入第一次Kalman滤波时的所有参数
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


def get_old_init_params():
    """获取老代码进入第一次Kalman滤波的参数"""
    data = create_test_data()
    n_factors = 3
    max_lags = 1

    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    n_obs = data.shape[1]
    n_time = data.shape[0]
    n_shocks = n_factors

    # 数据预处理
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    # PCA
    U, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    factors_init_df = pd.DataFrame(factors_init, index=z.index, columns=[f'Factor{i+1}' for i in range(n_factors)])

    # Lambda
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda_current = calculate_factor_loadings(obs_centered, factors_init_df)

    # VAR
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(max_lags)
    A_current = var_results.coefs[0]
    Q_current = np.cov(var_results.resid, rowvar=False)
    Q_current = np.diag(np.maximum(np.diag(Q_current), 1e-6))

    # R
    V = Vh.T
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag_current = psi_diag * (obs_std**2).to_numpy()
    R_diag_current = np.maximum(R_diag_current, 1e-6)
    R_current = np.diag(R_diag_current)

    # 其他参数
    x0_current = np.zeros(n_factors)
    P0_current = np.eye(n_factors)
    B_current = np.eye(n_factors)

    # error_df
    np.random.seed(DFM_SEED)
    u_data = np.random.randn(n_time, n_shocks)
    error_df = pd.DataFrame(u_data, columns=[f'shock{i+1}' for i in range(n_shocks)], index=data.index)

    return {
        'obs_centered': obs_centered,
        'error_df': error_df,
        'A': A_current,
        'B': B_current,
        'H': Lambda_current,
        'x0': x0_current,
        'P0': P0_current,
        'Q': Q_current,
        'R': R_current
    }


def get_new_init_params():
    """获取新代码进入第一次Kalman滤波的参数"""
    data = create_test_data()
    n_factors = 3
    max_lags = 1

    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    # 数据预处理
    means = data.mean(skipna=True).values
    stds = data.std(skipna=True).values
    stds = np.where(stds > 0, stds, 1.0)
    obs_centered = data - means
    Z_standardized = (obs_centered / stds).fillna(0).values

    # PCA
    U, s, Vh = np.linalg.svd(Z_standardized, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_df = pd.DataFrame(
        factors_init,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings
    initial_loadings = estimate_loadings(obs_centered, factors_df)

    # 使用VAR估计A和Q（完全匹配factor_model.py的实现）
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_df.dropna())
    var_results = var_model.fit(max_lags)

    if max_lags == 1:
        # VAR(1): 直接使用系数矩阵
        A = var_results.coefs[0]
    else:
        # VAR(p): 构造companion form矩阵
        n_factors_orig = factors_df.shape[1]
        A = np.zeros((n_factors_orig * max_lags, n_factors_orig * max_lags))
        # 填充VAR系数
        for lag in range(max_lags):
            A[:n_factors_orig, lag*n_factors_orig:(lag+1)*n_factors_orig] = var_results.coefs[lag]
        # 构造companion form下半部分
        if max_lags > 1:
            A[n_factors_orig:, :-n_factors_orig] = np.eye(n_factors_orig * (max_lags - 1))

    # 使用VAR残差计算Q矩阵（匹配老代码）
    Q = np.cov(var_results.resid, rowvar=False)
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    # R
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = Z_standardized - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (stds ** 2)
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # Kalman参数
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    Lambda = initial_loadings.copy()
    B = np.eye(n_factors)

    # U矩阵
    n_time = data.shape[0]
    U_matrix = np.zeros((n_factors, n_time))

    # 转换obs_centered为Kalman输入格式
    Z = obs_centered.values.T  # (n_obs, n_time)

    return {
        'Z': Z,
        'U': U_matrix,
        'A': A,
        'B': B,
        'H': Lambda,
        'x0': x0,
        'P0': P0,
        'Q': Q,
        'R': R,
        'obs_centered': obs_centered
    }


if __name__ == "__main__":
    print("="*80)
    print("精确对比新老代码第一次Kalman滤波的输入参数")
    print("="*80)

    old_params = get_old_init_params()
    new_params = get_new_init_params()

    print("\n观测数据obs_centered对比:")
    obs_old = old_params['obs_centered'].values
    obs_new = new_params['obs_centered'].values
    obs_diff = np.abs(obs_old - obs_new)
    print(f"  Shape: {obs_old.shape} vs {obs_new.shape}")
    print(f"  最大差异: {np.max(obs_diff):.15f}")
    print(f"  老代码前3行前3列:\n{obs_old[:3, :3]}")
    print(f"  新代码前3行前3列:\n{obs_new[:3, :3]}")

    print("\n载荷矩阵H对比:")
    H_diff = np.abs(old_params['H'] - new_params['H'])
    print(f"  Shape: {old_params['H'].shape} vs {new_params['H'].shape}")
    print(f"  最大差异: {np.max(H_diff):.15f}")
    print(f"  老代码前3行:\n{old_params['H'][:3]}")
    print(f"  新代码前3行:\n{new_params['H'][:3]}")

    print("\n状态转移矩阵A对比:")
    A_diff = np.abs(old_params['A'] - new_params['A'])
    print(f"  最大差异: {np.max(A_diff):.15f}")
    print(f"  老代码:\n{old_params['A']}")
    print(f"  新代码:\n{new_params['A']}")

    print("\n过程噪声Q对比:")
    Q_diff = np.abs(old_params['Q'] - new_params['Q'])
    print(f"  最大差异: {np.max(Q_diff):.15f}")
    print(f"  老代码对角线: {np.diag(old_params['Q'])}")
    print(f"  新代码对角线: {np.diag(new_params['Q'])}")

    print("\n观测噪声R对比:")
    R_diff = np.abs(old_params['R'] - new_params['R'])
    print(f"  最大差异: {np.max(R_diff):.15f}")
    print(f"  老代码对角线前5个: {np.diag(old_params['R'])[:5]}")
    print(f"  新代码对角线前5个: {np.diag(new_params['R'])[:5]}")

    print("\n初始状态x0对比:")
    x0_diff = np.abs(old_params['x0'] - new_params['x0'])
    print(f"  最大差异: {np.max(x0_diff):.15f}")
    print(f"  老代码: {old_params['x0']}")
    print(f"  新代码: {new_params['x0']}")

    print("\n初始协方差P0对比:")
    P0_diff = np.abs(old_params['P0'] - new_params['P0'])
    print(f"  最大差异: {np.max(P0_diff):.15f}")
    print(f"  老代码对角线: {np.diag(old_params['P0'])}")
    print(f"  新代码对角线: {np.diag(new_params['P0'])}")

    print("\n" + "="*80)
    print("总结")
    print("="*80)

    all_diffs = {
        'obs_centered': np.max(obs_diff),
        'H': np.max(H_diff),
        'A': np.max(A_diff),
        'Q': np.max(Q_diff),
        'R': np.max(R_diff),
        'x0': np.max(x0_diff),
        'P0': np.max(P0_diff)
    }

    for name, diff in all_diffs.items():
        status = "[OK]" if diff < 1e-10 else "[DIFF]"
        print(f"  {status} {name}: {diff:.15f}")
