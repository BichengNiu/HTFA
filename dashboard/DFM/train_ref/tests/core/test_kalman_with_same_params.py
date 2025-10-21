# -*- coding: utf-8 -*-
"""
使用完全相同的参数对比新老Kalman滤波
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


def test_kalman_with_same_params():
    """使用完全相同的参数测试新老Kalman滤波"""
    print("="*80)
    print("使用完全相同参数对比新老Kalman滤波")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    # 初始化（使用老代码的方法）
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    # PCA
    U, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    factors_init_df = pd.DataFrame(factors_init, index=z.index, columns=[f'Factor{i+1}' for i in range(n_factors)])

    # Lambda（使用老代码函数）
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda = calculate_factor_loadings(obs_centered, factors_init_df)

    # VAR（使用老代码方法）
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(max_lags)
    A = var_results.coefs[0]
    Q = np.cov(var_results.resid, rowvar=False)
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    # R矩阵
    V = Vh.T
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # 其他参数
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)

    print(f"\n使用的参数:")
    print(f"  Lambda shape: {Lambda.shape}")
    print(f"  A:\n{A}")
    print(f"  Q对角线: {np.diag(Q)}")
    print(f"  R对角线前3个: {np.diag(R)[:3]}")

    # ========== 老代码Kalman滤波 ==========
    print("\n" + "="*80)
    print("老代码Kalman滤波")
    print("="*80)

    from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKalmanFilter, FIS

    # error_df
    np.random.seed(DFM_SEED)
    u_data = np.random.randn(len(data), n_factors)
    error_df = pd.DataFrame(u_data, columns=[f'shock{i+1}' for i in range(n_factors)], index=data.index)

    state_names = [f'Factor{i+1}' for i in range(n_factors)]

    kf_old = OldKalmanFilter(
        Z=obs_centered,
        U=error_df,
        A=A,
        B=B,
        H=Lambda,
        state_names=state_names,
        x0=x0,
        P0=P0,
        Q=Q,
        R=R
    )
    fis_old = FIS(kf_old)

    print(f"\n老代码平滑因子前3行:")
    print(fis_old.x_sm.iloc[:3])

    # ========== 新代码Kalman滤波 ==========
    print("\n" + "="*80)
    print("新代码Kalman滤波")
    print("="*80)

    from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKalmanFilter

    # 转换数据格式
    Z = obs_centered.values.T  # (n_obs, n_time)

    # U矩阵（使用与老代码相同的error_df）
    U = error_df.values.T  # (n_factors, n_time)

    kf_new = NewKalmanFilter(A, B, Lambda, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z, U)
    smoother_result_new = kf_new.smooth(filter_result_new)

    factors_smoothed_new = smoother_result_new.x_smoothed[:n_factors, :].T

    print(f"\n新代码平滑因子前3行:")
    print(factors_smoothed_new[:3])

    # ========== 对比 ==========
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    factors_diff = np.abs(factors_smoothed_new - fis_old.x_sm.values)

    print(f"\n平滑因子差异:")
    print(f"  最大差异: {np.max(factors_diff):.15f}")
    print(f"  平均差异: {np.mean(factors_diff):.15f}")

    if np.max(factors_diff) < 1e-10:
        print("\n[通过] 使用相同参数时，新老Kalman滤波完全一致！")
    else:
        print(f"\n[失败] 仍有差异: {np.max(factors_diff):.15f}")
        print("\n差异最大的3行:")
        max_row_idx = np.where(factors_diff == np.max(factors_diff))[0][0]
        for i in range(max(0, max_row_idx-1), min(len(factors_diff), max_row_idx+2)):
            print(f"  行{i}: 老={fis_old.x_sm.iloc[i].values}, 新={factors_smoothed_new[i]}")


if __name__ == "__main__":
    test_kalman_with_same_params()
