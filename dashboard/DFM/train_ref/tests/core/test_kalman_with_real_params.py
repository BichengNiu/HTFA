# -*- coding: utf-8 -*-
"""
使用真实DFM初始化参数测试新老Kalman滤波器
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKF
from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKF


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


def test_kalman_with_real_params():
    """使用真实参数测试Kalman滤波器"""
    print("="*80)
    print("使用真实DFM参数测试新老Kalman滤波器")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    n_time = len(data)
    n_obs = data.shape[1]

    # 预处理数据
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    # PCA初始化
    U_svd, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U_svd[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_init_df = pd.DataFrame(
        factors_init,
        index=z.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda初始化
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda = calculate_factor_loadings(obs_centered, factors_init_df)

    # VAR估计A和Q
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(1)
    A = var_results.coefs[0]
    Q = np.cov(var_results.resid, rowvar=False)
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    # R矩阵
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # 初始状态
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)

    # U矩阵（零冲击）
    U_zeros = np.zeros((n_factors, n_time))

    print(f"\n参数:")
    print(f"  Lambda[:2, :] =\n{Lambda[:2, :]}")
    print(f"  A =\n{A}")
    print(f"  Q_diag = {np.diag(Q)}")
    print(f"  R_diag[:5] = {np.diag(R)[:5]}")

    # ========== 新代码 ==========
    print("\n" + "="*80)
    print("新代码Kalman滤波")
    print("="*80)

    Z_new = obs_centered.values.T  # (n_obs, n_time)
    U_new = U_zeros  # (n_factors, n_time)

    kf_new = NewKF(A, B, Lambda, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z_new, U_new)
    smoother_result_new = kf_new.smooth(filter_result_new)

    factors_new = smoother_result_new.x_smoothed[:n_factors, :].T

    print(f"  平滑因子前3行:\n{factors_new[:3]}")
    print(f"  平滑因子范围: [{factors_new.min():.6f}, {factors_new.max():.6f}]")

    # ========== 老代码 ==========
    print("\n" + "="*80)
    print("老代码Kalman滤波")
    print("="*80)

    Z_old = obs_centered  # DataFrame (n_time, n_obs)
    U_old = pd.DataFrame(
        U_zeros.T,  # (n_time, n_factors)
        index=obs_centered.index,
        columns=[f'shock{i+1}' for i in range(n_factors)]
    )

    state_names = [f'Factor{i+1}' for i in range(n_factors)]

    kf_old = OldKF(Z_old, U_old, A, B, Lambda, state_names, x0, P0, Q, R)

    from dashboard.DFM.train_model.DiscreteKalmanFilter import FIS
    fis_old = FIS(kf_old)

    factors_old = fis_old.x_sm.values

    print(f"  平滑因子前3行:\n{factors_old[:3]}")
    print(f"  平滑因子范围: [{factors_old.min():.6f}, {factors_old.max():.6f}]")

    # ========== 对比 ==========
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    diff = np.abs(factors_new - factors_old)
    print(f"\n因子差异:")
    print(f"  最大差异: {np.max(diff):.15f}")
    print(f"  平均差异: {np.mean(diff):.15f}")

    # 找出最大差异的位置
    if np.max(diff) >= 1e-10:
        print(f"\n差异详情:")
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  最大差异位置: 时刻{idx[0]}, 因子{idx[1]}")
        print(f"  新代码值: {factors_new[idx]:.15f}")
        print(f"  老代码值: {factors_old[idx]:.15f}")

        # 打印前几个时刻的详细对比
        print(f"\n前5个时刻的逐元素对比:")
        for t in range(min(5, n_time)):
            print(f"\n  时刻{t}:")
            print(f"    新代码: {factors_new[t]}")
            print(f"    老代码: {factors_old[t]}")
            print(f"    差异: {diff[t]}")
            print(f"    差异最大值: {np.max(diff[t]):.15f}")

    if np.max(diff) < 1e-10:
        print(f"\n[通过] 新老Kalman滤波器结果完全一致!")
    else:
        print(f"\n[失败] 存在差异，需要进一步调查")


if __name__ == "__main__":
    test_kalman_with_real_params()
