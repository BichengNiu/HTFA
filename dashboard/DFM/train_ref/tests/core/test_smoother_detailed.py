# -*- coding: utf-8 -*-
"""
详细对比新老Kalman平滑器的计算过程
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKF
from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKF, FIS


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


def test_smoother():
    """对比平滑器"""
    print("="*80)
    print("详细对比新老Kalman平滑器")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    n_time = len(data)
    n_obs = data.shape[1]

    # 预处理
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    # PCA
    U_svd, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U_svd[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_init_df = pd.DataFrame(
        factors_init,
        index=z.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda = calculate_factor_loadings(obs_centered, factors_init_df)

    # VAR
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(1)
    A = var_results.coefs[0]
    Q = np.cov(var_results.resid, rowvar=False)
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    # R
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # 初始状态
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors) * 0.1
    U_zeros = np.zeros((n_factors, n_time))

    # ========== 新代码：完整执行 ==========
    print("\n" + "="*80)
    print("新代码：完整执行滤波和平滑")
    print("="*80)

    Z_new = obs_centered.values.T
    kf_new = NewKF(A, B, Lambda, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z_new, U_zeros)
    smoother_result_new = kf_new.smooth(filter_result_new)

    x_filtered_new = filter_result_new.x_filtered.T  # (n_time, n_factors)
    x_smoothed_new = smoother_result_new.x_smoothed.T  # (n_time, n_factors)

    print(f"  滤波因子前3行:\n{x_filtered_new[:3]}")
    print(f"  平滑因子前3行:\n{x_smoothed_new[:3]}")
    print(f"  滤波因子[0]: {x_filtered_new[0]}")
    print(f"  平滑因子[0]: {x_smoothed_new[0]}")

    # ========== 老代码：完整执行 ==========
    print("\n" + "="*80)
    print("老代码：完整执行滤波和平滑")
    print("="*80)

    Z_old = obs_centered
    U_old = pd.DataFrame(
        U_zeros.T,
        index=obs_centered.index,
        columns=[f'shock{i+1}' for i in range(n_factors)]
    )
    state_names = [f'Factor{i+1}' for i in range(n_factors)]

    kf_old = OldKF(Z_old, U_old, A, B, Lambda, state_names, x0, P0, Q, R)
    fis_old = FIS(kf_old)

    x_filtered_old = kf_old.x.values  # (n_time, n_factors)
    x_smoothed_old = fis_old.x_sm.values  # (n_time, n_factors)

    print(f"  滤波因子前3行:\n{x_filtered_old[:3]}")
    print(f"  平滑因子前3行:\n{x_smoothed_old[:3]}")
    print(f"  滤波因子[0]: {x_filtered_old[0]}")
    print(f"  平滑因子[0]: {x_smoothed_old[0]}")

    # ========== 对比 ==========
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    diff_filtered = np.abs(x_filtered_new - x_filtered_old)
    diff_smoothed = np.abs(x_smoothed_new - x_smoothed_old)

    print(f"\n滤波因子差异:")
    print(f"  最大差异: {np.max(diff_filtered):.15e}")
    print(f"  平均差异: {np.mean(diff_filtered):.15e}")
    print(f"  前3个时刻最大差异: {np.max(diff_filtered[:3]):.15e}")

    print(f"\n平滑因子差异:")
    print(f"  最大差异: {np.max(diff_smoothed):.15e}")
    print(f"  平均差异: {np.mean(diff_smoothed):.15e}")
    print(f"  前3个时刻最大差异: {np.max(diff_smoothed[:3]):.15e}")

    # 逐时刻对比
    print(f"\n前5个时刻的逐元素对比:")
    for t in range(min(5, n_time)):
        diff_t_filt = np.max(diff_filtered[t])
        diff_t_smooth = np.max(diff_smoothed[t])
        print(f"  时刻{t}: 滤波差异={diff_t_filt:.15e}, 平滑差异={diff_t_smooth:.15e}")

    # 判断
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if np.max(diff_filtered) < 1e-10:
        print("\n[通过] 滤波因子完全一致!")
    else:
        print(f"\n[失败] 滤波因子存在差异: {np.max(diff_filtered):.15e}")

    if np.max(diff_smoothed) < 1e-10:
        print("\n[通过] 平滑因子完全一致!")
    else:
        print(f"\n[失败] 平滑因子存在差异: {np.max(diff_smoothed):.15e}")

        # 找出差异最大的时刻
        idx = np.unravel_index(np.argmax(diff_smoothed), diff_smoothed.shape)
        print(f"\n差异最大的位置: 时刻{idx[0]}, 因子{idx[1]}")
        print(f"  新代码值: {x_smoothed_new[idx]:.15f}")
        print(f"  老代码值: {x_smoothed_old[idx]:.15f}")


if __name__ == "__main__":
    test_smoother()
