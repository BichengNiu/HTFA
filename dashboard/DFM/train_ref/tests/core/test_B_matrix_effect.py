# -*- coding: utf-8 -*-
"""
测试B矩阵差异对Kalman滤波的影响
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKF


def test_B_matrix_effect():
    """测试B矩阵对Kalman滤波的影响"""
    print("="*80)
    print("测试B矩阵差异的影响")
    print("="*80)

    # 设置参数
    n_factors = 3
    n_obs = 5
    n_time = 10

    np.random.seed(42)

    # 创建测试数据
    data = pd.DataFrame(
        np.random.randn(n_time, n_obs),
        columns=[f'var{i}' for i in range(n_obs)]
    )
    obs_mean = data.mean(skipna=True)
    obs_centered = data - obs_mean
    Z = obs_centered.values.T  # (n_obs, n_time)

    # 固定参数
    Lambda = np.random.randn(n_obs, n_factors) * 0.5
    A = np.random.randn(n_factors, n_factors) * 0.3
    Q = np.eye(n_factors) * 0.1
    R = np.eye(n_obs) * 0.2
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)

    # 零矩阵U
    U_zeros = np.zeros((n_factors, n_time))

    print(f"\n测试1: B = I (单位矩阵)")
    B1 = np.eye(n_factors)
    kf1 = NewKF(A, B1, Lambda, Q, R, x0, P0)
    result1 = kf1.filter(Z, U_zeros)
    smoother_result1 = kf1.smooth(result1)
    factors1 = smoother_result1.x_smoothed[:n_factors, :].T

    print(f"  平滑因子前3行:\n{factors1[:3]}")

    print(f"\n测试2: B = I * 0.1")
    B2 = np.eye(n_factors) * 0.1
    kf2 = NewKF(A, B2, Lambda, Q, R, x0, P0)
    result2 = kf2.filter(Z, U_zeros)
    smoother_result2 = kf2.smooth(result2)
    factors2 = smoother_result2.x_smoothed[:n_factors, :].T

    print(f"  平滑因子前3行:\n{factors2[:3]}")

    print(f"\n对比结果:")
    diff = np.abs(factors1 - factors2)
    print(f"  最大差异: {np.max(diff):.15f}")
    print(f"  平均差异: {np.mean(diff):.15f}")

    if np.max(diff) < 1e-10:
        print(f"\n[结论] B矩阵差异对结果没有影响（U全为0时）")
    else:
        print(f"\n[结论] B矩阵差异影响了结果！")
        print(f"  这说明即使U全为0，B矩阵仍然影响Kalman滤波！")


if __name__ == "__main__":
    test_B_matrix_effect()
