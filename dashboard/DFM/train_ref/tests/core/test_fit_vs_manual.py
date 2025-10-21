# -*- coding: utf-8 -*-
"""
对比DFMModel.fit()与手动执行EM步骤的结果
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.factor_model import DFMModel


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


def test_fit_vs_manual():
    """对比fit()方法与手动执行的结果"""
    print("="*80)
    print("对比DFMModel.fit()与手动执行EM步骤")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # 使用fit()方法
    print("\n使用DFMModel.fit(max_iter=1)...")
    model = DFMModel(n_factors=n_factors, max_lags=1, max_iter=1)
    result_fit = model.fit(data)

    print(f"  拟合完成: 迭代{result_fit.n_iter}次")
    print(f"  Lambda shape: {result_fit.loadings.shape}")
    print(f"  Factors shape: {result_fit.factors.shape}")
    print(f"  Lambda范围: [{result_fit.loadings.min():.4f}, {result_fit.loadings.max():.4f}]")
    print(f"  Factors范围: [{result_fit.factors.values.min():.4f}, {result_fit.factors.values.max():.4f}]")

    # 手动执行EM步骤(参考test_first_em_iteration)
    print("\n手动执行EM步骤...")

    # 数据预处理
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = ((data - obs_mean) / obs_std).fillna(0)

    # SVD
    U, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    V = Vh.T

    # 初始载荷
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings
    factors_df_init = pd.DataFrame(factors_init, columns=[f'Factor{i+1}' for i in range(n_factors)])
    Lambda_init = estimate_loadings(obs_centered, factors_df_init)

    print(f"  初始Lambda范围: [{Lambda_init.min():.4f}, {Lambda_init.max():.4f}]")

    # VAR估计A和Q
    from dashboard.DFM.train_ref.core.estimator import estimate_transition_matrix
    A_init = estimate_transition_matrix(factors_init, max_lags=1)
    Q_init = np.eye(n_factors) * 0.1

    # R矩阵
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std ** 2).values
    R_diag = np.maximum(R_diag, 1e-6)
    R_init = np.diag(R_diag)

    # E步：Kalman滤波
    from dashboard.DFM.train_ref.core.kalman import KalmanFilter

    Z = obs_centered.values.T  # (n_obs, n_time)
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)
    U = np.zeros((n_factors, len(data)))

    kf = KalmanFilter(A_init, B, Lambda_init, Q_init, R_init, x0, P0)
    filter_result = kf.filter(Z, U)
    smoother_result = kf.smooth(filter_result)

    factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
    factors_df_smoothed = pd.DataFrame(
        factors_smoothed,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # M步：更新参数
    from dashboard.DFM.train_ref.core.estimator import estimate_covariance_matrices

    Lambda_manual = estimate_loadings(obs_centered, factors_df_smoothed)
    A_manual = estimate_transition_matrix(factors_smoothed, max_lags=1)
    Q_manual, R_manual = estimate_covariance_matrices(
        smoother_result,
        obs_centered,
        Lambda_manual,
        n_factors,
        A_manual
    )

    print(f"  手动Lambda范围: [{Lambda_manual.min():.4f}, {Lambda_manual.max():.4f}]")
    print(f"  手动Factors范围: [{factors_smoothed.min():.4f}, {factors_smoothed.max():.4f}]")

    # 对比结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    Lambda_diff = np.abs(result_fit.loadings - Lambda_manual)
    factors_diff = np.abs(result_fit.factors.values - factors_smoothed)

    print(f"\nLambda差异: 最大={np.max(Lambda_diff):.15f}, 平均={np.mean(Lambda_diff):.15f}")
    print(f"Factors差异: 最大={np.max(factors_diff):.15f}, 平均={np.mean(factors_diff):.15f}")

    if np.max(Lambda_diff) < 1e-10 and np.max(factors_diff) < 1e-10:
        print("\n[通过] DFMModel.fit()与手动执行完全一致!")
    else:
        print("\n[失败] 存在差异!")
        print("\nLambda前3个元素对比:")
        for i in range(min(3, Lambda_manual.shape[0])):
            for j in range(n_factors):
                print(f"  Lambda[{i},{j}]: fit={result_fit.loadings[i,j]:.10f} vs manual={Lambda_manual[i,j]:.10f}")


if __name__ == "__main__":
    test_fit_vs_manual()
