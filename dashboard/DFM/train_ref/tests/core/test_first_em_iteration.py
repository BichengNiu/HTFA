# -*- coding: utf-8 -*-
"""
调试EM算法第一次迭代

详细对比新旧代码在第一次EM迭代中的每个步骤
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.factor_model import DFMModel
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo
from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKalmanFilter, FIS
from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKalmanFilter


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


def test_first_em_iteration():
    """测试第一次EM迭代"""
    print("="*80)
    print("调试：EM算法第一次迭代")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    # 设置随机种子
    np.random.seed(42)
    import random
    random.seed(42)

    # ===== 初始化阶段 =====
    print("\n初始化参数...")

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

    # 载荷
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    factors_df = pd.DataFrame(factors_init, index=data.index, columns=[f'Factor{i+1}' for i in range(n_factors)])
    Lambda = calculate_factor_loadings(obs_centered, factors_df)

    # VAR模型
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_df.dropna())
    var_results = var_model.fit(1)
    A = var_results.coefs[0]
    Q = np.cov(var_results.resid, rowvar=False)

    # R矩阵
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std.fillna(1.0)**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # 确保正定性
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))
    R = np.diag(np.maximum(np.diag(R), 1e-6))

    print(f"初始化完成:")
    print(f"  Lambda: {Lambda.shape}, 范围: [{Lambda.min():.4f}, {Lambda.max():.4f}]")
    print(f"  A: {A.shape}")
    print(f"  Q: {Q.shape}, 对角线: {np.diag(Q)[:3]}")
    print(f"  R: {R.shape}, 对角线: {np.diag(R)[:3]}")

    # ===== E步：Kalman滤波和平滑 =====
    print("\n" + "="*80)
    print("E步：Kalman滤波和平滑")
    print("="*80)

    # 准备数据
    state_names = [f'Factor{i+1}' for i in range(n_factors)]
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)

    # 老代码
    print("\n运行老代码Kalman滤波...")
    np.random.seed(42)
    u_data_old = np.random.randn(len(data), n_factors)
    error_df_old = pd.DataFrame(u_data_old, columns=[f'shock{i+1}' for i in range(n_factors)], index=data.index)

    kf_old = OldKalmanFilter(
        Z=obs_centered,
        U=error_df_old,
        A=A,
        B=B,
        H=Lambda,
        state_names=state_names,
        x0=x0,
        P0=P0,
        Q=Q,
        R=R
    )

    # 平滑
    fis_old = FIS(kf_old)

    print(f"老代码滤波完成")
    print(f"  平滑因子形状: {fis_old.x_sm.shape}")
    print(f"  平滑因子前3个值:\n{fis_old.x_sm.iloc[:3].values}")

    # 新代码
    print("\n运行新代码Kalman滤波...")
    np.random.seed(42)
    u_data_new = np.random.randn(len(data), n_factors)
    U_new = u_data_new.T  # (n_factors, n_time)

    Z_new = obs_centered.values.T  # (n_obs, n_time)

    kf_new = NewKalmanFilter(A, B, Lambda, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z_new, U_new)
    smoother_result_new = kf_new.smooth(filter_result_new)

    x_sm_new = smoother_result_new.x_smoothed[:n_factors, :].T  # (n_time, n_factors)

    print(f"新代码滤波完成")
    print(f"  平滑因子形状: {x_sm_new.shape}")
    print(f"  平滑因子前3个值:\n{x_sm_new[:3]}")

    # 对比
    diff_factors = np.abs(x_sm_new - fis_old.x_sm.values)
    print(f"\n平滑因子差异:")
    print(f"  最大差异: {np.max(diff_factors):.15f}")
    print(f"  平均差异: {np.mean(diff_factors):.15f}")

    # ===== M步：参数更新 =====
    print("\n" + "="*80)
    print("M步：参数更新")
    print("="*80)

    # 老代码的M步
    from dashboard.DFM.train_model.DiscreteKalmanFilter import EMstep
    em_old = EMstep(fis_old, n_factors)

    print("\n老代码M步更新:")
    print(f"  Lambda: {em_old.Lambda.shape}, 范围: [{em_old.Lambda.min():.4f}, {em_old.Lambda.max():.4f}]")
    print(f"  A: {em_old.A.shape}")
    print(f"  Q对角线: {np.diag(em_old.Q)[:3]}")
    print(f"  R对角线: {np.diag(em_old.R)[:3]}")

    # 新代码的M步
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings, estimate_transition_matrix, estimate_covariance_matrices

    factors_df_new = pd.DataFrame(x_sm_new, columns=[f'Factor{i+1}' for i in range(n_factors)])
    Lambda_new = estimate_loadings(obs_centered, factors_df_new)
    A_new = estimate_transition_matrix(x_sm_new, max_lags=1)
    Q_new, R_new = estimate_covariance_matrices(
        smoother_result_new,
        obs_centered,
        Lambda_new,
        n_factors,
        A_new  # 传入A矩阵用于Q矩阵计算
    )

    print("\n新代码M步更新:")
    print(f"  Lambda: {Lambda_new.shape}, 范围: [{Lambda_new.min():.4f}, {Lambda_new.max():.4f}]")
    print(f"  A: {A_new.shape}")
    print(f"  Q对角线: {np.diag(Q_new)[:3]}")
    print(f"  R对角线: {np.diag(R_new)[:3]}")

    # 对比更新后的参数
    print("\n" + "="*80)
    print("参数差异对比")
    print("="*80)

    diff_Lambda = np.abs(Lambda_new - em_old.Lambda)
    diff_A = np.abs(A_new - em_old.A)
    diff_Q = np.abs(Q_new - em_old.Q)
    diff_R = np.abs(R_new - em_old.R)

    print(f"\nLambda差异: 最大={np.max(diff_Lambda):.10f}, 平均={np.mean(diff_Lambda):.10f}")
    print(f"A差异: 最大={np.max(diff_A):.10f}, 平均={np.mean(diff_A):.10f}")
    print(f"Q差异: 最大={np.max(diff_Q):.10f}, 平均={np.mean(diff_Q):.10f}")
    print(f"R差异: 最大={np.max(diff_R):.10f}, 平均={np.mean(diff_R):.10f}")

    # 详细对比Lambda前几个元素
    print("\nLambda前5个元素对比:")
    for i in range(min(5, Lambda_new.shape[0])):
        for j in range(n_factors):
            print(f"  Lambda[{i},{j}]: {Lambda_new[i,j]:.10f} vs {em_old.Lambda[i,j]:.10f} (diff={diff_Lambda[i,j]:.10f})")


if __name__ == "__main__":
    test_first_em_iteration()
