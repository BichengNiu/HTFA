# -*- coding: utf-8 -*-
"""
调试Q矩阵计算的详细步骤
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


def debug_q_matrix_calculation():
    """详细调试Q矩阵计算的每个步骤"""
    print("="*80)
    print("调试：Q矩阵计算详细步骤")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    # 设置随机种子
    np.random.seed(42)
    import random
    random.seed(42)

    # ===== 初始化 =====
    print("\n初始化参数...")
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
    Q = np.diag(np.maximum(np.diag(Q), 1e-6))

    # R矩阵
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (obs_std.fillna(1.0)**2).to_numpy()
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    # ===== 第一次E步 =====
    print("\n运行第一次Kalman滤波和平滑...")

    # 老代码
    state_names = [f'Factor{i+1}' for i in range(n_factors)]
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)

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
    fis_old = FIS(kf_old)

    # 新代码
    np.random.seed(42)
    u_data_new = np.random.randn(len(data), n_factors)
    U_new = u_data_new.T

    Z_new = obs_centered.values.T

    kf_new = NewKalmanFilter(A, B, Lambda, Q, R, x0, P0)
    filter_result_new = kf_new.filter(Z_new, U_new)
    smoother_result_new = kf_new.smooth(filter_result_new)

    x_sm_new = smoother_result_new.x_smoothed[:n_factors, :].T

    # ===== M步：Q矩阵计算详细步骤 =====
    print("\n" + "="*80)
    print("M步：Q矩阵计算详细步骤")
    print("="*80)

    # 老代码Q矩阵计算
    print("\n老代码Q矩阵计算:")
    F_old = fis_old.x_sm.values
    print(f"  F shape: {F_old.shape}")

    F_tm1_old = F_old[:-1, :]
    F_t_old = F_old[1:, :]
    print(f"  F_tm1 shape: {F_tm1_old.shape}, F_t shape: {F_t_old.shape}")

    temp_old = F_tm1_old.T @ F_tm1_old
    print(f"  F_tm1.T @ F_tm1 shape: {temp_old.shape}")
    print(f"  F_tm1.T @ F_tm1 前3个对角线元素: {np.diag(temp_old)[:3]}")

    term1_old = (F_t_old.T @ F_t_old) / (len(F_old) - 1)
    print(f"  term1 = (F_t.T @ F_t) / (n-1):")
    print(f"    shape: {term1_old.shape}")
    print(f"    对角线: {np.diag(term1_old)}")

    # 使用老代码计算A矩阵
    from dashboard.DFM.train_model.DiscreteKalmanFilter import _calculate_prediction_matrix
    A_old_for_Q = _calculate_prediction_matrix(fis_old.x_sm)
    print(f"\n  老代码A矩阵 (用于Q计算):")
    print(f"    shape: {A_old_for_Q.shape}")
    print(f"    矩阵值:\n{A_old_for_Q}")

    term2_old = A_old_for_Q @ (temp_old / (len(F_old) - 1)) @ A_old_for_Q.T
    print(f"\n  term2 = A @ (F_tm1.T @ F_tm1 / (n-1)) @ A.T:")
    print(f"    shape: {term2_old.shape}")
    print(f"    对角线: {np.diag(term2_old)}")

    Sigma_old = term1_old - term2_old
    print(f"\n  Sigma = term1 - term2:")
    print(f"    对角线: {np.diag(Sigma_old)}")
    print(f"    是否对称: {np.allclose(Sigma_old, Sigma_old.T)}")
    print(f"    最大非对称性: {np.max(np.abs(Sigma_old - Sigma_old.T))}")

    # 老代码的特征值分解
    eigenvalues_old, eigenvectors_old = np.linalg.eigh(Sigma_old)
    print(f"\n  特征值 (修正前): {eigenvalues_old}")
    eigenvalues_corrected_old = np.maximum(eigenvalues_old, 1e-7)
    print(f"  特征值 (修正后): {eigenvalues_corrected_old}")
    Q_old = eigenvectors_old @ np.diag(eigenvalues_corrected_old) @ eigenvectors_old.T
    print(f"  Q_old对角线: {np.diag(Q_old)}")

    # 新代码Q矩阵计算
    print("\n" + "-"*80)
    print("新代码Q矩阵计算:")
    F_new = x_sm_new
    print(f"  F shape: {F_new.shape}")

    F_tm1_new = F_new[:-1, :]
    F_t_new = F_new[1:, :]
    print(f"  F_tm1 shape: {F_tm1_new.shape}, F_t shape: {F_t_new.shape}")

    temp_new = F_tm1_new.T @ F_tm1_new
    print(f"  F_tm1.T @ F_tm1 shape: {temp_new.shape}")
    print(f"  F_tm1.T @ F_tm1 前3个对角线元素: {np.diag(temp_new)[:3]}")

    term1_new = (F_t_new.T @ F_t_new) / (len(F_new) - 1)
    print(f"  term1 = (F_t.T @ F_t) / (n-1):")
    print(f"    shape: {term1_new.shape}")
    print(f"    对角线: {np.diag(term1_new)}")

    # 使用新代码计算A矩阵
    from dashboard.DFM.train_ref.core.estimator import estimate_transition_matrix
    A_new_for_Q = estimate_transition_matrix(x_sm_new, max_lags=1)
    print(f"\n  新代码A矩阵 (用于Q计算):")
    print(f"    shape: {A_new_for_Q.shape}")
    print(f"    矩阵值:\n{A_new_for_Q}")

    term2_new = A_new_for_Q @ (temp_new / (len(F_new) - 1)) @ A_new_for_Q.T
    print(f"\n  term2 = A @ (F_tm1.T @ F_tm1 / (n-1)) @ A.T:")
    print(f"    shape: {term2_new.shape}")
    print(f"    对角线: {np.diag(term2_new)}")

    Sigma_new = term1_new - term2_new
    print(f"\n  Sigma = term1 - term2:")
    print(f"    对角线: {np.diag(Sigma_new)}")
    print(f"    是否对称: {np.allclose(Sigma_new, Sigma_new.T)}")
    print(f"    最大非对称性: {np.max(np.abs(Sigma_new - Sigma_new.T))}")

    # 新代码的对称化
    Sigma_new_symmetrized = (Sigma_new + Sigma_new.T) / 2
    print(f"\n  Sigma对称化后对角线: {np.diag(Sigma_new_symmetrized)}")
    print(f"  对称化引起的对角线变化: {np.diag(Sigma_new_symmetrized) - np.diag(Sigma_new)}")

    # 新代码的特征值分解
    eigenvalues_new, eigenvectors_new = np.linalg.eigh(Sigma_new_symmetrized)
    print(f"\n  特征值 (修正前): {eigenvalues_new}")
    eigenvalues_corrected_new = np.maximum(eigenvalues_new, 1e-7)
    print(f"  特征值 (修正后): {eigenvalues_corrected_new}")
    Q_new = eigenvectors_new @ np.diag(eigenvalues_corrected_new) @ eigenvectors_new.T
    print(f"  Q_new对角线: {np.diag(Q_new)}")

    # 对比
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)

    print(f"\nF矩阵差异: {np.max(np.abs(F_new - F_old)):.15f}")
    print(f"A矩阵差异: {np.max(np.abs(A_new_for_Q - A_old_for_Q)):.15f}")
    print(f"term1差异: {np.max(np.abs(term1_new - term1_old)):.15f}")
    print(f"term2差异: {np.max(np.abs(term2_new - term2_old)):.15f}")
    print(f"Sigma差异 (对称化前): {np.max(np.abs(Sigma_new - Sigma_old)):.15f}")
    print(f"Sigma差异 (对称化后): {np.max(np.abs(Sigma_new_symmetrized - Sigma_old)):.15f}")
    print(f"特征值差异 (修正后): {np.max(np.abs(eigenvalues_corrected_new - eigenvalues_corrected_old)):.15f}")
    print(f"Q矩阵差异: {np.max(np.abs(Q_new - Q_old)):.15f}")


if __name__ == "__main__":
    debug_q_matrix_calculation()
