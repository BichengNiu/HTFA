# -*- coding: utf-8 -*-
"""
详细对比新老Kalman滤波器的内部计算过程
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


def test_kalman_internal():
    """对比Kalman滤波器内部计算"""
    print("="*80)
    print("详细对比新老Kalman滤波器第一个时间步的内部计算")
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

    print("\n初始参数验证:")
    print(f"  Lambda[:2, :] =\n{Lambda[:2, :]}")
    print(f"  A对角线 = {np.diag(A)}")
    print(f"  Q对角线 = {np.diag(Q)}")
    print(f"  R对角线[:5] = {np.diag(R)[:5]}")

    # ========== 新代码：手动执行Kalman滤波第一步 ==========
    print("\n" + "="*80)
    print("新代码：手动执行Kalman滤波第1-2步")
    print("="*80)

    Z_new = obs_centered.values.T  # (n_obs, n_time)
    U_new = np.zeros((n_factors, n_time))

    # 初始化
    x_filt_new = np.zeros((n_factors, n_time))
    P_filt_new = [np.zeros((n_factors, n_factors)) for _ in range(n_time)]
    x_pred_new = np.zeros((n_factors, n_time))
    P_pred_new = [np.zeros((n_factors, n_factors)) for _ in range(n_time)]

    x_filt_new[:, 0] = x0
    P_filt_new[0] = P0.copy()
    x_pred_new[:, 0] = x0
    P_pred_new[0] = P0.copy()

    print(f"\n时刻0 (初始化):")
    print(f"  x_filt[:, 0] = {x_filt_new[:, 0]}")
    print(f"  P_filt[0] 对角线 = {np.diag(P_filt_new[0])}")

    # 时刻1
    t = 1
    print(f"\n时刻{t}:")

    # 预测步
    x_pred_new[:, t] = A @ x_filt_new[:, t-1] + B @ U_new[:, t]
    P_pred_raw = A @ P_filt_new[t-1] @ A.T + Q
    p_jitter = np.eye(n_factors) * 1e-6
    P_pred_new[t] = P_pred_raw + p_jitter

    print(f"  预测步:")
    print(f"    x_pred[:, {t}] = {x_pred_new[:, t]}")
    print(f"    P_pred[{t}] 对角线 = {np.diag(P_pred_new[t])}")

    # 更新步
    ix = np.where(~np.isnan(Z_new[:, t]))[0]
    print(f"  观测数量: {len(ix)}")

    z_t = Z_new[ix, t]
    H_t = Lambda[ix, :]
    R_t = R[np.ix_(ix, ix)]

    innov_t = z_t - H_t @ x_pred_new[:, t]
    S_t = H_t @ P_pred_new[t] @ H_t.T + R_t
    jitter = np.eye(S_t.shape[0]) * 1e-4

    import scipy.linalg
    K_t = scipy.linalg.solve((S_t + jitter).T, (P_pred_new[t] @ H_t.T).T, assume_a='pos').T

    x_filt_new[:, t] = x_pred_new[:, t] + K_t @ innov_t
    P_filt_new[t] = (np.eye(n_factors) - K_t @ H_t) @ P_pred_new[t]
    P_filt_new[t] = (P_filt_new[t] + P_filt_new[t].T) / 2.0

    print(f"  更新步:")
    print(f"    innov_t[:3] = {innov_t[:3]}")
    print(f"    S_t[:2, :2] =\n{S_t[:2, :2]}")
    print(f"    K_t[:, :2] =\n{K_t[:, :2]}")
    print(f"    x_filt[:, {t}] = {x_filt_new[:, t]}")

    # ========== 老代码：手动执行Kalman滤波第一步 ==========
    print("\n" + "="*80)
    print("老代码：手动执行Kalman滤波第1-2步")
    print("="*80)

    z_old = obs_centered.to_numpy()
    u_old = np.zeros((n_time, n_factors))

    x_old = np.zeros(shape=(n_time, n_factors))
    x_old[0, :] = x0
    x_minus_old = np.zeros(shape=(n_time, n_factors))
    x_minus_old[0, :] = x0

    P_old = [np.zeros_like(P0) for _ in range(n_time)]
    P_old[0] = P0.copy()
    P_minus_old = [np.zeros_like(P0) for _ in range(n_time)]
    P_minus_old[0] = P0.copy()

    print(f"\n时刻0 (初始化):")
    print(f"  x[0, :] = {x_old[0, :]}")
    print(f"  P[0] 对角线 = {np.diag(P_old[0])}")

    # 时刻1
    i = 1
    print(f"\n时刻{i}:")

    ix = np.where(~np.isnan(z_old[i, :]))[0]
    print(f"  观测数量: {len(ix)}")

    # 预测步
    x_prev_col = x_old[i-1, :].reshape(-1, 1)
    u_col = u_old[i, :].reshape(-1, 1)

    x_minus_pred = A @ x_prev_col + B @ u_col
    x_minus_old[i, :] = x_minus_pred.flatten()

    P_minus_raw = A @ P_old[i-1] @ A.T + Q
    p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6
    P_minus_old[i] = P_minus_raw + p_jitter

    print(f"  预测步:")
    print(f"    x_minus[{i}, :] = {x_minus_old[i, :]}")
    print(f"    P_minus[{i}] 对角线 = {np.diag(P_minus_old[i])}")

    # 更新步
    z_t = z_old[i, ix]
    H_t = Lambda[ix, :]
    R_t = R[np.ix_(ix, ix)]

    innovation_cov = H_t @ P_minus_old[i] @ H_t.T + R_t
    jitter = np.eye(innovation_cov.shape[0]) * 1e-4

    K_t_effective = scipy.linalg.solve(
        (innovation_cov + jitter).T,
        (P_minus_old[i] @ H_t.T).T,
        assume_a='pos'
    ).T

    x_minus_col = x_minus_old[i, :].reshape(-1, 1)
    z_t_col = z_t.reshape(-1, 1)
    innovation = z_t_col - H_t @ x_minus_col

    x_updated = x_minus_col + K_t_effective @ innovation
    x_old[i, :] = x_updated.flatten()

    I_mat = np.eye(n_factors)
    P_old[i] = (I_mat - K_t_effective @ H_t) @ P_minus_old[i]
    P_old[i] = (P_old[i] + P_old[i].T) / 2.0

    print(f"  更新步:")
    print(f"    innovation[:3] = {innovation.flatten()[:3]}")
    print(f"    innovation_cov[:2, :2] =\n{innovation_cov[:2, :2]}")
    print(f"    K_t[:, :2] =\n{K_t_effective[:, :2]}")
    print(f"    x[{i}, :] = {x_old[i, :]}")

    # ========== 对比 ==========
    print("\n" + "="*80)
    print("对比第1步的中间值")
    print("="*80)

    print(f"\n预测值 x_pred/x_minus:")
    diff_pred = np.abs(x_pred_new[:, 1] - x_minus_old[1, :])
    print(f"  新代码: {x_pred_new[:, 1]}")
    print(f"  老代码: {x_minus_old[1, :]}")
    print(f"  差异: {diff_pred}")
    print(f"  最大差异: {np.max(diff_pred):.15e}")

    print(f"\n新息 innov/innovation:")
    diff_innov = np.abs(innov_t - innovation.flatten())
    print(f"  新代码: {innov_t[:3]}")
    print(f"  老代码: {innovation.flatten()[:3]}")
    print(f"  差异[:3]: {diff_innov[:3]}")
    print(f"  最大差异: {np.max(diff_innov):.15e}")

    print(f"\n新息协方差 S/innovation_cov:")
    diff_S = np.abs(S_t - innovation_cov)
    print(f"  最大差异: {np.max(diff_S):.15e}")
    print(f"  对角线差异: {np.abs(np.diag(S_t) - np.diag(innovation_cov))}")

    print(f"\nKalman增益 K:")
    diff_K = np.abs(K_t - K_t_effective)
    print(f"  最大差异: {np.max(diff_K):.15e}")
    print(f"  第一列差异: {diff_K[:, 0]}")

    print(f"\n滤波值 x_filt/x:")
    diff_filt = np.abs(x_filt_new[:, 1] - x_old[1, :])
    print(f"  新代码: {x_filt_new[:, 1]}")
    print(f"  老代码: {x_old[1, :]}")
    print(f"  差异: {diff_filt}")
    print(f"  最大差异: {np.max(diff_filt):.15e}")

    print(f"\n协方差 P_filt/P:")
    diff_P = np.abs(P_filt_new[1] - P_old[1])
    print(f"  最大差异: {np.max(diff_P):.15e}")
    print(f"  对角线差异: {np.abs(np.diag(P_filt_new[1]) - np.diag(P_old[1]))}")

    # 判断
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if np.max(diff_filt) < 1e-10:
        print("\n[通过] 第1步滤波结果完全一致!")
    else:
        print(f"\n[失败] 第1步滤波结果存在差异: {np.max(diff_filt):.15e}")


if __name__ == "__main__":
    test_kalman_internal()
