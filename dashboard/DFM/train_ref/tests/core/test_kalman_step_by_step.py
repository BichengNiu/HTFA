# -*- coding: utf-8 -*-
"""
逐步对比新老Kalman滤波器的每一步计算
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
    n_time = 10  # 使用较小的时间步数便于调试
    n_obs = 5
    dates = pd.date_range('2015-01-01', periods=n_time, freq='ME')
    data = pd.DataFrame(
        np.random.randn(n_time, n_obs),
        index=dates,
        columns=[f'var{i}' for i in range(n_obs)]
    )
    return data


def test_kalman_step_by_step():
    """逐步对比Kalman滤波"""
    print("="*80)
    print("逐步对比新老Kalman滤波器")
    print("="*80)

    # 设置参数
    n_factors = 3
    n_obs = 5
    n_time = 10

    np.random.seed(42)

    # 创建测试数据
    data = create_test_data()
    obs_mean = data.mean(skipna=True)
    obs_centered = data - obs_mean

    # 定义固定参数
    Lambda = np.random.randn(n_obs, n_factors) * 0.5
    A = np.random.randn(n_factors, n_factors) * 0.3
    Q = np.eye(n_factors) * 0.1
    R = np.eye(n_obs) * 0.2
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    B = np.eye(n_factors)
    U_zeros = np.zeros((n_factors, n_time))

    print(f"\n初始参数:")
    print(f"  Lambda[:2, :] =\n{Lambda[:2, :]}")
    print(f"  A =\n{A}")
    print(f"  Q_diag = {np.diag(Q)}")
    print(f"  R_diag = {np.diag(R)}")
    print(f"  x0 = {x0}")
    print(f"  P0_diag = {np.diag(P0)}")

    # ========== 新代码 ==========
    print("\n" + "="*80)
    print("新代码Kalman滤波")
    print("="*80)

    Z_new = obs_centered.values.T  # (n_obs, n_time)
    U_new = U_zeros  # (n_factors, n_time)

    print(f"\n新代码输入:")
    print(f"  Z.shape = {Z_new.shape}")
    print(f"  U.shape = {U_new.shape}")
    print(f"  Z[:, 0] = {Z_new[:, 0]}")
    print(f"  U[:, 0] = {U_new[:, 0]}")

    kf_new = NewKF(A, B, Lambda, Q, R, x0, P0)

    # 手动执行滤波的前几步以打印中间值
    print("\n手动执行新代码滤波前3步:")
    n_states = A.shape[0]
    x_filt = np.zeros((n_states, n_time))
    P_filt = [np.zeros((n_states, n_states)) for _ in range(n_time)]
    x_pred = np.zeros((n_states, n_time))
    P_pred = [np.zeros((n_states, n_states)) for _ in range(n_time)]

    # 初始化
    x_filt[:, 0] = x0
    P_filt[0] = P0.copy()
    x_pred[:, 0] = x0
    P_pred[0] = P0.copy()

    print(f"\n时刻0 (初始化):")
    print(f"  x_filt[:, 0] = {x_filt[:, 0]}")
    print(f"  x_pred[:, 0] = {x_pred[:, 0]}")

    # 前3个时间步
    for t in range(1, min(4, n_time)):
        print(f"\n时刻{t}:")

        # 预测步
        x_pred[:, t] = A @ x_filt[:, t-1] + B @ U_new[:, t]
        P_pred_raw = A @ P_filt[t-1] @ A.T + Q
        p_jitter = np.eye(n_states) * 1e-6
        P_pred[t] = P_pred_raw + p_jitter

        print(f"  预测步:")
        print(f"    x_filt[:, {t-1}] = {x_filt[:, t-1]}")
        print(f"    U[:, {t}] = {U_new[:, t]}")
        print(f"    x_pred[:, {t}] = A @ x_filt[:, {t-1}] + B @ U[:, {t}] = {x_pred[:, t]}")
        print(f"    P_pred[{t}] 对角线 = {np.diag(P_pred[t])}")

        # 检查观测
        ix = np.where(~np.isnan(Z_new[:, t]))[0]
        print(f"  观测索引: {ix}, 共{len(ix)}个观测")

        if len(ix) == 0:
            x_filt[:, t] = x_pred[:, t]
            P_filt[t] = P_pred[t].copy()
            print(f"  没有观测，x_filt = x_pred")
            continue

        # 更新步
        z_t = Z_new[ix, t]
        H_t = Lambda[ix, :]
        R_t = R[np.ix_(ix, ix)]

        innov_t = z_t - H_t @ x_pred[:, t]
        S_t = H_t @ P_pred[t] @ H_t.T + R_t
        jitter = np.eye(S_t.shape[0]) * 1e-4

        import scipy.linalg
        K_t = scipy.linalg.solve((S_t + jitter).T, (P_pred[t] @ H_t.T).T, assume_a='pos').T

        x_filt[:, t] = x_pred[:, t] + K_t @ innov_t
        P_filt[t] = (np.eye(n_states) - K_t @ H_t) @ P_pred[t]
        P_filt[t] = (P_filt[t] + P_filt[t].T) / 2.0

        print(f"  更新步:")
        print(f"    z_t = {z_t}")
        print(f"    H_t @ x_pred = {H_t @ x_pred[:, t]}")
        print(f"    innov_t = z_t - H_t @ x_pred = {innov_t}")
        print(f"    K_t[:, 0] = {K_t[:, 0]}")
        print(f"    x_filt[:, {t}] = x_pred + K @ innov = {x_filt[:, t]}")

    # ========== 老代码 ==========
    print("\n" + "="*80)
    print("老代码Kalman滤波")
    print("="*80)

    # 老代码需要DataFrame输入
    U_old_df = pd.DataFrame(
        U_zeros.T,  # 转置为 (n_time, n_factors)
        index=obs_centered.index,
        columns=[f'shock{i+1}' for i in range(n_factors)]
    )

    print(f"\n老代码输入:")
    print(f"  Z.shape = {obs_centered.shape} (DataFrame)")
    print(f"  U.shape = {U_old_df.shape} (DataFrame)")
    print(f"  Z.iloc[0] = {obs_centered.iloc[0].values}")
    print(f"  U.iloc[0] = {U_old_df.iloc[0].values}")

    # 手动执行老代码滤波的前几步
    print("\n手动执行老代码滤波前3步:")

    z = obs_centered.to_numpy()
    u = U_old_df.to_numpy()

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
    print(f"  x_minus[0, :] = {x_minus_old[0, :]}")

    for i in range(1, min(4, n_time)):
        print(f"\n时刻{i}:")

        ix = np.where(~np.isnan(z[i, :]))[0]
        print(f"  观测索引: {ix}, 共{len(ix)}个观测")

        if len(ix) == 0:
            x_prev_col = x_old[i-1, :].reshape(-1, 1)
            u_col = u[i, :].reshape(-1, 1)
            x_minus_pred = A @ x_prev_col + B @ u_col
            x_minus_old[i, :] = x_minus_pred.flatten()
            P_minus_raw = A @ P_old[i-1] @ A.T + Q
            p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6
            P_minus_old[i] = P_minus_raw + p_jitter
            x_old[i, :] = x_minus_old[i, :]
            P_old[i] = P_minus_old[i]
            print(f"  没有观测，x = x_minus")
            continue

        # 预测步
        x_prev_col = x_old[i-1, :].reshape(-1, 1)
        u_col = u[i, :].reshape(-1, 1)

        x_minus_pred = A @ x_prev_col + B @ u_col
        x_minus_old[i, :] = x_minus_pred.flatten()

        P_minus_raw = A @ P_old[i-1] @ A.T + Q
        p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6
        P_minus_old[i] = P_minus_raw + p_jitter

        print(f"  预测步:")
        print(f"    x[{i-1}, :] = {x_old[i-1, :]}")
        print(f"    u[{i}, :] = {u[i, :]}")
        print(f"    x_minus[{i}, :] = A @ x[{i-1}] + B @ u[{i}] = {x_minus_old[i, :]}")
        print(f"    P_minus[{i}] 对角线 = {np.diag(P_minus_old[i])}")

        # 更新步
        z_t = z[i, ix]
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
        print(f"    z_t = {z_t}")
        print(f"    H_t @ x_minus = {(H_t @ x_minus_col).flatten()}")
        print(f"    innovation = z_t - H_t @ x_minus = {innovation.flatten()}")
        print(f"    K_t[:, 0] = {K_t_effective[:, 0]}")
        print(f"    x[{i}, :] = x_minus + K @ innovation = {x_old[i, :]}")

    # ========== 对比结果 ==========
    print("\n" + "="*80)
    print("对比前3步的结果")
    print("="*80)

    for t in range(min(4, n_time)):
        print(f"\n时刻{t}:")
        diff_x_filt = np.abs(x_filt[:, t] - x_old[t, :])
        print(f"  新代码 x_filt[:, {t}] = {x_filt[:, t]}")
        print(f"  老代码 x[{t}, :] = {x_old[t, :]}")
        print(f"  差异 = {diff_x_filt}")
        print(f"  最大差异 = {np.max(diff_x_filt):.15f}")

        if t > 0:
            diff_x_pred = np.abs(x_pred[:, t] - x_minus_old[t, :])
            print(f"  预测值差异:")
            print(f"    新代码 x_pred[:, {t}] = {x_pred[:, t]}")
            print(f"    老代码 x_minus[{t}, :] = {x_minus_old[t, :]}")
            print(f"    差异 = {diff_x_pred}")
            print(f"    最大差异 = {np.max(diff_x_pred):.15f}")


if __name__ == "__main__":
    test_kalman_step_by_step()
