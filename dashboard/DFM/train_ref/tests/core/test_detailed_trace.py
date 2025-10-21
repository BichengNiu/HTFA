# -*- coding: utf-8 -*-
"""
详细追踪新老代码每个步骤的差异
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


def trace_old_code_step_by_step():
    """逐步追踪老代码的执行"""
    print("="*80)
    print("追踪老代码执行流程")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1
    n_iter = 1

    # 设置随机种子
    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    n_obs = data.shape[1]
    n_time = data.shape[0]
    n_shocks = n_factors

    # Step 1: 数据预处理
    print("\nStep 1: 数据预处理")
    obs_mean = data.mean(skipna=True)
    obs_std = data.std(skipna=True)
    obs_std[obs_std == 0] = 1.0
    obs_centered = data - obs_mean
    z = (obs_centered / obs_std).fillna(0)

    print(f"  obs_centered shape: {obs_centered.shape}")
    print(f"  obs_centered前3行前3列:\n{obs_centered.iloc[:3, :3]}")

    # Step 2: PCA初始化
    print("\nStep 2: PCA初始化")
    U, s, Vh = np.linalg.svd(z, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    factors_init_df = pd.DataFrame(factors_init, index=z.index, columns=[f'Factor{i+1}' for i in range(n_factors)])

    print(f"  factors_init_df shape: {factors_init_df.shape}")
    print(f"  factors_init_df前3行:\n{factors_init_df.iloc[:3]}")
    print(f"  factors_init_df index前3个: {factors_init_df.index[:3].tolist()}")

    # Lambda初始化
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda_current = calculate_factor_loadings(obs_centered, factors_init_df)

    print(f"\n  Lambda_current shape: {Lambda_current.shape}")
    print(f"  Lambda_current前3行:\n{Lambda_current[:3]}")

    # VAR初始化
    from statsmodels.tsa.api import VAR
    var_model = VAR(factors_init_df.dropna())
    var_results = var_model.fit(max_lags)
    A_current = var_results.coefs[0]
    Q_current = np.cov(var_results.resid, rowvar=False)
    Q_current = np.diag(np.maximum(np.diag(Q_current), 1e-6))

    print(f"\n  A_current shape: {A_current.shape}")
    print(f"  A_current:\n{A_current}")
    print(f"  Q_current对角线: {np.diag(Q_current)}")

    # R矩阵初始化
    V = Vh.T
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = z.values - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag_current = psi_diag * (obs_std**2).to_numpy()
    R_diag_current = np.maximum(R_diag_current, 1e-6)
    R_current = np.diag(R_diag_current)

    print(f"\n  R_current对角线前3个: {np.diag(R_current)[:3]}")

    # 其他初始化
    state_names = [f'Factor{i+1}' for i in range(n_factors)]
    x0_current = np.zeros(n_factors)
    P0_current = np.eye(n_factors)
    B_current = np.eye(n_factors)

    # 生成error_df
    np.random.seed(DFM_SEED)
    u_data = np.random.randn(n_time, n_shocks)
    error_df = pd.DataFrame(u_data, columns=[f'shock{i+1}' for i in range(n_shocks)], index=data.index)

    print(f"\n  初始化完成，准备进入EM循环")
    print(f"  x0_current: {x0_current}")
    print(f"  P0_current对角线: {np.diag(P0_current)}")

    # Step 3: EM循环（只执行1次）
    print("\n" + "="*80)
    print("Step 3: EM循环 (迭代1次)")
    print("="*80)

    from dashboard.DFM.train_model.DiscreteKalmanFilter import KalmanFilter as OldKalmanFilter, FIS, EMstep

    for i in range(n_iter):
        print(f"\n迭代 {i+1}:")

        # E步：Kalman滤波和平滑
        print("  E步: Kalman滤波和平滑")
        kf = OldKalmanFilter(
            Z=obs_centered,
            U=error_df,
            A=A_current,
            B=B_current,
            H=Lambda_current,
            state_names=state_names,
            x0=x0_current,
            P0=P0_current,
            Q=Q_current,
            R=R_current
        )
        fis = FIS(kf)

        print(f"    平滑因子shape: {fis.x_sm.shape}")
        print(f"    平滑因子前3行:\n{fis.x_sm.iloc[:3]}")

        # M步：参数更新
        print("  M步: 参数更新")
        em = EMstep(fis, n_shocks)

        print(f"    更新后Lambda shape: {em.Lambda.shape}")
        print(f"    更新后Lambda前3行:\n{em.Lambda[:3]}")
        print(f"    更新后A:\n{em.A}")
        print(f"    更新后Q对角线: {np.diag(em.Q)}")
        print(f"    更新后R对角线前3个: {np.diag(em.R)[:3]}")

        # 保存更新后的参数
        A_current = np.array(em.A)
        B_current = np.array(em.B)
        Lambda_current = np.array(em.Lambda)
        Q_current = np.array(em.Q)
        R_current = np.array(em.R)

        # 更新初始状态
        x0_current = np.array(em.x_sm.iloc[0])
        P0_current = fis.P_sm[0]

        print(f"    下次迭代的x0: {x0_current}")
        print(f"    下次迭代的P0对角线: {np.diag(P0_current)}")

    # Step 4: 最终结果
    print("\n" + "="*80)
    print("Step 4: 最终结果")
    print("="*80)

    print(f"  最终返回的Lambda shape: {Lambda_current.shape}")
    print(f"  最终返回的Lambda前3行:\n{Lambda_current[:3]}")
    print(f"  最终返回的因子(em.x_sm) shape: {em.x_sm.shape}")
    print(f"  最终返回的因子前3行:\n{em.x_sm.iloc[:3]}")

    return {
        'Lambda': Lambda_current,
        'factors': em.x_sm.values,
        'A': A_current,
        'Q': Q_current,
        'R': R_current
    }


def trace_new_code_step_by_step():
    """逐步追踪新代码的执行"""
    print("\n" + "="*80)
    print("追踪新代码执行流程")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_lags = 1

    # 设置随机种子
    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    # Step 1: 数据预处理
    print("\nStep 1: 数据预处理")
    means = data.mean(skipna=True).values
    stds = data.std(skipna=True).values
    stds = np.where(stds > 0, stds, 1.0)
    obs_centered = data - means
    Z_standardized = (obs_centered / stds).fillna(0).values

    print(f"  obs_centered shape: {obs_centered.shape}")
    print(f"  obs_centered前3行前3列:\n{obs_centered.iloc[:3, :3]}")

    # Step 2: PCA初始化
    print("\nStep 2: PCA初始化")
    U, s, Vh = np.linalg.svd(Z_standardized, full_matrices=False)
    factors_init = U[:, :n_factors] * s[:n_factors]
    V = Vh.T

    factors_df = pd.DataFrame(
        factors_init,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    print(f"  factors_df shape: {factors_df.shape}")
    print(f"  factors_df前3行:\n{factors_df.iloc[:3]}")

    # Lambda初始化
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings
    initial_loadings = estimate_loadings(obs_centered, factors_df)

    print(f"\n  initial_loadings shape: {initial_loadings.shape}")
    print(f"  initial_loadings前3行:\n{initial_loadings[:3]}")

    # A和Q初始化
    from dashboard.DFM.train_ref.core.estimator import estimate_transition_matrix
    A = estimate_transition_matrix(factors_init, max_lags)
    Q = np.eye(n_factors) * 0.1

    print(f"\n  A shape: {A.shape}")
    print(f"  A:\n{A}")
    print(f"  Q对角线: {np.diag(Q)}")

    # R矩阵初始化
    reconstructed_z = factors_init @ V[:, :n_factors].T
    residuals_z = Z_standardized - reconstructed_z
    psi_diag = np.nanvar(residuals_z, axis=0)
    R_diag = psi_diag * (stds ** 2)
    R_diag = np.maximum(R_diag, 1e-6)
    R = np.diag(R_diag)

    print(f"\n  R对角线前3个: {np.diag(R)[:3]}")

    # 初始化Kalman参数
    x0 = np.zeros(n_factors)
    P0 = np.eye(n_factors)
    Lambda = initial_loadings.copy()

    print(f"\n  初始化完成，准备进入EM循环")
    print(f"  x0: {x0}")
    print(f"  P0对角线: {np.diag(P0)}")

    # Step 3: EM循环（只执行1次）
    print("\n" + "="*80)
    print("Step 3: EM循环 (迭代1次)")
    print("="*80)

    from dashboard.DFM.train_ref.core.kalman import KalmanFilter as NewKalmanFilter
    from dashboard.DFM.train_ref.core.estimator import estimate_covariance_matrices

    n_time = data.shape[0]
    n_obs = data.shape[1]

    for iteration in range(1):
        print(f"\n迭代 {iteration+1}:")

        # E步：Kalman滤波和平滑
        print("  E步: Kalman滤波和平滑")

        H = np.zeros((n_obs, n_factors))
        H[:, :n_factors] = Lambda
        B = np.eye(n_factors)
        U = np.zeros((n_factors, n_time))
        Z = obs_centered.values.T

        kf = NewKalmanFilter(A, B, H, Q, R, x0, P0)
        filter_result = kf.filter(Z, U)
        smoother_result = kf.smooth(filter_result)

        factors_smoothed = smoother_result.x_smoothed[:n_factors, :].T
        factors_df_smoothed = pd.DataFrame(
            factors_smoothed,
            columns=[f'Factor{i+1}' for i in range(n_factors)]
        )

        print(f"    平滑因子shape: {factors_smoothed.shape}")
        print(f"    平滑因子前3行:\n{factors_smoothed[:3]}")

        # M步：参数更新
        print("  M步: 参数更新")
        Lambda = estimate_loadings(obs_centered, factors_df_smoothed)
        A = estimate_transition_matrix(factors_smoothed, max_lags)
        Q, R = estimate_covariance_matrices(
            smoother_result,
            obs_centered,
            Lambda,
            n_factors,
            A
        )

        print(f"    更新后Lambda shape: {Lambda.shape}")
        print(f"    更新后Lambda前3行:\n{Lambda[:3]}")
        print(f"    更新后A:\n{A}")
        print(f"    更新后Q对角线: {np.diag(Q)}")
        print(f"    更新后R对角线前3个: {np.diag(R)[:3]}")

        # 更新初始状态
        x0 = smoother_result.x_smoothed[:, 0].copy()
        P0 = smoother_result.P_smoothed[:, :, 0].copy()

        print(f"    下次迭代的x0: {x0}")
        print(f"    下次迭代的P0对角线: {np.diag(P0)}")

    # Step 4: 最终结果
    print("\n" + "="*80)
    print("Step 4: 最终结果")
    print("="*80)

    print(f"  最终返回的Lambda shape: {Lambda.shape}")
    print(f"  最终返回的Lambda前3行:\n{Lambda[:3]}")
    print(f"  最终返回的因子shape: {factors_smoothed.shape}")
    print(f"  最终返回的因子前3行:\n{factors_smoothed[:3]}")

    return {
        'Lambda': Lambda,
        'factors': factors_smoothed,
        'A': A,
        'Q': Q,
        'R': R
    }


if __name__ == "__main__":
    print("="*80)
    print("详细追踪新老代码执行流程对比")
    print("="*80)

    old_result = trace_old_code_step_by_step()
    new_result = trace_new_code_step_by_step()

    # 对比最终结果
    print("\n" + "="*80)
    print("最终结果对比")
    print("="*80)

    Lambda_diff = np.abs(new_result['Lambda'] - old_result['Lambda'])
    factors_diff = np.abs(new_result['factors'] - old_result['factors'])
    A_diff = np.abs(new_result['A'] - old_result['A'])
    Q_diff = np.abs(new_result['Q'] - old_result['Q'])
    R_diff = np.abs(new_result['R'] - old_result['R'])

    print(f"\nLambda差异: 最大={np.max(Lambda_diff):.15f}, 平均={np.mean(Lambda_diff):.15f}")
    print(f"Factors差异: 最大={np.max(factors_diff):.15f}, 平均={np.mean(factors_diff):.15f}")
    print(f"A差异: 最大={np.max(A_diff):.15f}")
    print(f"Q差异: 最大={np.max(Q_diff):.15f}")
    print(f"R差异: 最大={np.max(R_diff):.15f}")
