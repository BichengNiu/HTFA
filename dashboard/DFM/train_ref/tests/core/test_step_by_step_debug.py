# -*- coding: utf-8 -*-
"""
逐步调试测试：对比新旧代码的每个步骤

详细对比初始化和EM迭代的中间值
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新代码
from dashboard.DFM.train_ref.core.factor_model import DFMModel

# 导入老代码
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo


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


def test_initialization_step_by_step():
    """逐步测试初始化过程"""
    print("="*80)
    print("逐步调试：初始化阶段")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    print(f"\n数据: {data.shape}")

    # ===== 步骤1: 数据标准化 =====
    print("\n" + "="*80)
    print("步骤1: 数据标准化")
    print("="*80)

    # 新代码
    model_new = DFMModel(n_factors=n_factors, max_lags=1, max_iter=10)
    means_new = data.mean(skipna=True).values
    stds_new = data.std(skipna=True).values
    stds_new = np.where(stds_new > 0, stds_new, 1.0)
    obs_centered_new = data - means_new
    z_new = (obs_centered_new / stds_new).fillna(0).values

    # 老代码
    obs_mean_old = data.mean(skipna=True)
    obs_std_old = data.std(skipna=True)
    obs_std_old[obs_std_old == 0] = 1.0
    obs_centered_old = data - obs_mean_old
    z_old = ((data - obs_mean_old) / obs_std_old).fillna(0)

    print(f"\n均值差异: {np.max(np.abs(means_new - obs_mean_old.values)):.15f}")
    print(f"标准差差异: {np.max(np.abs(stds_new - obs_std_old.values)):.15f}")
    print(f"标准化数据差异: {np.max(np.abs(z_new - z_old.values)):.15f}")

    # ===== 步骤2: SVD分解 =====
    print("\n" + "="*80)
    print("步骤2: SVD分解")
    print("="*80)

    # 新代码
    U_new, s_new, Vh_new = np.linalg.svd(z_new, full_matrices=False)
    factors_init_new = U_new[:, :n_factors] * s_new[:n_factors]
    V_new = Vh_new.T

    # 老代码
    U_old, s_old, Vh_old = np.linalg.svd(z_old, full_matrices=False)
    factors_init_old = U_old[:, :n_factors] * s_old[:n_factors]
    V_old = Vh_old.T

    print(f"\nU矩阵差异: {np.max(np.abs(np.abs(U_new[:, :n_factors]) - np.abs(U_old[:, :n_factors]))):.15f}")
    print(f"奇异值差异: {np.max(np.abs(s_new[:n_factors] - s_old[:n_factors])):.15f}")
    print(f"V矩阵差异: {np.max(np.abs(np.abs(V_new) - np.abs(V_old))):.15f}")

    # 对比初始因子（考虑符号差异）
    for i in range(n_factors):
        corr = np.corrcoef(factors_init_new[:, i], factors_init_old[:, i])[0, 1]
        print(f"因子{i}相关系数: {corr:.10f}")

    # ===== 步骤3: 载荷矩阵计算 =====
    print("\n" + "="*80)
    print("步骤3: 载荷矩阵计算（使用中心化数据）")
    print("="*80)

    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings

    # 新代码
    factors_df_new = pd.DataFrame(factors_init_new, columns=[f'Factor{i+1}' for i in range(n_factors)])
    Lambda_new = estimate_loadings(obs_centered_new, factors_df_new)

    # 老代码
    factors_df_old = pd.DataFrame(factors_init_old, index=data.index, columns=[f'Factor{i+1}' for i in range(n_factors)])
    Lambda_old = calculate_factor_loadings(obs_centered_old, factors_df_old)

    print(f"\n载荷矩阵形状: {Lambda_new.shape} vs {Lambda_old.shape}")
    print(f"载荷矩阵最大差异: {np.max(np.abs(Lambda_new - Lambda_old)):.10f}")
    print(f"载荷矩阵平均差异: {np.mean(np.abs(Lambda_new - Lambda_old)):.10f}")

    # 显示前几个元素
    print("\n前5个载荷值对比:")
    for i in range(min(5, Lambda_new.shape[0])):
        for j in range(n_factors):
            print(f"  Lambda[{i},{j}]: {Lambda_new[i,j]:.10f} vs {Lambda_old[i,j]:.10f} (diff={abs(Lambda_new[i,j]-Lambda_old[i,j]):.10f})")

    # ===== 步骤4: R矩阵计算 =====
    print("\n" + "="*80)
    print("步骤4: R矩阵计算")
    print("="*80)

    # 新代码的R矩阵计算
    reconstructed_z_new = factors_init_new @ V_new[:, :n_factors].T
    residuals_z_new = z_new - reconstructed_z_new
    psi_diag_new = np.nanvar(residuals_z_new, axis=0)
    R_diag_new = psi_diag_new * (stds_new ** 2)
    R_diag_new = np.maximum(R_diag_new, 1e-6)

    # 老代码的R矩阵计算
    reconstructed_z_old = factors_init_old @ V_old[:, :n_factors].T
    residuals_z_old = z_old.values - reconstructed_z_old
    psi_diag_old = np.nanvar(residuals_z_old, axis=0)
    original_std_sq_old = obs_std_old.fillna(1.0)**2
    R_diag_old = psi_diag_old * original_std_sq_old.to_numpy()
    R_diag_old = np.maximum(R_diag_old, 1e-6)

    print(f"\npsi_diag差异: {np.max(np.abs(psi_diag_new - psi_diag_old)):.10f}")
    print(f"R对角线差异: {np.max(np.abs(R_diag_new - R_diag_old)):.10f}")

    for i in range(min(5, len(R_diag_new))):
        print(f"  R[{i},{i}]: {R_diag_new[i]:.10f} vs {R_diag_old[i]:.10f} (diff={abs(R_diag_new[i]-R_diag_old[i]):.10f})")

    # ===== 步骤5: A矩阵和Q矩阵（VAR模型）=====
    print("\n" + "="*80)
    print("步骤5: A矩阵和Q矩阵（VAR模型）")
    print("="*80)

    from statsmodels.tsa.api import VAR
    from dashboard.DFM.train_ref.core.estimator import estimate_transition_matrix

    # 新代码
    A_new = estimate_transition_matrix(factors_init_new, max_lags=1)

    # 老代码
    var_model_old = VAR(factors_df_old.dropna())
    var_results_old = var_model_old.fit(1)
    A_old = var_results_old.coefs[0]
    Q_old = np.cov(var_results_old.resid, rowvar=False)

    print(f"\nA矩阵差异: {np.max(np.abs(A_new - A_old)):.10f}")

    # VAR残差
    try:
        var_model_new = VAR(factors_init_new)
        var_results_new = var_model_new.fit(maxlags=1, ic=None, trend='n')
        Q_new = np.cov(var_results_new.resid, rowvar=False)
        print(f"Q矩阵差异: {np.max(np.abs(Q_new - Q_old)):.10f}")
    except Exception as e:
        print(f"Q矩阵计算失败: {e}")

    print("\n" + "="*80)
    print("初始化阶段对比完成")
    print("="*80)

    return {
        'data': data,
        'factors_new': factors_init_new,
        'factors_old': factors_init_old,
        'Lambda_new': Lambda_new,
        'Lambda_old': Lambda_old,
        'R_new': np.diag(R_diag_new),
        'R_old': np.diag(R_diag_old),
        'A_new': A_new,
        'A_old': A_old
    }


if __name__ == "__main__":
    results = test_initialization_step_by_step()
