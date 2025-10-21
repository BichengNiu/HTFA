# -*- coding: utf-8 -*-
"""
对比新老代码的PCA初始化过程
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


def test_pca_initialization():
    """对比PCA初始化"""
    print("="*80)
    print("对比新老代码的PCA初始化过程")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # ========== 新代码：模拟DFMModel的初始化 ==========
    print("\n" + "="*80)
    print("新代码：模拟DFMModel的PCA初始化")
    print("="*80)

    # 预处理（匹配factor_model.py）
    train_data = data  # 没有train/test split
    means_new = train_data.mean(skipna=True).values
    stds_new = train_data.std(skipna=True).values
    stds_new = np.where(stds_new > 0, stds_new, 1.0)

    obs_centered_new = data - means_new
    Z_standardized_new = (obs_centered_new / stds_new).fillna(0).values

    print(f"  means[:5] = {means_new[:5]}")
    print(f"  stds[:5] = {stds_new[:5]}")
    print(f"  obs_centered[0, :3] = {obs_centered_new.iloc[0, :3].values}")
    print(f"  Z_standardized[0, :3] = {Z_standardized_new[0, :3]}")

    # PCA
    U_new, s_new, Vh_new = np.linalg.svd(Z_standardized_new, full_matrices=False)
    factors_init_new = U_new[:, :n_factors] * s_new[:n_factors]

    print(f"  factors_init shape: {factors_init_new.shape}")
    print(f"  factors_init[0]: {factors_init_new[0]}")
    print(f"  factors_init[:3, 0]: {factors_init_new[:3, 0]}")

    factors_df_new = pd.DataFrame(
        factors_init_new,
        index=data.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda
    from dashboard.DFM.train_ref.core.estimator import estimate_loadings
    Lambda_new = estimate_loadings(obs_centered_new, factors_df_new)

    print(f"  Lambda shape: {Lambda_new.shape}")
    print(f"  Lambda[:2, :] =\n{Lambda_new[:2, :]}")

    # ========== 老代码：模拟DFM_EMalgo的初始化 ==========
    print("\n" + "="*80)
    print("老代码：模拟DFM_EMalgo的PCA初始化")
    print("="*80)

    # 预处理（匹配DynamicFactorModel.py）
    DFM_SEED = 42
    np.random.seed(DFM_SEED)
    import random
    random.seed(DFM_SEED)

    observation = data
    obs_mean_old = observation.mean(skipna=True)
    obs_std_old = observation.std(skipna=True)
    obs_std_old[obs_std_old == 0] = 1.0

    obs_centered_old = observation - obs_mean_old
    z_old = (obs_centered_old / obs_std_old).fillna(0)

    print(f"  obs_mean[:5] = {obs_mean_old.values[:5]}")
    print(f"  obs_std[:5] = {obs_std_old.values[:5]}")
    print(f"  obs_centered.iloc[0, :3] = {obs_centered_old.iloc[0, :3].values}")
    print(f"  z.iloc[0, :3] = {z_old.iloc[0, :3].values}")

    # PCA
    U_old, s_old, Vh_old = np.linalg.svd(z_old, full_matrices=False)
    factors_init_old = U_old[:, :n_factors] * s_old[:n_factors]

    print(f"  factors_init shape: {factors_init_old.shape}")
    print(f"  factors_init[0]: {factors_init_old[0]}")
    print(f"  factors_init[:3, 0]: {factors_init_old[:3, 0]}")

    factors_df_old = pd.DataFrame(
        factors_init_old,
        index=z_old.index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Lambda
    from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings
    Lambda_old = calculate_factor_loadings(obs_centered_old, factors_df_old)

    print(f"  Lambda shape: {Lambda_old.shape}")
    print(f"  Lambda[:2, :] =\n{Lambda_old[:2, :]}")

    # ========== 对比 ==========
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    print(f"\nmeans差异:")
    diff_means = np.abs(means_new - obs_mean_old.values)
    print(f"  最大差异: {np.max(diff_means):.15e}")

    print(f"\nstds差异:")
    diff_stds = np.abs(stds_new - obs_std_old.values)
    print(f"  最大差异: {np.max(diff_stds):.15e}")

    print(f"\nobs_centered差异:")
    diff_obs_centered = np.abs(obs_centered_new.values - obs_centered_old.values)
    print(f"  最大差异: {np.max(diff_obs_centered):.15e}")

    print(f"\nZ_standardized差异:")
    diff_z = np.abs(Z_standardized_new - z_old.values)
    print(f"  最大差异: {np.max(diff_z):.15e}")

    print(f"\nfactors_init差异:")
    diff_factors = np.abs(factors_init_new - factors_init_old)
    print(f"  最大差异: {np.max(diff_factors):.15e}")
    print(f"  平均差异: {np.mean(diff_factors):.15e}")
    print(f"  第一个因子前3个值差异: {diff_factors[:3, 0]}")

    print(f"\nLambda差异:")
    diff_Lambda = np.abs(Lambda_new - Lambda_old)
    print(f"  最大差异: {np.max(diff_Lambda):.15e}")
    print(f"  平均差异: {np.mean(diff_Lambda):.15e}")

    # 判断
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if np.max(diff_Lambda) < 1e-10:
        print("\n[通过] PCA初始化完全一致!")
    else:
        print(f"\n[失败] PCA初始化存在差异: {np.max(diff_Lambda):.15e}")


if __name__ == "__main__":
    test_pca_initialization()
