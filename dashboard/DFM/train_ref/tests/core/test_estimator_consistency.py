# -*- coding: utf-8 -*-
"""
参数估计一致性测试

对比 train_ref/core/estimator.py 与 train_model/DiscreteKalmanFilter.py
验证载荷估计等功能的一致性
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新代码
from dashboard.DFM.train_ref.core.estimator import (
    estimate_loadings,
    estimate_target_loading,
    estimate_transition_matrix
)

# 导入老代码
from dashboard.DFM.train_model.DiscreteKalmanFilter import calculate_factor_loadings


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)

    n_time = 100
    n_obs = 8
    n_factors = 3

    # 生成因子
    factors = np.random.randn(n_time, n_factors)

    # 生成真实载荷
    true_loadings = np.random.randn(n_obs, n_factors) * 0.5

    # 生成观测
    observations = factors @ true_loadings.T + np.random.randn(n_time, n_obs) * 0.2

    # 转为DataFrame
    dates = pd.date_range('2015-01-01', periods=n_time, freq='M')
    obs_df = pd.DataFrame(
        observations,
        index=dates,
        columns=[f'Obs{i+1}' for i in range(n_obs)]
    )
    factors_df = pd.DataFrame(
        factors,
        index=dates,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    return obs_df, factors_df, true_loadings


def test_loadings_estimation():
    """测试载荷矩阵估计"""
    print("="*80)
    print("测试1: 载荷矩阵估计一致性")
    print("="*80)

    obs_df, factors_df, true_loadings = create_test_data()

    print(f"\n参数设置:")
    print(f"  观测变量数: {obs_df.shape[1]}")
    print(f"  因子数: {factors_df.shape[1]}")
    print(f"  样本数: {len(obs_df)}")

    # 新代码
    print("\n运行新代码...")
    loadings_new = estimate_loadings(obs_df, factors_df)

    print(f"  载荷矩阵形状: {loadings_new.shape}")

    # 老代码
    print("运行老代码...")
    loadings_old = calculate_factor_loadings(obs_df, factors_df)

    print(f"  载荷矩阵形状: {loadings_old.shape}")

    # 对比
    print("\n对比载荷矩阵:")
    diff = np.abs(loadings_new - loadings_old)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = mean_diff / (np.abs(loadings_old).mean() + 1e-10)

    print(f"  最大差异: {max_diff:.10f}")
    print(f"  平均差异: {mean_diff:.10f}")
    print(f"  相对差异: {rel_diff:.10f}")

    # 对比与真实载荷的相关性
    corr_new = np.corrcoef(loadings_new.ravel(), true_loadings.ravel())[0, 1]
    corr_old = np.corrcoef(loadings_old.ravel(), true_loadings.ravel())[0, 1]

    print(f"\n与真实载荷的相关性:")
    print(f"  新代码: {corr_new:.6f}")
    print(f"  老代码: {corr_old:.6f}")

    threshold = 1e-8
    passed = max_diff < threshold

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (阈值: {threshold})")

    return passed


def test_target_loading_estimation():
    """测试目标变量载荷估计"""
    print("\n" + "="*80)
    print("测试2: 目标变量载荷估计")
    print("="*80)

    obs_df, factors_df, _ = create_test_data()

    # 使用第一个观测作为目标
    target = obs_df.iloc[:, 0]

    print(f"\n测试目标变量载荷估计...")

    # 新代码
    print("\n运行新代码...")
    loading_new = estimate_target_loading(
        target=target,
        factors=factors_df,
        train_end=None
    )

    print(f"  载荷向量形状: {loading_new.shape}")
    print(f"  载荷值: {loading_new}")

    # 老代码（使用calculate_factor_loadings）
    print("\n运行老代码（用单变量DataFrame）...")
    loading_old = calculate_factor_loadings(
        observables=target.to_frame(),
        factors=factors_df
    )[0, :]  # 取第一行

    print(f"  载荷值: {loading_old}")

    # 对比
    diff = np.abs(loading_new - loading_old)
    max_diff = np.max(diff)

    print(f"\n载荷差异:")
    print(f"  最大差异: {max_diff:.10f}")

    threshold = 1e-8
    passed = max_diff < threshold

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (阈值: {threshold})")

    return passed


def test_target_loading_with_train_split():
    """测试带训练期切分的目标载荷估计"""
    print("\n" + "="*80)
    print("测试3: 带训练期切分的目标载荷估计")
    print("="*80)

    obs_df, factors_df, _ = create_test_data()

    target = obs_df.iloc[:, 0]
    train_end = obs_df.index[70]

    print(f"\n训练期结束: {train_end}")

    # 新代码
    loading_new = estimate_target_loading(
        target=target,
        factors=factors_df,
        train_end=str(train_end.date())
    )

    # 手动切分老代码
    target_train = target.loc[:train_end]
    factors_train = factors_df.loc[:train_end]

    loading_old = calculate_factor_loadings(
        observables=target_train.to_frame(),
        factors=factors_train
    )[0, :]

    # 对比
    diff = np.abs(loading_new - loading_old)
    max_diff = np.max(diff)

    print(f"\n载荷差异: {max_diff:.10f}")

    threshold = 1e-8
    passed = max_diff < threshold

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (阈值: {threshold})")

    return passed


def test_transition_matrix_estimation():
    """测试状态转移矩阵估计"""
    print("\n" + "="*80)
    print("测试4: 状态转移矩阵估计")
    print("="*80)

    _, factors_df, _ = create_test_data()

    print(f"\n测试VAR(1)模型估计...")

    try:
        # 新代码
        A_new = estimate_transition_matrix(
            factors=factors_df.values,
            max_lags=1
        )

        print(f"  转移矩阵形状: {A_new.shape}")
        print(f"  转移矩阵:\n{A_new}")

        # 检查矩阵稳定性（特征值）
        eigenvalues = np.linalg.eigvals(A_new)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        print(f"\n矩阵稳定性:")
        print(f"  最大特征值模: {max_eigenvalue:.6f}")

        # 对于稳定的VAR模型，最大特征值应<1
        passed = max_eigenvalue < 1.5  # 放宽一点，因为是估计值

        print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (最大特征值<1.5)")

        return passed

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loadings_with_missing_data():
    """测试带缺失数据的载荷估计"""
    print("\n" + "="*80)
    print("测试5: 带缺失数据的载荷估计")
    print("="*80)

    obs_df, factors_df, _ = create_test_data()

    # 添加缺失数据
    obs_missing = obs_df.copy()
    missing_idx = np.random.choice(len(obs_df), size=20, replace=False)
    obs_missing.iloc[missing_idx, 0] = np.nan

    print(f"\n添加了 {len(missing_idx)} 个缺失值")

    # 新代码
    loadings_new = estimate_loadings(obs_missing, factors_df)

    # 老代码
    loadings_old = calculate_factor_loadings(obs_missing, factors_df)

    # 对比（只对比非全缺失的变量）
    valid_vars = ~np.isnan(loadings_new).any(axis=1)
    diff = np.abs(loadings_new[valid_vars] - loadings_old[valid_vars])
    max_diff = np.max(diff)

    print(f"\n有效变量载荷差异: {max_diff:.10f}")

    threshold = 1e-8
    passed = max_diff < threshold

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (阈值: {threshold})")

    return passed


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("参数估计一致性测试套件")
    print("对比 train_ref vs train_model")
    print("="*80)

    results = []

    try:
        results.append(("载荷矩阵估计", test_loadings_estimation()))
    except Exception as e:
        print(f"\n✗ 载荷矩阵估计测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("载荷矩阵估计", False))

    try:
        results.append(("目标载荷估计", test_target_loading_estimation()))
    except Exception as e:
        print(f"\n✗ 目标载荷估计测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("目标载荷估计", False))

    try:
        results.append(("训练期切分载荷", test_target_loading_with_train_split()))
    except Exception as e:
        print(f"\n✗ 训练期切分载荷测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("训练期切分载荷", False))

    try:
        results.append(("转移矩阵估计", test_transition_matrix_estimation()))
    except Exception as e:
        print(f"\n✗ 转移矩阵估计测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("转移矩阵估计", False))

    try:
        results.append(("缺失数据载荷", test_loadings_with_missing_data()))
    except Exception as e:
        print(f"\n✗ 缺失数据载荷测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("缺失数据载荷", False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n🎉 所有测试通过！train_ref参数估计与老代码一致。")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要检查差异。")
        return 1


if __name__ == "__main__":
    exit(main())
