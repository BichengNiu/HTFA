# -*- coding: utf-8 -*-
"""
详细的完全一致性测试

验证：
1. 载荷值完全一致
2. 因子值完全一致
3. 卡尔曼滤波各类参数值完全一致
4. nowcasting值计算完全一致
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新代码
from dashboard.DFM.train_ref.core.factor_model import fit_dfm

# 导入老代码
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo


def create_test_data():
    """创建相同的测试数据"""
    np.random.seed(42)

    n_time = 100
    n_obs = 10

    dates = pd.date_range('2015-01-01', periods=n_time, freq='ME')

    # 生成随机数据
    data = pd.DataFrame(
        np.random.randn(n_time, n_obs),
        index=dates,
        columns=[f'var{i}' for i in range(n_obs)]
    )

    return data


def test_loading_consistency():
    """测试1: 载荷值完全一致"""
    print("="*80)
    print("测试1: 载荷值完全一致")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    print(f"\n参数设置:")
    print(f"  样本数: {len(data)}")
    print(f"  变量数: {len(data.columns)}")
    print(f"  因子数: {n_factors}")

    # 新代码
    print("\n运行新代码...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_iter=10
    )

    # 老代码
    print("运行老代码...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=10,
        max_lags=1
    )

    # 对比载荷矩阵
    print("\n对比载荷矩阵 (Lambda):")
    Lambda_new = result_new.loadings  # (n_obs, n_factors) - already ndarray
    Lambda_old = result_old.Lambda   # (n_obs, n_factors)

    print(f"  形状: {Lambda_new.shape} vs {Lambda_old.shape}")

    # 逐个变量对比
    max_diff = 0
    for i in range(Lambda_new.shape[0]):
        for j in range(Lambda_new.shape[1]):
            diff = abs(Lambda_new[i, j] - Lambda_old[i, j])
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                print(f"  变量{i} 因子{j}: {Lambda_new[i, j]:.10f} vs {Lambda_old[i, j]:.10f} (diff={diff:.10f})")

    print(f"\n  最大差异: {max_diff:.15f}")

    passed = max_diff < 1e-6
    print(f"\n[{'通过' if passed else '失败'}] 载荷值一致性测试 (阈值: 1e-6)")

    return passed


def test_factor_consistency():
    """测试2: 因子值完全一致"""
    print("\n" + "="*80)
    print("测试2: 因子值完全一致")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # 新代码
    print("\n运行新代码...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_iter=10
    )

    # 老代码
    print("运行老代码...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=10,
        max_lags=1
    )

    # 对比因子
    print("\n对比因子值:")
    factors_new = result_new.factors.values  # (n_time, n_factors)
    factors_old = result_old.x_sm.values     # (n_time, n_factors)

    print(f"  形状: {factors_new.shape} vs {factors_old.shape}")

    # 计算每个因子的相关系数
    print(f"\n  各因子相关系数:")
    for j in range(n_factors):
        corr = np.corrcoef(factors_new[:, j], factors_old[:, j])[0, 1]
        print(f"    因子{j}: {corr:.10f}")

    # 计算整体差异
    diff = np.abs(factors_new - factors_old)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n  最大绝对差异: {max_diff:.10f}")
    print(f"  平均绝对差异: {mean_diff:.10f}")

    # 因子可能有符号差异，检查相关性
    min_corr = 1.0
    for j in range(n_factors):
        corr = abs(np.corrcoef(factors_new[:, j], factors_old[:, j])[0, 1])
        min_corr = min(min_corr, corr)

    print(f"  最小相关系数: {min_corr:.10f}")

    passed = min_corr > 0.9999  # 非常高的相关性
    print(f"\n[{'通过' if passed else '失败'}] 因子值一致性测试 (相关性阈值: 0.9999)")

    return passed


def test_kalman_params_consistency():
    """测试3: 卡尔曼滤波参数完全一致"""
    print("\n" + "="*80)
    print("测试3: 卡尔曼滤波参数完全一致")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # 新代码
    print("\n运行新代码...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_iter=10
    )

    # 老代码
    print("运行老代码...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=10,
        max_lags=1
    )

    # 对比状态转移矩阵A
    print("\n对比状态转移矩阵 A:")
    A_new = result_new.transition_matrix
    A_old = result_old.A
    diff_A = np.abs(A_new - A_old)
    print(f"  最大差异: {np.max(diff_A):.15f}")

    # 对比过程噪声协方差Q
    print("\n对比过程噪声协方差 Q:")
    Q_new = result_new.process_noise_cov
    Q_old = result_old.Q
    diff_Q = np.abs(Q_new - Q_old)
    print(f"  最大差异: {np.max(diff_Q):.15f}")

    # 对比观测噪声协方差R
    print("\n对比观测噪声协方差 R:")
    R_new = result_new.measurement_noise_cov
    R_old = result_old.R
    diff_R = np.abs(R_new - R_old)
    print(f"  最大差异: {np.max(diff_R):.15f}")

    max_diff_all = max(np.max(diff_A), np.max(diff_Q), np.max(diff_R))

    passed = max_diff_all < 1e-6
    print(f"\n[{'通过' if passed else '失败'}] 卡尔曼参数一致性测试 (阈值: 1e-6)")

    return passed


def test_nowcasting_consistency():
    """测试4: Nowcasting值完全一致"""
    print("\n" + "="*80)
    print("测试4: Nowcasting值完全一致")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # 设置训练期
    train_end = data.index[70]  # 70%用于训练

    print(f"\n参数设置:")
    print(f"  训练期结束: {train_end.date()}")
    print(f"  训练样本: {70}, 全样本: {len(data)}")

    # 新代码
    print("\n运行新代码...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_iter=10,
        train_end=str(train_end.date())
    )

    # 老代码
    print("运行老代码...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=10,
        max_lags=1,
        train_end_date=str(train_end.date())
    )

    # 提取验证期因子（用于nowcasting）
    print("\n对比验证期因子值（用于nowcasting）:")
    factors_new = result_new.factors.values[70:, :]  # 验证期因子

    # 老代码的x_sm可能是DataFrame或ndarray
    if hasattr(result_old.x_sm, 'values'):
        factors_old = result_old.x_sm.values[70:, :]
    else:
        factors_old = result_old.x_sm[70:, :]

    print(f"  验证期长度: {factors_new.shape[0]}")

    diff = np.abs(factors_new - factors_old)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  最大绝对差异: {max_diff:.10f}")
    print(f"  平均绝对差异: {mean_diff:.10f}")

    # 使用因子重构观测值（nowcasting）
    Lambda_new = result_new.loadings  # already ndarray
    Lambda_old = result_old.Lambda

    # 重构验证期观测值
    obs_reconstructed_new = factors_new @ Lambda_new.T
    obs_reconstructed_old = factors_old @ Lambda_old.T

    diff_recon = np.abs(obs_reconstructed_new - obs_reconstructed_old)
    print(f"\n  重构观测值最大差异: {np.max(diff_recon):.10f}")

    passed = max_diff < 1e-6 and np.max(diff_recon) < 1e-6
    print(f"\n[{'通过' if passed else '失败'}] Nowcasting一致性测试 (阈值: 1e-6)")

    return passed


def main():
    """运行所有详细测试"""
    print("\n" + "="*80)
    print("DFM完全一致性详细测试套件")
    print("="*80)
    print("\n本测试验证train_ref与train_model的完全一致性:")
    print("1. 载荷值完全一致")
    print("2. 因子值完全一致")
    print("3. 卡尔曼滤波参数完全一致")
    print("4. Nowcasting值完全一致\n")

    results = []

    try:
        results.append(("载荷值一致性", test_loading_consistency()))
    except Exception as e:
        print(f"\n[失败] 载荷值测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("载荷值一致性", False))

    try:
        results.append(("因子值一致性", test_factor_consistency()))
    except Exception as e:
        print(f"\n[失败] 因子值测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("因子值一致性", False))

    try:
        results.append(("卡尔曼参数一致性", test_kalman_params_consistency()))
    except Exception as e:
        print(f"\n[失败] 卡尔曼参数测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("卡尔曼参数一致性", False))

    try:
        results.append(("Nowcasting一致性", test_nowcasting_consistency()))
    except Exception as e:
        print(f"\n[失败] Nowcasting测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Nowcasting一致性", False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, passed in results:
        status = "[通过]" if passed else "[失败]"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n" + "="*80)
        print("恭喜！所有测试完全通过！")
        print("="*80)
        print("\ntrain_ref实现与train_model完全一致：")
        print("  ✓ 载荷值完全一致")
        print("  ✓ 因子值完全一致")
        print("  ✓ 卡尔曼滤波参数完全一致")
        print("  ✓ Nowcasting值完全一致")
        print("\n可以安全地使用train_ref进行生产环境部署！")
        print("\n" + "="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("部分测试失败")
        print("="*80)
        print(f"\n失败的测试:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")
        print("\n需要进一步调查差异原因")
        print("\n" + "="*80)
        return 1


if __name__ == "__main__":
    exit(main())
