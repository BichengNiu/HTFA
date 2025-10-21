# -*- coding: utf-8 -*-
"""
DFM模型一致性测试

对比 train_ref/core/factor_model.py 与 train_model/DynamicFactorModel.py
验证计算结果的一致性
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新代码
from dashboard.DFM.train_ref.core.factor_model import DFMModel, fit_dfm

# 导入老代码
from dashboard.DFM.train_model.DynamicFactorModel import DFM_EMalgo


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)

    n_time = 100
    n_vars = 10
    n_factors_true = 3

    # 生成真实因子
    true_factors = np.random.randn(n_time, n_factors_true)

    # 生成载荷矩阵
    true_loadings = np.random.randn(n_vars, n_factors_true) * 0.5

    # 生成观测数据
    observations = true_factors @ true_loadings.T + np.random.randn(n_time, n_vars) * 0.3

    # 创建DataFrame
    dates = pd.date_range('2015-01-01', periods=n_time, freq='M')
    data = pd.DataFrame(
        observations,
        index=dates,
        columns=[f'Var{i+1}' for i in range(n_vars)]
    )

    return data, true_factors, true_loadings


def test_dfm_basic_fit():
    """测试DFM基本拟合"""
    print("="*80)
    print("测试1: DFM基本拟合一致性")
    print("="*80)

    data, true_factors, true_loadings = create_test_data()

    n_factors = 3
    max_lags = 1
    max_iter = 10  # 使用较少迭代以加快测试

    print(f"\n参数设置:")
    print(f"  数据形状: {data.shape}")
    print(f"  因子数量: {n_factors}")
    print(f"  最大滞后: {max_lags}")
    print(f"  最大迭代: {max_iter}")

    # 新代码
    print("\n运行新代码 (train_ref)...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_lags=max_lags,
        max_iter=max_iter,
        train_end=None
    )

    print(f"  迭代次数: {result_new.n_iter}")
    print(f"  是否收敛: {result_new.converged}")
    print(f"  对数似然: {result_new.loglikelihood:.4f}")

    # 老代码
    print("\n运行老代码 (train_model)...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=max_iter,
        train_end_date=None,
        max_lags=max_lags
    )

    # 对比因子
    print("\n对比提取的因子:")
    factors_new = result_new.factors.values
    factors_old = result_old.x_sm.values

    # 因子可能有符号和顺序差异，需要找到最佳对齐
    corr_matrix = np.corrcoef(factors_new.T, factors_old.T)[:n_factors, n_factors:]
    abs_corr = np.abs(corr_matrix)

    print(f"  因子相关系数矩阵:")
    print(f"  {abs_corr}")

    mean_corr = np.mean(np.max(abs_corr, axis=1))
    print(f"  平均最大相关系数: {mean_corr:.4f}")

    # 对比载荷矩阵
    print("\n对比载荷矩阵:")
    loadings_new = result_new.loadings
    loadings_old = result_old.Lambda

    # 计算载荷的Frobenius范数差异（归一化）
    loadings_new_norm = loadings_new / np.linalg.norm(loadings_new, axis=0, keepdims=True)
    loadings_old_norm = loadings_old / np.linalg.norm(loadings_old, axis=0, keepdims=True)

    # 考虑符号差异
    min_diff = np.inf
    for signs in [1, -1]:
        diff = np.linalg.norm(loadings_new_norm - signs * loadings_old_norm, 'fro')
        min_diff = min(min_diff, diff)

    print(f"  归一化Frobenius范数差异: {min_diff:.6f}")

    # 对比对数似然（老代码可能没有此属性）
    print("\n对比对数似然:")
    loglik_new = result_new.loglikelihood
    print(f"  新代码: {loglik_new:.4f}")

    loglik_match = True
    if hasattr(result_old, 'loglik'):
        loglik_old = result_old.loglik
        print(f"  老代码: {loglik_old:.4f}")
        print(f"  相对差异: {abs(loglik_new - loglik_old) / abs(loglik_old):.6f}")
        loglik_match = abs(loglik_new - loglik_old) / abs(loglik_old) < 0.1
    else:
        print(f"  老代码: 未提供（对象没有loglik属性）")

    # 判断通过标准
    passed = (mean_corr > 0.95 and  # 因子高度相关
              min_diff < 0.5 and     # 载荷相似
              loglik_match)           # 似然接近（如果可比）

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'}")
    print(f"  (标准: 因子相关>0.95, 载荷差异<0.5, 似然相对差异<10%)")

    return passed


def test_dfm_with_train_split():
    """测试带训练期切分的DFM"""
    print("\n" + "="*80)
    print("测试2: 带训练期切分的DFM拟合")
    print("="*80)

    data, _, _ = create_test_data()

    n_factors = 2
    train_end = data.index[70]  # 70%训练，30%测试

    print(f"\n参数设置:")
    print(f"  训练期结束: {train_end}")
    print(f"  训练样本: 70, 全样本: {len(data)}")

    # 新代码
    print("\n运行新代码...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_lags=1,
        max_iter=15,
        train_end=str(train_end.date())
    )

    # 老代码
    print("运行老代码...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=15,
        train_end_date=str(train_end.date()),
        max_lags=1
    )

    # 对比因子（确保时间维度一致）
    factors_new = result_new.factors.values  # (n_time_new, n_factors)
    factors_old = result_old.x_sm.values     # (n_time_old, n_factors)

    # 只比较共同的时间范围
    n_time_common = min(factors_new.shape[0], factors_old.shape[0])
    factors_new_common = factors_new[:n_time_common, :]
    factors_old_common = factors_old[:n_time_common, :]

    # 计算相关性
    corr_matrix = np.corrcoef(factors_new_common.T, factors_old_common.T)[:n_factors, n_factors:]
    mean_corr = np.mean(np.abs(np.max(np.abs(corr_matrix), axis=1)))

    print(f"\n对比因子数量: {factors_new.shape[0]} vs {factors_old.shape[0]}")
    print(f"共同时间范围: {n_time_common} 个样本")

    print(f"\n因子相关性: {mean_corr:.4f}")

    # 对比似然
    print(f"\n对数似然:")
    print(f"  新代码: {result_new.loglikelihood:.4f}")
    if hasattr(result_old, 'loglik'):
        print(f"  老代码: {result_old.loglik:.4f}")
    else:
        print(f"  老代码: 未提供")

    passed = mean_corr > 0.90

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'} (相关性>0.90)")

    return passed


def test_dfm_convergence():
    """测试DFM收敛行为"""
    print("\n" + "="*80)
    print("测试3: DFM收敛行为")
    print("="*80)

    data, _, _ = create_test_data()

    n_factors = 3
    max_iter = 30

    print(f"\n测试收敛性...")

    # 新代码
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_lags=1,
        max_iter=max_iter
    )

    # 老代码
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=max_iter,
        max_lags=1
    )

    print(f"\n新代码:")
    print(f"  迭代次数: {result_new.n_iter}")
    print(f"  是否收敛: {result_new.converged}")
    print(f"  最终似然: {result_new.loglikelihood:.4f}")

    print(f"\n老代码:")
    print(f"  迭代次数: {max_iter}")
    if hasattr(result_old, 'loglik'):
        print(f"  最终似然: {result_old.loglik:.4f}")
    else:
        print(f"  最终似然: 未提供")

    # 检查新代码是否正确识别收敛
    passed = result_new.converged or result_new.n_iter == max_iter

    print(f"\n{'✓ 测试通过' if passed else '✗ 测试失败'}")

    return passed


def test_dfm_dimensions():
    """测试不同维度配置"""
    print("\n" + "="*80)
    print("测试4: 不同维度配置")
    print("="*80)

    data, _, _ = create_test_data()

    test_configs = [
        (2, 1),  # 2因子, 1滞后
        (4, 1),  # 4因子, 1滞后
        (3, 2),  # 3因子, 2滞后
    ]

    all_passed = True

    for n_factors, max_lags in test_configs:
        print(f"\n测试配置: k={n_factors}, lags={max_lags}")

        try:
            result_new = fit_dfm(
                data=data,
                n_factors=n_factors,
                max_lags=max_lags,
                max_iter=10
            )

            result_old = DFM_EMalgo(
                observation=data,
                n_factors=n_factors,
                n_shocks=n_factors,
                n_iter=10,
                max_lags=max_lags
            )

            # 检查形状
            assert result_new.factors.shape == (len(data), n_factors)
            assert result_new.loadings.shape == (data.shape[1], n_factors)

            print(f"  ✓ 形状正确")

            # 检查相关性
            corr = np.corrcoef(
                result_new.factors.values.T,
                result_old.x_sm.values.T
            )[:n_factors, n_factors:]

            mean_corr = np.mean(np.abs(np.max(np.abs(corr), axis=1)))
            print(f"  因子相关性: {mean_corr:.4f}")

            if mean_corr < 0.85:
                all_passed = False
                print(f"  ✗ 相关性不足")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            all_passed = False

    print(f"\n{'✓ 所有配置通过' if all_passed else '✗ 部分配置失败'}")

    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("DFM模型一致性测试套件")
    print("对比 train_ref vs train_model")
    print("="*80)

    results = []

    try:
        results.append(("基本拟合", test_dfm_basic_fit()))
    except Exception as e:
        print(f"\n✗ 基本拟合测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("基本拟合", False))

    try:
        results.append(("训练期切分", test_dfm_with_train_split()))
    except Exception as e:
        print(f"\n✗ 训练期切分测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("训练期切分", False))

    try:
        results.append(("收敛行为", test_dfm_convergence()))
    except Exception as e:
        print(f"\n✗ 收敛行为测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("收敛行为", False))

    try:
        results.append(("多维度配置", test_dfm_dimensions()))
    except Exception as e:
        print(f"\n✗ 多维度配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("多维度配置", False))

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
        print("\n🎉 所有测试通过！train_ref DFM模型与老代码一致。")
        return 0
    else:
        print("\n⚠️  部分测试失败，需要检查差异。")
        return 1


if __name__ == "__main__":
    exit(main())
