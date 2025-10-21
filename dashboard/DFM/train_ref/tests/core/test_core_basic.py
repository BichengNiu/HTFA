# -*- coding: utf-8 -*-
"""
核心层基本功能测试

验证train_ref核心层的基本功能是否正常工作
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目路径 - 需要向上6层到达HTFA根目录
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.kalman import KalmanFilter
from dashboard.DFM.train_ref.core.estimator import estimate_loadings, estimate_target_loading
from dashboard.DFM.train_ref.core.factor_model import fit_dfm


def test_kalman_filter_basic():
    """测试卡尔曼滤波基本功能"""
    print("="*80)
    print("测试1: 卡尔曼滤波基本功能")
    print("="*80)

    np.random.seed(42)

    # 简单的状态空间模型
    A = np.array([[0.9, 0.1], [0, 0.8]])
    B = np.eye(2)
    H = np.array([[1, 0], [0, 1], [1, 1]])
    Q = np.eye(2) * 0.1
    R = np.eye(3) * 0.2
    x0 = np.zeros(2)
    P0 = np.eye(2)

    # 生成测试数据
    n_time = 30
    Z = np.random.randn(3, n_time)
    U = np.zeros((2, n_time))

    print(f"\n配置: 状态维度={A.shape[0]}, 观测维度={H.shape[0]}, 时间步={n_time}")

    try:
        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        result = kf.filter(Z, U)

        print(f"\n滤波结果:")
        print(f"  滤波状态形状: {result.x_filtered.shape}")
        print(f"  预测状态形状: {result.x_predicted.shape}")
        print(f"  对数似然: {result.loglikelihood:.4f}")

        # 检查形状
        assert result.x_filtered.shape == (2, n_time)
        assert result.x_predicted.shape == (2, n_time)
        assert not np.isnan(result.loglikelihood)

        print("\n[通过] 卡尔曼滤波功能正常")
        return True

    except Exception as e:
        print(f"\n[失败] 卡尔曼滤波测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kalman_smoother_basic():
    """测试卡尔曼平滑基本功能"""
    print("\n" + "="*80)
    print("测试2: 卡尔曼平滑基本功能")
    print("="*80)

    np.random.seed(42)

    A = np.array([[0.9, 0.1], [0, 0.8]])
    B = np.eye(2)
    H = np.array([[1, 0], [0, 1]])
    Q = np.eye(2) * 0.1
    R = np.eye(2) * 0.2
    x0 = np.zeros(2)
    P0 = np.eye(2)

    n_time = 30
    Z = np.random.randn(2, n_time)
    U = np.zeros((2, n_time))

    try:
        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        filter_result = kf.filter(Z, U)
        smoother_result = kf.smooth(filter_result)

        print(f"\n平滑结果:")
        print(f"  平滑状态形状: {smoother_result.x_smoothed.shape}")
        print(f"  平滑协方差形状: {smoother_result.P_smoothed.shape}")

        # 检查形状
        assert smoother_result.x_smoothed.shape == (2, n_time)
        assert smoother_result.P_smoothed.shape == (2, 2, n_time)

        print("\n[通过] 卡尔曼平滑功能正常")
        return True

    except Exception as e:
        print(f"\n[失败] 卡尔曼平滑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loadings_estimation_basic():
    """测试载荷估计基本功能"""
    print("\n" + "="*80)
    print("测试3: 载荷估计基本功能")
    print("="*80)

    np.random.seed(42)

    n_time = 50
    n_obs = 6
    n_factors = 2

    # 生成测试数据
    factors = np.random.randn(n_time, n_factors)
    true_loadings = np.random.randn(n_obs, n_factors)
    observations = factors @ true_loadings.T + np.random.randn(n_time, n_obs) * 0.1

    dates = pd.date_range('2020-01-01', periods=n_time, freq='M')
    obs_df = pd.DataFrame(observations, index=dates, columns=[f'Obs{i}' for i in range(n_obs)])
    factors_df = pd.DataFrame(factors, index=dates, columns=[f'F{i}' for i in range(n_factors)])

    try:
        estimated_loadings = estimate_loadings(obs_df, factors_df)

        print(f"\n估计结果:")
        print(f"  载荷矩阵形状: {estimated_loadings.shape}")
        print(f"  真实载荷形状: {true_loadings.shape}")

        # 计算相关性
        corr = np.corrcoef(estimated_loadings.ravel(), true_loadings.ravel())[0, 1]
        print(f"  与真实载荷的相关性: {corr:.4f}")

        # 检查
        assert estimated_loadings.shape == true_loadings.shape
        assert corr > 0.85  # 应该高度相关

        print("\n[通过] 载荷估计功能正常")
        return True

    except Exception as e:
        print(f"\n[失败] 载荷估计测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dfm_fit_basic():
    """测试DFM拟合基本功能"""
    print("\n" + "="*80)
    print("测试4: DFM拟合基本功能")
    print("="*80)

    np.random.seed(42)

    n_time = 80
    n_vars = 8
    n_factors = 2

    # 生成测试数据
    true_factors = np.random.randn(n_time, n_factors)
    true_loadings = np.random.randn(n_vars, n_factors) * 0.5
    observations = true_factors @ true_loadings.T + np.random.randn(n_time, n_vars) * 0.3

    dates = pd.date_range('2015-01-01', periods=n_time, freq='M')
    data = pd.DataFrame(observations, index=dates, columns=[f'Var{i}' for i in range(n_vars)])

    print(f"\n数据: {data.shape}, 因子数={n_factors}")

    try:
        result = fit_dfm(
            data=data,
            n_factors=n_factors,
            max_lags=1,
            max_iter=15
        )

        print(f"\nDFM结果:")
        print(f"  迭代次数: {result.n_iter}")
        print(f"  是否收敛: {result.converged}")
        print(f"  因子形状: {result.factors.shape}")
        print(f"  载荷形状: {result.loadings.shape}")
        print(f"  对数似然: {result.loglikelihood:.4f}")

        # 检查
        assert result.factors.shape == (n_time, n_factors)
        assert result.loadings.shape == (n_vars, n_factors)
        assert result.n_iter <= 15

        # 检查提取的因子与真实因子的相关性
        corr_matrix = np.corrcoef(result.factors.values.T, true_factors.T)[:n_factors, n_factors:]
        max_corr = np.max(np.abs(corr_matrix), axis=1).mean()

        print(f"  因子相关性: {max_corr:.4f}")

        assert max_corr > 0.7  # 应该能恢复大部分真实因子信息

        print("\n[通过] DFM拟合功能正常")
        return True

    except Exception as e:
        print(f"\n[失败] DFM拟合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dfm_with_train_split():
    """测试DFM带训练期切分"""
    print("\n" + "="*80)
    print("测试5: DFM训练期切分功能")
    print("="*80)

    np.random.seed(42)

    n_time = 60
    data = pd.DataFrame(
        np.random.randn(n_time, 6),
        index=pd.date_range('2018-01-01', periods=n_time, freq='M'),
        columns=[f'V{i}' for i in range(6)]
    )

    train_end = data.index[40]

    print(f"\n训练期结束: {train_end}, 全样本: {len(data)}")

    try:
        result = fit_dfm(
            data=data,
            n_factors=2,
            max_lags=1,
            max_iter=10,
            train_end=str(train_end.date())
        )

        print(f"\n结果: 迭代{result.n_iter}次")
        print(f"  因子形状: {result.factors.shape}")
        print(f"  因子时间范围: {result.factors.index[0]} 到 {result.factors.index[-1]}")

        # 因子应该只覆盖训练期
        assert result.factors.shape[0] == 41  # 训练期长度
        assert result.factors.index[-1] <= pd.to_datetime(train_end)

        print("\n[通过] 训练期切分功能正常")
        return True

    except Exception as e:
        print(f"\n[失败] 训练期切分测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("train_ref 核心层基本功能测试")
    print("="*80)

    tests = [
        ("卡尔曼滤波", test_kalman_filter_basic),
        ("卡尔曼平滑", test_kalman_smoother_basic),
        ("载荷估计", test_loadings_estimation_basic),
        ("DFM拟合", test_dfm_fit_basic),
        ("训练期切分", test_dfm_with_train_split),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[失败] {name}测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

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
        print("\n>>> 所有测试通过！train_ref核心层功能正常。")
        return 0
    else:
        print("\n>>> 部分测试失败，需要修复问题。")
        return 1


if __name__ == "__main__":
    exit(main())
