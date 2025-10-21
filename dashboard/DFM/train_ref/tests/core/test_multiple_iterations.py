# -*- coding: utf-8 -*-
"""
测试多次EM迭代的一致性
找出差异何时开始累积
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.factor_model import fit_dfm
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


def test_iterations(max_iter):
    """测试指定次数的EM迭代"""
    print(f"\n{'='*80}")
    print(f"测试{max_iter}次EM迭代")
    print(f"{'='*80}")

    data = create_test_data()
    n_factors = 3

    # 新代码
    print(f"\n运行新代码 (max_iter={max_iter})...")
    result_new = fit_dfm(
        data=data,
        n_factors=n_factors,
        max_iter=max_iter
    )

    # 老代码
    print(f"运行老代码 (max_iter={max_iter})...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=max_iter,
        max_lags=1
    )

    # 对比结果
    print(f"\n对比结果:")
    print(f"  迭代次数: 新={result_new.n_iter}, 老={max_iter}")
    print(f"  收敛状态: 新={result_new.converged}")

    # 载荷差异
    Lambda_diff = np.abs(result_new.loadings - result_old.Lambda)
    print(f"\n  Lambda差异: 最大={np.max(Lambda_diff):.10f}, 平均={np.mean(Lambda_diff):.10f}")

    # 因子差异
    factors_new = result_new.factors.values
    factors_old = result_old.x_sm.values
    factors_diff = np.abs(factors_new - factors_old)
    print(f"  因子差异: 最大={np.max(factors_diff):.10f}, 平均={np.mean(factors_diff):.10f}")

    # 因子相关性
    print(f"  因子相关系数:")
    for i in range(n_factors):
        corr = np.corrcoef(factors_new[:, i], factors_old[:, i])[0, 1]
        print(f"    因子{i}: {corr:.10f}")

    # 参数差异
    A_diff = np.abs(result_new.transition_matrix - result_old.A)
    Q_diff = np.abs(result_new.process_noise_cov - result_old.Q)
    R_diff = np.abs(result_new.measurement_noise_cov - result_old.R)

    print(f"\n  A差异: 最大={np.max(A_diff):.10f}")
    print(f"  Q差异: 最大={np.max(Q_diff):.10f}")
    print(f"  R差异: 最大={np.max(R_diff):.10f}")

    return {
        'max_iter': max_iter,
        'lambda_max_diff': np.max(Lambda_diff),
        'factor_max_diff': np.max(factors_diff),
        'min_corr': min([np.corrcoef(factors_new[:, i], factors_old[:, i])[0, 1] for i in range(n_factors)])
    }


if __name__ == "__main__":
    print("="*80)
    print("多次EM迭代一致性测试")
    print("="*80)
    print("\n测试不同迭代次数,观察差异何时开始累积")

    results = []
    for n_iter in [1, 2, 3, 5, 10]:
        result = test_iterations(n_iter)
        results.append(result)

    # 总结
    print("\n" + "="*80)
    print("差异累积总结")
    print("="*80)
    print(f"\n{'迭代次数':<10} {'Lambda最大差异':<20} {'因子最大差异':<20} {'最小相关系数':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['max_iter']:<10} {r['lambda_max_diff']:<20.10f} {r['factor_max_diff']:<20.10f} {r['min_corr']:<15.10f}")
