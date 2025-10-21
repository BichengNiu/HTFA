# -*- coding: utf-8 -*-
"""
对比DFMModel.fit(max_iter=1)与老代码DFM_EMalgo(n_iter=1)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.DFM.train_ref.core.factor_model import DFMModel
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


if __name__ == "__main__":
    print("="*80)
    print("对比DFMModel.fit(max_iter=1)与老代码DFM_EMalgo(n_iter=1)")
    print("="*80)

    data = create_test_data()
    n_factors = 3

    # 新代码
    print("\n运行新代码 DFMModel.fit(max_iter=1)...")
    model_new = DFMModel(n_factors=n_factors, max_lags=1, max_iter=1)
    result_new = model_new.fit(data)

    print(f"  迭代次数: {result_new.n_iter}")
    print(f"  因子shape: {result_new.factors.shape}")
    print(f"  因子前3行:\n{result_new.factors.iloc[:3]}")
    print(f"  Lambda shape: {result_new.loadings.shape}")
    print(f"  Lambda前3行:\n{result_new.loadings[:3]}")

    # 老代码
    print("\n运行老代码 DFM_EMalgo(n_iter=1)...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=1,
        max_lags=1
    )

    print(f"  迭代次数: 1")
    print(f"  因子shape: {result_old.x_sm.shape}")
    print(f"  因子前3行:\n{result_old.x_sm.iloc[:3]}")
    print(f"  Lambda shape: {result_old.Lambda.shape}")
    print(f"  Lambda前3行:\n{result_old.Lambda[:3]}")

    # 对比
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    factors_diff = np.abs(result_new.factors.values - result_old.x_sm.values)
    Lambda_diff = np.abs(result_new.loadings - result_old.Lambda)

    print(f"\n因子差异:")
    print(f"  最大差异: {np.max(factors_diff):.15f}")
    print(f"  平均差异: {np.mean(factors_diff):.15f}")
    print(f"  相关系数: {np.corrcoef(result_new.factors.values.flatten(), result_old.x_sm.values.flatten())[0,1]:.15f}")

    print(f"\nLambda差异:")
    print(f"  最大差异: {np.max(Lambda_diff):.15f}")
    print(f"  平均差异: {np.mean(Lambda_diff):.15f}")

    if np.max(factors_diff) < 1e-10 and np.max(Lambda_diff) < 1e-10:
        print("\n[通过] DFMModel.fit(max_iter=1)与老代码完全一致!")
    else:
        print("\n[失败] 存在差异")
        print(f"\n差异最大的因子位置:")
        idx = np.unravel_index(np.argmax(factors_diff), factors_diff.shape)
        print(f"  位置: {idx}")
        print(f"  新代码值: {result_new.factors.values[idx]:.15f}")
        print(f"  老代码值: {result_old.x_sm.values[idx]:.15f}")
