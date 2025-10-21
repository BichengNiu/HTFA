# -*- coding: utf-8 -*-
"""
对比新老DFM多次迭代的收敛情况
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
    print("对比新老DFM多次迭代的收敛情况")
    print("="*80)

    data = create_test_data()
    n_factors = 3
    max_iter = 10  # 运行10次迭代

    # 新代码
    print(f"\n运行新代码 DFMModel.fit(max_iter={max_iter})...")
    model_new = DFMModel(n_factors=n_factors, max_lags=1, max_iter=max_iter, tolerance=1e-6)
    result_new = model_new.fit(data)

    print(f"  实际迭代次数: {result_new.n_iter}")
    print(f"  因子shape: {result_new.factors.shape}")
    print(f"  因子前3行:\n{result_new.factors.iloc[:3]}")
    print(f"  Lambda shape: {result_new.loadings.shape}")

    # 老代码
    print(f"\n运行老代码 DFM_EMalgo(n_iter={max_iter})...")
    result_old = DFM_EMalgo(
        observation=data,
        n_factors=n_factors,
        n_shocks=n_factors,
        n_iter=max_iter,
        max_lags=1
    )

    print(f"  实际迭代次数: {max_iter}")
    print(f"  因子shape: {result_old.x_sm.shape}")
    print(f"  因子前3行:\n{result_old.x_sm.iloc[:3]}")
    print(f"  Lambda shape: {result_old.Lambda.shape}")

    # 对比
    print("\n" + "="*80)
    print("对比最终结果")
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

    # 检查收敛性
    if np.max(factors_diff) < 0.01 and np.max(Lambda_diff) < 0.01:
        print(f"\n[接近] 最终结果非常接近（差异 < 0.01）")
        print(f"  因子相关性: {np.corrcoef(result_new.factors.values.flatten(), result_old.x_sm.values.flatten())[0,1]:.6f}")
        if np.corrcoef(result_new.factors.values.flatten(), result_old.x_sm.values.flatten())[0,1] > 0.999:
            print(f"  相关性极高（> 0.999），可以认为基本一致")
    elif np.max(factors_diff) < 0.1:
        print(f"\n[可接受] 最终结果接近（差异 < 0.1）")
    else:
        print(f"\n[失败] 最终结果差异较大")
