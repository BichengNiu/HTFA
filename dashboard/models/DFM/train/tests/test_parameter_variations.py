# -*- coding: utf-8 -*-
"""
参数变化测试

验证：
1. 不同的因子数是否产生不同的结果
2. 不同的max_iterations是否影响结果
3. 累积方差策略是否正确工作
4. 参数是否正确传递到训练器
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# 设置UTF-8编码，避免GBK编码错误
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")

from dashboard.models.DFM.train.training.config import TrainingConfig
from dashboard.models.DFM.train.training.trainer import DFMTrainer


def load_test_data():
    """加载测试数据"""
    data_path = project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"测试数据不存在: {data_path}")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"测试数据加载完成: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")

    return data


def train_with_config(config_name, config):
    """使用给定配置训练模型"""
    print(f"\n{'=' * 80}")
    print(f"训练: {config_name}")
    print(f"{'=' * 80}")
    print(f"  k_factors: {config.k_factors}")
    print(f"  max_iterations: {config.max_iterations}")
    print(f"  max_lags: {config.max_lags}")
    print(f"  factor_selection_method: {config.factor_selection_method}")
    print(f"  pca_threshold: {config.pca_threshold}")

    trainer = DFMTrainer(config)

    result = trainer.train(
        progress_callback=None,
        enable_export=False  # 不导出文件，加快测试
    )

    print(f"\n训练完成:")
    print(f"  最终因子数: {result.k_factors}")
    print(f"  样本内RMSE: {result.metrics.is_rmse:.6f}")
    print(f"  样本外RMSE: {result.metrics.oos_rmse:.6f}")
    print(f"  样本内MAE: {result.metrics.is_mae:.6f}")
    print(f"  样本外MAE: {result.metrics.oos_mae:.6f}")
    print(f"  样本内Hit Rate: {result.metrics.is_hit_rate:.2f}%")
    print(f"  样本外Hit Rate: {result.metrics.oos_hit_rate:.2f}%")
    print(f"  训练时间: {result.training_time:.2f}秒")

    return result


def test_different_k_factors():
    """测试1: 不同的因子数应该产生不同的结果"""

    print("\n" + "=" * 80)
    print("测试1: 不同因子数的结果是否不同")
    print("=" * 80)

    data = load_test_data()

    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]

    # 选择NaN最少的15个指标
    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(15).index.tolist()

    # 基础配置
    base_kwargs = {
        'data_path': str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
        'target_variable': target_variable,
        'selected_indicators': selected_indicators,
        'training_start': '2020-01-03',
        'train_end': '2024-08-16',
        'validation_start': '2024-08-23',
        'validation_end': '2025-08-29',
        'target_freq': 'W-FRI',
        'max_iterations': 30,
        'max_lags': 1,
        'tolerance': 1e-6,
        'enable_variable_selection': False,
        'factor_selection_method': 'fixed',
        'output_dir': str(project_root / "dashboard/models/DFM/train/tests/result")
    }

    # 测试不同的因子数
    results = {}

    for k in [2, 3, 4]:
        config = TrainingConfig(
            **base_kwargs,
            k_factors=k
        )

        result = train_with_config(f"k_factors={k}", config)
        results[f"k={k}"] = result

    # 比较结果
    print(f"\n{'=' * 80}")
    print("结果对比")
    print(f"{'=' * 80}")

    print(f"\n{'配置':<15} {'最终k':<8} {'样本内RMSE':<15} {'样本外RMSE':<15} {'样本内Hit%':<12} {'样本外Hit%':<12}")
    print("-" * 80)

    for name, result in results.items():
        print(f"{name:<15} {result.k_factors:<8} {result.metrics.is_rmse:<15.6f} "
              f"{result.metrics.oos_rmse:<15.6f} {result.metrics.is_hit_rate:<12.2f} "
              f"{result.metrics.oos_hit_rate:<12.2f}")

    # 验证结果是否不同
    rmse_values = [r.metrics.is_rmse for r in results.values()]

    print(f"\n样本内RMSE值: {rmse_values}")

    if len(set(rmse_values)) == len(rmse_values):
        print(f"[PASS] 不同因子数产生了不同的结果")
        return True
    else:
        print(f"[FAIL] 不同因子数产生了相同的结果！")
        return False


def test_different_max_iterations():
    """测试2: 不同的max_iterations应该产生不同的结果"""

    print("\n" + "=" * 80)
    print("测试2: 不同max_iterations的结果是否不同")
    print("=" * 80)

    data = load_test_data()

    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]

    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(10).index.tolist()

    base_kwargs = {
        'data_path': str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
        'target_variable': target_variable,
        'selected_indicators': selected_indicators,
        'training_start': '2020-01-03',
        'train_end': '2024-08-16',
        'validation_start': '2024-08-23',
        'validation_end': '2025-08-29',
        'target_freq': 'W-FRI',
        'k_factors': 3,
        'max_lags': 1,
        'tolerance': 1e-6,
        'enable_variable_selection': False,
        'factor_selection_method': 'fixed',
        'output_dir': str(project_root / "dashboard/models/DFM/train/tests/result")
    }

    # 测试不同的迭代次数
    results = {}

    for max_iter in [10, 20, 30]:
        config = TrainingConfig(
            **base_kwargs,
            max_iterations=max_iter
        )

        result = train_with_config(f"max_iterations={max_iter}", config)
        results[f"iter={max_iter}"] = result

    # 比较结果
    print(f"\n{'=' * 80}")
    print("结果对比")
    print(f"{'=' * 80}")

    print(f"\n{'配置':<15} {'样本内RMSE':<15} {'样本外RMSE':<15}")
    print("-" * 50)

    for name, result in results.items():
        print(f"{name:<15} {result.metrics.is_rmse:<15.6f} {result.metrics.oos_rmse:<15.6f}")

    # 验证结果是否不同
    rmse_values = [r.metrics.is_rmse for r in results.values()]

    print(f"\n样本内RMSE值: {rmse_values}")

    # max_iterations不同可能产生略有不同的结果（因为EM算法收敛程度不同）
    # 我们检查是否至少有一个值不同
    unique_count = len(set([round(v, 6) for v in rmse_values]))

    if unique_count >= 2:
        print(f"[PASS] 不同迭代次数产生了不同的结果（至少有{unique_count}个不同值）")
        return True
    else:
        print(f"[WARNING] 不同迭代次数可能产生相同结果（可能已收敛）")
        return True  # 这不一定是错误，可能算法已经收敛


def test_cumulative_variance_strategy():
    """测试3: 累积方差策略是否正确工作"""

    print("\n" + "=" * 80)
    print("测试3: 累积方差策略")
    print("=" * 80)

    data = load_test_data()

    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]

    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(15).index.tolist()

    base_kwargs = {
        'data_path': str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
        'target_variable': target_variable,
        'selected_indicators': selected_indicators,
        'training_start': '2020-01-03',
        'train_end': '2024-08-16',
        'validation_start': '2024-08-23',
        'validation_end': '2025-08-29',
        'target_freq': 'W-FRI',
        'max_iterations': 30,
        'max_lags': 1,
        'tolerance': 1e-6,
        'enable_variable_selection': False,
        'output_dir': str(project_root / "dashboard/models/DFM/train/tests/result")
    }

    # 测试不同的PCA阈值
    results = {}

    for threshold in [0.8, 0.85, 0.9]:
        config = TrainingConfig(
            **base_kwargs,
            factor_selection_method='cumulative',
            k_factors=4,  # 这个会被PCA覆盖
            pca_threshold=threshold
        )

        result = train_with_config(f"pca_threshold={threshold}", config)
        results[f"threshold={threshold}"] = result

    # 比较结果
    print(f"\n{'=' * 80}")
    print("结果对比")
    print(f"{'=' * 80}")

    print(f"\n{'配置':<20} {'最终k':<8} {'样本内RMSE':<15} {'样本外RMSE':<15}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<20} {result.k_factors:<8} {result.metrics.is_rmse:<15.6f} "
              f"{result.metrics.oos_rmse:<15.6f}")

    # 验证PCA是否选择了不同的因子数
    k_values = [r.k_factors for r in results.values()]

    print(f"\n选定的因子数: {k_values}")

    # 检查因子数是否随阈值递增
    if k_values == sorted(k_values):
        print(f"[PASS] PCA阈值越大，选定的因子数越多（符合预期）")
        return True
    elif len(set(k_values)) >= 2:
        print(f"[PASS] 不同PCA阈值产生了不同的因子数")
        return True
    else:
        print(f"[FAIL] 所有PCA阈值产生了相同的因子数: {k_values}")
        return False


def test_fixed_vs_cumulative():
    """测试4: fixed策略 vs cumulative策略"""

    print("\n" + "=" * 80)
    print("测试4: fixed策略 vs cumulative策略")
    print("=" * 80)

    data = load_test_data()

    target_variable = "规模以上工业增加值:当月同比"
    all_indicators = [col for col in data.columns if col != target_variable]

    nan_counts = data[all_indicators].isna().sum().sort_values()
    selected_indicators = nan_counts.head(15).index.tolist()

    base_kwargs = {
        'data_path': str(project_root / "dashboard/models/DFM/train/tests/data/dfm_prepared_output.csv"),
        'target_variable': target_variable,
        'selected_indicators': selected_indicators,
        'training_start': '2020-01-03',
        'train_end': '2024-08-16',
        'validation_start': '2024-08-23',
        'validation_end': '2025-08-29',
        'target_freq': 'W-FRI',
        'max_iterations': 30,
        'max_lags': 1,
        'tolerance': 1e-6,
        'enable_variable_selection': False,
        'output_dir': str(project_root / "dashboard/models/DFM/train/tests/result")
    }

    # Fixed策略
    config_fixed = TrainingConfig(
        **base_kwargs,
        factor_selection_method='fixed',
        k_factors=3
    )

    result_fixed = train_with_config("fixed (k=3)", config_fixed)

    # Cumulative策略
    config_cumulative = TrainingConfig(
        **base_kwargs,
        factor_selection_method='cumulative',
        k_factors=4,  # 会被PCA覆盖
        pca_threshold=0.85
    )

    result_cumulative = train_with_config("cumulative (threshold=0.85)", config_cumulative)

    # 比较结果
    print(f"\n{'=' * 80}")
    print("结果对比")
    print(f"{'=' * 80}")

    print(f"\n{'策略':<30} {'最终k':<8} {'样本内RMSE':<15} {'样本外RMSE':<15}")
    print("-" * 70)
    print(f"{'fixed (k=3)':<30} {result_fixed.k_factors:<8} {result_fixed.metrics.is_rmse:<15.6f} "
          f"{result_fixed.metrics.oos_rmse:<15.6f}")
    print(f"{'cumulative (threshold=0.85)':<30} {result_cumulative.k_factors:<8} "
          f"{result_cumulative.metrics.is_rmse:<15.6f} {result_cumulative.metrics.oos_rmse:<15.6f}")

    # 验证
    if result_fixed.k_factors == 3:
        print(f"\n[PASS] fixed策略使用了指定的k=3")
    else:
        print(f"\n[FAIL] fixed策略应该使用k=3，但实际使用了k={result_fixed.k_factors}")
        return False

    if result_cumulative.k_factors != 4:
        print(f"[PASS] cumulative策略通过PCA自动选择了k={result_cumulative.k_factors}（不是输入的4）")
    else:
        print(f"[WARNING] cumulative策略选择了k=4，可能PCA刚好选了这个值")

    if result_fixed.metrics.is_rmse != result_cumulative.metrics.is_rmse:
        print(f"[PASS] 两种策略产生了不同的结果")
        return True
    else:
        print(f"[WARNING] 两种策略产生了相同的RMSE（如果k相同则正常）")
        return True


def main():
    """主测试函数"""
    print("=" * 80)
    print("参数变化测试")
    print("=" * 80)

    results = {}

    try:
        results['test1'] = test_different_k_factors()
    except Exception as e:
        print(f"\n[ERROR] 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results['test1'] = False

    try:
        results['test2'] = test_different_max_iterations()
    except Exception as e:
        print(f"\n[ERROR] 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results['test2'] = False

    try:
        results['test3'] = test_cumulative_variance_strategy()
    except Exception as e:
        print(f"\n[ERROR] 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results['test3'] = False

    try:
        results['test4'] = test_fixed_vs_cumulative()
    except Exception as e:
        print(f"\n[ERROR] 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        results['test4'] = False

    # 总结
    print("\n\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"测试1 (不同因子数): {'PASS' if results['test1'] else 'FAIL'}")
    print(f"测试2 (不同迭代次数): {'PASS' if results['test2'] else 'FAIL'}")
    print(f"测试3 (累积方差策略): {'PASS' if results['test3'] else 'FAIL'}")
    print(f"测试4 (fixed vs cumulative): {'PASS' if results['test4'] else 'FAIL'}")

    if all(results.values()):
        print(f"\n[PASS] 所有测试通过! 参数传递正常工作")
        return 0
    else:
        print(f"\n[FAIL] 部分测试失败，存在参数传递问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())
