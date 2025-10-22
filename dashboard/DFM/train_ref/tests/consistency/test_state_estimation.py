# -*- coding: utf-8 -*-
"""
状态估计对比测试

不依赖baseline，测试train_ref内部的状态估计（平滑因子）一致性和稳定性。
验证卡尔曼滤波+平滑算法产生的因子估计的数值稳定性。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.DFM.train_ref.core.factor_model import DFMModel
from dashboard.DFM.data_prep import prepare_data


class TestStateEstimation:
    """状态估计对比测试

    测试内容：
    1. 相同输入产生相同的平滑因子估计
    2. 因子估计矩阵的数值特性（有界性、有限性等）
    3. 时间点差异小于阈值（1e-6）
    """

    @pytest.fixture(scope="class")
    def prepared_data(self):
        """准备测试数据（所有测试共用）"""
        data_path = PROJECT_ROOT / "data" / "经济数据库1017.xlsx"

        if not data_path.exists():
            pytest.skip(f"数据文件不存在: {data_path}")

        # 使用data_prep准备数据
        processed_data, _, _, _ = prepare_data(
            excel_path=str(data_path),
            target_freq='W-FRI',
            target_sheet_name='工业增加值同比增速_月度_同花顺',
            target_variable_name='规模以上工业增加值:当月同比',
            consecutive_nan_threshold=10,
            data_start_date='2020-01-01',
            data_end_date='2025-07-03'
        )

        if processed_data is None:
            pytest.skip("数据预处理失败")

        return processed_data

    def test_state_estimation_reproducibility(self, prepared_data):
        """测试状态估计的可重现性：相同输入产生相同平滑因子"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 第一次估计
        np.random.seed(42)
        model1 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result1 = model1.fit(test_data)

        # 第二次估计（相同种子）
        np.random.seed(42)
        model2 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result2 = model2.fit(test_data)

        # 验证平滑因子一致（每个时间点差异 < 1e-10）
        factors1 = result1.factors.values
        factors2 = result2.factors.values

        np.testing.assert_allclose(factors1, factors2, rtol=1e-10, atol=1e-10,
                                   err_msg="平滑因子估计不一致")

        print(f"\n因子形状: {factors1.shape}")
        print(f"最大差异: {np.max(np.abs(factors1 - factors2))}")

    def test_smoothed_factors_properties(self, prepared_data):
        """测试平滑因子的数值特性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        np.random.seed(42)
        model = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        factors = result.factors.values

        # 验证因子形状
        n_time = len(result.factors)
        assert factors.shape == (n_time, 2), f"因子形状错误: {factors.shape}"

        # 验证所有值都是有限的
        assert np.all(np.isfinite(factors)), "因子包含NaN或Inf"

        # 验证因子值有界（一般应该在合理范围内）
        assert np.all(np.abs(factors) < 100), f"因子值过大: max={np.max(np.abs(factors))}"

        print(f"\n平滑因子形状: {factors.shape}")
        print(f"因子统计: mean={np.mean(np.abs(factors)):.2f}, std={np.std(factors):.2f}")
        print(f"因子范围: [{np.min(factors):.2f}, {np.max(factors):.2f}]")

    def test_time_point_consistency(self, prepared_data):
        """测试每个时间点的状态估计一致性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 多次运行（相同种子）
        seeds = [42, 42, 42]  # 相同种子应产生完全相同的结果
        results = []

        for seed in seeds:
            np.random.seed(seed)
            model = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(test_data)
            results.append(result.factors.values)

        # 验证每个时间点的差异 < 1e-10
        for i in range(1, len(results)):
            diff = np.abs(results[i] - results[0])
            max_diff = np.max(diff)
            assert max_diff < 1e-10, f"时间点差异过大: {max_diff}"

        print(f"\n时间点一致性测试通过，最大差异: {np.max(np.abs(results[1] - results[0]))}")

    def test_different_factor_numbers_states(self, prepared_data):
        """测试不同因子数的状态估计"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:8]
        test_data = prepared_data[selected_vars].dropna()

        n_time = len(test_data)

        for k in [1, 2, 3]:
            np.random.seed(42)
            model = DFMModel(n_factors=k, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(test_data)

            factors = result.factors.values

            # 验证因子维度正确
            assert factors.shape == (n_time, k), f"k={k}时因子形状错误: {factors.shape}"

            # 验证因子有效
            assert np.all(np.isfinite(factors)), f"k={k}时因子包含无效值"

            print(f"\nk={k}: 因子形状={factors.shape}, 范围=[{np.min(factors):.2f}, {np.max(factors):.2f}]")

    def test_single_factor_state_estimation(self, prepared_data):
        """专门测试单因子模型的状态估计"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        n_time = len(test_data)

        np.random.seed(42)
        model = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result = model.fit(test_data)

        factors = result.factors.values

        # 单因子时应该是 (n_time, 1) 形状
        assert factors.shape == (n_time, 1), f"单因子时形状错误: {factors.shape}"

        # 验证因子有效且有界
        assert np.all(np.isfinite(factors)), "单因子包含无效值"
        assert np.all(np.abs(factors) < 100), f"单因子值过大: max={np.max(np.abs(factors))}"

        print(f"\n单因子状态估计:")
        print(f"  形状: {factors.shape}")
        print(f"  前5个时间点: {factors[:5, 0]}")
        print(f"  统计: mean={np.mean(factors):.2f}, std={np.std(factors):.2f}")

    def test_factor_stability_across_data_subsets(self, prepared_data):
        """测试不同数据子集的因子估计稳定性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 使用不同长度的数据子集
        subsets = [
            test_data.iloc[:30],   # 前30个时间点
            test_data.iloc[-30:],  # 后30个时间点
        ]

        for i, subset in enumerate(subsets):
            if len(subset) < 10:  # 确保有足够的数据
                continue

            np.random.seed(42)
            model = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(subset)

            factors = result.factors.values

            # 验证因子有效
            assert np.all(np.isfinite(factors)), f"子集{i}的因子包含无效值"
            assert factors.shape[0] == len(subset), f"子集{i}因子长度不匹配"

            print(f"\n子集{i+1} (长度={len(subset)}): 因子形状={factors.shape}, "
                  f"范围=[{np.min(factors):.2f}, {np.max(factors):.2f}]")
