# -*- coding: utf-8 -*-
"""
评估指标对比测试

不依赖baseline，测试train_ref内部的评估指标计算一致性和稳定性。
验证RMSE、Hit Rate、相关系数等指标的数值稳定性。
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
from dashboard.DFM.train_ref.evaluation.metrics import calculate_hit_rate
from dashboard.DFM.data_prep import prepare_data
from sklearn.metrics import mean_squared_error


class TestMetrics:
    """评估指标对比测试

    测试内容：
    1. RMSE计算的可重现性和有效性
    2. Hit Rate计算的可重现性和有效性
    3. 相关系数计算的一致性
    4. 评估器整体输出的一致性
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

    def test_rmse_calculation_reproducibility(self, prepared_data):
        """测试RMSE计算的可重现性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 分割训练集和测试集
        split_idx = int(len(test_data) * 0.7)
        train_data = test_data.iloc[:split_idx]
        test_set = test_data.iloc[split_idx:]

        # 第一次训练和预测
        np.random.seed(42)
        model1 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result1 = model1.fit(train_data)
        pred1 = result1.factors.iloc[-len(test_set):, 0]
        true_values = test_set[target_col].values[:len(pred1)]

        # 第二次训练和预测（相同种子）
        np.random.seed(42)
        model2 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result2 = model2.fit(train_data)
        pred2 = result2.factors.iloc[-len(test_set):, 0]

        # 计算RMSE
        rmse1 = np.sqrt(mean_squared_error(true_values, pred1.values[:len(true_values)]))
        rmse2 = np.sqrt(mean_squared_error(true_values, pred2.values[:len(true_values)]))

        # 验证RMSE一致（差异 < 1e-10）
        assert np.abs(rmse1 - rmse2) < 1e-10, f"RMSE不一致: {rmse1} vs {rmse2}"
        assert np.isfinite(rmse1), "RMSE不是有限值"
        assert rmse1 >= 0, "RMSE应该非负"

        print(f"\nRMSE1: {rmse1:.6f}")
        print(f"RMSE2: {rmse2:.6f}")
        print(f"差异: {np.abs(rmse1 - rmse2):.2e}")

    def test_hit_rate_calculation_reproducibility(self, prepared_data):
        """测试Hit Rate计算的可重现性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 分割训练集和测试集
        split_idx = int(len(test_data) * 0.7)
        train_data = test_data.iloc[:split_idx]
        test_set = test_data.iloc[split_idx:]

        # 第一次训练和预测
        np.random.seed(42)
        model1 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result1 = model1.fit(train_data)
        pred1 = result1.factors.iloc[-len(test_set):, 0]
        true_series = test_set[target_col].iloc[:len(pred1)]

        # 第二次训练和预测（相同种子）
        np.random.seed(42)
        model2 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result2 = model2.fit(train_data)
        pred2 = result2.factors.iloc[-len(test_set):, 0]

        # 计算Hit Rate（重置索引以确保对齐）
        true_reset = true_series.reset_index(drop=True)
        pred1_reset = pred1.iloc[:len(true_series)].reset_index(drop=True)
        pred2_reset = pred2.iloc[:len(true_series)].reset_index(drop=True)

        hr1 = calculate_hit_rate(true_reset, pred1_reset)
        hr2 = calculate_hit_rate(true_reset, pred2_reset)

        # 验证Hit Rate一致
        if np.isfinite(hr1) and np.isfinite(hr2):
            assert np.abs(hr1 - hr2) < 1e-10, f"Hit Rate不一致: {hr1} vs {hr2}"
            assert 0 <= hr1 <= 1, "Hit Rate应该在0-1之间"
            print(f"\nHit Rate1: {hr1:.4f}")
            print(f"Hit Rate2: {hr2:.4f}")
            print(f"差异: {np.abs(hr1 - hr2):.2e}")
        else:
            print(f"\nHit Rate无效（数据不足）: hr1={hr1}, hr2={hr2}")
            assert hr1 == hr2 or (np.isnan(hr1) and np.isnan(hr2)), "无效Hit Rate应该相同"

    def test_correlation_coefficient_consistency(self, prepared_data):
        """测试相关系数计算的一致性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 分割训练集和测试集
        split_idx = int(len(test_data) * 0.7)
        train_data = test_data.iloc[:split_idx]
        test_set = test_data.iloc[split_idx:]

        # 第一次训练和预测
        np.random.seed(42)
        model1 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result1 = model1.fit(train_data)
        pred1 = result1.factors.iloc[-len(test_set):, 0]
        true_series = test_set[target_col].iloc[:len(pred1)]

        # 第二次训练和预测（相同种子）
        np.random.seed(42)
        model2 = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
        result2 = model2.fit(train_data)
        pred2 = result2.factors.iloc[-len(test_set):, 0]

        # 计算相关系数
        corr1 = np.corrcoef(true_series.values, pred1.values[:len(true_series)])[0, 1]
        corr2 = np.corrcoef(true_series.values, pred2.values[:len(true_series)])[0, 1]

        # 验证相关系数一致（差异 < 1e-10）
        if np.isfinite(corr1) and np.isfinite(corr2):
            assert np.abs(corr1 - corr2) < 1e-10, f"相关系数不一致: {corr1} vs {corr2}"
            assert -1 <= corr1 <= 1, "相关系数应该在-1到1之间"
            print(f"\n相关系数1: {corr1:.6f}")
            print(f"相关系数2: {corr2:.6f}")
            print(f"差异: {np.abs(corr1 - corr2):.2e}")
        else:
            print(f"\n相关系数无效: corr1={corr1}, corr2={corr2}")

    def test_metrics_with_different_factor_numbers(self, prepared_data):
        """测试不同因子数下的指标计算"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:8]
        test_data = prepared_data[selected_vars].dropna()

        # 分割数据
        split_idx = int(len(test_data) * 0.7)
        train_data = test_data.iloc[:split_idx]
        test_set = test_data.iloc[split_idx:]

        for k in [1, 2]:
            np.random.seed(42)
            model = DFMModel(n_factors=k, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(train_data)

            # 使用第一个因子作为预测
            pred = result.factors.iloc[-len(test_set):, 0]
            true_series = test_set[target_col].iloc[:len(pred)]

            # 计算指标（重置索引以确保对齐）
            true_reset = true_series.reset_index(drop=True)
            pred_reset = pred.iloc[:len(true_series)].reset_index(drop=True)

            rmse = np.sqrt(mean_squared_error(true_reset.values, pred_reset.values))
            hr = calculate_hit_rate(true_reset, pred_reset)
            corr = np.corrcoef(true_reset.values, pred_reset.values)[0, 1]

            # 验证指标有效
            assert np.isfinite(rmse) and rmse >= 0, f"k={k}时RMSE无效: {rmse}"

            print(f"\nk={k}: RMSE={rmse:.4f}, Hit Rate={hr:.2%}, Corr={corr:.4f}")

    def test_hit_rate_function_properties(self):
        """测试Hit Rate函数的数学特性"""
        # 创建测试数据
        y_true = pd.Series([1, 2, 3, 2, 1])
        y_pred = pd.Series([1, 2, 3, 2, 1])

        # 完美预测应该有100%命中率
        hr_perfect = calculate_hit_rate(y_true, y_pred)
        assert hr_perfect == 1.0, f"完美预测的Hit Rate应该是1.0，实际: {hr_perfect}"

        # 完全相反的预测
        y_pred_opposite = pd.Series([1, 0, -1, 0, 1])
        hr_opposite = calculate_hit_rate(y_true, y_pred_opposite)
        assert hr_opposite == 0.0, f"相反预测的Hit Rate应该是0.0，实际: {hr_opposite}"

        print(f"\n完美预测Hit Rate: {hr_perfect:.2%}")
        print(f"相反预测Hit Rate: {hr_opposite:.2%}")

    def test_metrics_stability_across_runs(self, prepared_data):
        """测试多次运行的指标稳定性"""
        target_col = '规模以上工业增加值:当月同比'
        available_cols = [col for col in prepared_data.columns if col != target_col]
        selected_vars = [target_col] + available_cols[:5]
        test_data = prepared_data[selected_vars].dropna()

        # 分割数据
        split_idx = int(len(test_data) * 0.7)
        train_data = test_data.iloc[:split_idx]
        test_set = test_data.iloc[split_idx:]

        # 多次运行（相同种子）
        rmse_list = []
        for _ in range(3):
            np.random.seed(42)
            model = DFMModel(n_factors=1, max_lags=1, max_iter=10, tolerance=1e-6)
            result = model.fit(train_data)
            pred = result.factors.iloc[-len(test_set):, 0]
            true_series = test_set[target_col].iloc[:len(pred)]
            rmse = np.sqrt(mean_squared_error(true_series.values, pred.values[:len(true_series)]))
            rmse_list.append(rmse)

        # 验证所有RMSE相同
        for i in range(1, len(rmse_list)):
            assert np.abs(rmse_list[i] - rmse_list[0]) < 1e-10, \
                f"第{i}次运行RMSE不一致: {rmse_list[i]} vs {rmse_list[0]}"

        print(f"\nRMSE稳定性: {rmse_list}")
        print(f"最大差异: {max(np.abs(np.array(rmse_list) - rmse_list[0])):.2e}")
