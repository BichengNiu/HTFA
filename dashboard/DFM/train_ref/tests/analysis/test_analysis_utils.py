# -*- coding: utf-8 -*-
"""
分析工具函数单元测试

测试analysis_utils.py中的各种指标计算函数
"""

import pytest
import numpy as np
import pandas as pd
from dashboard.DFM.train_ref.analysis.analysis_utils import (
    calculate_rmse,
    calculate_hit_rate,
    calculate_correlation,
    calculate_metrics_with_lagged_target,
    calculate_factor_contributions,
    calculate_individual_variable_r2,
    calculate_industry_r2,
    calculate_pca_variance,
    calculate_monthly_friday_metrics
)


class TestCalculateRMSE:
    """测试RMSE计算"""

    def test_basic_rmse(self):
        """测试基础RMSE计算"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        rmse = calculate_rmse(actual, predicted)

        # 手动计算验证
        expected_rmse = np.sqrt(np.mean([(1.0-1.1)**2, (2.0-2.2)**2,
                                         (3.0-2.9)**2, (4.0-4.1)**2,
                                         (5.0-4.8)**2]))
        np.testing.assert_almost_equal(rmse, expected_rmse, decimal=10)

    def test_rmse_with_nan(self):
        """测试含NaN的RMSE计算"""
        actual = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        predicted = np.array([1.1, np.nan, 3.0, 4.1, 4.8])

        rmse = calculate_rmse(actual, predicted)

        # 只有[1.0, 4.0, 5.0]和[1.1, 4.1, 4.8]有效
        expected_rmse = np.sqrt(np.mean([(1.0-1.1)**2, (4.0-4.1)**2, (5.0-4.8)**2]))
        np.testing.assert_almost_equal(rmse, expected_rmse, decimal=10)

    def test_rmse_all_nan(self):
        """测试全NaN情况"""
        actual = np.array([np.nan, np.nan, np.nan])
        predicted = np.array([1.0, 2.0, 3.0])

        rmse = calculate_rmse(actual, predicted)

        assert np.isnan(rmse)

    def test_rmse_perfect_fit(self):
        """测试完美拟合"""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = actual.copy()

        rmse = calculate_rmse(actual, predicted)

        assert rmse == 0.0


class TestCalculateHitRate:
    """测试命中率计算"""

    def test_basic_hit_rate(self):
        """测试基础命中率"""
        actual = np.array([1.0, 2.0, 3.0, 2.5, 3.5, 4.0])
        predicted = np.array([1.0, 1.8, 3.2, 2.3, 3.8, 4.2])

        hit_rate = calculate_hit_rate(actual, predicted, lag=1)

        # 变化方向: actual [+1, +1, -0.5, +1, +0.5]
        #           predicted [+0.8, +1.4, -0.9, +1.5, +0.4]
        # 全部同号，命中率100%
        assert hit_rate == 100.0

    def test_hit_rate_with_lag(self):
        """测试不同滞后期的命中率"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.1, 2.9, 4.2, 4.8])

        hit_rate = calculate_hit_rate(actual, predicted, lag=2)

        # lag=2: 比较[3,4,5]和[1,2,3] -> diff=[2,2,2]
        # predicted: [2.9,4.2,4.8]和[1,2.1,2.9] -> diff=[1.9,2.1,1.9]
        # 全部正值，命中率100%
        assert hit_rate == 100.0

    def test_hit_rate_with_nan(self):
        """测试含NaN的命中率"""
        actual = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        predicted = np.array([1.0, 2.1, 3.0, 4.2, 4.8])

        hit_rate = calculate_hit_rate(actual, predicted)

        # 有NaN时会跳过
        assert not np.isnan(hit_rate)

    def test_hit_rate_insufficient_data(self):
        """测试数据不足"""
        actual = np.array([1.0])
        predicted = np.array([1.1])

        hit_rate = calculate_hit_rate(actual, predicted, lag=1)

        assert np.isnan(hit_rate)


class TestCalculateCorrelation:
    """测试相关系数计算"""

    def test_perfect_correlation(self):
        """测试完美正相关"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual * 2  # 完美线性关系

        corr = calculate_correlation(actual, predicted)

        np.testing.assert_almost_equal(corr, 1.0, decimal=10)

    def test_negative_correlation(self):
        """测试负相关"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = -actual

        corr = calculate_correlation(actual, predicted)

        np.testing.assert_almost_equal(corr, -1.0, decimal=10)

    def test_correlation_with_nan(self):
        """测试含NaN的相关系数"""
        actual = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 3.3, np.nan, 5.5])

        corr = calculate_correlation(actual, predicted)

        # 只有[1.0, 2.0]和[1.1, 2.2]有效
        assert not np.isnan(corr)

    def test_correlation_insufficient_samples(self):
        """测试样本不足"""
        actual = np.array([1.0])
        predicted = np.array([1.1])

        corr = calculate_correlation(actual, predicted)

        assert np.isnan(corr)


class TestCalculateFactorContributions:
    """测试因子贡献度计算"""

    def test_basic_contributions(self):
        """测试基础贡献度计算"""
        n_time = 10
        n_factors = 2

        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            columns=['Factor1', 'Factor2']
        )
        loadings = np.random.randn(5, n_factors)
        target_loading = np.array([1.5, -0.8])

        contributions = calculate_factor_contributions(factors, loadings, target_loading)

        # 验证结构
        assert isinstance(contributions, pd.DataFrame)
        assert contributions.shape[0] == n_time
        assert 'Factor1_contribution' in contributions.columns
        assert 'Factor2_contribution' in contributions.columns
        assert 'Total_forecast' in contributions.columns

        # 验证总和
        manual_total = (factors.iloc[:, 0].values * target_loading[0] +
                       factors.iloc[:, 1].values * target_loading[1])
        np.testing.assert_allclose(
            contributions['Total_forecast'].values,
            manual_total,
            rtol=1e-10
        )


class TestCalculateIndividualVariableR2:
    """测试个体变量R²计算"""

    def test_basic_r2(self):
        """测试基础R²计算"""
        np.random.seed(42)
        n_time = 100
        n_factors = 2
        n_obs = 3

        factors = pd.DataFrame(np.random.randn(n_time, n_factors))
        loadings = np.random.randn(n_obs, n_factors)
        observables = pd.DataFrame(
            factors.values @ loadings.T + np.random.randn(n_time, n_obs) * 0.1,
            columns=['Var1', 'Var2', 'Var3']
        )

        r2 = calculate_individual_variable_r2(observables, factors, loadings)

        # 验证结构
        assert isinstance(r2, pd.Series)
        assert len(r2) == n_obs
        assert r2.index.tolist() == ['Var1', 'Var2', 'Var3']

        # R²应该都很高（因为噪声小）
        assert np.all(r2 > 0.8)

    def test_r2_with_missing_data(self):
        """测试含缺失值的R²"""
        n_time = 50
        n_factors = 2
        n_obs = 2

        factors = pd.DataFrame(np.random.randn(n_time, n_factors))
        loadings = np.random.randn(n_obs, n_factors)
        observables = pd.DataFrame(
            factors.values @ loadings.T,
            columns=['Var1', 'Var2']
        )

        # 添加缺失值
        observables.iloc[10:20, 0] = np.nan

        r2 = calculate_individual_variable_r2(observables, factors, loadings)

        assert len(r2) == n_obs
        # Var1有缺失但仍能计算
        assert not np.isnan(r2['Var1'])


class TestCalculateIndustryR2:
    """测试行业R²计算"""

    def test_basic_industry_r2(self):
        """测试基础行业R²"""
        np.random.seed(42)
        n_time = 100
        n_factors = 2

        factors = pd.DataFrame(np.random.randn(n_time, n_factors))
        loadings = np.random.randn(4, n_factors)
        observables = pd.DataFrame(
            factors.values @ loadings.T + np.random.randn(n_time, 4) * 0.1,
            columns=['钢铁1', '钢铁2', '有色1', '有色2']
        )

        variable_industry_map = {
            '钢铁1': '钢铁',
            '钢铁2': '钢铁',
            '有色1': '有色金属',
            '有色2': '有色金属'
        }

        industry_r2 = calculate_industry_r2(
            observables, factors, loadings, variable_industry_map
        )

        # 验证结构
        assert isinstance(industry_r2, pd.Series)
        assert '钢铁' in industry_r2.index
        assert '有色金属' in industry_r2.index

        # R²应该合理
        assert np.all(industry_r2 > 0.5)

    def test_industry_r2_no_map(self):
        """测试无映射情况"""
        factors = pd.DataFrame(np.random.randn(50, 2))
        loadings = np.random.randn(3, 2)
        observables = pd.DataFrame(np.random.randn(50, 3))

        industry_r2 = calculate_industry_r2(observables, factors, loadings, None)

        # 无映射时返回空Series
        assert len(industry_r2) == 0


class TestCalculatePCAVariance:
    """测试PCA方差计算"""

    def test_basic_pca(self):
        """测试基础PCA计算"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # 生成相关数据
        data = pd.DataFrame(np.random.randn(n_samples, n_features))
        data.iloc[:, 1] = data.iloc[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2

        explained_var, explained_ratio, cumulative_ratio = calculate_pca_variance(
            data, n_components=3
        )

        # 验证形状
        assert len(explained_var) == 3
        assert len(explained_ratio) == 3
        assert len(cumulative_ratio) == 3

        # 验证性质
        assert np.all(explained_var > 0)
        assert np.all(explained_ratio > 0)
        assert np.all(explained_ratio < 1)
        assert cumulative_ratio[0] == explained_ratio[0]
        assert cumulative_ratio[-1] <= 1.0

        # 累计方差应该递增
        assert np.all(np.diff(cumulative_ratio) >= 0)

    def test_pca_default_components(self):
        """测试默认主成分数"""
        data = pd.DataFrame(np.random.randn(50, 10))

        explained_var, explained_ratio, cumulative_ratio = calculate_pca_variance(data)

        # 默认使用min(n_samples, n_features)
        expected_components = min(data.shape)
        assert len(explained_var) == expected_components


class TestCalculateMonthlyFridayMetrics:
    """测试月度周五指标计算"""

    def test_basic_monthly_metrics(self):
        """测试基础月度指标计算"""
        # 生成周五数据
        dates = pd.date_range('2020-01-03', periods=52, freq='W-FRI')
        forecast = pd.Series(np.random.randn(52), index=dates)
        actual = forecast + np.random.randn(52) * 0.1

        monthly = calculate_monthly_friday_metrics(forecast, actual, freq='W-FRI')

        # 验证结构
        assert isinstance(monthly, pd.DataFrame)
        assert 'forecast' in monthly.columns
        assert 'actual' in monthly.columns
        assert 'error' in monthly.columns
        assert 'abs_error' in monthly.columns
        assert 'squared_error' in monthly.columns

        # 验证误差计算
        np.testing.assert_allclose(
            monthly['error'].values,
            monthly['actual'].values - monthly['forecast'].values,
            rtol=1e-10
        )

    def test_non_friday_warning(self):
        """测试非周五数据警告"""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        forecast = pd.Series(np.random.randn(30), index=dates)
        actual = pd.Series(np.random.randn(30), index=dates)

        monthly = calculate_monthly_friday_metrics(forecast, actual, freq='D')

        # 非周五数据返回空DataFrame
        assert monthly.empty


class TestCalculateMetricsWithLaggedTarget:
    """测试带滞后目标的指标计算"""

    def test_basic_metrics_with_lag(self):
        """测试基础带滞后指标"""
        np.random.seed(42)
        n_time = 200
        n_factors = 2

        dates = pd.date_range('2020-01-01', periods=n_time, freq='D')
        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            index=dates
        )
        target_loading = np.array([1.5, -0.8])
        target = pd.Series(
            factors.values @ target_loading + np.random.randn(n_time) * 0.1,
            index=dates
        )

        train_end = '2020-05-31'
        validation_start = '2020-06-01'
        validation_end = '2020-06-30'

        metrics = calculate_metrics_with_lagged_target(
            factors, target, target_loading,
            train_end, validation_start, validation_end
        )

        # 验证结构
        assert 'is_rmse' in metrics
        assert 'oos_rmse' in metrics
        assert 'is_hit_rate' in metrics
        assert 'oos_hit_rate' in metrics
        assert 'is_correlation' in metrics
        assert 'oos_correlation' in metrics

        # 验证数值合理性
        assert metrics['is_rmse'] > 0
        assert metrics['oos_rmse'] > 0
        assert 0 <= metrics['is_hit_rate'] <= 100
        assert 0 <= metrics['oos_hit_rate'] <= 100
        assert -1 <= metrics['is_correlation'] <= 1
        assert -1 <= metrics['oos_correlation'] <= 1
