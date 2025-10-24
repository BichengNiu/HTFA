# -*- coding: utf-8 -*-
"""
PrecomputeEngine单元测试

测试内容:
1. PrecomputedContext数据类
2. 标准化统计量预计算
3. 协方差矩阵预计算
4. 相关系数矩阵预计算
5. 上下文验证
6. 应用预计算统计量
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import is_dataclass

from dashboard.DFM.train_ref.utils.precompute import (
    PrecomputeEngine,
    PrecomputedContext
)


class TestPrecomputedContext:
    """PrecomputedContext数据类测试"""

    def test_precomputed_context_is_dataclass(self):
        """验证PrecomputedContext是dataclass"""
        assert is_dataclass(PrecomputedContext)

    def test_precomputed_context_creation_empty(self):
        """测试创建空上下文"""
        context = PrecomputedContext()

        assert context.standardization_stats == {}
        assert context.covariance_matrix is None
        assert context.correlation_matrix is None
        assert context.data_shape == (0, 0)
        assert context.variable_names == []

    def test_precomputed_context_creation_with_data(self):
        """测试创建包含数据的上下文"""
        stats = {'var1': (0.5, 1.0), 'var2': (1.0, 2.0)}
        cov_matrix = np.eye(2)
        corr_matrix = np.eye(2)

        context = PrecomputedContext(
            standardization_stats=stats,
            covariance_matrix=cov_matrix,
            correlation_matrix=corr_matrix,
            data_shape=(100, 2),
            variable_names=['var1', 'var2']
        )

        assert context.standardization_stats == stats
        assert np.array_equal(context.covariance_matrix, cov_matrix)
        assert np.array_equal(context.correlation_matrix, corr_matrix)
        assert context.data_shape == (100, 2)
        assert context.variable_names == ['var1', 'var2']


class TestPrecomputeEngineInit:
    """PrecomputeEngine初始化测试"""

    def test_init(self):
        """测试初始化"""
        engine = PrecomputeEngine()
        assert engine is not None
        assert hasattr(engine, 'logger')


class TestPrecomputeEngineStandardization:
    """PrecomputeEngine标准化统计量测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(100) + 5.0,
            'var2': np.random.randn(100) * 2.0 + 10.0,
            'var3': np.random.randn(100) * 0.5
        })
        return data

    def test_precompute_standardization_stats(self, sample_data):
        """测试预计算标准化统计量"""
        engine = PrecomputeEngine()
        stats = engine.precompute_standardization_stats(sample_data)

        assert len(stats) == 3
        assert 'var1' in stats
        assert 'var2' in stats
        assert 'var3' in stats

        for var in ['var1', 'var2', 'var3']:
            mean, std = stats[var]
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std > 0

    def test_precompute_standardization_stats_correctness(self, sample_data):
        """测试统计量计算正确性"""
        engine = PrecomputeEngine()
        stats = engine.precompute_standardization_stats(sample_data)

        for var in sample_data.columns:
            expected_mean = sample_data[var].mean()
            expected_std = sample_data[var].std()
            computed_mean, computed_std = stats[var]

            assert computed_mean == pytest.approx(expected_mean, abs=1e-6)
            assert computed_std == pytest.approx(expected_std, abs=1e-6)

    def test_precompute_standardization_stats_with_nan(self):
        """测试包含NaN的数据"""
        data = pd.DataFrame({
            'var1': [1.0, 2.0, np.nan, 4.0],
            'var2': [5.0, 6.0, 7.0, 8.0]
        })

        engine = PrecomputeEngine()
        stats = engine.precompute_standardization_stats(data)

        assert len(stats) == 2
        # var1的统计量应该排除NaN计算
        assert stats['var1'][1] > 0  # std > 0
        assert stats['var2'][1] > 0

    def test_precompute_standardization_stats_constant_series(self):
        """测试常数序列"""
        data = pd.DataFrame({
            'var1': [5.0] * 10,
            'var2': [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        })

        engine = PrecomputeEngine()
        stats = engine.precompute_standardization_stats(data)

        # 常数序列的std应该被设为1.0
        assert stats['var1'][1] == 1.0
        assert stats['var2'][1] > 0


class TestPrecomputeEngineCovariance:
    """PrecomputeEngine协方差矩阵测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })
        return data

    def test_precompute_covariance(self, sample_data):
        """测试预计算协方差矩阵"""
        engine = PrecomputeEngine()
        cov_matrix = engine.precompute_covariance(sample_data)

        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)  # 对称矩阵

    def test_precompute_covariance_correctness(self, sample_data):
        """测试协方差矩阵计算正确性"""
        engine = PrecomputeEngine()
        cov_matrix = engine.precompute_covariance(sample_data)

        expected_cov = sample_data.cov().values
        assert np.allclose(cov_matrix, expected_cov, atol=1e-10)

    def test_precompute_covariance_with_nan(self):
        """测试包含NaN的数据"""
        data = pd.DataFrame({
            'var1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'var2': [5.0, 6.0, 7.0, np.nan, 9.0]
        })

        engine = PrecomputeEngine()
        cov_matrix = engine.precompute_covariance(data)

        # 应该移除NaN后计算
        assert cov_matrix.shape == (2, 2)
        assert np.isfinite(cov_matrix).all()

    def test_precompute_covariance_insufficient_data(self):
        """测试数据不足的情况"""
        data = pd.DataFrame({
            'var1': [1.0],
            'var2': [2.0]
        })

        engine = PrecomputeEngine()
        cov_matrix = engine.precompute_covariance(data)

        # 应该返回零矩阵
        assert cov_matrix.shape == (2, 2)
        assert np.allclose(cov_matrix, 0.0)


class TestPrecomputeEngineCorrelation:
    """PrecomputeEngine相关系数矩阵测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })
        return data

    def test_precompute_correlation(self, sample_data):
        """测试预计算相关系数矩阵"""
        engine = PrecomputeEngine()
        corr_matrix = engine.precompute_correlation(sample_data)

        assert corr_matrix.shape == (3, 3)
        assert np.allclose(corr_matrix, corr_matrix.T)  # 对称矩阵
        assert np.allclose(np.diag(corr_matrix), 1.0)  # 对角线为1

    def test_precompute_correlation_correctness(self, sample_data):
        """测试相关系数矩阵计算正确性"""
        engine = PrecomputeEngine()
        corr_matrix = engine.precompute_correlation(sample_data)

        expected_corr = sample_data.corr().values
        assert np.allclose(corr_matrix, expected_corr, atol=1e-10)

    def test_precompute_correlation_range(self, sample_data):
        """测试相关系数范围"""
        engine = PrecomputeEngine()
        corr_matrix = engine.precompute_correlation(sample_data)

        # 非对角元素应该在[-1, 1]范围内
        off_diagonal = corr_matrix[~np.eye(3, dtype=bool)]
        assert (off_diagonal >= -1.0).all()
        assert (off_diagonal <= 1.0).all()


class TestPrecomputeEngineAll:
    """PrecomputeEngine完整预计算测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(100) + 5.0,
            'var2': np.random.randn(100) * 2.0,
        })
        return data

    def test_precompute_all(self, sample_data):
        """测试完整预计算"""
        engine = PrecomputeEngine()
        context = engine.precompute_all(sample_data)

        assert isinstance(context, PrecomputedContext)
        assert len(context.standardization_stats) == 2
        assert context.covariance_matrix is not None
        assert context.correlation_matrix is not None
        assert context.data_shape == sample_data.shape
        assert context.variable_names == list(sample_data.columns)

    def test_precompute_all_selective(self, sample_data):
        """测试选择性预计算"""
        engine = PrecomputeEngine()

        # 只计算协方差
        context1 = engine.precompute_all(
            sample_data,
            compute_covariance=True,
            compute_correlation=False
        )
        assert context1.covariance_matrix is not None
        assert context1.correlation_matrix is None

        # 只计算相关系数
        context2 = engine.precompute_all(
            sample_data,
            compute_covariance=False,
            compute_correlation=True
        )
        assert context2.covariance_matrix is None
        assert context2.correlation_matrix is not None


class TestPrecomputeEngineValidation:
    """PrecomputeEngine验证功能测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100)
        })
        return data

    def test_validate_context_valid(self, sample_data):
        """测试有效上下文验证"""
        engine = PrecomputeEngine()
        context = engine.precompute_all(sample_data)

        is_valid = engine.validate_context(context, sample_data)
        assert is_valid is True

    def test_validate_context_different_variables(self, sample_data):
        """测试变量不匹配"""
        engine = PrecomputeEngine()
        context = engine.precompute_all(sample_data)

        # 创建变量名不同的数据
        different_data = sample_data.rename(columns={'var1': 'var3'})
        is_valid = engine.validate_context(context, different_data)
        assert is_valid is False

    def test_validate_context_different_column_count(self, sample_data):
        """测试变量数不匹配"""
        engine = PrecomputeEngine()
        context = engine.precompute_all(sample_data)

        # 创建列数不同的数据
        subset_data = sample_data[['var1']]
        is_valid = engine.validate_context(context, subset_data)
        assert is_valid is False


class TestPrecomputeEngineApplyStandardization:
    """PrecomputeEngine应用标准化测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'var1': np.random.randn(50) + 5.0,
            'var2': np.random.randn(50) * 2.0 + 10.0
        })
        return data

    def test_apply_standardization(self, sample_data):
        """测试应用预计算统计量标准化"""
        engine = PrecomputeEngine()

        # 预计算统计量
        stats = engine.precompute_standardization_stats(sample_data)

        # 应用标准化
        standardized = engine.apply_standardization(sample_data, stats)

        assert standardized.shape == sample_data.shape
        # 标准化后的均值应接近0，标准差应接近1
        for col in standardized.columns:
            assert standardized[col].mean() == pytest.approx(0.0, abs=1e-10)
            assert standardized[col].std() == pytest.approx(1.0, abs=1e-10)

    def test_apply_standardization_to_new_data(self, sample_data):
        """测试将训练集统计量应用到新数据"""
        engine = PrecomputeEngine()

        # 使用训练集计算统计量
        train_data = sample_data.iloc[:30]
        stats = engine.precompute_standardization_stats(train_data)

        # 应用到测试集
        test_data = sample_data.iloc[30:]
        standardized_test = engine.apply_standardization(test_data, stats)

        assert standardized_test.shape == test_data.shape
        # 测试集标准化后均值和标准差不一定是0和1
        # 但应该使用训练集的统计量
        for col in test_data.columns:
            train_mean, train_std = stats[col]
            expected = (test_data[col] - train_mean) / train_std
            assert np.allclose(standardized_test[col], expected)

    def test_apply_standardization_missing_stats(self):
        """测试缺少统计量的情况"""
        data = pd.DataFrame({
            'var1': [1.0, 2.0, 3.0],
            'var2': [4.0, 5.0, 6.0],
            'var3': [7.0, 8.0, 9.0]
        })

        stats = {'var1': (2.0, 1.0), 'var2': (5.0, 1.0)}  # 缺少var3

        engine = PrecomputeEngine()
        standardized = engine.apply_standardization(data, stats)

        # var1和var2应该被标准化
        assert standardized.shape == data.shape
        expected_var1 = (data['var1'] - 2.0) / 1.0
        assert np.allclose(standardized['var1'].values, expected_var1.values)

        # var3保持原样（没有统计量）
        assert np.array_equal(standardized['var3'].values, data['var3'].values)
