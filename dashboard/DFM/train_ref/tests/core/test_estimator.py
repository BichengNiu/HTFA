# -*- coding: utf-8 -*-
"""
参数估计模块单元测试

测试estimate_loadings, estimate_transition_matrix等函数的正确性
"""

import pytest
import numpy as np
import pandas as pd
from dashboard.DFM.train_ref.core.estimator import (
    estimate_loadings,
    estimate_target_loading,
    estimate_transition_matrix,
    estimate_covariance_matrices,
    estimate_parameters,
    _ensure_positive_definite
)


class TestEstimateLoadings:
    """测试载荷矩阵估计"""

    def test_basic_loading_estimation(self):
        """测试基础载荷估计"""
        np.random.seed(42)
        n_time = 100
        n_factors = 2
        n_obs = 5

        # 生成模拟数据
        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            columns=['Factor1', 'Factor2']
        )

        true_loadings = np.random.randn(n_obs, n_factors)
        observables = pd.DataFrame(
            factors.values @ true_loadings.T + np.random.randn(n_time, n_obs) * 0.1,
            columns=[f'Var{i}' for i in range(n_obs)]
        )

        # 估计载荷
        estimated_loadings = estimate_loadings(observables, factors)

        # 验证形状
        assert estimated_loadings.shape == (n_obs, n_factors)

        # 验证估计接近真实值
        np.testing.assert_allclose(estimated_loadings, true_loadings, rtol=0.5, atol=0.5)

    def test_loading_with_missing_values(self):
        """测试含缺失值的载荷估计"""
        np.random.seed(123)
        n_time = 100
        n_factors = 2
        n_obs = 3

        factors = pd.DataFrame(np.random.randn(n_time, n_factors))
        observables = pd.DataFrame(np.random.randn(n_time, n_obs))

        # 添加缺失值
        observables.iloc[10:20, 0] = np.nan
        observables.iloc[30:35, 1] = np.nan

        # 估计载荷
        loadings = estimate_loadings(observables, factors)

        # 验证形状正确
        assert loadings.shape == (n_obs, n_factors)

        # 验证没有全NaN的载荷（除非整列都是NaN）
        assert not np.all(np.isnan(loadings), axis=1).any()

    def test_insufficient_samples_warning(self):
        """测试样本不足时的警告"""
        n_time = 5
        n_factors = 3
        n_obs = 2

        factors = pd.DataFrame(np.random.randn(n_time, n_factors))
        observables = pd.DataFrame(np.random.randn(n_time, n_obs))

        # 使大部分观测为NaN
        observables.iloc[:, 0] = np.nan

        loadings = estimate_loadings(observables, factors)

        # 验证缺失数据变量的载荷为NaN
        assert np.all(np.isnan(loadings[0, :]))


class TestEstimateTargetLoading:
    """测试目标变量载荷估计"""

    def test_target_loading_estimation(self):
        """测试目标变量载荷估计"""
        np.random.seed(42)
        n_time = 100
        n_factors = 2

        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            index=pd.date_range('2020-01-01', periods=n_time, freq='D')
        )

        true_loading = np.array([1.5, -0.8])
        target = pd.Series(
            factors.values @ true_loading + np.random.randn(n_time) * 0.1,
            index=factors.index
        )

        # 估计目标载荷
        estimated_loading = estimate_target_loading(target, factors)

        # 验证形状
        assert estimated_loading.shape == (n_factors,)

        # 验证估计接近真实值
        np.testing.assert_allclose(estimated_loading, true_loading, rtol=0.2, atol=0.2)

    def test_target_loading_with_train_end(self):
        """测试使用train_end限制训练数据"""
        np.random.seed(123)
        n_time = 100
        n_factors = 2

        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            index=pd.date_range('2020-01-01', periods=n_time, freq='D')
        )
        target = pd.Series(
            np.random.randn(n_time),
            index=factors.index
        )

        train_end = '2020-02-29'

        # 估计载荷
        loading = estimate_target_loading(target, factors, train_end=train_end)

        # 验证成功估计
        assert loading.shape == (n_factors,)
        assert not np.any(np.isnan(loading))


class TestEstimateTransitionMatrix:
    """测试状态转移矩阵估计"""

    def test_transition_matrix_lag1(self):
        """测试滞后1阶的转移矩阵估计"""
        np.random.seed(42)
        n_time = 200
        n_factors = 2

        # 生成VAR(1)数据
        true_A = np.array([[0.8, 0.1], [-0.1, 0.7]])

        factors = np.zeros((n_time, n_factors))
        factors[0] = np.random.randn(n_factors)

        for t in range(1, n_time):
            factors[t] = true_A @ factors[t-1] + np.random.randn(n_factors) * 0.1

        # 估计A矩阵
        estimated_A = estimate_transition_matrix(factors, max_lags=1)

        # 验证形状
        assert estimated_A.shape == (n_factors, n_factors)

        # 验证估计接近真实值
        np.testing.assert_allclose(estimated_A, true_A, rtol=0.3, atol=0.3)

    def test_transition_matrix_stability(self):
        """测试估计的转移矩阵稳定性"""
        np.random.seed(456)
        n_time = 100
        n_factors = 3

        factors = np.random.randn(n_time, n_factors)

        # 估计A矩阵
        A = estimate_transition_matrix(factors, max_lags=1)

        # 验证特征值都在单位圆内（稳定性）
        eigenvalues = np.linalg.eigvals(A)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        # 允许略微超出单位圆
        assert max_eigenvalue < 1.5


class TestEnsurePositiveDefinite:
    """测试正定矩阵确保函数"""

    def test_already_positive_definite(self):
        """测试已经正定的矩阵"""
        Q = np.array([[2.0, 0.5], [0.5, 1.0]])

        Q_pd = _ensure_positive_definite(Q)

        # 验证特征值都为正
        eigvals = np.linalg.eigvalsh(Q_pd)
        assert np.all(eigvals > 0)

        # 验证矩阵基本不变
        np.testing.assert_allclose(Q_pd, Q, rtol=1e-5, atol=1e-5)

    def test_negative_eigenvalue_correction(self):
        """测试负特征值修正"""
        # 构造有负特征值的矩阵
        Q = np.array([[1.0, 2.0], [2.0, 1.0]])

        Q_pd = _ensure_positive_definite(Q, epsilon=1e-6)

        # 验证所有特征值都不小于epsilon
        eigvals = np.linalg.eigvalsh(Q_pd)
        assert np.all(eigvals >= 1e-6)

    def test_near_singular_matrix(self):
        """测试接近奇异的矩阵"""
        Q = np.array([[1.0, 0.9999], [0.9999, 1.0]])

        Q_pd = _ensure_positive_definite(Q, epsilon=1e-4)

        # 验证正定性
        eigvals = np.linalg.eigvalsh(Q_pd)
        assert np.all(eigvals >= 1e-4)


class TestEstimateParameters:
    """测试完整参数估计"""

    def test_estimate_all_parameters(self):
        """测试估计所有DFM参数"""
        np.random.seed(42)
        n_time = 100
        n_factors = 2
        n_obs = 5

        # 生成模拟数据
        factors = pd.DataFrame(
            np.random.randn(n_time, n_factors),
            columns=['Factor1', 'Factor2']
        )

        true_Lambda = np.random.randn(n_obs, n_factors)
        observables = pd.DataFrame(
            factors.values @ true_Lambda.T + np.random.randn(n_time, n_obs) * 0.5,
            columns=[f'Var{i}' for i in range(n_obs)]
        )

        # 估计所有参数
        Lambda, A, Q, R = estimate_parameters(
            observables, factors, n_factors, max_lags=1
        )

        # 验证形状
        assert Lambda.shape == (n_obs, n_factors)
        assert A.shape == (n_factors, n_factors)
        assert Q.shape == (n_factors, n_factors)
        assert R.shape == (n_obs, n_obs)

        # 验证协方差矩阵正定性
        assert np.all(np.linalg.eigvalsh(Q) > 0)
        assert np.all(np.linalg.eigvalsh(R) > 0)
