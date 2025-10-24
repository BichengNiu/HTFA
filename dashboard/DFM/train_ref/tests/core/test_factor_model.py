# -*- coding: utf-8 -*-
'''
DFM模型单元测试

测试DFMModel类的完整训练流程
'''

import pytest
import numpy as np
import pandas as pd
from dashboard.DFM.train_ref.core.factor_model import (
    DFMModel,
    DFMResults,
    fit_dfm
)


class TestDFMModel:
    '''DFM模型测试'''

    @pytest.fixture
    def synthetic_data(self):
        '''生成合成DFM数据'''
        np.random.seed(42)
        n_time = 200
        n_factors = 2
        n_obs = 10

        # 真实参数
        true_A = np.array([[0.8, 0.1], [-0.1, 0.7]])
        true_Lambda = np.random.randn(n_obs, n_factors)

        # 生成因子
        factors = np.zeros((n_time, n_factors))
        factors[0] = np.random.randn(n_factors)
        for t in range(1, n_time):
            factors[t] = true_A @ factors[t-1] + np.random.randn(n_factors) * 0.3

        # 生成观测
        observations = factors @ true_Lambda.T + np.random.randn(n_time, n_obs) * 0.5

        data = pd.DataFrame(
            observations,
            index=pd.date_range('2020-01-01', periods=n_time, freq='D'),
            columns=[f'Var{i}' for i in range(n_obs)]
        )

        return {
            'data': data,
            'true_factors': factors,
            'true_A': true_A,
            'true_Lambda': true_Lambda,
            'n_factors': n_factors
        }

    def test_dfm_basic_fit(self, synthetic_data):
        '''测试DFM基本拟合功能'''
        model = DFMModel(
            n_factors=synthetic_data['n_factors'],
            max_lags=1,
            max_iter=10,
            tolerance=1e-4
        )

        results = model.fit(synthetic_data['data'])

        # 验证结果结构
        assert isinstance(results, DFMResults)
        assert results.factors.shape[1] == synthetic_data['n_factors']
        assert results.loadings.shape[0] == synthetic_data['data'].shape[1]
        assert results.transition_matrix.shape == (synthetic_data['n_factors'], synthetic_data['n_factors'])

        # 验证对数似然
        assert not np.isnan(results.loglikelihood)
        assert results.n_iter > 0

    def test_dfm_single_factor(self):
        '''测试单因子DFM（k=1）'''
        np.random.seed(123)
        n_time = 100
        n_obs = 5

        # 生成单因子数据
        factor = np.cumsum(np.random.randn(n_time)) * 0.1
        loadings = np.random.randn(n_obs)
        data = pd.DataFrame(
            factor[:, np.newaxis] @ loadings[np.newaxis, :] + np.random.randn(n_time, n_obs) * 0.2,
            columns=[f'Var{i}' for i in range(n_obs)]
        )

        model = DFMModel(n_factors=1, max_lags=1, max_iter=10)
        results = model.fit(data)

        # 验证单因子结果
        assert results.factors.shape[1] == 1
        assert results.transition_matrix.shape == (1, 1)

    def test_dfm_multiple_factors(self):
        '''测试多因子DFM（k=3）'''
        np.random.seed(456)
        n_time = 150
        n_factors = 3
        n_obs = 12

        # 生成多因子数据
        factors = np.random.randn(n_time, n_factors)
        for t in range(1, n_time):
            factors[t] += 0.7 * factors[t-1]

        loadings = np.random.randn(n_obs, n_factors)
        observations = factors @ loadings.T + np.random.randn(n_time, n_obs) * 0.3

        data = pd.DataFrame(observations, columns=[f'Var{i}' for i in range(n_obs)])

        model = DFMModel(n_factors=n_factors, max_lags=1, max_iter=10)
        results = model.fit(data)

        # 验证多因子结果
        assert results.factors.shape[1] == n_factors
        assert results.loadings.shape == (n_obs, n_factors)

    def test_dfm_reproducibility(self, synthetic_data):
        '''测试可重现性'''
        model1 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        results1 = model1.fit(synthetic_data['data'])

        model2 = DFMModel(n_factors=2, max_lags=1, max_iter=10, tolerance=1e-6)
        results2 = model2.fit(synthetic_data['data'])

        # 验证两次训练结果一致
        np.testing.assert_allclose(
            results1.factors.values,
            results2.factors.values,
            rtol=1e-10,
            atol=1e-10
        )

    def test_fit_dfm_function(self, synthetic_data):
        '''测试fit_dfm函数接口'''
        results = fit_dfm(
            data=synthetic_data['data'],
            n_factors=2,
            max_lags=1,
            max_iter=10
        )

        assert isinstance(results, DFMResults)
        assert results.factors.shape[1] == 2

    def test_dfm_covariance_matrices(self, synthetic_data):
        '''测试协方差矩阵的正定性'''
        model = DFMModel(n_factors=2, max_lags=1, max_iter=10)
        results = model.fit(synthetic_data['data'])

        # 验证Q矩阵正定
        Q_eigvals = np.linalg.eigvalsh(results.process_noise_cov)
        assert np.all(Q_eigvals > -1e-10)

        # 验证R矩阵正定
        R_eigvals = np.linalg.eigvalsh(results.measurement_noise_cov)
        assert np.all(R_eigvals > -1e-10)
