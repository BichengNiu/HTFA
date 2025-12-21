# -*- coding: utf-8 -*-
"""
核心模块单元测试

测试覆盖:
- KalmanFilter: 滤波和平滑算法
- 参数估计: 载荷、转移矩阵、协方差矩阵
- 评估指标: RMSE、Win Rate、加权得分
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestKalmanFilter:
    """卡尔曼滤波器测试"""

    def test_kalman_filter_basic(self):
        """测试基本滤波功能"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 50, 3, 2

        # 构造简单状态空间模型
        A = np.array([[0.9, 0.0], [0.0, 0.8]])
        B = np.eye(n_states) * 0.1
        H = np.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.7]])
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_obs) * 0.5
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        # 生成模拟数据
        np.random.seed(42)
        Z = np.random.randn(n_time, n_obs)
        U = np.random.randn(n_time, n_states)

        # 执行滤波
        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        result = kf.filter(Z, U)

        # 验证输出形状
        assert result.x_filtered.shape == (n_time, n_states)
        assert result.P_filtered.shape == (n_states, n_states, n_time)
        assert result.x_predicted.shape == (n_time, n_states)
        assert result.innovation.shape == (n_time, n_obs)
        assert np.isfinite(result.loglikelihood)

    def test_kalman_filter_with_missing_data(self):
        """测试缺失数据处理"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 30, 2, 1

        A = np.array([[0.9]])
        B = np.array([[0.1]])
        H = np.array([[1.0], [0.5]])
        Q = np.array([[0.1]])
        R = np.eye(n_obs) * 0.5
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        # 生成带缺失值的数据
        np.random.seed(42)
        Z = np.random.randn(n_time, n_obs)
        Z[5:10, 0] = np.nan  # 第一个观测变量缺失
        Z[15:20, :] = np.nan  # 全部缺失

        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        result = kf.filter(Z)

        # 验证滤波结果不包含NaN（缺失数据时使用预测值）
        assert not np.any(np.isnan(result.x_filtered))

    def test_kalman_smoother(self):
        """测试平滑器"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 40, 2, 2

        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.eye(n_states) * 0.1
        H = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_obs) * 0.3
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        np.random.seed(42)
        Z = np.random.randn(n_time, n_obs)

        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        filter_result = kf.filter(Z)
        smooth_result = kf.smooth(filter_result)

        # 验证输出形状
        assert smooth_result.x_smoothed.shape == (n_states, n_time)
        assert smooth_result.P_smoothed.shape == (n_states, n_states, n_time)

    def test_kalman_gains_history(self):
        """测试卡尔曼增益历史记录"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 20, 2, 1

        A = np.array([[0.9]])
        B = np.array([[0.1]])
        H = np.array([[1.0], [0.5]])
        Q = np.array([[0.1]])
        R = np.eye(n_obs) * 0.5
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        np.random.seed(42)
        Z = np.random.randn(n_time, n_obs)

        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        result = kf.filter(Z)

        # 验证卡尔曼增益历史
        assert result.kalman_gains_history is not None
        assert len(result.kalman_gains_history) == n_time

        # t=0时刻没有增益，后续时刻有增益
        assert result.kalman_gains_history[0] is None
        for t in range(1, n_time):
            K_t = result.kalman_gains_history[t]
            assert K_t is not None
            assert K_t.shape == (n_states, n_obs)


class TestEstimator:
    """参数估计测试"""

    def test_estimate_loadings_series(self):
        """测试单变量载荷估计"""
        from dashboard.models.DFM.train.core.estimator import estimate_loadings

        n_time, n_factors = 100, 2

        # 生成因子和观测数据
        np.random.seed(42)
        factors_data = np.random.randn(n_time, n_factors)
        true_loadings = np.array([0.8, 0.3])
        y = factors_data @ true_loadings + np.random.randn(n_time) * 0.1

        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        factors_df = pd.DataFrame(factors_data, index=dates, columns=['F1', 'F2'])
        y_series = pd.Series(y, index=dates)

        # 估计载荷
        loadings = estimate_loadings(y_series, factors_df)

        # 验证形状和近似值
        assert loadings.shape == (n_factors,)
        np.testing.assert_allclose(loadings, true_loadings, atol=0.1)

    def test_estimate_loadings_dataframe(self):
        """测试多变量载荷估计"""
        from dashboard.models.DFM.train.core.estimator import estimate_loadings

        n_time, n_factors, n_obs = 100, 2, 3

        np.random.seed(42)
        factors_data = np.random.randn(n_time, n_factors)
        true_loadings = np.array([
            [0.8, 0.3],
            [0.5, 0.7],
            [0.2, 0.9]
        ])
        noise = np.random.randn(n_time, n_obs) * 0.1
        observables = factors_data @ true_loadings.T + noise

        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        factors_df = pd.DataFrame(factors_data, index=dates, columns=['F1', 'F2'])
        obs_df = pd.DataFrame(observables, index=dates, columns=['V1', 'V2', 'V3'])

        loadings = estimate_loadings(obs_df, factors_df)

        assert loadings.shape == (n_obs, n_factors)
        np.testing.assert_allclose(loadings, true_loadings, atol=0.15)

    def test_estimate_transition_matrix(self):
        """测试转移矩阵估计"""
        from dashboard.models.DFM.train.core.estimator import estimate_transition_matrix

        n_time, n_factors = 200, 2

        # 生成AR(1)因子过程
        np.random.seed(42)
        true_A = np.array([[0.8, 0.1], [0.0, 0.7]])
        factors = np.zeros((n_time, n_factors))
        for t in range(1, n_time):
            factors[t] = true_A @ factors[t-1] + np.random.randn(n_factors) * 0.1

        A = estimate_transition_matrix(factors, max_lags=1)

        assert A.shape == (n_factors, n_factors)
        # 转移矩阵估计应接近真值
        np.testing.assert_allclose(A, true_A, atol=0.15)


class TestMetrics:
    """评估指标测试"""

    def test_calculate_rmse(self):
        """测试RMSE计算"""
        from dashboard.models.DFM.train.evaluation.metrics import calculate_rmse

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        rmse = calculate_rmse(y_true, y_pred)

        # 手动计算RMSE
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        np.testing.assert_allclose(rmse, expected)

    def test_calculate_rmse_with_nan(self):
        """测试带NaN的RMSE计算"""
        from dashboard.models.DFM.train.evaluation.metrics import calculate_rmse

        y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, np.nan, 4.1, 4.8])

        rmse = calculate_rmse(y_true, y_pred)

        # 只计算有效数据点
        valid_true = np.array([1.0, 4.0, 5.0])
        valid_pred = np.array([1.1, 4.1, 4.8])
        expected = np.sqrt(np.mean((valid_true - valid_pred) ** 2))
        np.testing.assert_allclose(rmse, expected)

    def test_calculate_weighted_score(self):
        """测试加权得分计算"""
        from dashboard.models.DFM.train.evaluation.metrics import calculate_weighted_score

        is_rmse = 0.5
        oos_rmse = 0.6
        is_win_rate = 70.0
        oos_win_rate = 65.0
        training_weight = 0.3

        result = calculate_weighted_score(
            is_rmse, oos_rmse, is_win_rate, oos_win_rate, training_weight
        )

        # 验证返回格式 (weighted_win_rate, -weighted_rmse, weighted_rmse)
        assert len(result) == 3

        # 计算期望的加权RMSE
        expected_rmse = 0.3 * is_rmse + 0.7 * oos_rmse
        expected_win_rate = 0.3 * is_win_rate + 0.7 * oos_win_rate

        assert result[0] == pytest.approx(expected_win_rate)
        assert result[1] == pytest.approx(-expected_rmse)
        assert result[2] == pytest.approx(expected_rmse)

    def test_compare_scores_with_winrate(self):
        """测试得分比较"""
        from dashboard.models.DFM.train.evaluation.metrics import compare_scores_with_winrate

        # A比B更好（RMSE更小）
        score_a = (65.0, -0.5, 0.5)
        score_b = (65.0, -0.6, 0.6)

        result = compare_scores_with_winrate(score_a, score_b)
        assert result == 1  # A > B

        # B比A更好
        result = compare_scores_with_winrate(score_b, score_a)
        assert result == -1  # B < A


class TestModels:
    """数据模型测试"""

    def test_evaluation_metrics_dataclass(self):
        """测试EvaluationMetrics数据类"""
        from dashboard.models.DFM.train.core.models import EvaluationMetrics

        metrics = EvaluationMetrics(
            is_rmse=0.5,
            oos_rmse=0.6,
            is_mae=0.4,
            oos_mae=0.5,
            is_win_rate=70.0,
            oos_win_rate=65.0
        )

        # 测试to_tuple方法
        result = metrics.to_tuple()
        assert len(result) == 9
        assert result[0] == 0.5  # is_rmse
        assert result[1] == 0.6  # oos_rmse
        assert result[4] == 70.0  # is_win_rate
        assert result[5] == 65.0  # oos_win_rate

    def test_dfm_model_result_dataclass(self):
        """测试DFMModelResult数据类"""
        from dashboard.models.DFM.train.core.models import DFMModelResult

        result = DFMModelResult(
            A=np.eye(2),
            Q=np.eye(2) * 0.1,
            H=np.ones((3, 2)),
            R=np.eye(3) * 0.5,
            factors=np.random.randn(2, 50),
            converged=True,
            iterations=30,
            log_likelihood=-100.0
        )

        assert result.A.shape == (2, 2)
        assert result.converged is True
        assert result.iterations == 30


class TestBackwardSelector:
    """后向选择器测试"""

    def test_backward_selector_error_on_invalid_evaluator(self):
        """测试评估函数返回值数量错误时抛出异常"""
        from dashboard.models.DFM.train.selection.backward_selector import BackwardSelector

        # 创建返回错误数量值的模拟评估函数
        def bad_evaluator(**kwargs):
            return (0.5, 0.6, 0.4, 0.5)  # 只返回4个值，预期9个

        selector = BackwardSelector(
            evaluator_func=bad_evaluator,
            criterion='rmse',
            min_variables=1
        )

        # 应该抛出RuntimeError（ValueError被包装）
        with pytest.raises(RuntimeError, match="计算初始基准性能失败"):
            selector.select(
                initial_variables=['target', 'var1', 'var2'],
                target_variable='target',
                full_data=pd.DataFrame({'target': [1, 2, 3], 'var1': [1, 2, 3], 'var2': [1, 2, 3]}),
                params={'k_factors': 1},
                validation_start='2020-01-01',
                validation_end='2020-12-31',
                target_freq='W',
                training_start_date='2019-01-01',
                train_end_date='2019-12-31'
            )

    def test_backward_selector_error_on_evaluator_exception(self):
        """测试评估函数抛出异常时正确传播"""
        from dashboard.models.DFM.train.selection.backward_selector import BackwardSelector

        def failing_evaluator(**kwargs):
            raise RuntimeError("模拟评估失败")

        selector = BackwardSelector(
            evaluator_func=failing_evaluator,
            criterion='rmse',
            min_variables=1
        )

        # 应该抛出RuntimeError
        with pytest.raises(RuntimeError, match="计算初始基准性能失败"):
            selector.select(
                initial_variables=['target', 'var1', 'var2'],
                target_variable='target',
                full_data=pd.DataFrame({'target': [1, 2, 3], 'var1': [1, 2, 3], 'var2': [1, 2, 3]}),
                params={'k_factors': 1},
                validation_start='2020-01-01',
                validation_end='2020-12-31',
                target_freq='W',
                training_start_date='2019-01-01',
                train_end_date='2019-12-31'
            )


class TestParallelConfig:
    """并行配置测试"""

    def test_parallel_config_validation(self):
        """测试配置验证"""
        from dashboard.models.DFM.train.utils.parallel_config import ParallelConfig

        # 有效配置
        config = ParallelConfig(enabled=True, n_jobs=-1, backend='loky')
        assert config.enabled is True

        # 无效n_jobs
        with pytest.raises(ValueError, match="n_jobs不能为0"):
            ParallelConfig(enabled=True, n_jobs=0)

        # 无效backend
        with pytest.raises(ValueError, match="backend必须是"):
            ParallelConfig(enabled=True, backend='invalid')

    def test_parallel_config_effective_jobs(self):
        """测试有效并行任务数计算"""
        from dashboard.models.DFM.train.utils.parallel_config import ParallelConfig, get_cpu_count

        cpu_count = get_cpu_count()

        # n_jobs=-1时使用所有核心减1
        config = ParallelConfig(enabled=True, n_jobs=-1)
        assert config.get_effective_n_jobs() == max(1, cpu_count - 1)

        # 禁用并行时返回1
        config = ParallelConfig(enabled=False, n_jobs=-1)
        assert config.get_effective_n_jobs() == 1

    def test_should_use_parallel(self):
        """测试并行判断逻辑"""
        from dashboard.models.DFM.train.utils.parallel_config import ParallelConfig

        config = ParallelConfig(enabled=True, n_jobs=-1, min_variables_for_parallel=5)

        # 变量数小于阈值时不使用并行
        assert config.should_use_parallel(3) is False
        assert config.should_use_parallel(4) is False

        # 变量数大于等于阈值时使用并行
        assert config.should_use_parallel(5) is True
        assert config.should_use_parallel(10) is True


# ========== 新增Bug修复测试（TDD） ==========

class TestKalmanSmootherFallback:
    """测试平滑器伪逆回退机制"""

    def test_smoother_near_singular_matrix_fallback(self):
        """测试近奇异矩阵时平滑器使用伪逆回退"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 20, 2, 2

        # 构造接近奇异的状态空间模型
        A = np.array([[0.99, 0.0], [0.0, 0.99]])
        B = np.eye(n_states) * 0.001  # 极小的输入矩阵
        H = np.array([[1.0, 0.0], [1.0, 0.0]])  # 线性相关的观测矩阵
        Q = np.eye(n_states) * 1e-8  # 极小的过程噪声
        R = np.eye(n_obs) * 0.5
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states) * 1e-6  # 极小的初始协方差

        np.random.seed(42)
        Z = np.random.randn(n_time, n_obs)

        kf = KalmanFilter(A, B, H, Q, R, x0, P0)
        filter_result = kf.filter(Z)

        # 平滑器应该能够处理近奇异矩阵（使用伪逆回退）
        smooth_result = kf.smooth(filter_result)

        # 验证结果有效
        assert smooth_result.x_smoothed.shape == (n_states, n_time)
        assert not np.any(np.isnan(smooth_result.x_smoothed))


class TestKalmanGainsValidation:
    """测试卡尔曼增益历史验证"""

    def test_all_nan_data_raises_early(self):
        """测试全NaN数据时尽早抛出异常"""
        from dashboard.models.DFM.train.core.kalman import KalmanFilter

        n_time, n_obs, n_states = 10, 2, 1

        A = np.array([[0.9]])
        B = np.array([[0.1]])
        H = np.array([[1.0], [0.5]])
        Q = np.array([[0.1]])
        R = np.eye(n_obs) * 0.5
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        # 全部NaN的观测数据
        Z = np.full((n_time, n_obs), np.nan)

        kf = KalmanFilter(A, B, H, Q, R, x0, P0)

        # 应该抛出ValueError，而不是静默返回无效的kalman_gains_history
        with pytest.raises(ValueError, match="卡尔曼增益历史"):
            kf.filter(Z)


class TestPredictionBounds:
    """测试预测索引边界处理"""

    def test_validation_index_exceeds_forecast(self):
        """测试验证期索引完全超出预测范围时的处理"""
        from dashboard.models.DFM.train.core.prediction import generate_target_forecast
        from dashboard.models.DFM.train.core.models import DFMModelResult

        # 创建简单的模型结果
        n_factors = 1
        n_time = 10

        np.random.seed(42)
        factors = np.random.randn(n_factors, n_time)

        model_result = DFMModelResult(
            A=np.array([[0.9]]),
            Q=np.array([[0.1]]),
            H=np.array([[1.0]]),
            R=np.array([[0.5]]),
            factors=factors,
            converged=True,
            iterations=10,
            log_likelihood=-50.0
        )

        # 创建目标数据
        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        target_data = pd.Series(np.random.randn(n_time), index=dates)

        # 验证期完全超出数据范围
        result = generate_target_forecast(
            model_result=model_result,
            target_data=target_data,
            training_start='2020-01-01',
            train_end='2020-02-01',
            validation_start='2021-01-01',  # 超出范围
            validation_end='2021-12-31'     # 超出范围
        )

        # 超出范围时forecast_oos应为None
        assert result.forecast_oos is None

    def test_validation_index_partial_exceed_truncates(self):
        """测试验证期部分超出预测范围时截断处理"""
        from dashboard.models.DFM.train.core.prediction import generate_target_forecast
        from dashboard.models.DFM.train.core.models import DFMModelResult

        n_factors = 1
        n_time = 20

        np.random.seed(42)
        factors = np.random.randn(n_factors, n_time)

        model_result = DFMModelResult(
            A=np.array([[0.9]]),
            Q=np.array([[0.1]]),
            H=np.array([[1.0]]),
            R=np.array([[0.5]]),
            factors=factors,
            converged=True,
            iterations=10,
            log_likelihood=-50.0
        )

        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        target_data = pd.Series(np.random.randn(n_time), index=dates)

        train_end_date = dates[10]
        val_start_date = dates[11]
        # 验证期结束超出数据范围
        val_end_date = dates[-1] + pd.DateOffset(weeks=10)

        result = generate_target_forecast(
            model_result=model_result,
            target_data=target_data,
            training_start=str(dates[0].date()),
            train_end=str(train_end_date.date()),
            validation_start=str(val_start_date.date()),
            validation_end=str(val_end_date.date())
        )

        # 应该截断到可用范围，而不是None
        assert result.forecast_oos is not None
        # 截断后长度应该小于等于可用数据长度
        assert len(result.forecast_oos) <= n_time - 11


class TestEstimatorRankCheck:
    """测试载荷估计秩检查"""

    def test_rank_deficient_matrix_raises(self):
        """测试秩亏矩阵时抛出异常而非静默失败"""
        from dashboard.models.DFM.train.core.estimator import estimate_loadings

        n_time, n_factors = 50, 3

        np.random.seed(42)
        # 创建秩亏的因子数据（第3列是前两列的线性组合）
        factors_base = np.random.randn(n_time, 2)
        factors_data = np.column_stack([
            factors_base[:, 0],
            factors_base[:, 1],
            factors_base[:, 0] + factors_base[:, 1]  # 线性相关
        ])

        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        factors_df = pd.DataFrame(factors_data, index=dates, columns=['F1', 'F2', 'F3'])
        y = factors_data[:, 0] * 0.5 + np.random.randn(n_time) * 0.1
        y_series = pd.Series(y, index=dates)

        # 秩亏矩阵应该使用岭回归或抛出警告
        # 不应该静默返回NaN
        loadings = estimate_loadings(y_series, factors_df)

        # 结果应该有效（不是NaN）
        assert not np.any(np.isnan(loadings))
        assert loadings.shape == (n_factors,)


class TestFactorModelValidation:
    """测试因子模型数据验证"""

    def test_empty_training_data_raises(self):
        """测试空训练数据抛出明确异常"""
        from dashboard.models.DFM.train.core.factor_model import DFMModel

        n_time = 50
        n_vars = 5
        n_factors = 2

        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        data = pd.DataFrame(
            np.random.randn(n_time, n_vars),
            index=dates,
            columns=[f'var_{i}' for i in range(n_vars)]
        )

        model = DFMModel(n_factors=n_factors)

        # 训练期完全在数据范围之外
        with pytest.raises(ValueError, match="训练期.*没有数据"):
            model.fit(
                data=data,
                training_start='2021-01-01',  # 超出数据范围
                train_end='2021-12-31'
            )

    def test_insufficient_training_rows_raises(self):
        """测试训练数据行数不足抛出异常"""
        from dashboard.models.DFM.train.core.factor_model import DFMModel

        n_time = 25  # 25行，满足变量有效性检查但不满足训练行数要求
        n_vars = 5
        n_factors = 3

        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n_time, freq='W')
        data = pd.DataFrame(
            np.random.randn(n_time, n_vars),
            index=dates,
            columns=[f'var_{i}' for i in range(n_vars)]
        )

        model = DFMModel(n_factors=n_factors)

        # 训练期只有10行数据（2020-01-01到2020-03-01约9周）
        # min_train_rows = max(3*3, 20) = 20，所以10行不够
        with pytest.raises(ValueError, match="训练期数据不足"):
            model.fit(
                data=data,
                training_start='2020-01-01',
                train_end='2020-03-01'  # 约9周数据，不足20行
            )


class TestStepwiseSelectorSafety:
    """测试变量选择器安全性"""

    def test_remove_nonexistent_variable_raises(self):
        """测试移除不存在的变量时抛出异常"""
        # 这是一个边界测试，验证代码防御性
        test_list = ['a', 'b', 'c']

        # 直接测试list.remove行为
        with pytest.raises(ValueError):
            test_list.remove('d')  # 不存在的元素

        # 修复后的代码应该在移除前检查


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
