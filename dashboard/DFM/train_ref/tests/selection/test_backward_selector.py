# -*- coding: utf-8 -*-
"""
BackwardSelector单元测试

测试内容:
1. 后向选择逻辑正确性
2. 边界情况处理(单变量、无改进停止、空输入等)
3. 评估函数调用正确性
4. SelectionResult返回值正确性
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, call
from typing import Tuple

from dashboard.DFM.train_ref.selection.backward_selector import (
    BackwardSelector,
    SelectionResult
)


class TestBackwardSelectorBasic:
    """BackwardSelector基础功能测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        evaluator = Mock()
        selector = BackwardSelector(evaluator_func=evaluator)

        assert selector.evaluator_func == evaluator
        assert selector.criterion == 'rmse'
        assert selector.min_variables == 1

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        evaluator = Mock()
        selector = BackwardSelector(
            evaluator_func=evaluator,
            criterion='hit_rate',
            min_variables=3
        )

        assert selector.criterion == 'hit_rate'
        assert selector.min_variables == 3

    def test_init_min_variables_lower_bound(self):
        """测试min_variables最小值为1"""
        evaluator = Mock()
        selector = BackwardSelector(
            evaluator_func=evaluator,
            min_variables=0  # 应该被修正为1
        )

        assert selector.min_variables == 1


class TestBackwardSelectorSelection:
    """BackwardSelector选择逻辑测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='W-FRI')
        data = pd.DataFrame({
            'target': np.random.randn(100),
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100),
        }, index=dates)
        return data

    @pytest.fixture
    def mock_evaluator_improving(self):
        """
        创建模拟评估器 - 移除变量会改善性能

        返回格式: (is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _)
        """
        def evaluator(variables, **kwargs):
            # 变量越少,性能越好(模拟过拟合场景)
            num_vars = len([v for v in variables if v != kwargs.get('target_variable', 'target')])

            # RMSE随变量数增加
            rmse = 0.5 + num_vars * 0.1
            # Hit Rate随变量数减少而提高
            hit_rate = 60.0 - num_vars * 5.0

            return (rmse, rmse, None, None, hit_rate, hit_rate, False, None, None)

        return Mock(side_effect=evaluator)

    @pytest.fixture
    def mock_evaluator_no_improvement(self):
        """
        创建模拟评估器 - 移除变量不会改善性能

        所有评估返回相同性能,触发"无改进停止"条件
        """
        def evaluator(variables, **kwargs):
            # 固定性能,不随变量数变化
            return (0.5, 0.5, None, None, 60.0, 60.0, False, None, None)

        return Mock(side_effect=evaluator)

    def test_select_basic_flow(self, sample_data, mock_evaluator_improving):
        """测试基本选择流程"""
        selector = BackwardSelector(
            evaluator_func=mock_evaluator_improving,
            min_variables=1
        )

        initial_vars = ['target', 'var1', 'var2', 'var3']
        params = {'k_factors': 2}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10,
            max_lags=1
        )

        # 验证返回对象类型
        assert isinstance(result, SelectionResult)

        # 验证选中的变量数量减少(因为mock评估器鼓励移除变量)
        selected_predictors = [v for v in result.selected_variables if v != 'target']
        initial_predictors = [v for v in initial_vars if v != 'target']
        assert len(selected_predictors) < len(initial_predictors)

        # 验证目标变量总是包含在最终变量列表中
        assert 'target' in result.selected_variables

        # 验证历史记录
        assert isinstance(result.selection_history, list)
        assert len(result.selection_history) > 0

        # 验证评估次数
        assert result.total_evaluations > 0

    def test_select_no_improvement_stops(self, sample_data, mock_evaluator_no_improvement):
        """测试无改进时停止选择"""
        selector = BackwardSelector(
            evaluator_func=mock_evaluator_no_improvement,
            min_variables=1
        )

        initial_vars = ['target', 'var1', 'var2', 'var3']
        params = {'k_factors': 2}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 由于无改进,应该在第一轮就停止
        # 最终变量应该是初始变量
        assert set(result.selected_variables) == set(initial_vars)

        # 历史记录应该为空(未进行任何移除)
        assert len(result.selection_history) == 0

    def test_select_min_variables_constraint(self, sample_data, mock_evaluator_improving):
        """测试最小变量数约束"""
        selector = BackwardSelector(
            evaluator_func=mock_evaluator_improving,
            min_variables=2  # 至少保留2个预测变量
        )

        initial_vars = ['target', 'var1', 'var2', 'var3']
        params = {'k_factors': 2}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 验证最终预测变量数量 >= min_variables
        selected_predictors = [v for v in result.selected_variables if v != 'target']
        assert len(selected_predictors) >= 2


class TestBackwardSelectorEdgeCases:
    """BackwardSelector边界情况测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='W-FRI')
        data = pd.DataFrame({
            'target': np.random.randn(100),
            'var1': np.random.randn(100),
        }, index=dates)
        return data

    def test_select_single_predictor(self, sample_data):
        """测试只有单个预测变量时的行为"""
        evaluator = Mock(return_value=(0.5, 0.5, None, None, 60.0, 60.0, False, None, None))
        selector = BackwardSelector(
            evaluator_func=evaluator,
            min_variables=1
        )

        initial_vars = ['target', 'var1']
        params = {'k_factors': 1}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 由于只有一个预测变量且min_variables=1,应该直接停止
        assert set(result.selected_variables) == set(initial_vars)
        assert len(result.selection_history) == 0

    def test_select_empty_predictors(self, sample_data):
        """测试空预测变量列表"""
        evaluator = Mock()
        selector = BackwardSelector(evaluator_func=evaluator)

        # 只有目标变量,没有预测变量
        initial_vars = ['target']
        params = {'k_factors': 1}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 应该直接返回初始变量,并记录错误
        assert result.selected_variables == initial_vars
        assert result.total_evaluations == 0
        assert len(result.selection_history) == 0

    def test_select_with_svd_errors(self, sample_data):
        """测试包含SVD错误的评估"""
        # 模拟评估器,第一次调用返回SVD错误,后续正常
        call_count = [0]

        def evaluator_with_svd_error(variables, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一次调用返回SVD错误
                return (0.5, 0.5, None, None, 60.0, 60.0, True, None, None)
            else:
                # 后续调用正常
                return (0.5, 0.5, None, None, 60.0, 60.0, False, None, None)

        evaluator = Mock(side_effect=evaluator_with_svd_error)
        selector = BackwardSelector(evaluator_func=evaluator)

        initial_vars = ['target', 'var1']
        params = {'k_factors': 1}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 验证SVD错误被正确记录
        assert result.svd_error_count >= 1

    def test_select_with_progress_callback(self, sample_data):
        """测试进度回调功能"""
        evaluator = Mock(return_value=(0.5, 0.5, None, None, 60.0, 60.0, False, None, None))
        selector = BackwardSelector(evaluator_func=evaluator)

        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)

        initial_vars = ['target', 'var1']
        params = {'k_factors': 1}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10,
            progress_callback=progress_callback
        )

        # 验证进度回调被调用
        assert len(progress_messages) > 0
        # 验证消息包含SELECTION标记
        assert any('[SELECTION]' in msg for msg in progress_messages)


class TestBackwardSelectorEvaluatorInteraction:
    """BackwardSelector与评估器交互测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='W-FRI')
        data = pd.DataFrame({
            'target': np.random.randn(100),
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
        }, index=dates)
        return data

    def test_evaluator_called_with_correct_params(self, sample_data):
        """测试评估器被正确调用"""
        evaluator = Mock(return_value=(0.5, 0.5, None, None, 60.0, 60.0, False, None, None))
        selector = BackwardSelector(evaluator_func=evaluator)

        initial_vars = ['target', 'var1', 'var2']
        params = {'k_factors': 2}
        validation_start = '2020-07-01'
        validation_end = '2020-12-31'
        target_freq = 'W-FRI'
        train_end_date = '2020-06-30'
        target_mean = 0.0
        target_std = 1.0
        max_iter = 10
        max_lags = 1

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start=validation_start,
            validation_end=validation_end,
            target_freq=target_freq,
            train_end_date=train_end_date,
            target_mean_original=target_mean,
            target_std_original=target_std,
            max_iter=max_iter,
            max_lags=max_lags
        )

        # 验证评估器被调用
        assert evaluator.call_count > 0

        # 验证第一次调用的参数
        first_call = evaluator.call_args_list[0]
        assert 'variables' in first_call[1]
        assert 'full_data' in first_call[1]
        assert 'target_variable' in first_call[1]
        assert first_call[1]['target_variable'] == 'target'
        assert first_call[1]['validation_start'] == validation_start
        assert first_call[1]['validation_end'] == validation_end
        assert first_call[1]['max_iter'] == max_iter

    def test_evaluator_return_invalid_format(self, sample_data):
        """测试评估器返回无效格式时的处理"""
        # 返回错误数量的元素
        evaluator = Mock(return_value=(0.5, 0.5, None))  # 只有3个元素,预期9个
        selector = BackwardSelector(evaluator_func=evaluator)

        initial_vars = ['target', 'var1']
        params = {'k_factors': 1}

        result = selector.select(
            initial_variables=initial_vars,
            target_variable='target',
            full_data=sample_data,
            params=params,
            validation_start='2020-07-01',
            validation_end='2020-12-31',
            target_freq='W-FRI',
            train_end_date='2020-06-30',
            target_mean_original=0.0,
            target_std_original=1.0,
            max_iter=10
        )

        # 应该返回初始变量,标记为错误
        assert result.selected_variables == initial_vars
        assert result.final_score == (-np.inf, np.inf)


class TestSelectionResult:
    """SelectionResult数据类测试"""

    def test_selection_result_creation(self):
        """测试SelectionResult对象创建"""
        result = SelectionResult(
            selected_variables=['target', 'var1', 'var2'],
            selection_history=[
                {'removed': 'var3', 'score': (70.0, -0.4)},
                {'removed': 'var4', 'score': (75.0, -0.3)}
            ],
            final_score=(75.0, -0.3),
            total_evaluations=10,
            svd_error_count=1
        )

        assert result.selected_variables == ['target', 'var1', 'var2']
        assert len(result.selection_history) == 2
        assert result.final_score == (75.0, -0.3)
        assert result.total_evaluations == 10
        assert result.svd_error_count == 1

    def test_selection_result_is_dataclass(self):
        """验证SelectionResult是dataclass"""
        from dataclasses import is_dataclass
        assert is_dataclass(SelectionResult)
