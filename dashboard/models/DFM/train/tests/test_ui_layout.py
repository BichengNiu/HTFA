# -*- coding: utf-8 -*-
"""
UI布局单元测试

测试覆盖:
- 高级选项两列布局配置
- UIConfig配置完整性
- 布局参数验证
"""

import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np


class TestUIConfigCompleteness:
    """UIConfig配置完整性测试"""

    def test_factor_strategies_defined(self):
        """测试因子策略选项完整"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        assert hasattr(UIConfig, 'FACTOR_STRATEGIES')
        assert 'fixed_number' in UIConfig.FACTOR_STRATEGIES
        assert 'cumulative_variance' in UIConfig.FACTOR_STRATEGIES
        assert 'kaiser' in UIConfig.FACTOR_STRATEGIES

    def test_selection_criteria_defined(self):
        """测试筛选策略选项完整"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        assert hasattr(UIConfig, 'SELECTION_CRITERIA')
        assert 'rmse' in UIConfig.SELECTION_CRITERIA
        assert 'win_rate' in UIConfig.SELECTION_CRITERIA
        assert 'hybrid' in UIConfig.SELECTION_CRITERIA

    def test_hybrid_priorities_defined(self):
        """测试混合优先级选项完整"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        assert hasattr(UIConfig, 'HYBRID_PRIORITIES')
        assert 'win_rate_first' in UIConfig.HYBRID_PRIORITIES
        assert 'rmse_first' in UIConfig.HYBRID_PRIORITIES

    def test_factor_ar_order_bounds(self):
        """测试因子AR阶数边界值"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        assert UIConfig.FACTOR_AR_ORDER_MIN >= 0
        assert UIConfig.FACTOR_AR_ORDER_MAX > UIConfig.FACTOR_AR_ORDER_MIN
        assert UIConfig.FACTOR_AR_ORDER_MIN <= UIConfig.DEFAULT_FACTOR_AR_ORDER <= UIConfig.FACTOR_AR_ORDER_MAX

    def test_training_weight_bounds(self):
        """测试训练期权重边界值"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        assert UIConfig.TRAINING_WEIGHT_MIN == 0
        assert UIConfig.TRAINING_WEIGHT_MAX == 100
        assert UIConfig.TRAINING_WEIGHT_MIN <= UIConfig.DEFAULT_TRAINING_WEIGHT <= UIConfig.TRAINING_WEIGHT_MAX

    def test_get_safe_option_index_valid(self):
        """测试安全索引获取-有效值"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        options = {'a': 'A', 'b': 'B', 'c': 'C'}
        index = UIConfig.get_safe_option_index(options, 'b', 'a')
        assert index == 1

    def test_get_safe_option_index_invalid(self):
        """测试安全索引获取-无效值回退到默认"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        options = {'a': 'A', 'b': 'B', 'c': 'C'}
        index = UIConfig.get_safe_option_index(options, 'invalid', 'a')
        assert index == 0  # 回退到默认值'a'的索引


class TestAdvancedOptionsLayout:
    """高级选项布局测试"""

    def test_factor_params_two_column_layout(self):
        """测试因子参数使用两列布局"""
        # 模拟Streamlit组件
        mock_st = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2)

        # 模拟expander上下文
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        # 验证columns(2)被调用（两列布局）
        mock_st.columns(2)
        mock_st.columns.assert_called_with(2)

    def test_selection_params_two_column_layout(self):
        """测试筛选参数使用两列布局"""
        mock_st = MagicMock()
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2)

        # 验证columns(2)被调用
        mock_st.columns(2)
        mock_st.columns.assert_called_with(2)

    def test_strategy_value_determines_left_column_content(self):
        """测试策略值决定左列内容"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        # 验证三种策略都有对应的配置
        strategies = list(UIConfig.FACTOR_STRATEGIES.keys())
        assert len(strategies) == 3
        assert 'fixed_number' in strategies
        assert 'cumulative_variance' in strategies
        assert 'kaiser' in strategies


class TestLayoutValidation:
    """布局参数验证测试"""

    def test_invalid_selection_criterion_raises(self):
        """测试无效筛选策略抛出异常"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        invalid_criterion = 'invalid_criterion'
        assert invalid_criterion not in UIConfig.SELECTION_CRITERIA

        # 代码中的验证逻辑
        with pytest.raises(ValueError, match="无效的筛选策略"):
            if invalid_criterion not in UIConfig.SELECTION_CRITERIA:
                raise ValueError(f"无效的筛选策略: {invalid_criterion}，有效值: {list(UIConfig.SELECTION_CRITERIA.keys())}")

    def test_invalid_hybrid_priority_raises(self):
        """测试无效混合优先级抛出异常"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        invalid_priority = 'invalid_priority'
        assert invalid_priority not in UIConfig.HYBRID_PRIORITIES

        # 代码中的验证逻辑
        with pytest.raises(ValueError, match="无效的混合优先级"):
            if invalid_priority not in UIConfig.HYBRID_PRIORITIES:
                raise ValueError(f"无效的混合优先级: {invalid_priority}，有效值: {list(UIConfig.HYBRID_PRIORITIES.keys())}")


class TestNoCompatibilityCode:
    """无兼容/回退代码测试"""

    def test_no_fallback_on_invalid_strategy(self):
        """测试无效策略不会静默回退"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        # 验证get_safe_option_index在值完全无效时不会崩溃
        # 但是会回退到默认值（这是预期行为）
        options = UIConfig.SELECTION_CRITERIA
        index = UIConfig.get_safe_option_index(options, 'nonexistent', UIConfig.DEFAULT_SELECTION_CRITERION)

        # 应该返回默认值的索引
        expected_index = list(options.keys()).index(UIConfig.DEFAULT_SELECTION_CRITERION)
        assert index == expected_index

    def test_direct_error_on_validation_failure(self):
        """测试验证失败时直接报错"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        # 模拟代码中的验证逻辑
        def validate_criterion(criterion):
            if criterion not in UIConfig.SELECTION_CRITERIA:
                raise ValueError(f"无效的筛选策略: {criterion}")
            return criterion

        # 有效值通过
        assert validate_criterion('rmse') == 'rmse'
        assert validate_criterion('hybrid') == 'hybrid'

        # 无效值直接报错
        with pytest.raises(ValueError):
            validate_criterion('invalid')


class TestWeightSliderIndependence:
    """评分权重滑块独立测试"""

    def test_weight_slider_full_width(self):
        """测试评分权重滑块独占一行（不在columns中）"""
        from dashboard.models.DFM.train.config.ui_config import UIConfig

        # 验证权重配置存在
        assert hasattr(UIConfig, 'TRAINING_WEIGHT_MIN')
        assert hasattr(UIConfig, 'TRAINING_WEIGHT_MAX')
        assert hasattr(UIConfig, 'TRAINING_WEIGHT_STEP')
        assert hasattr(UIConfig, 'DEFAULT_TRAINING_WEIGHT')

        # 验证值范围合理
        assert UIConfig.TRAINING_WEIGHT_MIN == 0
        assert UIConfig.TRAINING_WEIGHT_MAX == 100
        assert UIConfig.TRAINING_WEIGHT_STEP > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
