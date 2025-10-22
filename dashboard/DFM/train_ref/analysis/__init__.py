# -*- coding: utf-8 -*-
"""
分析输出层

提供DFM模型结果的分析、报告生成和可视化功能
"""

# 报告生成
from dashboard.DFM.train_ref.analysis.reporter import (
    AnalysisReporter,
    generate_report
)

# 可视化
from dashboard.DFM.train_ref.analysis.visualizer import (
    ResultVisualizer,
    plot_results
)

# 分析工具函数
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

__all__ = [
    # 报告生成
    'AnalysisReporter',
    'generate_report',

    # 可视化
    'ResultVisualizer',
    'plot_results',

    # 分析工具
    'calculate_rmse',
    'calculate_hit_rate',
    'calculate_correlation',
    'calculate_metrics_with_lagged_target',
    'calculate_factor_contributions',
    'calculate_individual_variable_r2',
    'calculate_industry_r2',
    'calculate_pca_variance',
    'calculate_monthly_friday_metrics',
]
