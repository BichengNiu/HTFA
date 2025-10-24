# -*- coding: utf-8 -*-
"""
DFM训练模块 - 重构版本
基于KISS, DRY, YAGNI, SOC, SRP原则的全新架构

主要组件：
- DFMTrainer: 统一训练接口（完整两阶段训练流程）
- TrainingConfig: 扁平化配置管理
- DFMModel: 核心DFM算法实现
- KalmanFilter: 卡尔曼滤波和平滑
- BackwardSelector: 后向逐步变量选择
- AnalysisReporter: 分析报告生成
"""

# 主训练器和配置
from dashboard.DFM.train_ref.training.trainer import DFMTrainer, TrainingResult, EvaluationMetrics
from dashboard.DFM.train_ref.training.config import TrainingConfig

# 核心算法
from dashboard.DFM.train_ref.core.factor_model import DFMModel, DFMResults
from dashboard.DFM.train_ref.core.kalman import KalmanFilter
from dashboard.DFM.train_ref.core.estimator import estimate_parameters

# 变量选择
from dashboard.DFM.train_ref.selection.backward_selector import BackwardSelector, SelectionResult

# 分析和报告
from dashboard.DFM.train_ref.analysis.reporter import AnalysisReporter
from dashboard.DFM.train_ref.analysis.visualizer import ResultVisualizer

__version__ = '2.0.0'

__all__ = [
    # 主接口
    'DFMTrainer',
    'TrainingConfig',
    'TrainingResult',
    'EvaluationMetrics',

    # 核心算法
    'DFMModel',
    'DFMResults',
    'KalmanFilter',
    'estimate_parameters',

    # 变量选择
    'BackwardSelector',
    'SelectionResult',

    # 分析报告
    'AnalysisReporter',
    'ResultVisualizer',
]
