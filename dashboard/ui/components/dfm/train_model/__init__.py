# -*- coding: utf-8 -*-
"""
DFM模型训练UI组件模块

提供模型训练相关的UI组件，包括：
- 变量选择组件
- 参数配置组件  
- 训练执行组件
- 结果展示组件
"""

from dashboard.ui.components.dfm.train_model.variable_selection import VariableSelectionComponent
from dashboard.ui.components.dfm.train_model.date_range import DateRangeComponent
from dashboard.ui.components.dfm.train_model.model_parameters import ModelParametersComponent
from dashboard.ui.components.dfm.train_model.training_status import TrainingStatusComponent

__all__ = [
    'VariableSelectionComponent',
    'DateRangeComponent',
    'ModelParametersComponent',
    'TrainingStatusComponent'
]
