# -*- coding: utf-8 -*-
"""
DFM训练模块配置管理系统

提供统一的配置管理，包括模型配置、训练配置和变量选择配置
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class DFMConfig:
    """DFM模型配置
    
    管理动态因子模型的核心参数
    """
    k_factors: int
    max_iterations: int = 30
    tolerance: float = 1e-6
    convergence_check_interval: int = 5
    verbose: bool = False
    
    def __post_init__(self):
        """验证配置参数"""
        if self.k_factors <= 0:
            raise ValueError(f"k_factors must be positive, got {self.k_factors}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DFMConfig':
        """从字典创建配置"""
        return cls(**data)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DFMConfig has no attribute '{key}'")
        self.__post_init__()  # 重新验证
    
    def copy_with(self, **kwargs) -> 'DFMConfig':
        """创建带有更新参数的新配置"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """训练配置
    
    管理训练过程的参数，包括数据路径、目标变量、日期范围等
    """
    data_path: str
    target_variable: str
    selected_indicators: List[str] = field(default_factory=list)
    date_ranges: Dict[str, str] = field(default_factory=dict)
    
    # 日期相关配置
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    validation_start: Optional[str] = None
    validation_end: Optional[str] = None
    
    # 其他训练参数
    batch_size: Optional[int] = None
    early_stopping: bool = False
    patience: int = 5
    
    def __post_init__(self):
        """处理日期范围"""
        # 如果提供了date_ranges，更新独立的日期字段
        if self.date_ranges:
            self.train_start = self.date_ranges.get('train_start', self.train_start)
            self.train_end = self.date_ranges.get('train_end', self.train_end)
            self.validation_start = self.date_ranges.get('validation_start', self.validation_start)
            self.validation_end = self.date_ranges.get('validation_end', self.validation_end)
        else:
            # 反向更新date_ranges
            self.date_ranges = {
                'train_start': self.train_start,
                'train_end': self.train_end,
                'validation_start': self.validation_start,
                'validation_end': self.validation_end
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        return cls(**data)


@dataclass
class SelectionConfig:
    """变量选择配置
    
    管理变量选择过程的参数
    """
    method: str  # "backward", "forward", "lasso", "stepwise"
    max_variables: int
    min_variables: int
    selection_criteria: str  # "aic", "bic", "rmse", "hit_rate"
    
    # 额外参数
    significance_level: float = 0.05
    cross_validation_folds: int = 5
    parallel_jobs: int = -1  # -1表示使用所有可用核心
    
    def __post_init__(self):
        """验证配置参数"""
        valid_methods = ["backward", "forward", "lasso", "stepwise"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        
        valid_criteria = ["aic", "bic", "rmse", "hit_rate"]
        if self.selection_criteria not in valid_criteria:
            raise ValueError(f"selection_criteria must be one of {valid_criteria}, got {self.selection_criteria}")
        
        if self.min_variables < 1:
            raise ValueError(f"min_variables must be at least 1, got {self.min_variables}")
        
        if self.max_variables < self.min_variables:
            raise ValueError(f"max_variables ({self.max_variables}) must be >= min_variables ({self.min_variables})")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelectionConfig':
        """从字典创建配置"""
        return cls(**data)


@dataclass
class CompleteConfig:
    """完整的DFM训练配置
    
    整合所有配置组件
    """
    dfm: DFMConfig
    training: TrainingConfig
    selection: Optional[SelectionConfig] = None
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        config_dict = {
            'dfm': self.dfm.to_dict(),
            'training': self.training.to_dict(),
            'selection': self.selection.to_dict() if self.selection else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'CompleteConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        dfm_config = DFMConfig.from_dict(config_dict['dfm'])
        training_config = TrainingConfig.from_dict(config_dict['training'])
        
        selection_config = None
        if config_dict.get('selection'):
            selection_config = SelectionConfig.from_dict(config_dict['selection'])
        
        return cls(dfm=dfm_config, training=training_config, selection=selection_config)


def get_default_dfm_config() -> DFMConfig:
    """获取默认的DFM配置"""
    return DFMConfig(
        k_factors=4,
        max_iterations=30,
        tolerance=1e-6,
        convergence_check_interval=5,
        verbose=False
    )


def get_default_training_config() -> TrainingConfig:
    """获取默认的训练配置"""
    return TrainingConfig(
        data_path="",
        target_variable="",
        selected_indicators=[],
        date_ranges={},
        early_stopping=False,
        patience=5
    )


def get_default_selection_config() -> SelectionConfig:
    """获取默认的变量选择配置"""
    return SelectionConfig(
        method="backward",
        max_variables=50,
        min_variables=5,
        selection_criteria="rmse",
        significance_level=0.05,
        cross_validation_folds=5,
        parallel_jobs=-1
    )


# 配置验证函数
def validate_config(config: Any) -> bool:
    """验证配置的完整性和有效性"""
    if isinstance(config, DFMConfig):
        return config.k_factors > 0 and config.max_iterations > 0
    elif isinstance(config, TrainingConfig):
        return bool(config.target_variable)
    elif isinstance(config, SelectionConfig):
        return config.max_variables >= config.min_variables
    elif isinstance(config, CompleteConfig):
        return all([
            validate_config(config.dfm),
            validate_config(config.training),
            validate_config(config.selection) if config.selection else True
        ])
    return False