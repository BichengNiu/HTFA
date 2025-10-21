# -*- coding: utf-8 -*-
"""
配置管理模块

提供统一的配置类和验证
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """DFM模型配置"""
    k_factors: int
    max_iter: int = 30
    max_lags: int = 1
    tolerance: float = 1e-6

    def __post_init__(self):
        if self.k_factors <= 0:
            raise ValueError(f"k_factors必须为正数，当前值: {self.k_factors}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter必须为正数，当前值: {self.max_iter}")
        if self.max_lags < 1:
            raise ValueError(f"max_lags必须>=1，当前值: {self.max_lags}")


@dataclass
class SelectionConfig:
    """变量选择配置"""
    method: str = 'backward'
    selection_criteria: str = 'hit_rate'
    min_variables: int = 5
    max_variables: Optional[int] = None

    def __post_init__(self):
        valid_methods = ['backward', 'forward', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"method必须是{valid_methods}之一，当前值: {self.method}")

        valid_criteria = ['rmse', 'mae', 'hit_rate', 'aic', 'bic']
        if self.selection_criteria not in valid_criteria:
            raise ValueError(f"selection_criteria必须是{valid_criteria}之一")

        if self.min_variables < 1:
            raise ValueError(f"min_variables必须>=1，当前值: {self.min_variables}")


@dataclass
class OptimizationConfig:
    """优化配置"""
    use_cache: bool = True
    use_precompute: bool = True
    cache_size: int = 1000
    enable_disk_cache: bool = False
    cache_dir: Optional[str] = None


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str
    target_variable: str
    selected_variables: List[str] = field(default_factory=list)
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    validation_start: Optional[str] = None
    validation_end: Optional[str] = None
    target_freq: str = 'M'

    def __post_init__(self):
        if not self.data_path:
            raise ValueError("data_path不能为空")
        if not self.target_variable:
            raise ValueError("target_variable不能为空")

        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")


@dataclass
class TrainingConfig:
    """完整训练配置"""
    data: DataConfig
    model: ModelConfig
    selection: Optional[SelectionConfig] = None
    optimization: Optional[OptimizationConfig] = None
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.selection is None:
            self.selection = SelectionConfig(method='none')

        if self.optimization is None:
            self.optimization = OptimizationConfig()

        if self.output_dir is None:
            self.output_dir = str(Path.cwd() / "dfm_output")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))

        selection_dict = config_dict.get('selection')
        selection_config = SelectionConfig(**selection_dict) if selection_dict else None

        optimization_dict = config_dict.get('optimization')
        optimization_config = OptimizationConfig(**optimization_dict) if optimization_dict else None

        return cls(
            data=data_config,
            model=model_config,
            selection=selection_config,
            optimization=optimization_config,
            output_dir=config_dict.get('output_dir')
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> List[str]:
        """验证配置完整性

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []

        try:
            self.data.__post_init__()
        except Exception as e:
            errors.append(f"数据配置错误: {e}")

        try:
            self.model.__post_init__()
        except Exception as e:
            errors.append(f"模型配置错误: {e}")

        if self.selection and self.selection.method != 'none':
            try:
                self.selection.__post_init__()
            except Exception as e:
                errors.append(f"选择配置错误: {e}")

        return errors


def create_default_config(
    data_path: str,
    target_variable: str,
    selected_variables: List[str] = None,
    k_factors: int = 4
) -> TrainingConfig:
    """创建默认配置

    Args:
        data_path: 数据文件路径
        target_variable: 目标变量名
        selected_variables: 已选变量列表
        k_factors: 因子数量

    Returns:
        TrainingConfig: 默认配置对象
    """
    data_config = DataConfig(
        data_path=data_path,
        target_variable=target_variable,
        selected_variables=selected_variables or []
    )

    model_config = ModelConfig(k_factors=k_factors)

    return TrainingConfig(
        data=data_config,
        model=model_config
    )
