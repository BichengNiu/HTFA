# -*- coding: utf-8 -*-
"""
配置管理模块

提供统一的训练配置类和验证
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from dashboard.models.DFM.train.utils.parallel_config import ParallelConfig, create_default_parallel_config


@dataclass
class TrainingConfig:
    """完整训练配置

    包含DFM模型训练所需的全部配置参数,与trainer.py配合使用
    """
    # ========== 必填字段（无默认值） ==========
    # 核心配置
    data_path: str
    target_variable: str

    # 训练/验证期配置
    training_start: str  # 训练期开始日期
    train_end: str  # 训练期结束日期
    validation_start: str  # 验证期开始日期
    validation_end: str  # 验证期结束日期

    # ========== 可选字段（有默认值） ==========
    # 核心配置
    selected_indicators: List[str] = field(default_factory=list)
    target_freq: str = 'W-FRI'

    # 模型参数
    k_factors: int = 4
    max_iterations: int = 30
    max_lags: int = 1
    tolerance: float = 1e-6

    # 变量选择配置
    enable_variable_selection: bool = False
    variable_selection_method: str = 'backward'
    min_variables_after_selection: Optional[int] = None

    # 因子数选择配置
    factor_selection_method: str = 'fixed'  # fixed, cumulative
    pca_threshold: Optional[float] = 0.9  # cumulative方法的阈值

    # 并行计算配置（2025-11-08重构后默认启用）
    enable_parallel: bool = True  # 是否启用并行计算（重构后已解决序列化问题）
    n_jobs: int = -1  # 并行任务数（-1=所有核心，1=串行）
    parallel_backend: str = 'loky'  # 并行后端（loky, multiprocessing, threading）
    min_variables_for_parallel: int = 5  # 启用并行的最小变量数

    # 输出配置
    output_dir: Optional[str] = None

    # 行业映射（变量名 -> 行业名）
    industry_map: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化验证"""
        # 设置默认输出目录
        if self.output_dir is None:
            self.output_dir = str(Path.cwd() / "dfm_output")

        # 验证必填字段
        if not self.data_path:
            raise ValueError("data_path不能为空")
        if not self.target_variable:
            raise ValueError("target_variable不能为空")

        # 验证数据文件存在
        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        # 验证模型参数
        if self.k_factors <= 0:
            raise ValueError(f"k_factors必须为正数,当前值: {self.k_factors}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations必须为正数,当前值: {self.max_iterations}")
        if self.max_lags < 1:
            raise ValueError(f"max_lags必须>=1,当前值: {self.max_lags}")

        # 验证因子选择方法
        valid_methods = ['fixed', 'cumulative']
        if self.factor_selection_method not in valid_methods:
            raise ValueError(
                f"factor_selection_method必须是{valid_methods}之一,"
                f"当前值: {self.factor_selection_method}"
            )

        # 验证变量选择方法
        if self.enable_variable_selection:
            valid_selection_methods = ['backward', 'forward', 'none']
            if self.variable_selection_method not in valid_selection_methods:
                raise ValueError(
                    f"variable_selection_method必须是{valid_selection_methods}之一,"
                    f"当前值: {self.variable_selection_method}"
                )

        # 验证并行配置
        if self.n_jobs == 0:
            raise ValueError("n_jobs不能为0，使用-1表示所有核心，1表示串行")
        valid_backends = ['loky', 'multiprocessing', 'threading']
        if self.parallel_backend not in valid_backends:
            raise ValueError(
                f"parallel_backend必须是{valid_backends}之一,"
                f"当前值: {self.parallel_backend}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置

        Args:
            config_dict: 配置字典

        Returns:
            TrainingConfig对象
        """
        # 直接使用字典的键值对创建配置对象
        # dataclass会自动处理字段映射
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            配置字典
        """
        from dataclasses import asdict
        return asdict(self)

    def validate(self) -> List[str]:
        """验证配置完整性

        Returns:
            错误信息列表,空列表表示验证通过
        """
        errors = []

        try:
            # 调用__post_init__进行验证
            self.__post_init__()
        except Exception as e:
            errors.append(str(e))

        return errors

    def get_parallel_config(self) -> ParallelConfig:
        """获取并行配置对象

        Returns:
            ParallelConfig对象
        """
        return ParallelConfig(
            enabled=self.enable_parallel,
            n_jobs=self.n_jobs,
            backend=self.parallel_backend,
            verbose=0,
            min_variables_for_parallel=self.min_variables_for_parallel
        )

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"TrainingConfig(\n"
            f"  data_path={self.data_path},\n"
            f"  target_variable={self.target_variable},\n"
            f"  indicators={len(self.selected_indicators)},\n"
            f"  k_factors={self.k_factors},\n"
            f"  factor_selection={self.factor_selection_method},\n"
            f"  variable_selection={self.enable_variable_selection}\n"
            f")"
        )


def create_default_config(
    data_path: str,
    target_variable: str,
    selected_indicators: List[str] = None,
    k_factors: int = 4,
    train_end: str = None,
    validation_start: str = None,
    validation_end: str = None
) -> TrainingConfig:
    """创建默认配置

    Args:
        data_path: 数据文件路径
        target_variable: 目标变量名
        selected_indicators: 已选变量列表
        k_factors: 因子数量
        train_end: 训练期结束日期
        validation_start: 验证期开始日期
        validation_end: 验证期结束日期

    Returns:
        TrainingConfig: 默认配置对象
    """
    return TrainingConfig(
        data_path=data_path,
        target_variable=target_variable,
        selected_indicators=selected_indicators or [],
        k_factors=k_factors,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end
    )
