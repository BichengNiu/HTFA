# -*- coding: utf-8 -*-
"""
配置管理模块

提供统一的训练配置类和验证
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from dashboard.models.DFM.train.utils.parallel_config import ParallelConfig


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
    factor_selection_method: str = 'fixed'  # fixed, cumulative, kaiser
    pca_threshold: Optional[float] = 0.9  # cumulative方法的阈值
    kaiser_threshold: Optional[float] = 1.0  # kaiser方法的特征值阈值

    # 并行计算配置（2025-11-08重构后默认启用）
    enable_parallel: bool = True  # 是否启用并行计算（重构后已解决序列化问题）
    n_jobs: int = -1  # 并行任务数（-1=所有核心，1=串行）
    parallel_backend: str = 'loky'  # 并行后端（loky, multiprocessing, threading）
    min_variables_for_parallel: int = 5  # 启用并行的最小变量数

    # 输出配置
    output_dir: Optional[str] = None

    # 行业映射（变量名 -> 行业名）
    industry_map: Optional[Dict[str, str]] = field(default_factory=dict)

    # 二次估计法配置（2025-11-09新增）
    estimation_method: str = 'single_stage'  # 估计方法: single_stage(一次估计) 或 two_stage(二次估计)
    industry_k_factors: Dict[str, int] = field(default_factory=dict)  # 二次估计法中各行业因子数映射
    second_stage_extra_predictors: List[str] = field(default_factory=list)  # 二次估计法第二阶段额外预测变量

    # 第一阶段并行配置（2025-11-12新增）
    enable_first_stage_parallel: bool = True  # 是否启用第一阶段（分行业训练）并行计算
    first_stage_n_jobs: int = -1  # 第一阶段并行任务数（-1=所有核心，1=串行）
    min_industries_for_parallel: int = 3  # 启用第一阶段并行的最小行业数

    # 一阶段目标变量映射（2025-12新增）
    first_stage_target_map: Dict[str, str] = field(default_factory=dict)  # 变量名(normalized) -> "是"

    # 目标变量配对模式（2025-12新增）
    target_alignment_mode: str = 'next_month'  # 'current_month'(本月) 或 'next_month'(下月)

    # Win Rate优化配置（2025-12-19新增）
    rmse_tolerance_percent: float = 1.0  # RMSE相近的容忍度（百分比）
    win_rate_tolerance_percent: float = 5.0  # Win Rate相近的容忍度（百分比），用于胜率优先策略

    # 筛选策略配置（2025-12-20新增）
    selection_criterion: str = 'hybrid'  # 筛选标准: 'rmse', 'win_rate', 'hybrid'
    prioritize_win_rate: Optional[bool] = True  # 混合策略时的优先级，仅hybrid模式有效

    # 训练期权重配置（2025-12-20新增）
    training_weight: float = 0.5  # 训练期权重 (0.0-1.0)，验证期权重自动为 1-training_weight

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
        valid_methods = ['fixed', 'cumulative', 'kaiser']
        if self.factor_selection_method not in valid_methods:
            raise ValueError(
                f"factor_selection_method必须是{valid_methods}之一,"
                f"当前值: {self.factor_selection_method}"
            )

        # 验证变量选择方法
        if self.enable_variable_selection:
            valid_selection_methods = ['backward', 'forward', 'stepwise', 'none']
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

        # 验证二次估计法配置（2025-11-09）
        valid_estimation_methods = ['single_stage', 'two_stage']
        if self.estimation_method not in valid_estimation_methods:
            raise ValueError(
                f"estimation_method必须是{valid_estimation_methods}之一,"
                f"当前值: {self.estimation_method}"
            )

        if self.estimation_method == 'two_stage':
            if not self.industry_k_factors:
                raise ValueError("二次估计法需要设置各行业因子数（industry_k_factors不能为空）")

            if not self.first_stage_target_map:
                raise ValueError(
                    "二次估计法需要提供一阶段目标映射（first_stage_target_map不能为空）。"
                    "请确保指标体系表的'一阶段目标'列中至少有一个变量标记为'是'"
                )

            # 验证first_stage_target_map中是否有变量标记为'是'
            has_valid_target = any(v == '是' for v in self.first_stage_target_map.values())
            if not has_valid_target:
                raise ValueError(
                    "first_stage_target_map中没有任何变量标记为'是'。"
                    "请确保指标体系表的'一阶段目标'列中至少有一个变量标记为'是'"
                )

            for industry, k in self.industry_k_factors.items():
                if not isinstance(k, int) or k <= 0:
                    raise ValueError(f"行业 {industry} 的因子数必须为正整数，当前值: {k}")

        # 验证第一阶段并行配置（2025-11-12）
        if self.first_stage_n_jobs == 0:
            raise ValueError("first_stage_n_jobs不能为0，使用-1表示所有核心，1表示串行")
        if self.min_industries_for_parallel < 1:
            raise ValueError(f"min_industries_for_parallel必须>=1，当前值: {self.min_industries_for_parallel}")

        # 验证目标配对模式（2025-12）
        valid_alignment_modes = ['current_month', 'next_month']
        if self.target_alignment_mode not in valid_alignment_modes:
            raise ValueError(
                f"target_alignment_mode必须是{valid_alignment_modes}之一,"
                f"当前值: {self.target_alignment_mode}"
            )

        # 验证Win Rate配置（2025-12-19）
        if self.rmse_tolerance_percent < 0 or self.rmse_tolerance_percent > 100:
            raise ValueError(
                f"rmse_tolerance_percent必须在0-100之间，"
                f"当前值: {self.rmse_tolerance_percent}"
            )
        if self.win_rate_tolerance_percent < 0 or self.win_rate_tolerance_percent > 100:
            raise ValueError(
                f"win_rate_tolerance_percent必须在0-100之间，"
                f"当前值: {self.win_rate_tolerance_percent}"
            )

        # 验证筛选策略配置（2025-12-20）
        valid_criteria = ['rmse', 'win_rate', 'hybrid']
        if self.selection_criterion not in valid_criteria:
            raise ValueError(
                f"selection_criterion必须是{valid_criteria}之一，"
                f"当前值: {self.selection_criterion}"
            )

        # 验证训练期权重（2025-12-20）
        if not 0.0 <= self.training_weight <= 1.0:
            raise ValueError(
                f"training_weight必须在0.0到1.0之间，"
                f"当前值: {self.training_weight}"
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
