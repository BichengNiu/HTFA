# -*- coding: utf-8 -*-
"""
分析配置类

提供统一的配置管理，简化函数参数，遵循KISS原则

基于YAGNI原则优化：
- 默认禁用参数验证（UI通常已保证参数有效）
- 提供validate_on_init参数控制是否在初始化时验证
- 提供独立的validate()方法供需要时手动调用
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LeadLagAnalysisConfig:
    """
    领先滞后分析配置

    使用配置类封装12个参数，简化函数调用
    """

    # 核心参数
    max_lags: int
    kl_bins: int = 10

    # 标准化配置
    standardize_for_kl: bool = True
    standardization_method: str = 'zscore'

    # 频率对齐配置
    enable_frequency_alignment: bool = True
    target_frequency: Optional[str] = None
    freq_agg_method: str = 'mean'
    time_column: Optional[str] = None

    # 验证控制（优化：默认不验证，减少开销）
    validate_on_init: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        """条件验证（基于validate_on_init）"""
        if self.validate_on_init:
            self.validate()

    def validate(self):
        """
        手动验证配置参数

        仅在需要时调用此方法（如从外部配置文件加载时）
        UI控制的参数通常不需要验证
        """
        if self.max_lags < 1:
            raise ValueError("max_lags必须大于0")

        if self.kl_bins < 2:
            raise ValueError("kl_bins必须大于等于2")

        if self.standardization_method not in ['zscore', 'minmax', 'none']:
            raise ValueError(
                f"不支持的标准化方法: {self.standardization_method}，"
                f"请使用 'zscore', 'minmax' 或 'none'"
            )

        valid_freq_agg = ['mean', 'last', 'first', 'sum', 'median']
        if self.freq_agg_method not in valid_freq_agg:
            raise ValueError(
                f"不支持的聚合方法: {self.freq_agg_method}，"
                f"请使用 {valid_freq_agg} 之一"
            )

        valid_freqs = [None, 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual']
        if self.target_frequency not in valid_freqs:
            raise ValueError(
                f"不支持的目标频率: {self.target_frequency}，"
                f"请使用 {[f for f in valid_freqs if f is not None]} 之一或None"
            )


@dataclass
class StationarityConfig:
    """
    平稳性分析配置

    封装平稳性检验的所有配置参数

    注意：当前未使用（stationarity.py直接使用原始参数）
    保留此类以备未来重构使用，或可考虑删除（YAGNI原则）
    """

    alpha: float = 0.05
    processing_method: str = 'keep'  # 'keep', 'diff', 'log_diff'
    diff_order: int = 1
    validate_on_init: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        """条件验证（基于validate_on_init）"""
        if self.validate_on_init:
            self.validate()

    def validate(self):
        """手动验证配置参数"""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha必须在(0, 1)之间")

        if self.diff_order < 1:
            raise ValueError("diff_order必须大于0")

        valid_methods = ['keep', 'diff', 'log_diff']
        if self.processing_method not in valid_methods:
            raise ValueError(
                f"不支持的处理方法: {self.processing_method}，"
                f"请使用 {valid_methods} 之一"
            )
