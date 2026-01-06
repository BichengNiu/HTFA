# -*- coding: utf-8 -*-
"""
变量选择参数模块

提供SelectionParams数据类，简化BackwardSelector的接口
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import pandas as pd


@dataclass
class SelectionParams:
    """
    变量选择参数

    将BackwardSelector.select()的16个参数封��为数据类，
    提高代码可读性和可维护性
    """
    # 必需参数
    initial_variables: List[str]
    target_variable: str
    full_data: pd.DataFrame
    training_start_date: str
    train_end_date: str
    validation_start: str
    validation_end: str

    # DFM参数
    k_factors: int = 2
    max_iter: int = 30
    max_lags: int = 1
    tolerance: float = 1e-4

    # 目标变量参数
    target_freq: str = 'M'
    target_mean_original: float = 0.0
    target_std_original: float = 1.0

    # 选择策略参数
    rmse_tolerance_percent: float = 1.0
    win_rate_tolerance_percent: float = 5.0
    selection_criterion: str = 'hybrid'
    prioritize_win_rate: bool = True
    training_weight: float = 0.5
    alignment_mode: str = 'next_month'

    # 动态因子选择参数
    factor_selection_method: str = 'fixed'
    pca_threshold: float = 0.9
    kaiser_threshold: float = 1.0

    # 回调函数
    progress_callback: Optional[Callable[[str], None]] = None

    # 其他参数
    use_optimization: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_params_dict(self) -> Dict[str, Any]:
        """
        转换为params字典格式（兼容旧接口）

        Returns:
            params字典
        """
        return {
            'k_factors': self.k_factors,
            'rmse_tolerance_percent': self.rmse_tolerance_percent,
            'win_rate_tolerance_percent': self.win_rate_tolerance_percent,
            'selection_criterion': self.selection_criterion,
            'prioritize_win_rate': self.prioritize_win_rate,
            'training_weight': self.training_weight,
            'factor_selection_method': self.factor_selection_method,
            'pca_threshold': self.pca_threshold,
            'kaiser_threshold': self.kaiser_threshold,
            **self.extra_params
        }

    def to_evaluator_config(self) -> Dict[str, Any]:
        """
        转换为评估器配置字典（用于并行评估）

        Returns:
            可序列化的评估器配置字典
        """
        return {
            'training_start': self.training_start_date,
            'train_end': self.train_end_date,
            'validation_start': self.validation_start,
            'validation_end': self.validation_end,
            'max_iterations': self.max_iter,
            'tolerance': self.tolerance,
            'alignment_mode': self.alignment_mode,
            'factor_selection_method': self.factor_selection_method,
            'pca_threshold': self.pca_threshold,
            'kaiser_threshold': self.kaiser_threshold
        }

    @classmethod
    def from_legacy_args(
        cls,
        initial_variables: List[str],
        target_variable: str,
        full_data: pd.DataFrame,
        params: Dict[str, Any],
        validation_start: str,
        validation_end: str,
        target_freq: str,
        training_start_date: str,
        train_end_date: str,
        target_mean_original: float = 0.0,
        target_std_original: float = 1.0,
        max_iter: int = 30,
        max_lags: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None,
        use_optimization: bool = False
    ) -> 'SelectionParams':
        """
        从旧接口参数创建SelectionParams

        Args:
            与BackwardSelector.select()相同的参数

        Returns:
            SelectionParams实例
        """
        return cls(
            initial_variables=initial_variables,
            target_variable=target_variable,
            full_data=full_data,
            training_start_date=training_start_date,
            train_end_date=train_end_date,
            validation_start=validation_start,
            validation_end=validation_end,
            k_factors=params.get('k_factors', 2),
            max_iter=max_iter,
            max_lags=max_lags,
            target_freq=target_freq,
            target_mean_original=target_mean_original,
            target_std_original=target_std_original,
            rmse_tolerance_percent=params.get('rmse_tolerance_percent', 1.0),
            win_rate_tolerance_percent=params.get('win_rate_tolerance_percent', 5.0),
            selection_criterion=params.get('selection_criterion', 'hybrid'),
            prioritize_win_rate=params.get('prioritize_win_rate', True),
            training_weight=params.get('training_weight', 0.5),
            factor_selection_method=params.get('factor_selection_method', 'fixed'),
            pca_threshold=params.get('pca_threshold', 0.9),
            kaiser_threshold=params.get('kaiser_threshold', 1.0),
            progress_callback=progress_callback,
            use_optimization=use_optimization,
            extra_params={k: v for k, v in params.items() if k not in [
                'k_factors', 'rmse_tolerance_percent', 'win_rate_tolerance_percent',
                'selection_criterion', 'prioritize_win_rate', 'training_weight',
                'factor_selection_method', 'pca_threshold', 'kaiser_threshold'
            ]}
        )


__all__ = ['SelectionParams']
