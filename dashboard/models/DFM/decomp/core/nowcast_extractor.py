# -*- coding: utf-8 -*-
"""
Nowcast提取器

从保存的DFM模型中提取nowcast时间序列和相关状态信息，
为影响分析提供基础数据和基准值。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.exceptions import ComputationError, DataFormatError
from ..utils.helpers import align_time_series, ensure_numerical_stability
from .model_loader import SavedNowcastData


@dataclass
class NowcastState:
    """Nowcast状态信息"""
    timestamp: pd.Timestamp
    nowcast_value: float
    factor_states: np.ndarray
    confidence_interval: Optional[Tuple[float, float]] = None
    prediction_components: Optional[Dict[str, float]] = None


class NowcastExtractor:
    """
    Nowcast数据提取器

    专门用于从已保存的DFM模型数据中提取nowcast相关信息，
    提供基准预测值和状态信息用于影响分析。
    """

    def __init__(self, saved_nowcast_data: SavedNowcastData):
        """
        初始化Nowcast提取器

        Args:
            saved_nowcast_data: 已保存的nowcast数据
        """
        self.data = saved_nowcast_data
        self._aligned_data: Optional[Dict[str, Any]] = None
        self._baseline_predictions: Optional[Dict[str, float]] = None

    def extract_nowcast_series(self, target_period: Optional[str] = None) -> pd.Series:
        """
        提取nowcast时间序列

        Args:
            target_period: 目标周期 (YYYY-MM格式)，None表示全部

        Returns:
            nowcast时间序列

        Raises:
            DataFormatError: 数据格式错误时抛出
        """
        try:
            if self.data.nowcast_series is None:
                raise DataFormatError("nowcast时间序列数据不可用")

            nowcast_series = self.data.nowcast_series.copy()

            # 过滤目标周期
            if target_period:
                target_date = pd.to_datetime(target_period)
                nowcast_series = nowcast_series[nowcast_series.index <= target_date]

            print(f"[NowcastExtractor] 提取nowcast序列: {len(nowcast_series)} 个数据点")
            if target_period:
                print(f"[NowcastExtractor] 目标周期: {target_period}")

            return nowcast_series

        except Exception as e:
            raise ComputationError(f"提取nowcast序列失败: {str(e)}", "nowcast_extraction")

    def get_kalman_gains_matrix(self) -> np.ndarray:
        """
        获取卡尔曼增益矩阵（已废弃）

        警告: 此方法已废弃，仅为向后兼容保留。
        新的实现应直接访问 self.data.kalman_gains_history 获取完整的K_t历史。

        Returns:
            最后一个可用的卡尔曼增益矩阵（如果有）或因子载荷矩阵H（向后兼容）

        Raises:
            DataFormatError: 数据不可用时抛出
        """
        print("[NowcastExtractor] 警告: get_kalman_gains_matrix()已废弃")

        # 新版本：尝试返回最后一个可用的K_t
        if self.data.kalman_gains_history is not None and len(self.data.kalman_gains_history) > 0:
            # 从后向前查找第一个非None的K_t
            for K_t in reversed(self.data.kalman_gains_history):
                if K_t is not None:
                    print(f"[NowcastExtractor] 返回最后可用的K_t矩阵: {K_t.shape}")
                    return K_t

        # 向后兼容：如果没有K_t历史，返回H矩阵（旧行为，但这是错误的！）
        if self.data.factor_loadings is not None:
            print("[NowcastExtractor] 警告: 返回H矩阵（观测矩阵）用于向后兼容")
            print("[NowcastExtractor] 注意: 这不是真正的卡尔曼增益！请重新训练模型。")
            return self.data.factor_loadings

        raise DataFormatError("卡尔曼增益历史和因子载荷矩阵都不可用")

    def extract_factor_loadings(self) -> np.ndarray:
        """
        提取因子载荷矩阵

        Returns:
            因子载荷矩阵

        Raises:
            DataFormatError: 数据不可用时抛出
        """
        if self.data.factor_loadings is None:
            raise DataFormatError("因子载荷矩阵不可用")

        print(f"[NowcastExtractor] 因子载荷矩阵: {self.data.factor_loadings.shape}")
        return self.data.factor_loadings

    def get_model_state_at_date(self, target_date: pd.Timestamp) -> Dict[str, np.ndarray]:
        """
        获取指定日期的模型状态

        Args:
            target_date: 目标日期

        Returns:
            模型状态字典

        Raises:
            ComputationError: 状态提取失败时抛出
        """
        try:
            # 对齐因子序列和nowcast序列
            if self._aligned_data is None:
                self._prepare_aligned_data()

            if self._aligned_data is None or 'factor_series' not in self._aligned_data:
                raise DataFormatError("对齐数据不可用")

            factor_series = self._aligned_data['factor_series']

            # 查找最接近目标日期的状态
            if target_date not in factor_series.index:
                # 找到最接近的日期
                available_dates = factor_series.index[factor_series.index <= target_date]
                if len(available_dates) == 0:
                    raise ComputationError(f"未找到 {target_date} 之前的可用状态")
                target_date = available_dates[-1]

            # 提取因子状态
            factor_states = factor_series.loc[target_date].values

            # 构建状态字典
            model_state = {
                'factor_states': factor_states,
                'timestamp': target_date,
                'nowcast_value': self._aligned_data['nowcast_series'].loc[target_date],
                'factor_loadings': self.data.factor_loadings,
                'kalman_gains': self.data.kalman_gains
            }

            print(f"[NowcastExtractor] 提取 {target_date} 的模型状态")
            return model_state

        except Exception as e:
            raise ComputationError(f"提取模型状态失败: {str(e)}", "state_extraction")

    def compute_baseline_prediction(self, target_date: pd.Timestamp) -> float:
        """
        计算基准预测值

        Args:
            target_date: 目标日期

        Returns:
            基准预测值

        Raises:
            ComputationError: 计算失败时抛出
        """
        try:
            if self._baseline_predictions is None:
                self._compute_baseline_predictions()

            if self._baseline_predictions is None or target_date not in self._baseline_predictions:
                # 如果没有预计算的基准值，使用最接近的nowcast值
                nowcast_series = self.extract_nowcast_series()
                available_dates = nowcast_series.index[nowcast_series.index <= target_date]
                if len(available_dates) == 0:
                    raise ComputationError(f"未找到 {target_date} 之前的基准预测")
                baseline_value = nowcast_series.loc[available_dates[-1]]
            else:
                baseline_value = self._baseline_predictions[target_date]

            print(f"[NowcastExtractor] 基准预测值 ({target_date}): {baseline_value:.4f}")
            return baseline_value

        except Exception as e:
            raise ComputationError(f"计算基准预测失败: {str(e)}", "baseline_computation")

    def analyze_prediction_components(self) -> Dict[str, float]:
        """
        分析预测的组成成分

        Returns:
            各成分的贡献字典

        Raises:
            ComputationError: 分析失败时抛出
        """
        try:
            components = {}

            # 因子贡献
            if self.data.factor_series is not None and self.data.factor_loadings is not None:
                # 使用最新的因子状态和载荷计算贡献
                latest_factors = self.data.factor_series.iloc[-1]
                factor_contributions = np.dot(latest_factors.values, self.data.factor_loadings.T)

                for i, contribution in enumerate(factor_contributions):
                    components[f'factor_{i+1}_contribution'] = contribution

            # 模型参数贡献
            if self.data.model_parameters:
                for param_name, param_value in self.data.model_parameters.items():
                    if isinstance(param_value, (int, float)):
                        components[f'param_{param_name}'] = param_value

            print(f"[NowcastExtractor] 分析了 {len(components)} 个预测成分")
            return components

        except Exception as e:
            raise ComputationError(f"分析预测成分失败: {str(e)}", "component_analysis")

    def extract_time_series_slice(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        提取指定时间范围内的数据切片

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            数据切片字典

        Raises:
            ComputationError: 提取失败时抛出
        """
        try:
            if self._aligned_data is None:
                self._prepare_aligned_data()

            if self._aligned_data is None:
                raise DataFormatError("对齐数据不可用")

            slice_data = {}

            # 应用时间过滤
            for key, data in self._aligned_data.items():
                if isinstance(data, (pd.Series, pd.DataFrame)):
                    filtered_data = data.copy()
                    if start_date:
                        filtered_data = filtered_data[filtered_data.index >= start_date]
                    if end_date:
                        filtered_data = filtered_data[filtered_data.index <= end_date]
                    slice_data[key] = filtered_data

            print(f"[NowcastExtractor] 提取时间切片: {start_date} 到 {end_date}")
            return slice_data

        except Exception as e:
            raise ComputationError(f"提取时间切片失败: {str(e)}", "time_slice_extraction")

    def _prepare_aligned_data(self) -> None:
        """准备对齐的数据"""
        try:
            aligned_data = {}

            # 对齐nowcast序列和因子序列
            if self.data.nowcast_series is not None and self.data.factor_series is not None:
                nowcast_aligned, factors_aligned = align_time_series(
                    self.data.nowcast_series,
                    self.data.factor_series,
                    method="inner"
                )
                aligned_data['nowcast_series'] = nowcast_aligned
                aligned_data['factor_series'] = factors_aligned
            else:
                if self.data.nowcast_series is not None:
                    aligned_data['nowcast_series'] = self.data.nowcast_series
                if self.data.factor_series is not None:
                    aligned_data['factor_series'] = self.data.factor_series

            self._aligned_data = aligned_data
            print(f"[NowcastExtractor] 数据对齐完成，包含 {len(aligned_data)} 个序列")

        except Exception as e:
            raise ComputationError(f"数据对齐失败: {str(e)}", "data_alignment")

    def _compute_baseline_predictions(self) -> None:
        """计算基准预测值"""
        try:
            if self._aligned_data is None:
                self._prepare_aligned_data()

            if self._aligned_data is None or 'nowcast_series' not in self._aligned_data:
                raise DataFormatError("无法计算基准预测：数据不可用")

            nowcast_series = self._aligned_data['nowcast_series']

            # 简单的基准预测：使用移动平均
            window_size = min(5, len(nowcast_series))
            baseline_predictions = nowcast_series.rolling(window=window_size, center=True).mean()

            # 填充边界值
            baseline_predictions = baseline_predictions.bfill().ffill()

            self._baseline_predictions = baseline_predictions.to_dict()
            print(f"[NowcastExtractor] 计算了 {len(self._baseline_predictions)} 个基准预测值")

        except Exception as e:
            raise ComputationError(f"计算基准预测失败: {str(e)}", "baseline_computation")

    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        获取数据提取摘要

        Returns:
            提取摘要字典
        """
        summary = {
            'target_variable': self.data.target_variable,
            'data_period': self.data.data_period,
            'nowcast_data_points': len(self.data.nowcast_series) if self.data.nowcast_series is not None else 0,
            'factor_count': self.data.kalman_gains.shape[1] if self.data.kalman_gains is not None else 0,
            'variable_count': self.data.kalman_gains.shape[0] if self.data.kalman_gains is not None else 0,
        }

        if self.data.nowcast_series is not None:
            summary.update({
                'nowcast_start_date': str(self.data.nowcast_series.index.min()),
                'nowcast_end_date': str(self.data.nowcast_series.index.max()),
                'nowcast_mean': float(self.data.nowcast_series.mean()),
                'nowcast_std': float(self.data.nowcast_series.std()),
            })

        if self.data.convergence_info:
            summary['convergence_info'] = self.data.convergence_info

        return summary