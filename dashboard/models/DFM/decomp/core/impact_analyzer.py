# -*- coding: utf-8 -*-
"""
影响分析器

计算新数据发布对nowcast值的边际影响，基于卡尔曼增益权重
进行影响分析和贡献分解。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils.exceptions import ComputationError, ValidationError
from ..utils.helpers import safe_matrix_division, ensure_numerical_stability
from .nowcast_extractor import NowcastExtractor


@dataclass
class DataRelease:
    """数据发布事件"""
    timestamp: pd.Timestamp
    variable_name: str
    observed_value: float
    expected_value: float
    measurement_error: Optional[float] = None
    variable_index: Optional[int] = None


@dataclass
class ImpactResult:
    """单次影响计算结果"""
    release: DataRelease
    impact_on_target: float
    contribution_percentage: float
    kalman_weight: float
    confidence_interval: Optional[Tuple[float, float]] = None
    calculation_details: Optional[Dict[str, Any]] = None


@dataclass
class SequentialImpactResult:
    """时序影响分析结果"""
    target_date: pd.Timestamp
    baseline_value: float
    final_value: float
    total_impact: float
    cumulative_impacts: pd.Series
    individual_impacts: List[ImpactResult]
    positive_impact_sum: float
    negative_impact_sum: float


class ImpactAnalyzer:
    """
    数据发布影响分析器

    基于卡尔曼滤波理论计算新数据发布对nowcast预测的边际影响。
    核心公式（修正版）：Δy = λ_y' × K_t[:, i] × v_i

    其中：
    - λ_y: 目标变量的因子载荷向量 (n_factors,)
    - K_t: 时刻t的卡尔曼增益矩阵 (n_factors, n_variables)
    - K_t[:, i]: 第i个观测变量对应的卡尔曼增益列向量 (n_factors,)
    - v_i: 第i个变量的新息（观测值 - 期望值）
    """

    def __init__(self, nowcast_extractor: NowcastExtractor):
        """
        初始化影响分析器

        Args:
            nowcast_extractor: Nowcast提取器实例
        """
        self.extractor = nowcast_extractor
        self._impact_cache: Optional[Dict[str, ImpactResult]] = None

        # 验证数据完整性
        if self.extractor.data.kalman_gains_history is None:
            print("[ImpactAnalyzer] 警告: 卡尔曼增益历史不可用，影响分析功能受限")
        if self.extractor.data.variable_index_map is None:
            print("[ImpactAnalyzer] 警告: 变量索引映射不可用，可能无法正确识别变量")

    def calculate_single_release_impact(self, release: DataRelease) -> ImpactResult:
        """
        计算单个数据发布的影响（修正版）

        使用正确的新闻分解公式：
        Δy = λ_y' × K_t[:, i] × v_i

        Args:
            release: 数据发布事件

        Returns:
            影响计算结果

        Raises:
            ComputationError: 计算失败时抛出
            ValidationError: 数据验证失败时抛出
        """
        try:
            # 检查数据完整性
            if self.extractor.data.kalman_gains_history is None:
                raise ComputationError(
                    "影响分解功能需要卡尔曼增益历史数据。\n"
                    "当前模型是使用旧版本训练模块生成的。\n"
                    "请使用新版本训练模块重新训练模型以支持此功能。"
                )

            # 1. 获取目标变量的因子载荷向量 λ_y
            lambda_y = self._get_target_variable_loading()  # (n_factors,)

            # 2. 获取观测变量在向量中的索引
            variable_index = self._get_variable_index(release.variable_name)

            # 3. 获取发布时刻的卡尔曼增益矩阵 K_t
            K_t = self._get_kalman_gain_at_time(release.timestamp)  # (n_factors, n_variables)

            # 4. 提取第i个变量对应的卡尔曼增益列向量
            K_col = K_t[:, variable_index]  # (n_factors,)

            # 5. 计算新息 v_i
            innovation = release.observed_value - release.expected_value

            # 6. 正确的影响计算公式：Δy = λ_y' × K_t[:, i] × v_i
            # 步骤：K_col * innovation → (n_factors,) 因子状态变化
            delta_f = K_col * innovation  # (n_factors,) 因子状态的变化量

            # 步骤：λ_y' × delta_f → 标量，映射到目标变量
            impact_on_target = np.dot(lambda_y, delta_f)  # 标量

            # 7. 计算有效卡尔曼权重（用于报告）
            effective_kalman_weight = np.linalg.norm(K_col)  # 向量的范数

            # 8. 计算贡献百分比（相对于标准化的基准）
            contribution_percentage = self._calculate_contribution_percentage(impact_on_target)

            # 9. 计算置信区间
            confidence_interval = self._calculate_confidence_interval(
                impact_on_target, release.measurement_error
            )

            # 10. 构建详细的计算信息
            calculation_details = {
                'innovation': float(innovation),
                'variable_index': int(variable_index),
                'kalman_gain_vector': K_col.tolist(),  # 完整的K_t[:, i]向量
                'factor_loading_vector': lambda_y.tolist(),  # λ_y向量
                'factor_state_change': delta_f.tolist(),  # Δf向量
                'effective_kalman_weight': float(effective_kalman_weight),
                'calculation_formula': 'Δy = λ_y\' × K_t[:, i] × v_i',
                'formula_explanation': {
                    'lambda_y': 'Target variable factor loadings (n_factors,)',
                    'K_t_col': 'Kalman gain vector for variable i (n_factors,)',
                    'v_i': 'Innovation (observed - expected)',
                    'delta_f': 'Factor state change K_t[:, i] * v_i',
                    'impact': 'Final impact λ_y\' × delta_f'
                }
            }

            result = ImpactResult(
                release=release,
                impact_on_target=impact_on_target,
                contribution_percentage=contribution_percentage,
                kalman_weight=effective_kalman_weight,  # 使用向量范数作为权重
                confidence_interval=confidence_interval,
                calculation_details=calculation_details
            )

            print(f"[ImpactAnalyzer] 单次影响计算: {release.variable_name} = {impact_on_target:.4f} "
                  f"(innovation={innovation:.4f}, ||K||={effective_kalman_weight:.4f})")
            return result

        except Exception as e:
            raise ComputationError(f"单次影响计算失败: {str(e)}", "single_impact_calculation")

    def analyze_sequential_impacts(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp
    ) -> SequentialImpactResult:
        """
        分析时序累积影响

        Args:
            releases: 数据发布列表（按时间排序）
            target_date: 目标分析日期

        Returns:
            时序影响分析结果

        Raises:
            ComputationError: 分析失败时抛出
        """
        try:
            # 按时间排序数据发布
            sorted_releases = sorted(releases, key=lambda x: x.timestamp)

            # 获取基准值
            baseline_value = self.extractor.compute_baseline_prediction(target_date)

            # 计算各次发布的影响
            individual_impacts = []
            cumulative_values = []

            current_value = baseline_value
            cumulative_impact = 0.0

            # 计算目标月份的结束日期
            target_month_start = pd.Timestamp(year=target_date.year, month=target_date.month, day=1)
            if target_date.month == 12:
                target_month_end = pd.Timestamp(year=target_date.year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                target_month_end = pd.Timestamp(year=target_date.year, month=target_date.month + 1, day=1) - pd.Timedelta(days=1)

            for release in sorted_releases:
                # 只处理目标月份范围内的数据发布
                if release.timestamp < target_month_start or release.timestamp > target_month_end:
                    continue

                impact_result = self.calculate_single_release_impact(release)
                individual_impacts.append(impact_result)

                # 更新累积影响
                cumulative_impact += impact_result.impact_on_target
                current_value = baseline_value + cumulative_impact
                cumulative_values.append({
                    'timestamp': release.timestamp,
                    'cumulative_impact': cumulative_impact,
                    'current_value': current_value,
                    'individual_impact': impact_result.impact_on_target
                })

            # 创建累积影响序列
            if cumulative_values:
                cumulative_df = pd.DataFrame(cumulative_values)
                cumulative_df.set_index('timestamp', inplace=True)
                cumulative_series = cumulative_df['cumulative_impact']
            else:
                cumulative_series = pd.Series([], dtype=float)

            # 计算正负影响总和
            positive_impact_sum = sum(imp.impact_on_target for imp in individual_impacts if imp.impact_on_target > 0)
            negative_impact_sum = sum(imp.impact_on_target for imp in individual_impacts if imp.impact_on_target < 0)

            result = SequentialImpactResult(
                target_date=target_date,
                baseline_value=baseline_value,
                final_value=current_value,
                total_impact=cumulative_impact,
                cumulative_impacts=cumulative_series,
                individual_impacts=individual_impacts,
                positive_impact_sum=positive_impact_sum,
                negative_impact_sum=negative_impact_sum
            )

            print(f"[ImpactAnalyzer] 时序影响分析完成: 总影响 = {cumulative_impact:.4f}")
            return result

        except Exception as e:
            raise ComputationError(f"时序影响分析失败: {str(e)}", "sequential_impact_analysis")

    def decompose_total_impact(
        self,
        target_date: pd.Timestamp,
        releases: List[DataRelease]
    ) -> Dict[str, Any]:
        """
        分解总影响为各数据发布的具体贡献

        Args:
            target_date: 目标日期
            releases: 数据发布列表

        Returns:
            影响分解结果

        Raises:
            ComputationError: 分解失败时抛出
        """
        try:
            # 计算时序影响
            sequential_result = self.analyze_sequential_impacts(releases, target_date)

            # 按变量分组影响
            variable_impacts = {}
            for impact in sequential_result.individual_impacts:
                var_name = impact.release.variable_name
                if var_name not in variable_impacts:
                    variable_impacts[var_name] = []
                variable_impacts[var_name].append(impact)

            # 计算每个变量的总影响
            variable_summary = {}
            total_abs_impact = sum(abs(imp.impact_on_target) for imp in sequential_result.individual_impacts)

            for var_name, impacts in variable_impacts.items():
                total_impact = sum(imp.impact_on_target for imp in impacts)
                contribution_pct = (abs(total_impact) / total_abs_impact * 100) if total_abs_impact > 0 else 0

                # 统计该变量的发布次数
                release_count = len(impacts)

                # 计算平均影响
                avg_impact = total_impact / release_count if release_count > 0 else 0

                variable_summary[var_name] = {
                    'total_impact': total_impact,
                    'contribution_percentage': contribution_pct,
                    'release_count': release_count,
                    'average_impact': avg_impact,
                    'impacts': impacts
                }

            # 按贡献度排序
            sorted_variables = sorted(
                variable_summary.items(),
                key=lambda x: abs(x[1]['total_impact']),
                reverse=True
            )

            # 构建分解结果
            decomposition_result = {
                'target_date': target_date,
                'total_impact': sequential_result.total_impact,
                'positive_impact_sum': sequential_result.positive_impact_sum,
                'negative_impact_sum': sequential_result.negative_impact_sum,
                'variable_contributions': dict(sorted_variables),
                'top_contributors': [var for var, _ in sorted_variables[:5]],
                'sequential_result': sequential_result,
                'impact_count': len(sequential_result.individual_impacts)
            }

            print(f"[ImpactAnalyzer] 影响分解完成: {len(variable_summary)} 个变量")
            return decomposition_result

        except Exception as e:
            raise ComputationError(f"影响分解失败: {str(e)}", "impact_decomposition")

    def simulate_counterfactual(
        self,
        releases: List[DataRelease],
        excluded_releases: List[str],
        target_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        反事实分析：模拟排除某些数据发布后的影响差异

        Args:
            releases: 完整的数据发布列表
            excluded_releases: 要排除的变量名列表
            target_date: 目标日期

        Returns:
            反事实分析结果

        Raises:
            ComputationError: 模拟失败时抛出
        """
        try:
            # 过滤掉排除的数据发布
            filtered_releases = [
                release for release in releases
                if release.variable_name not in excluded_releases
            ]

            # 计算完整影响
            full_result = self.analyze_sequential_impacts(releases, target_date)

            # 计算过滤后的影响
            filtered_result = self.analyze_sequential_impacts(filtered_releases, target_date)

            # 计算差异
            impact_difference = full_result.total_impact - filtered_result.total_impact

            # 分析排除变量的影响
            excluded_impacts = {}
            for var_name in excluded_releases:
                var_impacts = [
                    imp for imp in full_result.individual_impacts
                    if imp.release.variable_name == var_name
                ]
                if var_impacts:
                    excluded_impacts[var_name] = {
                        'total_impact': sum(imp.impact_on_target for imp in var_impacts),
                        'release_count': len(var_impacts),
                        'impacts': var_impacts
                    }

            counterfactual_result = {
                'target_date': target_date,
                'excluded_variables': excluded_releases,
                'full_impact': full_result.total_impact,
                'filtered_impact': filtered_result.total_impact,
                'impact_difference': impact_difference,
                'baseline_value': full_result.baseline_value,
                'full_final_value': full_result.final_value,
                'filtered_final_value': filtered_result.final_value,
                'excluded_impacts': excluded_impacts,
                'difference_percentage': (
                    abs(impact_difference) / abs(full_result.total_impact) * 100
                    if full_result.total_impact != 0 else 0
                )
            }

            print(f"[ImpactAnalyzer] 反事实分析完成: 影响差异 = {impact_difference:.4f}")
            return counterfactual_result

        except Exception as e:
            raise ComputationError(f"反事实分析失败: {str(e)}", "counterfactual_analysis")

    def _get_target_variable_loading(self) -> np.ndarray:
        """
        获取目标变量的因子载荷向量 λ_y

        Returns:
            目标变量的载荷向量 (n_factors,)

        Raises:
            ValidationError: 数据不可用时抛出
        """
        # 优先使用target_factor_loading字段
        if self.extractor.data.target_factor_loading is not None:
            lambda_y = self.extractor.data.target_factor_loading
            return lambda_y

        # 降级方案：从H矩阵通过索引提取（向后兼容旧模型）
        if self.extractor.data.factor_loadings is None:
            raise ValidationError("因子载荷矩阵（H矩阵）不可用")

        if self.extractor.data.target_variable_index is None:
            raise ValidationError("目标变量索引不可用，且缺少target_factor_loading字段")

        # H矩阵形状为 (n_variables, n_factors)
        # 提取目标变量对应的行，得到载荷向量
        target_index = self.extractor.data.target_variable_index
        lambda_y = self.extractor.data.factor_loadings[target_index, :]  # (n_factors,)

        return lambda_y

    def _get_kalman_gain_at_time(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        获取指定时刻最近的卡尔曼增益矩阵 K_t

        Args:
            timestamp: 目标时间戳

        Returns:
            卡尔曼增益矩阵 (n_factors, n_variables)

        Raises:
            ComputationError: 无法找到有效的卡尔曼增益时抛出
        """
        if self.extractor.data.kalman_gains_history is None:
            raise ComputationError("卡尔曼增益历史不可用")

        if self.extractor.data.nowcast_series is None:
            raise ComputationError("nowcast时间序列不可用")

        # 找到时间戳对应的索引
        nowcast_series = self.extractor.data.nowcast_series
        available_dates = nowcast_series.index[nowcast_series.index <= timestamp]

        if len(available_dates) == 0:
            raise ComputationError(f"未找到 {timestamp} 之前的卡尔曼增益")

        # 获取最接近的日期索引
        closest_date_index = len(available_dates) - 1

        # 边界检查
        if closest_date_index >= len(self.extractor.data.kalman_gains_history):
            closest_date_index = len(self.extractor.data.kalman_gains_history) - 1

        K_t = self.extractor.data.kalman_gains_history[closest_date_index]

        # 如果该时刻的K_t为None（无观测），向前查找最近的非None值
        if K_t is None:
            for idx in range(closest_date_index - 1, -1, -1):
                if self.extractor.data.kalman_gains_history[idx] is not None:
                    K_t = self.extractor.data.kalman_gains_history[idx]
                    print(f"[ImpactAnalyzer] 使用时刻 {idx} 的卡尔曼增益（最接近 {timestamp}）")
                    break

        if K_t is None:
            raise ComputationError(f"未找到可用的卡尔曼增益矩阵（目标时间: {timestamp}）")

        return K_t

    def _get_variable_index(self, variable_name: str) -> int:
        """
        从映射中获取变量索引（修正版）

        Args:
            variable_name: 变量名

        Returns:
            变量在观测向量中的索引

        Raises:
            ValidationError: 变量不在映射中时抛出
        """
        if self.extractor.data.variable_index_map is None:
            raise ValidationError(
                "变量索引映射不可用。\n"
                "这可能是因为模型元数据不完整。\n"
                "请确保模型包含完整的变量列表信息。"
            )

        if variable_name not in self.extractor.data.variable_index_map:
            # 提供详细的错误信息
            available_vars = list(self.extractor.data.variable_index_map.keys())
            raise ValidationError(
                f"变量 '{variable_name}' 不在变量映射中。\n"
                f"可用变量数: {len(available_vars)}\n"
                f"前5个可用变量: {available_vars[:5]}"
            )

        return self.extractor.data.variable_index_map[variable_name]

    def _calculate_contribution_percentage(self, impact: float) -> float:
        """计算贡献百分比"""
        # 简化的贡献度计算
        # 实际应用中应该基于基准预测的标准化
        baseline_std = 1.0  # 简化假设
        return (abs(impact) / baseline_std * 100) if baseline_std != 0 else 0

    def _calculate_confidence_interval(
        self,
        impact: float,
        measurement_error: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """计算置信区间"""
        try:
            # 简化的置信区间计算
            # 实际应用中应该基于完整的协方差矩阵
            if measurement_error is None:
                measurement_error = 0.1  # 默认测量误差

            standard_error = measurement_error * abs(impact) if impact != 0 else measurement_error
            margin = 1.96 * standard_error  # 95% 置信区间

            return (impact - margin, impact + margin)

        except Exception:
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取影响分析器摘要

        Returns:
            分析器摘要字典
        """
        summary = {
            'kalman_gains_shape': self._kalman_gains.shape if self._kalman_gains is not None else None,
            'variable_mapping_count': len(self._variable_mapping) if self._variable_mapping else 0,
            'cached_impacts_count': len(self._impact_cache) if self._impact_cache else 0,
        }

        if self._kalman_gains is not None:
            summary.update({
                'max_kalman_weight': float(np.max(np.abs(self._kalman_gains))),
                'mean_kalman_weight': float(np.mean(np.abs(self._kalman_gains))),
                'kalman_gains_condition_number': float(np.linalg.cond(self._kalman_gains)),
            })

        return summary