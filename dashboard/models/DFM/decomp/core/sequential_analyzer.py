# -*- coding: utf-8 -*-
"""
时序影响分析器

分析数据发布的时序累积影响，包括反事实分析、
影响路径分析和预测路径模拟。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..utils.exceptions import ComputationError, ValidationError
from ..utils.helpers import smooth_time_series, calculate_confidence_intervals
from .impact_analyzer import ImpactAnalyzer, DataRelease, ImpactResult, SequentialImpactResult


class ScenarioType(Enum):
    """分析场景类型"""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    COUNTERFACTUAL = "counterfactual"
    SENSITIVITY = "sensitivity"


@dataclass
class ImpactPath:
    """影响路径"""
    timestamp: pd.Timestamp
    baseline_value: float
    current_value: float
    cumulative_impact: float
    marginal_impact: float
    variable_name: str
    path_type: str  # "increment", "decrement", "volatile"


@dataclass
class CounterfactualScenario:
    """反事实场景"""
    scenario_name: str
    description: str
    excluded_releases: List[DataRelease]
    baseline_result: SequentialImpactResult
    counterfactual_result: SequentialImpactResult
    impact_difference: float
    difference_percentage: float


@dataclass
class SensitivityAnalysis:
    """敏感性分析结果"""
    variable_name: str
    sensitivity_coefficient: float
    impact_range: Tuple[float, float]
    confidence_level: float
    test_scenarios: List[Dict[str, Any]]


class SequentialAnalyzer:
    """
    时序影响分析器

    专门分析数据发布的时序累积影响，提供反事实分析、
    影响路径追踪和多种场景模拟功能。
    """

    def __init__(self, impact_analyzer: ImpactAnalyzer):
        """
        初始化时序影响分析器

        Args:
            impact_analyzer: 影响分析器实例
        """
        self.analyzer = impact_analyzer
        self._path_cache: Optional[Dict[str, List[ImpactPath]]] = None
        self._scenario_cache: Optional[Dict[str, CounterfactualScenario]] = None

    def analyze_cumulative_impact_path(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp,
        smoothing_method: str = "none"
    ) -> List[ImpactPath]:
        """
        分析累积影响路径

        Args:
            releases: 数据发布列表
            target_date: 目标日期
            smoothing_method: 平滑方法 ("none", "rolling", "exponential")

        Returns:
            影响路径列表

        Raises:
            ComputationError: 分析失败时抛出
        """
        try:
            # 按时间排序数据发布
            sorted_releases = sorted(releases, key=lambda x: x.timestamp)

            # 获取基准值
            baseline_value = self.analyzer.extractor.compute_baseline_prediction(target_date)

            # 计算影响路径
            paths = []
            current_value = baseline_value
            cumulative_impact = 0.0

            for release in sorted_releases:
                if release.timestamp > target_date:
                    break

                # 计算单次影响
                impact_result = self.analyzer.calculate_single_release_impact(release)

                # 更新累积影响
                cumulative_impact += impact_result.impact_on_target
                current_value = baseline_value + cumulative_impact

                # 确定路径类型
                if impact_result.impact_on_target > 0:
                    if paths and paths[-1].marginal_impact > 0:
                        path_type = "increment"
                    else:
                        path_type = "positive_turn"
                elif impact_result.impact_on_target < 0:
                    if paths and paths[-1].marginal_impact < 0:
                        path_type = "decrement"
                    else:
                        path_type = "negative_turn"
                else:
                    path_type = "neutral"

                path = ImpactPath(
                    timestamp=release.timestamp,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    cumulative_impact=cumulative_impact,
                    marginal_impact=impact_result.impact_on_target,
                    variable_name=release.variable_name,
                    path_type=path_type
                )

                paths.append(path)

            # 应用平滑
            if smoothing_method != "none":
                paths = self._smooth_impact_paths(paths, smoothing_method)

            print(f"[SequentialAnalyzer] 累积影响路径分析: {len(paths)} 个路径点")
            return paths

        except Exception as e:
            raise ComputationError(f"累积影响路径分析失败: {str(e)}", "cumulative_path_analysis")

    def simulate_counterfactual_scenarios(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp,
        scenarios: List[Dict[str, Any]]
    ) -> List[CounterfactualScenario]:
        """
        模拟反事实场景

        Args:
            releases: 完整的数据发布列表
            target_date: 目标日期
            scenarios: 场景配置列表

        Returns:
            反事实场景列表

        Raises:
            ComputationError: 模拟失败时抛出
        """
        try:
            # 计算基准结果
            baseline_result = self.analyzer.analyze_sequential_impacts(releases, target_date)

            counterfactual_scenarios = []

            for scenario_config in scenarios:
                scenario_name = scenario_config.get('name', 'unnamed')
                description = scenario_config.get('description', '')
                exclusion_criteria = scenario_config.get('exclusion_criteria', [])

                # 根据条件筛选要排除的数据发布
                excluded_releases = self._filter_releases_by_criteria(releases, exclusion_criteria)

                # 计算反事实结果
                counterfactual_result = self.analyzer.analyze_sequential_impacts(
                    [r for r in releases if r not in excluded_releases],
                    target_date
                )

                # 计算差异
                impact_difference = baseline_result.total_impact - counterfactual_result.total_impact
                difference_percentage = (
                    abs(impact_difference) / abs(baseline_result.total_impact) * 100
                    if baseline_result.total_impact != 0 else 0
                )

                scenario = CounterfactualScenario(
                    scenario_name=scenario_name,
                    description=description,
                    excluded_releases=excluded_releases,
                    baseline_result=baseline_result,
                    counterfactual_result=counterfactual_result,
                    impact_difference=impact_difference,
                    difference_percentage=difference_percentage
                )

                counterfactual_scenarios.append(scenario)

            print(f"[SequentialAnalyzer] 反事实场景模拟: {len(counterfactual_scenarios)} 个场景")
            return counterfactual_scenarios

        except Exception as e:
            raise ComputationError(f"反事实场景模拟失败: {str(e)}", "counterfactual_simulation")

    def analyze_early_vs_late_impact(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp,
        split_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        分析早期vs晚期数据发布的影响差异

        Args:
            releases: 数据发布列表
            target_date: 目标日期
            split_ratio: 分割比例（早期数据占比）

        Returns:
            早期晚期影响分析结果

        Raises:
            ComputationError: 分析失败时抛出
        """
        try:
            # 按时间排序
            sorted_releases = sorted(releases, key=lambda x: x.timestamp)
            valid_releases = [r for r in sorted_releases if r.timestamp <= target_date]

            if len(valid_releases) < 2:
                raise ValidationError("数据发布数量不足，无法进行早期晚期分析")

            # 分割数据
            split_index = int(len(valid_releases) * split_ratio)
            early_releases = valid_releases[:split_index]
            late_releases = valid_releases[split_index:]

            # 计算早期影响
            early_result = self.analyzer.analyze_sequential_impacts(early_releases, target_date)

            # 计算晚期影响（在早期影响基础上的增量）
            baseline_for_late = early_result.final_value
            late_result = self.analyzer.analyze_sequential_impacts(late_releases, target_date)

            # 分析差异
            early_vs_late = {
                'early_period': {
                    'start_date': early_releases[0].timestamp if early_releases else None,
                    'end_date': early_releases[-1].timestamp if early_releases else None,
                    'release_count': len(early_releases),
                    'total_impact': early_result.total_impact,
                    'final_value': early_result.final_value,
                    'avg_impact_per_release': early_result.total_impact / len(early_releases) if early_releases else 0
                },
                'late_period': {
                    'start_date': late_releases[0].timestamp if late_releases else None,
                    'end_date': late_releases[-1].timestamp if late_releases else None,
                    'release_count': len(late_releases),
                    'total_impact': late_result.total_impact,
                    'final_value': late_result.final_value,
                    'avg_impact_per_release': late_result.total_impact / len(late_releases) if late_releases else 0
                },
                'comparison': {
                    'impact_difference': late_result.total_impact - early_result.total_impact,
                    'efficiency_ratio': (
                        (late_result.total_impact / len(late_releases)) /
                        (early_result.total_impact / len(early_releases))
                        if early_releases and late_releases and early_result.total_impact != 0 else 0
                    ),
                    'early_contribution_pct': (
                        abs(early_result.total_impact) / (abs(early_result.total_impact) + abs(late_result.total_impact)) * 100
                        if (early_result.total_impact + late_result.total_impact) != 0 else 50
                    ),
                    'late_contribution_pct': (
                        abs(late_result.total_impact) / (abs(early_result.total_impact) + abs(late_result.total_impact)) * 100
                        if (early_result.total_impact + late_result.total_impact) != 0 else 50
                    )
                },
                'split_ratio': split_ratio
            }

            print(f"[SequentialAnalyzer] 早期晚期分析: 早期={len(early_releases)}, 晚期={len(late_releases)}")
            return early_vs_late

        except Exception as e:
            raise ComputationError(f"早期晚期影响分析失败: {str(e)}", "early_late_analysis")

    def calculate_impact_volatility(
        self,
        releases: List[DataRelease],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """
        计算影响的波动性

        Args:
            releases: 数据发布列表
            window_size: 滑动窗口大小

        Returns:
            波动性分析结果

        Raises:
            ComputationError: 计算失败时抛出
        """
        try:
            # 计算各次发布的影响
            impacts = []
            timestamps = []

            for release in sorted(releases, key=lambda x: x.timestamp):
                impact_result = self.analyzer.calculate_single_release_impact(release)
                impacts.append(impact_result.impact_on_target)
                timestamps.append(release.timestamp)

            if len(impacts) < window_size:
                raise ValidationError(f"数据发布数量不足，需要至少 {window_size} 个")

            impact_series = pd.Series(impacts, index=timestamps)

            # 计算滚动统计
            rolling_std = impact_series.rolling(window=window_size).std()
            rolling_mean = impact_series.rolling(window=window_size).mean()
            rolling_var = impact_series.rolling(window=window_size).var()

            # 计算波动性指标
            volatility_metrics = {
                'overall_volatility': impact_series.std(),
                'mean_volatility': rolling_std.mean(),
                'max_volatility': rolling_std.max(),
                'min_volatility': rolling_std.min(),
                'volatility_trend': np.polyfit(range(len(rolling_std.dropna())), rolling_std.dropna(), 1)[0],
                'relative_volatility': rolling_std / rolling_mean.abs().replace(0, np.nan),
                'volatility_periods': self._identify_volatility_periods(rolling_std, impact_series)
            }

            # 按变量分组计算波动性
            variable_volatility = {}
            variable_groups = {}
            for release in sorted(releases, key=lambda x: x.timestamp):
                var_name = release.variable_name
                if var_name not in variable_groups:
                    variable_groups[var_name] = []
                impact_result = self.analyzer.calculate_single_release_impact(release)
                variable_groups[var_name].append(impact_result.impact_on_target)

            for var_name, var_impacts in variable_groups.items():
                if len(var_impacts) >= 2:
                    var_series = pd.Series(var_impacts)
                    variable_volatility[var_name] = {
                        'volatility': var_series.std(),
                        'mean_impact': var_series.mean(),
                        'count': len(var_impacts),
                        'coefficient_of_variation': var_series.std() / abs(var_series.mean()) if var_series.mean() != 0 else np.inf
                    }

            volatility_analysis = {
                'metrics': volatility_metrics,
                'by_variable': variable_volatility,
                'time_series': {
                    'timestamps': timestamps,
                    'impacts': impacts,
                    'rolling_std': rolling_std.tolist(),
                    'rolling_mean': rolling_mean.tolist(),
                    'rolling_var': rolling_var.tolist()
                },
                'window_size': window_size
            }

            print(f"[SequentialAnalyzer] 波动性分析: {len(releases)} 个数据发布")
            return volatility_analysis

        except Exception as e:
            raise ComputationError(f"影响波动性计算失败: {str(e)}", "volatility_calculation")

    def generate_forecast_scenarios(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp,
        forecast_horizon: int = 30,
        scenario_configs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成预测场景

        Args:
            releases: 历史数据发布
            target_date: 当前日期
            forecast_horizon: 预测天数
            scenario_configs: 场景配置

        Returns:
            预测场景结果

        Raises:
            ComputationError: 生成失败时抛出
        """
        try:
            # 默认场景配置
            if scenario_configs is None:
                scenario_configs = [
                    {'name': 'baseline', 'multiplier': 1.0, 'description': '基准情景'},
                    {'name': 'optimistic', 'multiplier': 1.2, 'description': '乐观情景'},
                    {'name': 'pessimistic', 'multiplier': 0.8, 'description': '悲观情景'}
                ]

            # 计算历史基准
            historical_result = self.analyzer.analyze_sequential_impacts(releases, target_date)

            # 生成预测日期序列
            forecast_dates = pd.date_range(
                start=target_date + timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )

            # 分析历史模式
            daily_pattern = self._analyze_daily_impact_pattern(releases, target_date)

            scenarios = {}
            for config in scenario_configs:
                scenario_name = config['name']
                multiplier = config['multiplier']

                # 生成情景预测
                forecast_values = []
                cumulative_forecast = historical_result.final_value

                for i, date in enumerate(forecast_dates):
                    # 基于历史模式预测每日影响
                    expected_daily_impact = daily_pattern.get(date.dayofweek, 0.0) * multiplier

                    # 添加随机扰动
                    noise = np.random.normal(0, abs(expected_daily_impact) * 0.1)
                    daily_impact = expected_daily_impact + noise

                    cumulative_forecast += daily_impact
                    forecast_values.append(cumulative_forecast)

                scenarios[scenario_name] = {
                    'config': config,
                    'dates': forecast_dates.tolist(),
                    'values': forecast_values,
                    'final_value': forecast_values[-1] if forecast_values else historical_result.final_value,
                    'total_change': forecast_values[-1] - historical_result.final_value if forecast_values else 0
                }

            forecast_analysis = {
                'historical_baseline': historical_result,
                'scenarios': scenarios,
                'forecast_horizon': forecast_horizon,
                'generation_time': datetime.now().isoformat(),
                'pattern_analysis': daily_pattern
            }

            print(f"[SequentialAnalyzer] 预测场景生成: {len(scenarios)} 个情景")
            return forecast_analysis

        except Exception as e:
            raise ComputationError(f"预测场景生成失败: {str(e)}", "forecast_generation")

    def _smooth_impact_paths(
        self,
        paths: List[ImpactPath],
        method: str
    ) -> List[ImpactPath]:
        """平滑影响路径"""
        try:
            if len(paths) < 3:
                return paths

            # 提取数值序列
            timestamps = [p.timestamp for p in paths]
            current_values = [p.current_value for p in paths]
            cumulative_impacts = [p.cumulative_impact for p in paths]

            # 应用平滑
            smoothed_values = smooth_time_series(
                pd.Series(current_values, index=timestamps),
                method=method,
                window=min(3, len(paths))
            )

            smoothed_cumulative = smooth_time_series(
                pd.Series(cumulative_impacts, index=timestamps),
                method=method,
                window=min(3, len(paths))
            )

            # 更新路径
            smoothed_paths = []
            for i, path in enumerate(paths):
                smoothed_path = ImpactPath(
                    timestamp=path.timestamp,
                    baseline_value=path.baseline_value,
                    current_value=smoothed_values.iloc[i],
                    cumulative_impact=smoothed_cumulative.iloc[i],
                    marginal_impact=path.marginal_impact,
                    variable_name=path.variable_name,
                    path_type=path.path_type
                )
                smoothed_paths.append(smoothed_path)

            return smoothed_paths

        except Exception as e:
            print(f"[SequentialAnalyzer] 路径平滑警告: {str(e)}")
            return paths

    def _filter_releases_by_criteria(
        self,
        releases: List[DataRelease],
        criteria: List[Dict[str, Any]]
    ) -> List[DataRelease]:
        """根据条件筛选数据发布"""
        try:
            excluded_releases = []

            for criterion in criteria:
                criterion_type = criterion.get('type', 'variable_name')
                criterion_value = criterion.get('value')
                operator = criterion.get('operator', 'equals')

                for release in releases:
                    should_exclude = False

                    if criterion_type == 'variable_name':
                        if operator == 'equals' and release.variable_name == criterion_value:
                            should_exclude = True
                        elif operator == 'contains' and criterion_value in release.variable_name:
                            should_exclude = True
                        elif operator == 'in_list' and release.variable_name in criterion_value:
                            should_exclude = True

                    elif criterion_type == 'impact_threshold':
                        # 需要预先计算影响
                        impact_result = self.analyzer.calculate_single_release_impact(release)
                        impact_value = abs(impact_result.impact_on_target)

                        if operator == 'greater_than' and impact_value > criterion_value:
                            should_exclude = True
                        elif operator == 'less_than' and impact_value < criterion_value:
                            should_exclude = True

                    elif criterion_type == 'time_range':
                        start_time = pd.to_datetime(criterion_value.get('start'))
                        end_time = pd.to_datetime(criterion_value.get('end'))
                        if start_time <= release.timestamp <= end_time:
                            should_exclude = True

                    if should_exclude and release not in excluded_releases:
                        excluded_releases.append(release)

            return excluded_releases

        except Exception as e:
            print(f"[SequentialAnalyzer] 筛选条件处理警告: {str(e)}")
            return []

    def _identify_volatility_periods(
        self,
        rolling_std: pd.Series,
        impact_series: pd.Series
    ) -> List[Dict[str, Any]]:
        """识别高波动性时期"""
        try:
            # 计算波动性阈值
            volatility_threshold = rolling_std.mean() + rolling_std.std()

            high_volatility_periods = []
            in_high_volatility = False
            period_start = None

            for timestamp, volatility in rolling_std.items():
                if pd.notna(volatility):
                    if volatility > volatility_threshold and not in_high_volatility:
                        # 进入高波动期
                        in_high_volatility = True
                        period_start = timestamp
                    elif volatility <= volatility_threshold and in_high_volatility:
                        # 离开高波动期
                        in_high_volatility = False
                        if period_start:
                            period_impacts = impact_series.loc[period_start:timestamp]
                            high_volatility_periods.append({
                                'start_date': period_start,
                                'end_date': timestamp,
                                'duration_days': (timestamp - period_start).days,
                                'avg_volatility': rolling_std.loc[period_start:timestamp].mean(),
                                'max_impact': period_impacts.max(),
                                'min_impact': period_impacts.min(),
                                'impact_range': period_impacts.max() - period_impacts.min()
                            })

            return high_volatility_periods

        except Exception as e:
            print(f"[SequentialAnalyzer] 波动期识别警告: {str(e)}")
            return []

    def _analyze_daily_impact_pattern(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp
    ) -> Dict[int, float]:
        """分析每日影响模式"""
        try:
            # 计算各次发布的影响
            daily_impacts = {}
            for release in releases:
                if release.timestamp <= target_date:
                    impact_result = self.analyzer.calculate_single_release_impact(release)
                    day_of_week = release.timestamp.dayofweek
                    if day_of_week not in daily_impacts:
                        daily_impacts[day_of_week] = []
                    daily_impacts[day_of_week].append(impact_result.impact_on_target)

            # 计算平均每日影响
            daily_pattern = {}
            for day_of_week, impacts in daily_impacts.items():
                daily_pattern[day_of_week] = np.mean(impacts) if impacts else 0.0

            return daily_pattern

        except Exception as e:
            print(f"[SequentialAnalyzer] 每日模式分析警告: {str(e)}")
            return {i: 0.0 for i in range(7)}  # 默认无影响

    def get_sequential_analysis_summary(self) -> Dict[str, Any]:
        """
        获取时序分析摘要

        Returns:
            分析摘要字典
        """
        summary = {
            'analyzer_type': 'SequentialAnalyzer',
            'cache_status': {
                'path_cache_size': len(self._path_cache) if self._path_cache else 0,
                'scenario_cache_size': len(self._scenario_cache) if self._scenario_cache else 0
            },
            'available_methods': [
                'cumulative_impact_path',
                'counterfactual_scenarios',
                'early_vs_late_impact',
                'impact_volatility',
                'forecast_scenarios'
            ]
        }

        return summary