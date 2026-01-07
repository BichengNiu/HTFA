# -*- coding: utf-8 -*-
"""
新闻影响计算器

专门处理新闻项的边际贡献分析，包括贡献度排名、
正负影响分解和关键驱动变量识别。
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.exceptions import ComputationError, ValidationError
from ..utils.helpers import detect_outliers
from ..utils.constants import (
    KEY_DRIVERS_CONTRIBUTION_THRESHOLD,
    PRIMARY_DRIVERS_RANK_THRESHOLD,
    SECONDARY_DRIVERS_RANK_THRESHOLD,
    STABLE_POSITIVE_RATIO_THRESHOLD,
    STABLE_NEGATIVE_RATIO_THRESHOLD
)
from .impact_analyzer import ImpactResult, SequentialImpactResult, DataRelease

logger = logging.getLogger(__name__)


@dataclass
class NewsContribution:
    """新闻贡献项"""
    variable_name: str
    impact_value: float
    contribution_pct: float
    is_positive: bool
    release_date: pd.Timestamp
    observed_value: float
    expected_value: float
    kalman_weight: float
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class ContributionSummary:
    """贡献汇总信息"""
    total_impact: float
    positive_impact: float
    negative_impact: float
    net_impact: float
    positive_contributions: List[NewsContribution]
    negative_contributions: List[NewsContribution]
    contribution_count: int
    unique_variables: int


class NewsImpactCalculator:
    """
    新闻影响计算器

    专门分析数据发布对nowcast的边际贡献，提供详细的贡献度分析
    和变量排名功能。
    """

    def __init__(self, impact_analyzer):
        """
        初始化新闻影响计算器

        Args:
            impact_analyzer: 影响分析器实例
        """
        self.analyzer = impact_analyzer

    def calculate_news_contributions(
        self,
        releases: List[DataRelease],
        target_date: pd.Timestamp
    ) -> List[NewsContribution]:
        """
        计算各数据发布的贡献

        Args:
            releases: 数据发布列表
            target_date: 目标日期

        Returns:
            新闻贡献列表

        Raises:
            ComputationError: 计算失败时抛出
        """
        try:
            # 获取时序影响结果
            sequential_result = self.analyzer.analyze_sequential_impacts(releases, target_date)

            # 转换为新闻贡献格式
            contributions = []
            total_abs_impact = sum(abs(imp.impact_on_target) for imp in sequential_result.individual_impacts)

            for impact_result in sequential_result.individual_impacts:
                release = impact_result.release

                # 计算贡献百分比
                contribution_pct = (
                    abs(impact_result.impact_on_target) / total_abs_impact * 100
                    if total_abs_impact > 0 else 0
                )

                contribution = NewsContribution(
                    variable_name=release.variable_name,
                    impact_value=impact_result.impact_on_target,
                    contribution_pct=contribution_pct,
                    is_positive=impact_result.impact_on_target > 0,
                    release_date=release.timestamp,
                    observed_value=release.observed_value,
                    expected_value=release.expected_value,
                    kalman_weight=impact_result.kalman_weight,
                    confidence_interval=impact_result.confidence_interval
                )

                contributions.append(contribution)

            logger.info(f"计算了 {len(contributions)} 个新闻贡献")
            return contributions

        except Exception as e:
            raise ComputationError(f"计算新闻贡献失败: {str(e)}", "news_contributions_calculation")

    def rank_variables_by_impact(
        self,
        contributions: List[NewsContribution],
        method: str = "total_impact"
    ) -> pd.DataFrame:
        """
        按影响度对变量进行排名

        Args:
            contributions: 新闻贡献列表
            method: 排名方法 ("total_impact", "avg_impact", "contribution_pct", "frequency")

        Returns:
            排名数据框

        Raises:
            ComputationError: 排名失败时抛出
        """
        try:
            # 按变量分组
            variable_data = {}
            for contrib in contributions:
                var_name = contrib.variable_name
                if var_name not in variable_data:
                    variable_data[var_name] = []
                variable_data[var_name].append(contrib)

            # 计算每个变量的统计指标
            ranking_data = []
            for var_name, var_contributions in variable_data.items():
                total_impact = sum(c.impact_value for c in var_contributions)
                avg_impact = total_impact / len(var_contributions) if var_contributions else 0
                total_contribution_pct = sum(c.contribution_pct for c in var_contributions)
                positive_count = sum(1 for c in var_contributions if c.is_positive)
                negative_count = len(var_contributions) - positive_count

                # 计算一致性（影响的方差）
                impact_values = [c.impact_value for c in var_contributions]
                consistency = np.std(impact_values) if len(impact_values) > 1 else 0

                # 计算最大单次影响
                max_single_impact = max(abs(c.impact_value) for c in var_contributions)

                ranking_data.append({
                    'variable_name': var_name,
                    'total_impact': total_impact,
                    'avg_impact': avg_impact,
                    'total_contribution_pct': total_contribution_pct,
                    'release_count': len(var_contributions),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'consistency': consistency,
                    'max_single_impact': max_single_impact,
                    'impact_ratio': positive_count / len(var_contributions) if var_contributions else 0
                })

            # 创建数据框
            ranking_df = pd.DataFrame(ranking_data)

            # 如果没有数据，返回空DataFrame（带列名）
            if len(ranking_df) == 0:
                logger.warning("contributions为空，返回空排名DataFrame")
                empty_df = pd.DataFrame(columns=[
                    'rank', 'variable_name', 'total_impact', 'avg_impact',
                    'total_contribution_pct', 'release_count', 'positive_count',
                    'negative_count', 'consistency', 'max_single_impact', 'impact_ratio'
                ])
                return empty_df

            # 根据方法排名
            if method == "total_impact":
                ranking_df = ranking_df.sort_values('total_impact', key=abs, ascending=False)
            elif method == "avg_impact":
                ranking_df = ranking_df.sort_values('avg_impact', key=abs, ascending=False)
            elif method == "contribution_pct":
                ranking_df = ranking_df.sort_values('total_contribution_pct', ascending=False)
            elif method == "frequency":
                ranking_df = ranking_df.sort_values('release_count', ascending=False)
            else:
                raise ValidationError(f"未知的排名方法: {method}")

            # 添加排名列
            ranking_df['rank'] = range(1, len(ranking_df) + 1)

            logger.info(f"变量排名完成: {method} 方法")
            return ranking_df

        except Exception as e:
            raise ComputationError(f"变量排名失败: {str(e)}", "variable_ranking")

    def calculate_positive_negative_split(
        self,
        contributions: List[NewsContribution]
    ) -> Dict[str, Any]:
        """
        计算正负影响分解

        Args:
            contributions: 新闻贡献列表

        Returns:
            正负影响分解结果

        Raises:
            ComputationError: 计算失败时抛出
        """
        try:
            positive_contributions = [c for c in contributions if c.is_positive]
            negative_contributions = [c for c in contributions if not c.is_positive]

            total_positive = sum(c.impact_value for c in positive_contributions)
            total_negative = sum(c.impact_value for c in negative_contributions)
            net_impact = total_positive + total_negative

            # 计算占比（基于绝对值总和）
            # total_positive是正数，total_negative是负数（负影响之和）
            total_abs = total_positive + abs(total_negative)
            positive_ratio = total_positive / total_abs if total_abs > 0 else 0
            negative_ratio = abs(total_negative) / total_abs if total_abs > 0 else 0

            # 按变量分解正负影响
            positive_by_var = {}
            negative_by_var = {}

            for c in positive_contributions:
                var_name = c.variable_name
                positive_by_var[var_name] = positive_by_var.get(var_name, 0) + c.impact_value

            for c in negative_contributions:
                var_name = c.variable_name
                negative_by_var[var_name] = negative_by_var.get(var_name, 0) + c.impact_value

            # 识别主要贡献变量
            top_positive_vars = sorted(positive_by_var.items(), key=lambda x: x[1], reverse=True)[:5]
            top_negative_vars = sorted(negative_by_var.items(), key=lambda x: x[1])[:5]

            split_result = {
                'total_impact': net_impact,
                'positive_impact': total_positive,
                'negative_impact': total_negative,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'positive_count': len(positive_contributions),
                'negative_count': len(negative_contributions),
                'positive_by_variable': positive_by_var,
                'negative_by_variable': negative_by_var,
                'top_positive_variables': top_positive_vars,
                'top_negative_variables': top_negative_vars,
                'balance_factor': abs(total_positive / total_negative) if total_negative != 0 else float('inf')
            }

            logger.info(f"正负影响分解: 正向={total_positive:.4f}, 负向={total_negative:.4f}")
            return split_result

        except Exception as e:
            raise ComputationError(f"正负影响分解失败: {str(e)}", "positive_negative_split")

    def identify_key_drivers(
        self,
        contributions: List[NewsContribution],
        threshold: float = KEY_DRIVERS_CONTRIBUTION_THRESHOLD,
        top_n: int = SECONDARY_DRIVERS_RANK_THRESHOLD
    ) -> Dict[str, Any]:
        """
        识别关键驱动变量

        Args:
            contributions: 新闻贡献列表
            threshold: 贡献度阈值
            top_n: 返回前N个关键变量

        Returns:
            关键驱动变量分析结果

        Raises:
            ComputationError: 识别失败时抛出
        """
        try:
            # 获取变量排名
            ranking_df = self.rank_variables_by_impact(contributions, "contribution_pct")

            # 如果没有数据，返回空结果
            if len(ranking_df) == 0:
                logger.warning("无数据，返回空关键驱动结果")
                return {
                    'key_drivers': ranking_df,
                    'primary_drivers': ranking_df.copy(),
                    'secondary_drivers': ranking_df.copy(),
                    'driver_count': 0,
                    'primary_count': 0,
                    'secondary_count': 0,
                    'threshold_used': threshold,
                    'total_contribution_coverage': 0.0,
                    'driver_patterns': {},
                    'identification_time': datetime.now().isoformat()
                }

            # 筛选关键驱动变量
            key_drivers = ranking_df[
                (ranking_df['total_contribution_pct'] >= threshold * 100) |
                (ranking_df['rank'] <= top_n)
            ].copy()

            # 计算关键度评分（避免除以0）
            max_release_count = ranking_df['release_count'].max()
            if max_release_count == 0:
                max_release_count = 1  # 避免除以0

            key_drivers['key_score'] = (
                key_drivers['total_contribution_pct'] * 0.4 +
                (key_drivers['release_count'] / max_release_count) * 100 * 0.3 +
                (key_drivers['impact_ratio']) * 100 * 0.3
            )

            # 按关键度评分重新排序
            key_drivers = key_drivers.sort_values('key_score', ascending=False)

            # 分类关键驱动
            primary_drivers = key_drivers[key_drivers['rank'] <= PRIMARY_DRIVERS_RANK_THRESHOLD].copy()
            secondary_drivers = key_drivers[
                (key_drivers['rank'] > PRIMARY_DRIVERS_RANK_THRESHOLD) &
                (key_drivers['rank'] <= SECONDARY_DRIVERS_RANK_THRESHOLD)
            ].copy()

            # 分析驱动模式
            driver_patterns = self._analyze_driver_patterns(key_drivers, contributions)

            key_drivers_result = {
                'key_drivers': key_drivers,
                'primary_drivers': primary_drivers,
                'secondary_drivers': secondary_drivers,
                'driver_count': len(key_drivers),
                'primary_count': len(primary_drivers),
                'secondary_count': len(secondary_drivers),
                'threshold_used': threshold,
                'total_contribution_coverage': key_drivers['total_contribution_pct'].sum(),
                'driver_patterns': driver_patterns,
                'identification_time': datetime.now().isoformat()
            }

            logger.info(f"识别关键驱动: {len(key_drivers)} 个变量")
            return key_drivers_result

        except Exception as e:
            raise ComputationError(f"关键驱动识别失败: {str(e)}", "key_drivers_identification")

    def analyze_temporal_patterns(
        self,
        contributions: List[NewsContribution],
        time_window: str = "W"
    ) -> Dict[str, Any]:
        """
        分析时间模式

        Args:
            contributions: 新闻贡献列表
            time_window: 时间窗口 ("D", "W", "M", "Q")

        Returns:
            时间模式分析结果

        Raises:
            ComputationError: 分析失败时抛出
        """
        try:
            # 创建时间序列数据
            contribution_data = []
            for c in contributions:
                contribution_data.append({
                    'timestamp': c.release_date,
                    'impact_value': c.impact_value,
                    'variable_name': c.variable_name,
                    'is_positive': c.is_positive,
                    'contribution_pct': c.contribution_pct
                })

            df = pd.DataFrame(contribution_data)
            df.set_index('timestamp', inplace=True)

            # 按时间窗口聚合
            if time_window == "D":
                grouped = df.resample('D')
            elif time_window == "W":
                grouped = df.resample('W')
            elif time_window == "M":
                grouped = df.resample('M')
            elif time_window == "Q":
                grouped = df.resample('Q')
            else:
                raise ValidationError(f"不支持的时间窗口: {time_window}")

            # 计算各时间窗口的统计指标
            temporal_stats = []
            for timestamp, group in grouped:
                if len(group) > 0:
                    stats = {
                        'timestamp': timestamp,
                        'count': len(group),
                        'total_impact': group['impact_value'].sum(),
                        'avg_impact': group['impact_value'].mean(),
                        'positive_count': group['is_positive'].sum(),
                        'negative_count': len(group) - group['is_positive'].sum(),
                        'unique_variables': group['variable_name'].nunique(),
                        'max_single_impact': group['impact_value'].abs().max(),
                        'total_contribution_pct': group['contribution_pct'].sum()
                    }
                    temporal_stats.append(stats)

            temporal_df = pd.DataFrame(temporal_stats)
            temporal_df.set_index('timestamp', inplace=True)

            # 识别异常活跃期
            outlier_mask = detect_outliers(temporal_df['count'], method="zscore", threshold=2.0)
            active_periods = temporal_df[outlier_mask]

            # 计算趋势
            if len(temporal_df) > 1:
                impact_trend = np.polyfit(range(len(temporal_df)), temporal_df['total_impact'], 1)[0]
                frequency_trend = np.polyfit(range(len(temporal_df)), temporal_df['count'], 1)[0]
            else:
                impact_trend = 0
                frequency_trend = 0

            temporal_patterns = {
                'temporal_stats': temporal_df,
                'active_periods': active_periods,
                'time_window': time_window,
                'impact_trend': impact_trend,
                'frequency_trend': frequency_trend,
                'peak_period': temporal_df.loc[temporal_df['total_impact'].abs().idxmax()].to_dict() if len(temporal_df) > 0 else None,
                'most_active_period': temporal_df.loc[temporal_df['count'].idxmax()].to_dict() if len(temporal_df) > 0 else None
            }

            logger.info(f"时间模式分析: {time_window} 窗口")
            return temporal_patterns

        except Exception as e:
            raise ComputationError(f"时间模式分析失败: {str(e)}", "temporal_patterns_analysis")

    def _analyze_driver_patterns(
        self,
        key_drivers: pd.DataFrame,
        contributions: List[NewsContribution]
    ) -> Dict[str, Any]:
        """分析驱动变量模式"""
        try:
            patterns = {}

            # 持续性模式：频繁出现的影响
            frequent_drivers = key_drivers[key_drivers['release_count'] > key_drivers['release_count'].median()]
            patterns['frequent_drivers'] = frequent_drivers['variable_name'].tolist()

            # 高影响模式：单次影响很大的变量
            high_impact_drivers = key_drivers[key_drivers['max_single_impact'] > key_drivers['max_single_impact'].median()]
            patterns['high_impact_drivers'] = high_impact_drivers['variable_name'].tolist()

            # 稳定性模式：影响方向一致的变量
            stable_drivers = key_drivers[key_drivers['impact_ratio'] >= STABLE_POSITIVE_RATIO_THRESHOLD]
            patterns['stable_positive_drivers'] = stable_drivers['variable_name'].tolist()

            unstable_drivers = key_drivers[key_drivers['impact_ratio'] <= STABLE_NEGATIVE_RATIO_THRESHOLD]
            patterns['stable_negative_drivers'] = unstable_drivers['variable_name'].tolist()

            return patterns

        except Exception as e:
            logger.warning(f"驱动模式分析警告: {str(e)}")
            return {}

    def get_comprehensive_summary(
        self,
        contributions: List[NewsContribution]
    ) -> Dict[str, Any]:
        """
        获取综合摘要报告

        Args:
            contributions: 新闻贡献列表

        Returns:
            综合摘要报告
        """
        try:
            # 基础统计
            total_impact = sum(c.impact_value for c in contributions)
            positive_count = sum(1 for c in contributions if c.is_positive)
            negative_count = len(contributions) - positive_count
            unique_variables = len(set(c.variable_name for c in contributions))

            # 正负影响分解
            pn_split = self.calculate_positive_negative_split(contributions)

            # 变量排名
            ranking_df = self.rank_variables_by_impact(contributions)

            # 关键驱动
            key_drivers = self.identify_key_drivers(contributions)

            # 时间模式
            temporal_patterns = self.analyze_temporal_patterns(contributions)

            summary = {
                'basic_stats': {
                    'total_contributions': len(contributions),
                    'total_impact': total_impact,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'unique_variables': unique_variables,
                    'avg_impact': total_impact / len(contributions) if contributions else 0
                },
                'positive_negative_split': pn_split,
                'variable_ranking': ranking_df,
                'key_drivers': key_drivers,
                'temporal_patterns': temporal_patterns,
                'analysis_timestamp': datetime.now().isoformat()
            }

            return summary

        except Exception as e:
            raise ComputationError(f"综合摘要生成失败: {str(e)}", "comprehensive_summary")