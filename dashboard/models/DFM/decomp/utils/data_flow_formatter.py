# -*- coding: utf-8 -*-
"""
数据流格式化器

将新闻贡献数据格式化为按日期分组的数据流结构，
用于在UI中展示类似纽约联储的Data Flow表格。
"""

import pandas as pd
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from collections import defaultdict
from datetime import datetime

from .helpers import normalize_variable_name
from .constants import DEFAULT_INDUSTRY
from ..core.news_impact_calculator import NewsContribution

if TYPE_CHECKING:
    from .industry_aggregator import IndustryAggregator


class DataFlowFormatter:
    """
    数据流格式化器

    将新闻贡献数据转换为分层结构（日期 > 发布时间 > 变量），
    支持在Streamlit中使用expander展示。
    """

    def __init__(
        self,
        var_industry_map: Optional[Dict[str, str]] = None,
        industry_aggregator: Optional['IndustryAggregator'] = None
    ):
        """
        初始化数据流格式化器

        Args:
            var_industry_map: 变量名到行业的映射字典
            industry_aggregator: 行业聚合器实例(可选,优先使用)
        """
        self.var_industry_map = var_industry_map or {}
        self._industry_aggregator = industry_aggregator

    def _get_industry(self, variable_name: str) -> str:
        """
        获取变量所属行业

        Args:
            variable_name: 变量名

        Returns:
            行业名称
        """
        if self._industry_aggregator:
            return self._industry_aggregator.get_industry(variable_name)
        normalized_name = normalize_variable_name(variable_name)
        return self.var_industry_map.get(normalized_name, DEFAULT_INDUSTRY)

    def format_data_flow(
        self,
        contributions: List[NewsContribution],
        nowcast_series: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        格式化为数据流结构

        Args:
            contributions: 新闻贡献列表
            nowcast_series: Nowcast时间序列（可选，用于显示累积值）

        Returns:
            数据流列表，格式：
            [
                {
                    'date': '2025-10-24',
                    'nowcast_value': 2.35,
                    'releases': [
                        {
                            'time': '8:35AM',
                            'variable': 'CPI: All items less food and energy',
                            'industry': 'Prices',
                            'actual': 0.23,
                            'impact': 0.01,
                            'is_positive': True
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        if not contributions:
            return []

        # 按日期分组
        date_groups = defaultdict(list)
        for contrib in contributions:
            date_key = contrib.release_date.strftime('%Y-%m-%d')
            date_groups[date_key].append(contrib)

        # 排序日期
        sorted_dates = sorted(date_groups.keys(), reverse=True)  # 最新日期在前

        # 构建数据流
        data_flow = []
        for date_str in sorted_dates:
            contribs_at_date = date_groups[date_str]

            # 获取该日期的Nowcast值（如果有）
            nowcast_value = None
            if nowcast_series is not None:
                try:
                    date_ts = pd.to_datetime(date_str)
                    # 找到最接近的日期
                    if date_ts in nowcast_series.index:
                        nowcast_value = float(nowcast_series.loc[date_ts])
                    else:
                        # 找到不晚于该日期的最后一个值
                        earlier_dates = nowcast_series.index[nowcast_series.index <= date_ts]
                        if len(earlier_dates) > 0:
                            nowcast_value = float(nowcast_series.loc[earlier_dates[-1]])
                except Exception as e:
                    print(f"[DataFlowFormatter] 无法获取{date_str}的nowcast值: {e}")

            # 格式化发布数据
            releases = []
            for contrib in sorted(contribs_at_date, key=lambda x: x.release_date):
                release = {
                    'time': contrib.release_date.strftime('%H:%M'),
                    'variable': contrib.variable_name,
                    'industry': self._get_industry(contrib.variable_name),
                    'actual': float(contrib.observed_value),
                    'contribution': float(contrib.contribution_pct),
                    'impact': float(contrib.impact_value),
                    'is_positive': contrib.impact_value > 0
                }
                releases.append(release)

            date_entry = {
                'date': date_str,
                'nowcast_value': nowcast_value,
                'releases': releases
            }

            data_flow.append(date_entry)

        return data_flow

    def format_as_dataframe(
        self,
        contributions: List[NewsContribution]
    ) -> pd.DataFrame:
        """
        格式化为DataFrame

        Args:
            contributions: 新闻贡献列表

        Returns:
            DataFrame，包含所有发布信息
        """
        if not contributions:
            return pd.DataFrame()

        data = []
        for contrib in contributions:
            row = {
                '发布日期': contrib.release_date.strftime('%Y-%m-%d'),
                '发布时间': contrib.release_date.strftime('%H:%M'),
                '变量名称': contrib.variable_name,
                '所属行业': self._get_industry(contrib.variable_name),
                '观测值': contrib.observed_value,
                '影响值': contrib.impact_value,
                '贡献度(%)': contrib.contribution_pct,
                '影响方向': '正向' if contrib.impact_value > 0 else '负向' if contrib.impact_value < 0 else '中性'
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values(by='发布日期', ascending=False)

        return df

    def group_by_date(
        self,
        contributions: List[NewsContribution]
    ) -> Dict[str, List[NewsContribution]]:
        """
        按日期分组贡献数据

        Args:
            contributions: 新闻贡献列表

        Returns:
            日期到贡献列表的映射
        """
        date_groups = defaultdict(list)
        for contrib in contributions:
            date_key = contrib.release_date.strftime('%Y-%m-%d')
            date_groups[date_key].append(contrib)

        return dict(date_groups)

    def get_summary_by_date(
        self,
        contributions: List[NewsContribution]
    ) -> List[Dict[str, Any]]:
        """
        获取按日期的摘要统计

        Args:
            contributions: 新闻贡献列表

        Returns:
            日期摘要列表
        """
        date_groups = self.group_by_date(contributions)

        summaries = []
        for date_str in sorted(date_groups.keys(), reverse=True):
            contribs = date_groups[date_str]

            total_impact = sum(c.impact_value for c in contribs)
            positive_impact = sum(c.impact_value for c in contribs if c.impact_value > 0)
            negative_impact = sum(c.impact_value for c in contribs if c.impact_value < 0)

            summary = {
                'date': date_str,
                'release_count': len(contribs),
                'total_impact': total_impact,
                'positive_impact': positive_impact,
                'negative_impact': negative_impact
            }

            summaries.append(summary)

        return summaries

    def format_for_streamlit_display(
        self,
        contributions: List[NewsContribution],
        nowcast_series: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        格式化为Streamlit显示友好的数据结构

        Args:
            contributions: 新闻贡献列表
            nowcast_series: Nowcast时间序列

        Returns:
            包含格式化数据的字典
        """
        data_flow = self.format_data_flow(contributions, nowcast_series)
        date_summaries = self.get_summary_by_date(contributions)

        return {
            'data_flow': data_flow,
            'date_summaries': date_summaries,
            'total_dates': len(data_flow),
            'total_releases': len(contributions)
        }

    def create_release_timeline(
        self,
        contributions: List[NewsContribution]
    ) -> List[Dict[str, Any]]:
        """
        创建发布时间线

        Args:
            contributions: 新闻贡献列表

        Returns:
            时间线事件列表
        """
        timeline = []

        for contrib in sorted(contributions, key=lambda x: x.release_date):
            event = {
                'timestamp': contrib.release_date.isoformat(),
                'date': contrib.release_date.strftime('%Y-%m-%d'),
                'time': contrib.release_date.strftime('%H:%M'),
                'variable': contrib.variable_name,
                'industry': self._get_industry(contrib.variable_name),
                'impact': float(contrib.impact_value),
                'description': f"{contrib.variable_name}: {contrib.observed_value:.2f} (影响: {contrib.impact_value:+.4f})"
            }
            timeline.append(event)

        return timeline

    def filter_by_industry(
        self,
        contributions: List[NewsContribution],
        industry: str
    ) -> List[NewsContribution]:
        """
        按行业筛选贡献数据

        Args:
            contributions: 新闻贡献列表
            industry: 行业名称

        Returns:
            筛选后的贡献列表
        """
        filtered = [
            c for c in contributions
            if self._get_industry(c.variable_name) == industry
        ]
        return filtered

    def filter_by_date_range(
        self,
        contributions: List[NewsContribution],
        start_date: str,
        end_date: str
    ) -> List[NewsContribution]:
        """
        按日期范围筛选贡献数据

        Args:
            contributions: 新闻贡献列表
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            筛选后的贡献列表
        """
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        filtered = [
            c for c in contributions
            if start_ts <= c.release_date <= end_ts
        ]
        return filtered
