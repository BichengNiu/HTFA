# -*- coding: utf-8 -*-
"""
影响瀑布图生成器

生成数据发布影响的瀑布图，清晰展示各次发布对nowcast的
累积影响和贡献度分解。
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import os

from ..utils.exceptions import VisualizationError
from ..core.impact_analyzer import ImpactResult, SequentialImpactResult
from ..core.news_impact_calculator import NewsContribution


class ImpactWaterfallPlotter:
    """
    影响瀑布图生成器

    专门生成展示数据发布影响的瀑布图，包括累积效应、
    正负贡献分离和关键影响因素突出显示等功能。
    """

    def __init__(self):
        self.default_colors = {
            'positive': '#2E8B57',    # 海绿色
            'negative': '#DC143C',    # 深红色
            'neutral': '#708090',     # 石板灰
            'baseline': '#4169E1',    # 皇家蓝
            'total': '#FFD700'        # 金色
        }

    def create_waterfall_chart(
        self,
        contributions: List[NewsContribution],
        title: str = "数据发布影响瀑布图",
        show_details: bool = True,
        top_n: Optional[int] = None
    ) -> go.Figure:
        """
        创建瀑布图

        Args:
            contributions: 新闻贡献列表
            title: 图表标题
            show_details: 是否显示详细信息
            top_n: 显示前N个贡献，None表示全部

        Returns:
            Plotly图表对象

        Raises:
            VisualizationError: 图表生成失败时抛出
        """
        try:
            if not contributions:
                raise VisualizationError("贡献数据为空", "waterfall_chart")

            # 排序和筛选数据
            sorted_contributions = sorted(
                contributions,
                key=lambda x: abs(x.contribution_pct),
                reverse=True
            )

            if top_n is not None:
                sorted_contributions = sorted_contributions[:top_n]

            # 准备瀑布图数据
            waterfall_data = self._prepare_waterfall_data(sorted_contributions)

            # 创建基础图表
            fig = go.Figure()

            # 添加基准线
            fig.add_trace(go.Scatter(
                x=waterfall_data['x_labels'],
                y=waterfall_data['baseline_values'],
                mode='lines',
                line=dict(color=self.default_colors['baseline'], width=2, dash='dash'),
                name='基准值',
                hoverinfo='skip'
            ))

            # 添加累积影响线
            fig.add_trace(go.Scatter(
                x=waterfall_data['x_labels'],
                y=waterfall_data['cumulative_values'],
                mode='lines+markers',
                line=dict(color=self.default_colors['total'], width=3),
                marker=dict(size=6),
                name='累积影响',
                customdata=waterfall_data['custom_data'],
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    '时间: %{customdata[1]}<br>'
                    '累积影响: %{y:.4f}<br>'
                    '<extra></extra>'
                )
            ))

            # 添加贡献柱状图
            fig.add_trace(go.Bar(
                x=waterfall_data['x_labels'],
                y=waterfall_data['contribution_values'],
                marker_color=waterfall_data['colors'],
                name='边际贡献',
                customdata=waterfall_data['custom_data'],
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    '时间: %{customdata[1]}<br>'
                    '观测值: %{customdata[2]:.4f}<br>'
                    '期望值: %{customdata[3]:.4f}<br>'
                    '贡献: %{y:.4f}<br>'
                    '贡献度: %{customdata[4]:.2f}%<br>'
                    '<extra></extra>'
                ),
                text=waterfall_data['contribution_texts'],
                textposition='outside'
            ))

            # 更新布局
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16, color='#2c3e50')
                ),
                xaxis_title="数据发布",
                yaxis_title="Nowcast值",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600,
                template='plotly_white'
            )

            # 添加零线
            fig.add_hline(
                y=0,
                line_dash="dot",
                line_color="gray",
                annotation_text="零线",
                annotation_position="bottom right"
            )

            print(f"[WaterfallPlotter] 瀑布图创建完成: {len(sorted_contributions)} 个贡献")
            return fig

        except Exception as e:
            raise VisualizationError(f"瀑布图创建失败: {str(e)}", "waterfall_chart")

    def create_grouped_waterfall(
        self,
        contributions: List[NewsContribution],
        group_by: str = "variable",
        title: str = "分组影响瀑布图"
    ) -> go.Figure:
        """
        创建分组瀑布图

        Args:
            contributions: 新闻贡献列表
            group_by: 分组方式 ("variable", "time", "sign")
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        try:
            if not contributions:
                raise VisualizationError("贡献数据为空", "grouped_waterfall")

            # 按变量分组
            if group_by == "variable":
                grouped_data = self._group_by_variable(contributions)
            elif group_by == "time":
                grouped_data = self._group_by_time(contributions)
            elif group_by == "sign":
                grouped_data = self._group_by_sign(contributions)
            else:
                raise ValidationError(f"不支持的分组方式: {group_by}")

            # 创建子图
            fig = make_subplots(
                rows=len(grouped_data),
                cols=1,
                subplot_titles=list(grouped_data.keys()),
                vertical_spacing=0.05,
                shared_xaxes=True
            )

            for i, (group_name, group_contributions) in enumerate(grouped_data.items(), 1):
                # 准备组内数据
                waterfall_data = self._prepare_waterfall_data(group_contributions)

                # 添加累积线
                fig.add_trace(
                    go.Scatter(
                        x=waterfall_data['x_labels'],
                        y=waterfall_data['cumulative_values'],
                        mode='lines+markers',
                        line=dict(color=self.default_colors['total'], width=2),
                        marker=dict(size=4),
                        name=f'{group_name} 累积',
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

                # 添加贡献柱
                fig.add_trace(
                    go.Bar(
                        x=waterfall_data['x_labels'],
                        y=waterfall_data['contribution_values'],
                        marker_color=waterfall_data['colors'],
                        name=f'{group_name} 贡献',
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

            # 更新布局
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16, color='#2c3e50')
                ),
                height=200 * len(grouped_data) + 100,
                template='plotly_white'
            )

            print(f"[WaterfallPlotter] 分组瀑布图创建完成: {len(grouped_data)} 个组")
            return fig

        except Exception as e:
            raise VisualizationError(f"分组瀑布图创建失败: {str(e)}", "grouped_waterfall")

    def create_comparison_waterfall(
        self,
        baseline_contributions: List[NewsContribution],
        scenario_contributions: List[NewsContribution],
        scenario_name: str = "情景分析",
        title: str = "对比瀑布图"
    ) -> go.Figure:
        """
        创建对比瀑布图

        Args:
            baseline_contributions: 基准贡献列表
            scenario_contributions: 情景贡献列表
            scenario_name: 情景名称
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        try:
            # 准备基准数据
            baseline_data = self._prepare_waterfall_data(baseline_contributions)
            scenario_data = self._prepare_waterfall_data(scenario_contributions)

            # 创建对比图
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["基准情景", scenario_name],
                vertical_spacing=0.1
            )

            # 基准情景
            fig.add_trace(
                go.Bar(
                    x=baseline_data['x_labels'],
                    y=baseline_data['contribution_values'],
                    marker_color=baseline_data['colors'],
                    name="基准贡献",
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=baseline_data['x_labels'],
                    y=baseline_data['cumulative_values'],
                    mode='lines+markers',
                    line=dict(color=self.default_colors['total'], width=2),
                    name="基准累积",
                    showlegend=True
                ),
                row=1, col=1
            )

            # 情景
            fig.add_trace(
                go.Bar(
                    x=scenario_data['x_labels'],
                    y=scenario_data['contribution_values'],
                    marker_color=scenario_data['colors'],
                    name="情景贡献",
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=scenario_data['x_labels'],
                    y=scenario_data['cumulative_values'],
                    mode='lines+markers',
                    line=dict(color=self.default_colors['total'], width=2, dash='dash'),
                    name="情景累积",
                    showlegend=True
                ),
                row=2, col=1
            )

            # 更新布局
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16, color='#2c3e50')
                ),
                height=800,
                template='plotly_white'
            )

            print(f"[WaterfallPlotter] 对比瀑布图创建完成")
            return fig

        except Exception as e:
            raise VisualizationError(f"对比瀑布图创建失败: {str(e)}", "comparison_waterfall")

    def add_baseline_reference(
        self,
        fig: go.Figure,
        baseline_value: float,
        label: str = "基准线"
    ) -> go.Figure:
        """
        添加基准参考线

        Args:
            fig: 原始图表
            baseline_value: 基准值
            label: 标签文本

        Returns:
            更新后的图表
        """
        try:
            fig.add_hline(
                y=baseline_value,
                line_dash="dot",
                line_color=self.default_colors['baseline'],
                annotation_text=label,
                annotation_position="bottom right",
                annotation_font=dict(color=self.default_colors['baseline'])
            )

            return fig

        except Exception as e:
            raise VisualizationError(f"基准线添加失败: {str(e)}", "baseline_reference")

    def highlight_major_contributors(
        self,
        fig: go.Figure,
        contributions: List[NewsContribution],
        top_n: int = 5
    ) -> go.Figure:
        """
        突出显示主要贡献者

        Args:
            fig: 原始图表
            contributions: 贡献列表
            top_n: 突出显示的数量

        Returns:
            更新后的图表
        """
        try:
            # 识别主要贡献者
            top_contributors = sorted(
                contributions,
                key=lambda x: abs(x.contribution_pct),
                reverse=True
            )[:top_n]

            # 更新图表以突出主要贡献者
            # 这里需要根据具体的图表结构进行调整
            # 简化实现：添加注释
            for i, contrib in enumerate(top_contributors):
                fig.add_annotation(
                    x=i,
                    y=contrib.impact_value,
                    text=f"Top {i+1}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.default_colors['total']
                )

            return fig

        except Exception as e:
            raise VisualizationError(f"主要贡献者突出显示失败: {str(e)}", "highlight_contributors")

    def save_plot_as_html(
        self,
        fig: go.Figure,
        filename: Optional[str] = None,
        directory: Optional[str] = None
    ) -> str:
        """
        保存图表为HTML文件

        Args:
            fig: Plotly图表对象
            filename: 文件名（可选）
            directory: 保存目录（可选）

        Returns:
            保存的文件路径
        """
        try:
            if filename is None:
                filename = f"waterfall_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"

            if directory is None:
                directory = tempfile.gettempdir()

            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)

            fig.write_html(file_path, include_plotlyjs='cdn')

            print(f"[WaterfallPlotter] 图表已保存: {file_path}")
            return file_path

        except Exception as e:
            raise VisualizationError(f"图表保存失败: {str(e)}", "save_plot")

    def _prepare_waterfall_data(self, contributions: List[NewsContribution]) -> Dict[str, Any]:
        """准备瀑布图数据"""
        try:
            # 计算累积值
            cumulative_values = []
            current_value = 0.0

            contribution_values = []
            colors = []
            x_labels = []
            custom_data = []
            contribution_texts = []

            for contrib in contributions:
                # 累积影响
                current_value += contrib.impact_value
                cumulative_values.append(current_value)

                # 贡献值
                contribution_values.append(contrib.impact_value)

                # 颜色
                if contrib.impact_value > 0:
                    colors.append(self.default_colors['positive'])
                elif contrib.impact_value < 0:
                    colors.append(self.default_colors['negative'])
                else:
                    colors.append(self.default_colors['neutral'])

                # 标签
                x_labels.append(contrib.variable_name)

                # 自定义数据（用于悬停信息）
                custom_data.append([
                    contrib.variable_name,
                    contrib.release_date.strftime('%Y-%m-%d'),
                    contrib.observed_value,
                    contrib.expected_value,
                    contrib.contribution_pct
                ])

                # 文本标注
                contribution_texts.append(f"{contrib.contribution_pct:.1f}%")

            # 基准值（初始nowcast值）
            baseline_value = 0.0  # 简化假设
            baseline_values = [baseline_value] * len(contributions)

            return {
                'x_labels': x_labels,
                'contribution_values': contribution_values,
                'cumulative_values': cumulative_values,
                'baseline_values': baseline_values,
                'colors': colors,
                'custom_data': custom_data,
                'contribution_texts': contribution_texts
            }

        except Exception as e:
            raise VisualizationError(f"瀑布图数据准备失败: {str(e)}", "prepare_waterfall_data")

    def _group_by_variable(self, contributions: List[NewsContribution]) -> Dict[str, List[NewsContribution]]:
        """按变量分组"""
        grouped = {}
        for contrib in contributions:
            var_name = contrib.variable_name
            if var_name not in grouped:
                grouped[var_name] = []
            grouped[var_name].append(contrib)
        return grouped

    def _group_by_time(self, contributions: List[NewsContribution]) -> Dict[str, List[NewsContribution]]:
        """按时间分组"""
        grouped = {}
        for contrib in contributions:
            time_key = contrib.release_date.strftime('%Y-%m')
            if time_key not in grouped:
                grouped[time_key] = []
            grouped[time_key].append(contrib)
        return grouped

    def _group_by_sign(self, contributions: List[NewsContribution]) -> Dict[str, List[NewsContribution]]:
        """按正负分组"""
        grouped = {
            '正向影响': [],
            '负向影响': [],
            '中性影响': []
        }

        for contrib in contributions:
            if contrib.impact_value > 0:
                grouped['正向影响'].append(contrib)
            elif contrib.impact_value < 0:
                grouped['负向影响'].append(contrib)
            else:
                grouped['中性影响'].append(contrib)

        # 移除空组
        return {k: v for k, v in grouped.items() if v}