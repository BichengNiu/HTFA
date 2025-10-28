# -*- coding: utf-8 -*-
"""
纽约联储风格组合图表绘制器

生成类似纽约联储"Impact of Data Releases"的组合可视化，
包括时间序列折线图和按行业分类的堆叠柱状图。
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import tempfile
import os

from ..utils.exceptions import VisualizationError
from ..core.news_impact_calculator import NewsContribution
from ..utils.industry_aggregator import IndustryAggregator


class NYFedStylePlotter:
    """
    纽约联储风格组合图表绘制器

    创建包含以下元素的组合图表：
    1. Nowcast时间序列折线图（菱形标记）
    2. 参考线（Latest Estimate, Advance Estimate等）
    3. 按行业分类的堆叠柱状图（正向/负向分离）
    """

    def __init__(self, var_industry_map: Optional[Dict[str, str]] = None):
        """
        初始化绘图器

        Args:
            var_industry_map: 变量名到行业的映射字典
        """
        self.var_industry_map = var_industry_map or {}
        self.aggregator = IndustryAggregator(var_industry_map)

        # 默认颜色方案（参考纽约联储）
        self.default_colors = {
            'Construction': '#8B4513',
            'Production': '#FF8C00',
            'Surveys': '#1E3A8A',
            'Consumption': '#16A34A',
            'Income': '#4ADE80',
            'Labor': '#DC2626',
            'Trade': '#0EA5E9',
            'Prices': '#9333EA',
            'Other': '#9CA3AF'
        }

        # 折线样式
        self.line_styles = {
            'nowcast': {
                'color': '#1E3A8A',
                'width': 3,
                'symbol': 'diamond',
                'size': 8
            },
            'latest_estimate': {
                'color': '#60A5FA',
                'width': 2,
                'dash': 'dash',
                'symbol': 'square',
                'size': 6
            },
            'advance_estimate': {
                'color': '#93C5FD',
                'width': 2,
                'dash': 'dot',
                'symbol': 'circle',
                'size': 6
            }
        }

    def create_combined_chart(
        self,
        contributions: List[NewsContribution],
        nowcast_series: pd.Series,
        target_month: str,
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        actual_values: Optional[pd.Series] = None
    ) -> go.Figure:
        """
        创建纽约联储风格的组合图表

        Args:
            contributions: 新闻贡献列表
            nowcast_series: Nowcast时间序列（仅当月数据）
            target_month: 目标月份（用于标题和时间过滤，格式：YYYY-MM）
            reference_values: 参考值字典（如Latest Estimate, Advance Estimate）
            title: 图表标题
            actual_values: target变量的真实观测值序列（可选）

        Returns:
            Plotly图表对象
        """
        try:
            if not contributions:
                raise VisualizationError("贡献数据为空", "combined_chart")

            # 解析目标月份，计算当月日期范围
            target_date = pd.to_datetime(target_month + '-01')
            month_start = pd.Timestamp(year=target_date.year, month=target_date.month, day=1)
            month_end = month_start + pd.offsets.MonthEnd(1)

            # 过滤nowcast序列，只保留当月数据
            nowcast_filtered = nowcast_series[
                (nowcast_series.index >= month_start) &
                (nowcast_series.index <= month_end)
            ]

            # 过滤真实值序列（如果有）
            actual_filtered = None
            if actual_values is not None:
                actual_filtered = actual_values[
                    (actual_values.index >= month_start) &
                    (actual_values.index <= month_end)
                ]

            # 创建包含两个子图的布局
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": False}]],
                vertical_spacing=0.05
            )

            # 准备堆叠柱状图数据
            stacked_data = self.aggregator.prepare_stacked_bar_data(contributions)
            color_map = self.aggregator.get_industry_color_map(stacked_data['industries'])

            # 1. 添加堆叠柱状图（按行业）
            self._add_stacked_bars(fig, stacked_data, color_map)

            # 2. 添加Nowcast折线（仅当月数据）
            if len(nowcast_filtered) > 0:
                self._add_nowcast_line(fig, nowcast_filtered, nowcast_filtered.index)

            # 3. 添加真实值（红色散点）
            if actual_filtered is not None and len(actual_filtered) > 0:
                self._add_actual_values(fig, actual_filtered, actual_filtered.index)

            # 4. 添加参考线（如果有）
            if reference_values and len(nowcast_filtered) > 0:
                self._add_reference_lines(fig, reference_values, nowcast_filtered.index)

            # 5. 更新布局
            if title is None:
                title = f"数据发布影响 - {target_month}"

            self._update_layout(fig, title, stacked_data)

            print(f"[NYFedStylePlotter] 组合图表创建完成（当月数据：{month_start.strftime('%Y-%m-%d')} - {month_end.strftime('%Y-%m-%d')}）")
            return fig

        except Exception as e:
            raise VisualizationError(f"组合图表创建失败: {str(e)}", "combined_chart")

    def _add_stacked_bars(
        self,
        fig: go.Figure,
        stacked_data: Dict[str, Any],
        color_map: Dict[str, str]
    ) -> None:
        """添加堆叠柱状图"""
        dates = stacked_data['dates']
        industries = stacked_data['industries']
        positive_data = stacked_data['positive']
        negative_data = stacked_data['negative']

        # 添加正向影响的堆叠柱
        for industry in industries:
            pos_values = positive_data[industry]

            fig.add_trace(go.Bar(
                x=dates,
                y=pos_values,
                name=industry,
                marker_color=color_map.get(industry, '#9CA3AF'),
                hovertemplate=(
                    f'<b>{industry}</b><br>'
                    '日期: %{x}<br>'
                    '影响: %{y:.4f}<br>'
                    '<extra></extra>'
                ),
                showlegend=True,
                legendgroup=industry
            ))

        # 添加负向影响的堆叠柱（使用相同颜色但透明度降低）
        for industry in industries:
            neg_values = negative_data[industry]

            fig.add_trace(go.Bar(
                x=dates,
                y=neg_values,
                name=industry,
                marker_color=color_map.get(industry, '#9CA3AF'),
                hovertemplate=(
                    f'<b>{industry}</b><br>'
                    '日期: %{x}<br>'
                    '影响: %{y:.4f}<br>'
                    '<extra></extra>'
                ),
                showlegend=False,  # 不重复显示图例
                legendgroup=industry
            ))

    def _add_nowcast_line(
        self,
        fig: go.Figure,
        nowcast_series: pd.Series,
        x_range: pd.DatetimeIndex
    ) -> None:
        """添加Nowcast折线"""
        style = self.line_styles['nowcast']

        # 过滤nowcast_series以匹配x轴范围
        filtered_series = nowcast_series[nowcast_series.index.isin(x_range)]

        fig.add_trace(go.Scatter(
            x=filtered_series.index,
            y=filtered_series.values,
            mode='lines+markers',
            name='Nowcast',
            line=dict(
                color=style['color'],
                width=style['width']
            ),
            marker=dict(
                symbol=style['symbol'],
                size=10,
                color=style['color']
            ),
            hovertemplate=(
                '<b>Nowcast</b><br>'
                '日期: %{x}<br>'
                '值: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ))

    def _add_reference_lines(
        self,
        fig: go.Figure,
        reference_values: Dict[str, float],
        x_range: pd.DatetimeIndex
    ) -> None:
        """添加参考线"""
        ref_line_configs = {
            'latest_estimate': ('Latest GDP Estimate', self.line_styles['latest_estimate']),
            'advance_estimate': ('Advance GDP Estimate', self.line_styles['advance_estimate'])
        }

        for key, value in reference_values.items():
            if key in ref_line_configs:
                label, style = ref_line_configs[key]

                # 创建水平参考线（在整个x轴范围）
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[value] * len(x_range),
                    mode='lines+markers',
                    name=label,
                    line=dict(
                        color=style['color'],
                        width=style['width'],
                        dash=style.get('dash', 'solid')
                    ),
                    marker=dict(
                        symbol=style['symbol'],
                        size=style['size'],
                        color=style['color']
                    ),
                    hovertemplate=(
                        f'<b>{label}</b><br>'
                        '值: %{y:.4f}<br>'
                        '<extra></extra>'
                    )
                ))

    def _add_actual_values(
        self,
        fig: go.Figure,
        actual_values: pd.Series,
        x_range: pd.DatetimeIndex
    ) -> None:
        """添加target变量的真实观测值（红色散点）"""
        # 过滤actual_values以匹配x轴范围
        filtered_values = actual_values[actual_values.index.isin(x_range)]

        fig.add_trace(go.Scatter(
            x=filtered_values.index,
            y=filtered_values.values,
            mode='markers',
            name='真实值',
            marker=dict(
                symbol='circle',
                size=12,
                color='#DC143C',
                line=dict(
                    color='#8B0000',
                    width=1
                )
            ),
            hovertemplate=(
                '<b>真实值</b><br>'
                '日期: %{x}<br>'
                '值: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ))

    def _update_layout(
        self,
        fig: go.Figure,
        title: str,
        stacked_data: Dict[str, Any]
    ) -> None:
        """更新图表布局"""
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1F2937', family='Arial, sans-serif')
            ),
            xaxis=dict(
                title="",
                tickangle=-45,
                tickformat='%Y-%m-%d',
                gridcolor='#E5E7EB',
                showgrid=True
            ),
            yaxis=dict(
                title="百分比 (年化增长率)",
                gridcolor='#E5E7EB',
                showgrid=True,
                zeroline=True,
                zerolinecolor='#6B7280',
                zerolinewidth=2
            ),
            barmode='relative',
            bargap=0.05,
            height=800,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
                tracegroupgap=5
            ),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#F9FAFB',
            font=dict(family='Arial, sans-serif', size=12, color='#374151'),
            margin=dict(t=150, b=100, l=80, r=50)
        )

        # 添加零线标注
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="#6B7280",
            line_width=1.5,
            annotation_text="",
            annotation_position="right"
        )

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
                filename = f"ny_fed_impact_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"

            if directory is None:
                directory = tempfile.gettempdir()

            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)

            # 使用CDN版本的plotly.js以减小文件大小
            fig.write_html(
                file_path,
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            )

            print(f"[NYFedStylePlotter] 图表已保存: {file_path}")
            return file_path

        except Exception as e:
            raise VisualizationError(f"图表保存失败: {str(e)}", "save_plot")

    def create_simple_nowcast_line(
        self,
        nowcast_series: pd.Series,
        title: str = "Nowcast演变"
    ) -> go.Figure:
        """
        创建简单的Nowcast折线图

        Args:
            nowcast_series: Nowcast时间序列
            title: 图表标题

        Returns:
            Plotly图表对象
        """
        try:
            fig = go.Figure()

            style = self.line_styles['nowcast']

            fig.add_trace(go.Scatter(
                x=nowcast_series.index,
                y=nowcast_series.values,
                mode='lines+markers',
                name='Nowcast',
                line=dict(
                    color=style['color'],
                    width=style['width']
                ),
                marker=dict(
                    symbol=style['symbol'],
                    size=style['size']
                ),
                hovertemplate=(
                    '<b>Nowcast</b><br>'
                    '日期: %{x}<br>'
                    '值: %{y:.4f}<br>'
                    '<extra></extra>'
                )
            ))

            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, color='#1F2937')
                ),
                xaxis=dict(
                    title="日期",
                    gridcolor='#E5E7EB'
                ),
                yaxis=dict(
                    title="Nowcast值",
                    gridcolor='#E5E7EB',
                    zeroline=True,
                    zerolinecolor='#6B7280'
                ),
                height=400,
                hovermode='x unified',
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#F9FAFB'
            )

            return fig

        except Exception as e:
            raise VisualizationError(f"Nowcast折线图创建失败: {str(e)}", "nowcast_line")
