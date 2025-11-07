"""
Industrial Enterprise Operations Analysis Module
工业企业经营分析模块
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 导入统一的工具函数
from dashboard.analysis.industrial.utils import (
    convert_cumulative_to_yoy,
    convert_margin_to_yoy_diff,
    filter_data_by_time_range,
    load_enterprise_profit_data,
    create_excel_download_button
)


# ============================================================================
# 图表创建函数
# ============================================================================


def create_profit_contribution_chart(
    df_contribution: pd.DataFrame,
    total_growth: pd.Series,
    time_range: str = "3年",
    custom_start_date: Optional[str] = None,
    custom_end_date: Optional[str] = None
) -> Optional[go.Figure]:
    """
    创建工业企业利润分行业拉动率图表（堆叠柱状图+折线图）

    Args:
        df_contribution: 上中下游拉动率DataFrame
        total_growth: 总体增速Series
        time_range: 时间范围选择
        custom_start_date: 自定义开始日期
        custom_end_date: 自定义结束日期

    Returns:
        plotly Figure对象
    """
    try:
        from dashboard.analysis.industrial.utils import filter_data_by_time_range

        # 应用时间过滤
        filtered_contribution = filter_data_by_time_range(
            df_contribution, time_range, custom_start_date, custom_end_date
        )
        filtered_total_growth = filter_data_by_time_range(
            total_growth.to_frame('total'), time_range, custom_start_date, custom_end_date
        )['total']

        if filtered_contribution.empty or filtered_total_growth.empty:
            logger.warning("过滤后数据为空")
            return None

        # 创建图表
        fig = go.Figure()

        # 先添加总体增速折线图（确保在legend和hover中排第一）
        if not filtered_total_growth.empty:
            y_data_line = filtered_total_growth.dropna()

            if not y_data_line.empty:
                fig.add_trace(go.Scatter(
                    x=y_data_line.index,
                    y=y_data_line,
                    mode='lines+markers',
                    name='利润总额累计同比',
                    line=dict(width=4, color='#1f77b4'),
                    marker=dict(size=7),
                    connectgaps=False,
                    hovertemplate='<b>利润总额累计同比</b><br>' +
                                  '时间: %{x|%Y年%m月}<br>' +
                                  '数值: %{y:.2f}%<extra></extra>'
                ))

        # 定义颜色映射规则（按上游、中游、下游分类）
        def get_color_for_stream(col_name):
            """根据列名返回对应的颜色"""
            if '上游' in col_name:
                return '#d62728'  # 红色
            elif '中游' in col_name:
                return '#ff7f0e'  # 橙色
            elif '下游' in col_name:
                return '#2ca02c'  # 绿色
            else:
                return '#1f77b4'  # 默认蓝色

        # 获取所有上中下游列，并按上游、中游、下游排序
        stream_cols = [col for col in filtered_contribution.columns if col.startswith('上中下游_')]

        # 自定义排序：上游优先，然后中游，最后下游
        def stream_sort_key(col_name):
            if '上游' in col_name:
                return (0, col_name)
            elif '中游' in col_name:
                return (1, col_name)
            elif '下游' in col_name:
                return (2, col_name)
            else:
                return (3, col_name)

        stream_cols_sorted = sorted(stream_cols, key=stream_sort_key)

        # 添加堆叠柱状图
        for stream_col in stream_cols_sorted:
            y_data = filtered_contribution[stream_col].dropna()

            if not y_data.empty:
                # 获取图例名称（去掉前缀）
                legend_name = stream_col.replace('上中下游_', '')
                color = get_color_for_stream(stream_col)

                fig.add_trace(go.Bar(
                    x=y_data.index,
                    y=y_data,
                    name=legend_name,
                    marker_color=color,
                    opacity=0.8,
                    hovertemplate=f'<b>{legend_name}</b><br>' +
                                  '时间: %{x|%Y年%m月}<br>' +
                                  '拉动率: %{y:.2f}百分点<extra></extra>'
                ))

        # 计算数据范围
        min_date = filtered_contribution.index.min()
        max_date = filtered_contribution.index.max()

        # 配置x轴
        xaxis_config = dict(
            title=dict(text="", font=dict(size=16)),
            type="date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="M3",  # 3个月间隔
            tickformat="%Y-%m",
            hoverformat="%Y-%m",
            tickfont=dict(size=14)
        )

        if min_date and max_date:
            xaxis_config['range'] = [min_date, max_date]

        # 更新布局
        fig.update_layout(
            title=dict(text="工业企业利润结构:分上中下游行业", font=dict(size=18)),
            xaxis=xaxis_config,
            yaxis=dict(
                title=dict(text="%", font=dict(size=16)),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=14)
            ),
            barmode='relative',  # 相对堆积模式，正确处理正负值
            hovermode='x unified',
            height=600,
            margin=dict(l=80, r=50, t=50, b=120),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    except Exception as e:
        logger.error(f"创建利润拉动率图表时发生错误: {e}", exc_info=True)
        return None


def create_enterprise_operations_indicators_chart(
    df_operations: pd.DataFrame,
    time_range: str = "3年",
    custom_start_date: Optional[str] = None,
    custom_end_date: Optional[str] = None
) -> Optional[go.Figure]:
    """
    创建企业经营指标图表（四个子图：ROE、利润率、周转率、权益乘数）

    Args:
        df_operations: 包含企业经营数据的DataFrame，必须包含以下列：
            - ROE: 净资产收益率（%）
            - 利润率: 利润率（%）
            - 总资产周转率: 总资产周转率（次数）
            - 权益乘数: 权益乘数（倍数）
        time_range: 时间范围选择
        custom_start_date: 自定义开始日期
        custom_end_date: 自定义结束日期

    Returns:
        plotly Figure对象（包含4个子图）
    """
    try:
        from plotly.subplots import make_subplots

        # 应用时间过滤
        filtered_df = filter_data_by_time_range(
            df_operations, time_range, custom_start_date, custom_end_date
        )

        if filtered_df.empty:
            logger.warning("过滤后数据为空")
            return None

        # 定义四个指标及其配置
        indicators = [
            {'name': 'ROE', 'title': 'ROE', 'color': '#1f77b4', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '利润率', 'title': '利润率', 'color': '#ff7f0e', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '总资产周转率', 'title': '总资产周转率', 'color': '#2ca02c', 'suffix': '次', 'yaxis_title': '次数'},
            {'name': '权益乘数', 'title': '权益乘数', 'color': '#d62728', 'suffix': '倍', 'yaxis_title': '倍数'}
        ]

        # 检查哪些指标存在于数据中
        available_indicators = [ind for ind in indicators if ind['name'] in filtered_df.columns]

        if not available_indicators:
            logger.warning("未找到任何企业经营指标")
            return None

        # 创建2x2子图布局
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[ind['title'] for ind in indicators],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 为每个子图添加数据
        for idx, indicator in enumerate(indicators):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if indicator['name'] in filtered_df.columns:
                y_data = filtered_df[indicator['name']].dropna()

                if not y_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode='lines+markers',
                            name=indicator['title'],
                            line=dict(width=3, color=indicator['color']),
                            marker=dict(size=7),
                            showlegend=False,
                            connectgaps=False,
                            hovertemplate=(
                                f'<b>{indicator["title"]}</b><br>' +
                                '时间: %{x|%Y年%m月}<br>' +
                                f'数值: %{{y:.2f}}{indicator["suffix"]}<extra></extra>'
                            )
                        ),
                        row=row, col=col
                    )

        # 更新所有x轴（时间轴对齐）
        fig.update_xaxes(
            title_text='',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick="M3",
            tickformat='%Y-%m',
            tickfont=dict(size=11)
        )

        # 更新每个子图的y轴，并添加上下空间
        for idx, indicator in enumerate(indicators):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if indicator['name'] in filtered_df.columns:
                y_data = filtered_df[indicator['name']].dropna()

                if not y_data.empty:
                    # 计算数据范围并添加10%的边距
                    y_min = y_data.min()
                    y_max = y_data.max()
                    y_range = y_max - y_min
                    margin = y_range * 0.1 if y_range > 0 else 0.1

                    fig.update_yaxes(
                        title_text=indicator['yaxis_title'],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(size=11),
                        title_font=dict(size=12),
                        range=[y_min - margin, y_max + margin],
                        row=row, col=col
                    )
                else:
                    fig.update_yaxes(
                        title_text=indicator['yaxis_title'],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(size=11),
                        title_font=dict(size=12),
                        row=row, col=col
                    )

        # 更新整体布局
        fig.update_layout(
            height=700,
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(
                text='净资产收益率分析',
                x=0,
                xanchor='left',
                font=dict(size=18)
            )
        )

        # 更新子图标题样式
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color='#333')

        return fig

    except Exception as e:
        logger.error(f"创建企业经营指标图表时发生错误: {e}", exc_info=True)
        return None


def create_enterprise_efficiency_metrics_chart(
    df_efficiency: pd.DataFrame,
    time_range: str = "3年",
    custom_start_date: Optional[str] = None,
    custom_end_date: Optional[str] = None
) -> Optional[go.Figure]:
    """
    创建企业经营效率指标图表（六个子图：成本、费用、资产收入、人均收入、产成品周转天数、应收账款平均回收期的累计同比）

    Args:
        df_efficiency: 包含企业经营效率数据的DataFrame，必须包含以下列：
            - 每百元营业收入中的成本: 每百元营业收入中的成本累计同比（%）
            - 每百元营业收入中的费用: 每百元营业收入中的费用累计同比（%）
            - 每百元资产实现的营业收入: 每百元资产实现的营业收入累计同比（%）
            - 人均营业收入: 人均营业收入累计同比（%）
            - 产成品周转天数: 产成品周转天数累计同比（%）
            - 应收账款平均回收期: 应收账款平均回收期累计同比（%）
        time_range: 时间范围选择
        custom_start_date: 自定义开始日期
        custom_end_date: 自定义结束日期

    Returns:
        plotly Figure对象（包含6个子图）
    """
    try:
        from plotly.subplots import make_subplots

        # 应用时间过滤
        filtered_df = filter_data_by_time_range(
            df_efficiency, time_range, custom_start_date, custom_end_date
        )

        if filtered_df.empty:
            logger.warning("过滤后数据为空")
            return None

        # 定义六个指标及其配置（累计同比）
        indicators = [
            {'name': '每百元营业收入中的成本', 'title': '每百元营业收入中的成本:累计同比', 'color': '#1f77b4', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '每百元营业收入中的费用', 'title': '每百元营业收入中的费用:累计同比', 'color': '#ff7f0e', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '每百元资产实现的营业收入', 'title': '每百元资产实现的营业收入:累计同比', 'color': '#2ca02c', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '人均营业收入', 'title': '人均营业收入:累计同比', 'color': '#d62728', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '产成品周转天数', 'title': '产成品周转天数:累计同比', 'color': '#9467bd', 'suffix': '%', 'yaxis_title': '%'},
            {'name': '应收账款平均回收期', 'title': '应收账款平均回收期:累计同比', 'color': '#8c564b', 'suffix': '%', 'yaxis_title': '%'}
        ]

        # 检查哪些指标存在于数据中
        available_indicators = [ind for ind in indicators if ind['name'] in filtered_df.columns]

        if not available_indicators:
            logger.warning("未找到任何企业经营效率指标")
            return None

        # 创建3x2子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[ind['title'] for ind in indicators],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 为每个子图添加数据
        for idx, indicator in enumerate(indicators):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if indicator['name'] in filtered_df.columns:
                y_data = filtered_df[indicator['name']].dropna()

                if not y_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode='lines+markers',
                            name=indicator['title'],
                            line=dict(width=3, color=indicator['color']),
                            marker=dict(size=7),
                            showlegend=False,
                            connectgaps=False,
                            hovertemplate=(
                                f'<b>{indicator["title"]}</b><br>' +
                                '时间: %{x|%Y年%m月}<br>' +
                                f'数值: %{{y:.2f}}{indicator["suffix"]}<extra></extra>'
                            )
                        ),
                        row=row, col=col
                    )

        # 更新所有x轴（时间轴对齐）
        fig.update_xaxes(
            title_text='',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick="M3",
            tickformat='%Y-%m',
            tickfont=dict(size=11)
        )

        # 更新每个子图的y轴，并添加上下空间
        for idx, indicator in enumerate(indicators):
            row = idx // 2 + 1
            col = idx % 2 + 1

            if indicator['name'] in filtered_df.columns:
                y_data = filtered_df[indicator['name']].dropna()

                if not y_data.empty:
                    # 计算数据范围并添加10%的边距
                    y_min = y_data.min()
                    y_max = y_data.max()
                    y_range = y_max - y_min
                    margin = y_range * 0.1 if y_range > 0 else 0.1

                    fig.update_yaxes(
                        title_text=indicator['yaxis_title'],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(size=11),
                        title_font=dict(size=12),
                        range=[y_min - margin, y_max + margin],
                        row=row, col=col
                    )
                else:
                    fig.update_yaxes(
                        title_text=indicator['yaxis_title'],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        tickfont=dict(size=11),
                        title_font=dict(size=12),
                        row=row, col=col
                    )

        # 更新整体布局
        fig.update_layout(
            height=1000,
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(
                text='企业经营效率指标',
                x=0,
                xanchor='left',
                font=dict(size=18)
            )
        )

        # 更新子图标题样式
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color='#333')

        return fig

    except Exception as e:
        logger.error(f"创建企业经营效率指标图表时发生错误: {e}", exc_info=True)
        return None


def create_enterprise_indicators_chart(df_data: pd.DataFrame, time_range: str = "3年",
                                     custom_start_date: Optional[str] = None,
                                     custom_end_date: Optional[str] = None) -> Optional[go.Figure]:
    """
    创建企业经营四个指标的时间序列线图

    Args:
        df_data: 包含指标数据的DataFrame
        time_range: 时间范围选择
        custom_start_date: 自定义开始日期
        custom_end_date: 自定义结束日期

    Returns:
        plotly Figure对象
    """
    try:
        # Apply time filtering first
        filtered_df = filter_data_by_time_range(df_data, time_range, custom_start_date, custom_end_date)

        # 定义需要的四个指标（使用标准列名）
        from dashboard.analysis.industrial.constants import (
            PROFIT_TOTAL_COLUMN,
            CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
            PPI_COLUMN,
            PROFIT_MARGIN_COLUMN_YOY
        )

        required_indicators = [
            PROFIT_TOTAL_COLUMN,
            CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
            PPI_COLUMN,
            PROFIT_MARGIN_COLUMN_YOY
        ]

        # 检查哪些指标存在于数据中
        available_indicators = [ind for ind in required_indicators if ind in df_data.columns]

        if not available_indicators:
            logger.warning("没有可用指标")
            return None

        # 处理数据 - 改为更灵活的缺失数据策略
        complete_data_df = filtered_df[available_indicators].copy()

        # 转换所有列为数值型
        for col in complete_data_df.columns:
            complete_data_df[col] = pd.to_numeric(complete_data_df[col], errors='coerce')

        # 新策略：降低要求，只要有至少1个指标有数据就显示
        # 计算每行的非空值数量
        non_null_counts = complete_data_df.count(axis=1)

        # 保留至少有1个指标数据的行（降低要求）
        min_indicators = 1
        valid_rows = non_null_counts >= min_indicators

        complete_data_df = complete_data_df[valid_rows]

        if complete_data_df.empty:
            logger.warning("过滤后数据为空")
            return None

        # 创建图表
        fig = go.Figure()

        # 导入常量
        from dashboard.analysis.industrial.constants import (
            CHART_COLORS,
            ENTERPRISE_INDICATOR_LEGEND_MAPPING,
            BAR_CHART_INDICATORS,
            LINE_CHART_INDICATORS
        )

        colors = CHART_COLORS

        def get_legend_name(indicator):
            """将完整的指标名称转换为简化的图例名称"""
            return ENTERPRISE_INDICATOR_LEGEND_MAPPING.get(indicator, indicator)

        # 先添加线图指标
        for i, indicator in enumerate(available_indicators):
            if indicator in complete_data_df.columns:
                # 获取数据，保留NaN值用于正确处理缺失点
                y_data = complete_data_df[indicator].dropna()

                if not y_data.empty:
                    # 检查是否应该用线图
                    is_line_indicator = any(line_key in indicator for line_key in LINE_CHART_INDICATORS)

                    if is_line_indicator:
                        # 添加线图
                        fig.add_trace(go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode='lines+markers',
                            name=get_legend_name(indicator),
                            line=dict(width=4, color=colors[i % len(colors)]),
                            marker=dict(size=7),
                            connectgaps=False,  # 不连接缺失点
                            hovertemplate=f'<b>{get_legend_name(indicator)}</b><br>' +
                                          '时间: %{x|%Y年%m月}<br>' +
                                          '数值: %{y:.2f}%<extra></extra>'
                        ))

        # 添加条形图指标（恢复简单逻辑，让Plotly正确处理正负值堆叠）
        for i, indicator in enumerate(available_indicators):
            if indicator in complete_data_df.columns:
                # 获取数据，保留NaN值用于正确处理缺失点
                y_data = complete_data_df[indicator].dropna()

                if not y_data.empty:
                    # 检查是否应该用条形图
                    is_bar_indicator = any(bar_key in indicator for bar_key in BAR_CHART_INDICATORS)

                    if is_bar_indicator:
                        # 计算渐变透明度：按原始指标顺序
                        opacity_value = 0.9 - (i * 0.15)  # 0.9, 0.75, 0.6, 0.45...
                        opacity_value = max(0.5, opacity_value)  # 最小透明度为0.5

                        # 添加堆积条形图
                        fig.add_trace(go.Bar(
                            x=y_data.index,
                            y=y_data,
                            name=get_legend_name(indicator),
                            marker_color=colors[i % len(colors)],
                            opacity=opacity_value,
                            hovertemplate=f'<b>{get_legend_name(indicator)}</b><br>' +
                                          '时间: %{x|%Y年%m月}<br>' +
                                          '数值: %{y:.2f}%<extra></extra>'
                        ))

        # Calculate actual data range for x-axis using complete data
        if not complete_data_df.empty:
            min_date = complete_data_df.index.min()
            max_date = complete_data_df.index.max()
        else:
            min_date = max_date = None

        # Configure x-axis with 3-month intervals
        xaxis_config = dict(
            title=dict(text="", font=dict(size=16)),  # No x-axis title
            type="date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="M3",  # 3-month intervals
            tickformat="%Y-%m",  # 时间轴刻度显示格式：年-月
            hoverformat="%Y-%m",  # 鼠标悬停显示格式：年-月
            tickfont=dict(size=14)
        )

        # Set the range to actual data if available
        if min_date and max_date:
            xaxis_config['range'] = [min_date, max_date]

        # 更新布局 - 移除所有文字，设置堆积条形图
        fig.update_layout(
            title=dict(text="工业企业利润拆解", font=dict(size=18)),
            xaxis=xaxis_config,
            yaxis=dict(
                title=dict(text="%", font=dict(size=16)),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=14)
            ),
            barmode='relative',  # 设置为相对堆积模式，正确处理正负值堆叠
            hovermode='x unified',
            height=600,
            margin=dict(l=80, r=50, t=50, b=120),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    except KeyError as e:
        logger.warning(f"创建企业指标图表时列不存在: {e}")
        return None
    except ValueError as e:
        logger.warning(f"创建企业指标图表时数据值错误: {e}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning("企业指标数据为空")
        return None
    except Exception as e:
        logger.error(f"创建企业指标图表时发生未预期错误: {e}", exc_info=True)
        return None


def render_enterprise_operations_analysis_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    使用预加载数据渲染企业经营分析（用于统一模块）

    Args:
        st_obj: Streamlit对象
        df_macro: 宏观运行数据
        df_weights: 权重数据
        uploaded_file: 上传的Excel文件对象
    """
    # 如果没有传入上传的文件，尝试从统一状态管理器获取
    if uploaded_file is None:
        uploaded_file = st.session_state.get("analysis.industrial.unified_file_uploader")

    if uploaded_file is None:
        return

    # 读取企业利润拆解数据
    df_profit = load_enterprise_profit_data(uploaded_file)

    if df_profit is None:
        logger.error("无法加载企业利润数据")
        st_obj.error("错误：无法加载企业利润数据")
        return

    # 处理"规模以上工业企业:营业收入利润率:累计值"转换为年同比
    profit_margin_col = None
    for col in df_profit.columns:
        if '营业收入利润率' in str(col) and '累计值' in str(col):
            profit_margin_col = col
            break

    # load_enterprise_profit_data已经确保返回DatetimeIndex
    df_profit_with_index = df_profit

    if profit_margin_col:
        # 计算年同比：(当期值/去年同期值 - 1) × 100
        # convert_cumulative_to_yoy会自动将1月2月设为NaN
        yoy_data = convert_cumulative_to_yoy(df_profit_with_index[profit_margin_col])

        # 创建新的列名（将"累计值"替换为"累计同比"）
        yoy_col_name = profit_margin_col.replace('累计值', '累计同比')

        # 添加年同比数据
        df_profit_with_index[yoy_col_name] = yoy_data

    # 过滤掉所有1月和2月的数据（因为累计值在这两个月不具有可比性）
    jan_feb_mask = df_profit_with_index.index.month.isin([1, 2])
    df_profit = df_profit_with_index[~jan_feb_mask]

    if len(df_profit) == 0:
        logger.error("过滤1月2月后数据为空")
        st_obj.error("错误：过滤1月2月后，数据为空！请检查上传的Excel文件。")
        return

    # 图表1：工业企业利润拆解（添加时间筛选器）

    # 使用Fragment组件添加时间筛选器
    from dashboard.analysis.industrial.utils import (
        create_chart_with_time_selector_fragment,
        IndustrialStateManager
    )
    from dashboard.analysis.industrial.constants import (
        PROFIT_TOTAL_COLUMN,
        CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
        PPI_COLUMN,
        PROFIT_MARGIN_COLUMN_YOY
    )

    # 定义图表1的变量
    chart1_variables = [
        PROFIT_TOTAL_COLUMN,
        CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
        PPI_COLUMN,
        PROFIT_MARGIN_COLUMN_YOY
    ]

    # 过滤出存在的变量
    available_vars = [var for var in chart1_variables if var in df_profit.columns]

    if not available_vars:
        st_obj.error(f"错误：未找到任何必需的指标列！数据列名: {list(df_profit.columns[:5])}")
        return

    # 定义图表创建函数（包装create_enterprise_indicators_chart以符合Fragment接口）
    def create_chart1(df, variables, time_range, custom_start_date, custom_end_date):
        fig = create_enterprise_indicators_chart(
            df_data=df,
            time_range=time_range,
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        return fig

    # 使用Fragment组件
    current_time_range_1, custom_start_1, custom_end_1 = create_chart_with_time_selector_fragment(
        st_obj=st_obj,
        chart_id="enterprise_chart1",
        state_namespace="monitoring.industrial.enterprise",
        chart_title=None,
        chart_creator_func=create_chart1,
        chart_data=df_profit,
        chart_variables=available_vars,
        get_state_func=IndustrialStateManager.get,
        set_state_func=IndustrialStateManager.set
    )

    # 添加第一个图表的数据下载功能 - 下载所有时间的完整数据
    if not df_profit.empty:
        create_excel_download_button(
            st_obj=st_obj,
            data=df_profit,
            file_name="企业经营指标_全部数据.xlsx",
            sheet_name='企业经营指标',
            button_key="industrial_enterprise_operations_download_data_button",
            column_ratio=(1, 3)
        )

    # ============================================================================
    # 企业经营指标图表
    # ============================================================================
    st_obj.markdown("#### 企业经营指标")

    # 加载企业经营数据
    from dashboard.analysis.industrial.utils import load_enterprise_operations_data, convert_cumulative_to_current

    df_operations = load_enterprise_operations_data(uploaded_file)

    if df_operations is not None and not df_operations.empty:
        # 定义需要的列名
        profit_cumulative_col = '中国:利润总额:规模以上工业企业:累计值'
        revenue_cumulative_col = '中国:营业收入:规模以上工业企业:累计值'
        assets_col = '中国:资产合计:规模以上工业企业'
        equity_col = '中国:所有者权益合计:规模以上工业企业'

        # 检查必需的列是否存在
        required_cols = [profit_cumulative_col, revenue_cumulative_col, assets_col, equity_col]
        missing_cols = [col for col in required_cols if col not in df_operations.columns]

        if missing_cols:
            logger.warning(f"缺少必需列: {missing_cols}")
            st_obj.warning(f"数据中缺少部分企业经营指标列，无法生成完整图表")
        else:
            # 转换累计值为当期值
            df_operations['利润总额当期值'] = convert_cumulative_to_current(df_operations[profit_cumulative_col])
            df_operations['营业收入当期值'] = convert_cumulative_to_current(df_operations[revenue_cumulative_col])

            # 过滤掉1月和2月的数据（从3月开始计算）
            jan_feb_mask = df_operations.index.month.isin([1, 2])
            df_operations_filtered = df_operations[~jan_feb_mask].copy()

            # 计算4个指标（使用过滤后的数据）
            # 1. 净资产收益率 = 利润总额当期值 / 所有者权益合计 × 100
            df_operations_filtered['ROE'] = (df_operations_filtered['利润总额当期值'] / df_operations_filtered[equity_col]) * 100

            # 2. 利润率 = 利润总额当期值 / 营业收入当期值 × 100
            df_operations_filtered['利润率'] = (df_operations_filtered['利润总额当期值'] / df_operations_filtered['营业收入当期值']) * 100

            # 3. 总资产周转率 = 营业收入当期值 / 资产合计
            df_operations_filtered['总资产周转率'] = df_operations_filtered['营业收入当期值'] / df_operations_filtered[assets_col]

            # 4. 权益乘数 = 资产合计 / 所有者权益合计
            df_operations_filtered['权益乘数'] = df_operations_filtered[assets_col] / df_operations_filtered[equity_col]

            # 使用过滤后的数据
            df_operations = df_operations_filtered

            # 定义所有可用的指标变量
            all_indicators = ['ROE', '利润率', '总资产周转率', '权益乘数']

            # 添加指标选择器（默认选择前3个，不选权益乘数）
            default_indicators = ['ROE', '利润率', '总资产周转率']

            selected_indicators = st_obj.multiselect(
                "选择要显示的企业经营指标",
                options=all_indicators,
                default=default_indicators,
                key="enterprise_operations_indicator_selector"
            )

            # 如果没有选择任何指标，显示提示
            if not selected_indicators:
                st_obj.warning("请至少选择一个指标")
                return

            # 定义图表创建函数（使用选中的指标）
            def create_operations_chart(df, variables, time_range, custom_start_date, custom_end_date):
                fig = create_enterprise_operations_indicators_chart(
                    df_operations=df,
                    time_range=time_range,
                    custom_start_date=custom_start_date,
                    custom_end_date=custom_end_date
                )
                return fig

            # 使用Fragment组件创建图表
            current_time_range_ops, custom_start_ops, custom_end_ops = create_chart_with_time_selector_fragment(
                st_obj=st_obj,
                chart_id="enterprise_operations_indicators",
                state_namespace="monitoring.industrial.enterprise",
                chart_title=None,
                chart_creator_func=create_operations_chart,
                chart_data=df_operations,
                chart_variables=selected_indicators,
                get_state_func=IndustrialStateManager.get,
                set_state_func=IndustrialStateManager.set
            )

            # 添加数据下载功能
            if not df_operations.empty:
                # 准备下载数据：只包含时间和4个指标
                download_df = df_operations[all_indicators].copy()

                # 按时间降序排列
                download_df = download_df.sort_index(ascending=False)

                # 重置索引并格式化时间
                download_df = download_df.reset_index()
                first_col = download_df.columns[0]
                download_df.rename(columns={first_col: '时间'}, inplace=True)
                download_df['时间'] = download_df['时间'].dt.strftime('%Y-%m')

                # 确保只包含时间和4个指标列（移除其他列）
                download_columns = ['时间'] + all_indicators
                download_df = download_df[download_columns]

                create_excel_download_button(
                    st_obj=st_obj,
                    data=download_df,
                    file_name="企业经营指标.xlsx",
                    sheet_name='企业经营指标',
                    button_key="enterprise_operations_indicators_download_button",
                    column_ratio=(1, 3)
                )
    else:
        logger.warning("未能加载企业经营数据")
        st_obj.info("提示：数据模板中未找到'工业企业经营'sheet，跳过企业经营指标图表")

    # ============================================================================
    # 分割线 - 分行业利润拆解
    # ============================================================================
    st_obj.markdown("---")

    # 加载分行业利润数据
    from dashboard.analysis.industrial.utils import load_industry_profit_data
    from dashboard.analysis.industrial.utils.contribution_calculator import calculate_profit_contributions

    df_industry_profit = load_industry_profit_data(uploaded_file)

    if df_industry_profit is None:
        logger.error("无法加载分行业利润数据")
        st_obj.error("错误：无法加载分行业利润数据")
        return

    if df_weights is None:
        st_obj.error("错误：缺少权重数据，无法进行利润拆解分析")
        return

    # 计算利润拉动率
    try:
        profit_contribution_result = calculate_profit_contributions(
            df_industry_profit=df_industry_profit,
            df_weights=df_weights
        )

        stream_contribution_df = profit_contribution_result['stream_groups']
        individual_contribution_df = profit_contribution_result['individual']
        total_growth_series = profit_contribution_result['total_growth']
        validation_result = profit_contribution_result['validation']

        # 定义图表2的创建函数
        def create_chart2(df, variables, time_range, custom_start_date, custom_end_date):
            # df是stream_contribution_df，但我们还需要total_growth_series
            # 这里通过闭包访问total_growth_series
            fig = create_profit_contribution_chart(
                df_contribution=df,
                total_growth=total_growth_series,
                time_range=time_range,
                custom_start_date=custom_start_date,
                custom_end_date=custom_end_date
            )
            return fig

        # 使用Fragment组件创建图表2
        available_vars_chart2 = list(stream_contribution_df.columns)

        if available_vars_chart2:
            current_time_range_2, custom_start_2, custom_end_2 = create_chart_with_time_selector_fragment(
                st_obj=st_obj,
                chart_id="enterprise_chart2",
                state_namespace="monitoring.industrial.enterprise",
                chart_title=None,
                chart_creator_func=create_chart2,
                chart_data=stream_contribution_df,
                chart_variables=available_vars_chart2,
                get_state_func=IndustrialStateManager.get,
                set_state_func=IndustrialStateManager.set
            )
        else:
            st_obj.warning("未找到上中下游拉动率数据")

        # 添加拉动率数据下载功能
        if not stream_contribution_df.empty:
            # 合并拉动率数据和总体增速
            download_df = stream_contribution_df.copy()

            # 去掉列名中的"上中下游_"前缀
            download_df.columns = [col.replace('上中下游_', '') for col in download_df.columns]

            # 添加总体增速列（放在第一列）
            download_df.insert(0, '利润总额累计同比', total_growth_series)

            # 按时间降序排列（近到远）
            download_df = download_df.sort_index(ascending=False)

            # 重置索引，并将日期格式化为"年-月"
            download_df = download_df.reset_index()
            # 将第一列（索引列）重命名为'时间'
            first_col = download_df.columns[0]
            download_df.rename(columns={first_col: '时间'}, inplace=True)
            download_df['时间'] = download_df['时间'].dt.strftime('%Y-%m')

            create_excel_download_button(
                st_obj=st_obj,
                data=download_df,
                file_name="分行业利润拆解_拉动率数据.xlsx",
                sheet_name='拉动率',
                button_key="industrial_profit_contribution_download_button",
                column_ratio=(1, 3)
            )

    except Exception as e:
        logger.error(f"计算利润拉动率时发生错误: {e}", exc_info=True)
        st_obj.error(f"计算利润拉动率失败：{str(e)}")
        return


def render_enterprise_profit_analysis_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    渲染工业企业利润分析（包含利润拆解和分行业拆解）

    Args:
        st_obj: Streamlit对象
        df_macro: 宏观运行数据
        df_weights: 权重数据
        uploaded_file: 上传的Excel文件对象
    """
    # 如果没有传入上传的文件，尝试从统一状态管理器获取
    if uploaded_file is None:
        uploaded_file = st.session_state.get("analysis.industrial.unified_file_uploader")

    if uploaded_file is None:
        st_obj.info("请先上传Excel数据文件以开始工业企业利润分析")
        return

    # 读取企业利润拆解数据
    df_profit = load_enterprise_profit_data(uploaded_file)

    if df_profit is None:
        logger.error("无法加载企业利润数据")
        st_obj.error("错误：无法加载企业利润数据")
        return

    # 处理"规模以上工业企业:营业收入利润率:累计值"转换为年同比
    profit_margin_col = None
    for col in df_profit.columns:
        if '营业收入利润率' in str(col) and '累计值' in str(col):
            profit_margin_col = col
            break

    # load_enterprise_profit_data已经确保返回DatetimeIndex
    df_profit_with_index = df_profit

    if profit_margin_col:
        yoy_data = convert_cumulative_to_yoy(df_profit_with_index[profit_margin_col])
        yoy_col_name = profit_margin_col.replace('累计值', '累计同比')
        df_profit_with_index[yoy_col_name] = yoy_data

    # 过滤掉所有1月和2月的数据
    jan_feb_mask = df_profit_with_index.index.month.isin([1, 2])
    df_profit = df_profit_with_index[~jan_feb_mask]

    if len(df_profit) == 0:
        logger.error("过滤1月2月后数据为空")
        st_obj.error("错误：过滤1月2月后，数据为空！请检查上传的Excel文件。")
        return

    # 图表1：工业企业利润拆解

    from dashboard.analysis.industrial.utils import (
        create_chart_with_time_selector_fragment,
        IndustrialStateManager
    )
    from dashboard.analysis.industrial.constants import (
        PROFIT_TOTAL_COLUMN,
        CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
        PPI_COLUMN,
        PROFIT_MARGIN_COLUMN_YOY
    )

    chart1_variables = [
        PROFIT_TOTAL_COLUMN,
        CUMULATIVE_INDUSTRIAL_GROWTH_COLUMN,
        PPI_COLUMN,
        PROFIT_MARGIN_COLUMN_YOY
    ]

    available_vars = [var for var in chart1_variables if var in df_profit.columns]

    if not available_vars:
        st_obj.error(f"错误：未找到任何必需的指标列！")
        return

    def create_chart1(df, variables, time_range, custom_start_date, custom_end_date):
        fig = create_enterprise_indicators_chart(
            df_data=df,
            time_range=time_range,
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        return fig

    create_chart_with_time_selector_fragment(
        st_obj=st_obj,
        chart_id="enterprise_profit_chart1",
        state_namespace="monitoring.industrial.enterprise.profit",
        chart_title=None,
        chart_creator_func=create_chart1,
        chart_data=df_profit,
        chart_variables=available_vars,
        get_state_func=IndustrialStateManager.get,
        set_state_func=IndustrialStateManager.set
    )

    if not df_profit.empty:
        create_excel_download_button(
            st_obj=st_obj,
            data=df_profit,
            file_name="企业利润指标_全部数据.xlsx",
            sheet_name='企业利润指标',
            button_key="enterprise_profit_download_button",
            column_ratio=(1, 3)
        )

    # 分割线
    st_obj.markdown("---")

    # 图表2：分行业利润拆解
    from dashboard.analysis.industrial.utils import load_industry_profit_data
    from dashboard.analysis.industrial.utils.contribution_calculator import calculate_profit_contributions

    df_industry_profit = load_industry_profit_data(uploaded_file)

    if df_industry_profit is None:
        logger.error("无法加载分行业利润数据")
        st_obj.error("错误：无法加载分行业利润数据")
        return

    if df_weights is None:
        st_obj.error("错误：缺少权重数据，无法进行利润拆解分析")
        return

    try:
        profit_contribution_result = calculate_profit_contributions(
            df_industry_profit=df_industry_profit,
            df_weights=df_weights
        )

        stream_contribution_df = profit_contribution_result['stream_groups']
        total_growth_series = profit_contribution_result['total_growth']

        def create_chart2(df, variables, time_range, custom_start_date, custom_end_date):
            fig = create_profit_contribution_chart(
                df_contribution=df,
                total_growth=total_growth_series,
                time_range=time_range,
                custom_start_date=custom_start_date,
                custom_end_date=custom_end_date
            )
            return fig

        available_vars_chart2 = list(stream_contribution_df.columns)

        if available_vars_chart2:
            create_chart_with_time_selector_fragment(
                st_obj=st_obj,
                chart_id="enterprise_profit_chart2",
                state_namespace="monitoring.industrial.enterprise.profit",
                chart_title=None,
                chart_creator_func=create_chart2,
                chart_data=stream_contribution_df,
                chart_variables=available_vars_chart2,
                get_state_func=IndustrialStateManager.get,
                set_state_func=IndustrialStateManager.set
            )
        else:
            st_obj.warning("未找到上中下游拉动率数据")

        if not stream_contribution_df.empty:
            download_df = stream_contribution_df.copy()
            download_df.columns = [col.replace('上中下游_', '') for col in download_df.columns]
            download_df.insert(0, '利润总额累计同比', total_growth_series)
            download_df = download_df.sort_index(ascending=False)
            download_df = download_df.reset_index()
            first_col = download_df.columns[0]
            download_df.rename(columns={first_col: '时间'}, inplace=True)
            download_df['时间'] = download_df['时间'].dt.strftime('%Y-%m')

            create_excel_download_button(
                st_obj=st_obj,
                data=download_df,
                file_name="分行业利润拆解_拉动率数据.xlsx",
                sheet_name='拉动率',
                button_key="profit_contribution_download_button",
                column_ratio=(1, 3)
            )

    except Exception as e:
        logger.error(f"计算利润拉动率时发生错误: {e}", exc_info=True)
        st_obj.error(f"计算利润拉动率失败：{str(e)}")
        return


def render_enterprise_efficiency_analysis_with_data(st_obj, df_macro: Optional[pd.DataFrame], df_weights: Optional[pd.DataFrame], uploaded_file=None):
    """
    渲染工业企业经营效率分析（包含净资产收益率分析）

    Args:
        st_obj: Streamlit对象
        df_macro: 宏观运行数据
        df_weights: 权重数据
        uploaded_file: 上传的Excel文件对象
    """
    # 如果没有传入上传的文件，尝试从统一状态管理器获取
    if uploaded_file is None:
        uploaded_file = st.session_state.get("analysis.industrial.unified_file_uploader")

    if uploaded_file is None:
        st_obj.info("请先上传Excel数据文件以开始工业企业经营效率分析")
        return

    # 加载企业经营数据
    from dashboard.analysis.industrial.utils import load_enterprise_operations_data, convert_cumulative_to_current

    df_operations = load_enterprise_operations_data(uploaded_file)

    if df_operations is None or df_operations.empty:
        logger.warning("未能加载企业经营数据")
        st_obj.info("提示：数据模板中未找到'工业企业经营'sheet，无法生成经营效率分析")
        return

    # 定义需要的列名
    profit_cumulative_col = '中国:利润总额:规模以上工业企业:累计值'
    revenue_cumulative_col = '中国:营业收入:规模以上工业企业:累计值'
    assets_col = '中国:资产合计:规模以上工业企业'
    equity_col = '中国:所有者权益合计:规模以上工业企业'

    # 检查必需的列是否存在
    required_cols = [profit_cumulative_col, revenue_cumulative_col, assets_col, equity_col]
    missing_cols = [col for col in required_cols if col not in df_operations.columns]

    if missing_cols:
        logger.warning(f"缺少必需列: {missing_cols}")
        st_obj.warning(f"数据中缺少部分企业经营指标列，无法生成完整图表")
        return

    # 转换累计值为当期值
    df_operations['利润总额当期值'] = convert_cumulative_to_current(df_operations[profit_cumulative_col])
    df_operations['营业收入当期值'] = convert_cumulative_to_current(df_operations[revenue_cumulative_col])

    # 过滤掉1月和2月的数据
    jan_feb_mask = df_operations.index.month.isin([1, 2])
    df_operations_filtered = df_operations[~jan_feb_mask].copy()

    # 计算4个指标
    df_operations_filtered['ROE'] = (df_operations_filtered['利润总额当期值'] / df_operations_filtered[equity_col]) * 100
    df_operations_filtered['利润率'] = (df_operations_filtered['利润总额当期值'] / df_operations_filtered['营业收入当期值']) * 100
    df_operations_filtered['总资产周转率'] = df_operations_filtered['营业收入当期值'] / df_operations_filtered[assets_col]
    df_operations_filtered['权益乘数'] = df_operations_filtered[assets_col] / df_operations_filtered[equity_col]

    df_operations = df_operations_filtered

    # 定义所有可用的指标变量
    all_indicators = ['ROE', '利润率', '总资产周转率', '权益乘数']

    # 定义图表创建函数
    def create_operations_chart(df, variables, time_range, custom_start_date, custom_end_date):
        fig = create_enterprise_operations_indicators_chart(
            df_operations=df,
            time_range=time_range,
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        return fig

    # 导入工具函数
    from dashboard.analysis.industrial.utils import (
        create_chart_with_time_selector_fragment,
        IndustrialStateManager
    )

    # 使用Fragment组件创建图表（不使用变量选择器）
    create_chart_with_time_selector_fragment(
        st_obj=st_obj,
        chart_id="enterprise_efficiency_indicators",
        state_namespace="monitoring.industrial.enterprise.efficiency",
        chart_title=None,
        chart_creator_func=create_operations_chart,
        chart_data=df_operations,
        chart_variables=all_indicators,
        get_state_func=IndustrialStateManager.get,
        set_state_func=IndustrialStateManager.set
    )

    # 添加数据下载功能
    if not df_operations.empty:
        download_df = df_operations[all_indicators].copy()
        download_df = download_df.sort_index(ascending=False)
        download_df = download_df.reset_index()
        first_col = download_df.columns[0]
        download_df.rename(columns={first_col: '时间'}, inplace=True)
        download_df['时间'] = download_df['时间'].dt.strftime('%Y-%m')
        download_columns = ['时间'] + all_indicators
        download_df = download_df[download_columns]

        create_excel_download_button(
            st_obj=st_obj,
            data=download_df,
            file_name="企业经营效率指标.xlsx",
            sheet_name='企业经营效率指标',
            button_key="efficiency_indicators_download_button",
            column_ratio=(1, 3)
        )

    # ============================================================================
    # 企业经营效率指标分析（每百元成本/费用/资产/人均收入）
    # ============================================================================
    st_obj.markdown("---")

    # 定义需要的列名（累计值）
    cost_per_hundred_cumulative_col = '中国:每百元营业收入中的成本:规模以上工业企业:累计值'
    expense_per_hundred_cumulative_col = '中国:每百元营业收入中的费用:规模以上工业企业:累计值'
    asset_revenue_per_hundred_cumulative_col = '中国:每百元资产实现的营业收入:规模以上工业企业:累计值'
    revenue_per_capita_cumulative_col = '中国:人均营业收入:规模以上工业企业:累计值'
    turnover_days_cumulative_col = '中国:产成品周转天数:规模以上工业企业:累计值'
    receivable_period_cumulative_col = '中国:应收账款平均回收期:规模以上工业企业:累计值'

    # 检查必需的列是否存在
    required_efficiency_cols = [
        cost_per_hundred_cumulative_col,
        expense_per_hundred_cumulative_col,
        asset_revenue_per_hundred_cumulative_col,
        revenue_per_capita_cumulative_col,
        turnover_days_cumulative_col,
        receivable_period_cumulative_col
    ]

    # 重新加载数据（因为需要新的列）
    df_efficiency_metrics = load_enterprise_operations_data(uploaded_file)

    if df_efficiency_metrics is None or df_efficiency_metrics.empty:
        logger.warning("未能加载企业经营效率指标数据")
        st_obj.info("提示：数据模板中未找到'工业企业经营'sheet，无法生成企业经营效率指标分析")
    else:
        missing_efficiency_cols = [col for col in required_efficiency_cols if col not in df_efficiency_metrics.columns]

        if missing_efficiency_cols:
            logger.warning(f"企业经营效率指标分析缺少必需列: {missing_efficiency_cols}")
            st_obj.warning(f"数据中缺少部分企业经营效率指标列，无法生成完整图表")
        else:
            # 转换累计值为累计同比（年同比增长率，自动过滤1-2月）
            df_efficiency_metrics['每百元营业收入中的成本'] = convert_cumulative_to_yoy(df_efficiency_metrics[cost_per_hundred_cumulative_col])
            df_efficiency_metrics['每百元营业收入中的费用'] = convert_cumulative_to_yoy(df_efficiency_metrics[expense_per_hundred_cumulative_col])
            df_efficiency_metrics['每百元资产实现的营业收入'] = convert_cumulative_to_yoy(df_efficiency_metrics[asset_revenue_per_hundred_cumulative_col])
            df_efficiency_metrics['人均营业收入'] = convert_cumulative_to_yoy(df_efficiency_metrics[revenue_per_capita_cumulative_col])
            df_efficiency_metrics['产成品周转天数'] = convert_cumulative_to_yoy(df_efficiency_metrics[turnover_days_cumulative_col])
            df_efficiency_metrics['应收账款平均回收期'] = convert_cumulative_to_yoy(df_efficiency_metrics[receivable_period_cumulative_col])

            # 过滤掉1月和2月的数据（convert_cumulative_to_yoy已经将1-2月设为NaN）
            jan_feb_mask = df_efficiency_metrics.index.month.isin([1, 2])
            df_efficiency_metrics_filtered = df_efficiency_metrics[~jan_feb_mask].copy()

            df_efficiency_metrics = df_efficiency_metrics_filtered

            # 定义所有企业经营效率指标
            efficiency_indicators = ['每百元营业收入中的成本', '每百元营业收入中的费用', '每百元资产实现的营业收入', '人均营业收入', '产成品周转天数', '应收账款平均回收期']

            # 定义图表创建函数
            def create_efficiency_metrics_chart(df, variables, time_range, custom_start_date, custom_end_date):
                fig = create_enterprise_efficiency_metrics_chart(
                    df_efficiency=df,
                    time_range=time_range,
                    custom_start_date=custom_start_date,
                    custom_end_date=custom_end_date
                )
                return fig

            # 使用Fragment组件创建图表
            create_chart_with_time_selector_fragment(
                st_obj=st_obj,
                chart_id="enterprise_efficiency_metrics",
                state_namespace="monitoring.industrial.enterprise.efficiency_metrics",
                chart_title=None,
                chart_creator_func=create_efficiency_metrics_chart,
                chart_data=df_efficiency_metrics,
                chart_variables=efficiency_indicators,
                get_state_func=IndustrialStateManager.get,
                set_state_func=IndustrialStateManager.set
            )

            # 添加数据下载功能
            if not df_efficiency_metrics.empty:
                download_df = df_efficiency_metrics[efficiency_indicators].copy()
                download_df = download_df.sort_index(ascending=False)
                download_df = download_df.reset_index()
                first_col = download_df.columns[0]
                download_df.rename(columns={first_col: '时间'}, inplace=True)
                download_df['时间'] = download_df['时间'].dt.strftime('%Y-%m')
                download_columns = ['时间'] + efficiency_indicators
                download_df = download_df[download_columns]

                create_excel_download_button(
                    st_obj=st_obj,
                    data=download_df,
                    file_name="企业经营效率指标.xlsx",
                    sheet_name='企业经营效率指标',
                    button_key="efficiency_metrics_download_button",
                    column_ratio=(1, 3)
                )

