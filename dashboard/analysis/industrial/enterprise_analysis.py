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
    filter_data_by_time_range,
    load_profit_breakdown_data,
    load_enterprise_profit_data,
    create_excel_download_button
)

# 数据加载函数已移至utils.data_loader模块，删除重复代码
# 原函数：read_profit_breakdown_data → load_profit_breakdown_data
# 原函数：read_enterprise_profit_data → load_enterprise_profit_data


# ============================================================================
# 利润分组计算辅助函数（已重构 - 消除205行长函数）
# ============================================================================

def _build_industry_weights_mapping(df_weights: pd.DataFrame) -> dict:
    """
    构建行业名称到权重信息的映射

    Args:
        df_weights: 权重数据DataFrame

    Returns:
        dict: {指标名称: {'出口依赖': 值, '上中下游': 值}}
    """
    weights_mapping = {}
    for _, row in df_weights.iterrows():
        indicator_name = row['指标名称']
        if pd.notna(indicator_name):
            weights_mapping[indicator_name] = {
                '出口依赖': row['出口依赖'],
                '上中下游': row['上中下游']
            }
    return weights_mapping


def _match_industry_name(industry_col: str, weights_mapping: dict) -> Optional[str]:
    """
    匹配权重数据中的行业指标名称

    匹配策略：
    1. 精确匹配
    2. 正则表达式提取行业名称匹配
    3. 部分字符串匹配

    Args:
        industry_col: 利润拆解数据的列名
        weights_mapping: 权重映射字典

    Returns:
        匹配的指标名称，未找到则返回None
    """
    import re

    # 策略1：精确匹配
    if industry_col in weights_mapping:
        return industry_col

    # 策略2：使用正则表达式提取行业名称进行匹配
    # 从利润拆解列名中提取行业名称（格式: 规模以上工业企业:利润总额:行业名称:累计值）
    profit_match = re.search(r'规模以上工业企业:利润总额:([^:]+):累计值', industry_col)
    if profit_match:
        profit_industry = profit_match.group(1)

        # 在权重数据中查找匹配的行业
        for indicator in weights_mapping.keys():
            # 从权重指标名称中提取行业名称（格式: 规模以上工业增加值:行业名称:当月同比）
            weight_match = re.search(r'规模以上工业增加值:([^:]+):当月同比', indicator)
            if weight_match:
                weight_industry = weight_match.group(1)
                if profit_industry == weight_industry:
                    return indicator

    # 策略3：部分字符串匹配
    for indicator in weights_mapping.keys():
        if industry_col in indicator or indicator in industry_col:
            return indicator

    return None


def _classify_industries_by_groups(industry_columns: pd.Index, weights_mapping: dict) -> tuple:
    """
    按分组分类行业（出口依赖、上中下游）

    Args:
        industry_columns: 行业列索引
        weights_mapping: 权重映射字典

    Returns:
        (export_groups, stream_groups) 两个字典
        - export_groups: {分组名: [列名列表]}
        - stream_groups: {分组名: [列名列表]}
    """
    export_groups = {}
    stream_groups = {}

    for col in industry_columns:
        matched_indicator = _match_industry_name(col, weights_mapping)

        if matched_indicator and matched_indicator in weights_mapping:
            info = weights_mapping[matched_indicator]
            export_dep = info['出口依赖']
            stream_type = info['上中下游']

            # 分组到出口依赖
            if pd.notna(export_dep):
                if export_dep not in export_groups:
                    export_groups[export_dep] = []
                export_groups[export_dep].append(col)

            # 分组到上中下游
            if pd.notna(stream_type):
                if stream_type not in stream_groups:
                    stream_groups[stream_type] = []
                stream_groups[stream_type].append(col)

    return export_groups, stream_groups


def _aggregate_group_profits(df_profit_breakdown: pd.DataFrame,
                             export_groups: dict,
                             stream_groups: dict) -> tuple:
    """
    按照映射汇总各分组的利润累计值

    使用向量化操作提高性能（相比原实现使用循环）

    Args:
        df_profit_breakdown: 利润拆解数据
        export_groups: 出口依赖分组映射
        stream_groups: 上中下游分组映射

    Returns:
        (export_group_profits, stream_group_profits) 两个字典
        - export_group_profits: {分组名: pd.Series（利润总额）}
        - stream_group_profits: {分组名: pd.Series（利润总额）}
    """
    # 出口依赖分组利润汇总（向量化操作）
    export_group_profits = {}
    for group_name, columns in export_groups.items():
        if columns:
            # 使用向量化操作：直接对所有列求和（性能优化）
            valid_columns = [col for col in columns if col in df_profit_breakdown.columns]
            if valid_columns:
                group_profit = df_profit_breakdown[valid_columns].sum(axis=1)
                export_group_profits[group_name] = group_profit

    # 上中下游分组利润汇总（向量化操作）
    stream_group_profits = {}
    for group_name, columns in stream_groups.items():
        if columns:
            # 使用向量化操作
            valid_columns = [col for col in columns if col in df_profit_breakdown.columns]
            if valid_columns:
                group_profit = df_profit_breakdown[valid_columns].sum(axis=1)
                stream_group_profits[group_name] = group_profit

    return export_group_profits, stream_group_profits


def _calculate_group_yoy(group_profits: dict, prefix: str, result_df: pd.DataFrame) -> None:
    """
    计算各分组的年同比增速并添加到结果DataFrame

    Args:
        group_profits: {分组名: pd.Series（利润总额）}
        prefix: 列名前缀（'出口依赖_' 或 '上中下游_'）
        result_df: 结果DataFrame（会被原地修改）
    """
    for group_name, group_profit in group_profits.items():
        group_yoy = convert_cumulative_to_yoy(group_profit)
        result_df[f'{prefix}{group_name}_利润累计同比'] = group_yoy


def _calculate_total_profit_yoy(df_index: pd.Index,
                               stream_group_profits: dict,
                               result_df: pd.DataFrame) -> pd.Series:
    """
    计算总利润同比，使用上中下游分组的动态权重加权和

    公式：总同比 = Σ(分组同比 × 分组权重)
    其中，分组权重 = 分组利润 / 总利润

    Args:
        df_index: 时间索引
        stream_group_profits: 上中下游分组利润字典
        result_df: 包含分组同比的结果DataFrame

    Returns:
        pd.Series: 总利润累计同比
    """
    total_profit_yoy = pd.Series(index=df_index, dtype=float)
    total_profit_yoy[:] = 0.0

    for idx in df_index:
        # 1月和2月数据设为NaN（累计同比无意义）
        if hasattr(idx, 'month') and idx.month in [1, 2]:
            total_profit_yoy.loc[idx] = float('nan')
            continue

        # 计算总利润（所有上中下游分组利润之和）
        total_profit_at_idx = sum(
            stream_group_profits[group_name].loc[idx]
            for group_name in stream_group_profits.keys()
        )

        if total_profit_at_idx > 0:
            # 使用动态权重计算总同比
            total_weighted_sum = 0.0

            for group_name in stream_group_profits.keys():
                group_col = f'上中下游_{group_name}_利润累计同比'
                if group_col in result_df.columns:
                    group_profit = stream_group_profits[group_name].loc[idx]
                    group_yoy = result_df[group_col].loc[idx]

                    if pd.notna(group_yoy) and group_profit > 0:
                        # 动态权重 = 该分组利润 / 总利润
                        weight = group_profit / total_profit_at_idx
                        total_weighted_sum += weight * group_yoy

            total_profit_yoy.loc[idx] = total_weighted_sum
        else:
            total_profit_yoy.loc[idx] = float('nan')

    return total_profit_yoy


def calculate_grouped_profit_yoy(df_profit_breakdown: pd.DataFrame, df_weights: pd.DataFrame) -> pd.DataFrame:
    """
    根据权重数据中的分组映射，计算分组利润总额累计同比

    重构说明（P0-3优化）：
    - 原函数205行 → 拆分为6个辅助函数，每个函数职责单一
    - 提高了代码可读性和可维护性
    - 在 _aggregate_group_profits 中使用向量化操作，提升性能
    - 便于单元测试每个子步骤
    - 遵循KISS原则：每个函数只做一件事

    Args:
        df_profit_breakdown: 分上中下游利润拆解数据
        df_weights: 权重数据，包含出口依赖、上中下游映射

    Returns:
        DataFrame: 包含各分组利润总额累计同比的数据
    """
    try:
        # 数据验证
        if df_profit_breakdown is None or df_profit_breakdown.empty:
            return pd.DataFrame()
        if df_weights is None or df_weights.empty:
            return pd.DataFrame()
        if len(df_profit_breakdown.columns) < 2:
            return pd.DataFrame()

        # 获取分行业利润数据（第3列到最后一列）
        industry_columns = df_profit_breakdown.columns[2:]
        if len(industry_columns) == 0:
            return pd.DataFrame()

        # 初始化结果DataFrame
        result_df = pd.DataFrame(index=df_profit_breakdown.index)

        # 步骤1：构建行业权重映射
        weights_mapping = _build_industry_weights_mapping(df_weights)

        # 步骤2：按分组分类行业
        export_groups, stream_groups = _classify_industries_by_groups(
            industry_columns, weights_mapping
        )

        # 步骤3：按照映射汇总各分组的利润累计值（向量化操作）
        export_group_profits, stream_group_profits = _aggregate_group_profits(
            df_profit_breakdown, export_groups, stream_groups
        )

        # 步骤4：计算各分组的年同比
        _calculate_group_yoy(export_group_profits, '出口依赖_', result_df)
        _calculate_group_yoy(stream_group_profits, '上中下游_', result_df)

        # 步骤5：计算总利润同比（使用上中下游分组的动态权重加权和）
        total_profit_yoy = _calculate_total_profit_yoy(
            df_profit_breakdown.index, stream_group_profits, result_df
        )

        # 将总同比添加到结果的第一列
        result_df.insert(0, '工业企业累计利润总额累计同比', total_profit_yoy)

        return result_df

    except KeyError as e:
        logger.warning(f"数据列不存在，无法计算分组利润: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.warning(f"数据值错误，无法计算分组利润: {e}")
        return pd.DataFrame()
    except TypeError as e:
        logger.warning(f"数据类型错误，无法计算分组利润: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"计算分组利润时发生未预期错误: {e}", exc_info=True)
        return pd.DataFrame()


# ============================================================================
# 图表Fragment辅助函数（消除重复代码）
# ============================================================================

from typing import Callable, Tuple


def _create_enterprise_chart_fragment(
    st_obj,
    chart_title: str,
    state_key: str,
    chart_func: Callable,
    chart_data: pd.DataFrame,
    radio_key_base: str,
    chart_key: str,
    initialize_states: bool = False
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    企业经营分析图表Fragment辅助函数

    用于消除第1、2、3个图表中的重复代码（约136行 → ~21行）

    Args:
        st_obj: Streamlit对象
        chart_title: 图表标题（Markdown格式）
        state_key: 状态管理key
        chart_func: 图表创建函数
        chart_data: 图表数据
        radio_key_base: Radio控件key前缀
        chart_key: 图表显示key
        initialize_states: 是否预初始化状态（仅第1个图表需要）

    Returns:
        (time_range, custom_start, custom_end) 元组
    """
    @st_obj.fragment
    def render():
        st_obj.markdown(chart_title)

        from dashboard.analysis.industrial.industrial_analysis import get_industrial_state, set_industrial_state

        # 预初始化状态（仅第1个图表需要）
        if initialize_states:
            from dashboard.analysis.industrial.industrial_analysis import initialize_industrial_states
            initialize_industrial_states()

        current_time_range = get_industrial_state(state_key, "3年")
        time_range_options = ["1年", "3年", "5年", "全部", "自定义"]
        default_index = time_range_options.index(current_time_range) if current_time_range in time_range_options else 1

        time_range = st_obj.radio(
            "时间范围",
            time_range_options,
            index=default_index,
            horizontal=True,
            key=f"{radio_key_base}_time_range_selector",
            label_visibility="collapsed"
        )

        if time_range != current_time_range:
            set_industrial_state(state_key, time_range)

        custom_start = None
        custom_end = None
        if time_range == "自定义":
            col_start, col_end = st_obj.columns([1, 1])
            with col_start:
                custom_start = st_obj.text_input(
                    "开始年月",
                    placeholder="2020-01",
                    key=f"{radio_key_base}_custom_start_date"
                )
            with col_end:
                custom_end = st_obj.text_input(
                    "结束年月",
                    placeholder="2024-12",
                    key=f"{radio_key_base}_custom_end_date"
                )

        fig = chart_func(chart_data, time_range, custom_start, custom_end)

        if fig is not None:
            st_obj.plotly_chart(fig, use_container_width=True, key=chart_key)

        return time_range, custom_start, custom_end

    return render()


# ============================================================================
# 图表创建函数
# ============================================================================


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

        # 定义需要的四个指标
        required_indicators = [
            '规模以上工业企业:利润总额:累计同比',
            '规模以上工业增加值:累计同比',
            'PPI:累计同比',
            '规模以上工业企业:营业收入利润率:累计值年同比'
        ]

        # 检查哪些指标存在于数据中
        available_indicators = []

        # 首先尝试精确匹配
        for indicator in required_indicators:
            exact_match = [col for col in df_data.columns if col == indicator]
            if exact_match:
                available_indicators.append(exact_match[0])
                continue

            # 尝试关键词匹配
            keywords_to_match = []
            if '工业增加值' in indicator and '累计同比' in indicator:
                keywords_to_match = ['工业增加值', '累计同比']
            elif '利润总额' in indicator and '累计同比' in indicator:
                keywords_to_match = ['利润总额', '累计同比']
            elif '营业收入利润率' in indicator and '累计值年同比' in indicator:
                keywords_to_match = ['营业收入利润率', '累计值年同比']
            elif 'PPI' in indicator and '累计同比' in indicator:
                keywords_to_match = ['PPI', '累计同比']

            if keywords_to_match:
                partial_matches = [col for col in df_data.columns
                                 if all(keyword in str(col) for keyword in keywords_to_match)]
                if partial_matches:
                    available_indicators.append(partial_matches[0])
                    continue

            # 最后尝试单个关键词匹配
            single_keyword_matches = [col for col in df_data.columns
                                    if any(keyword in str(col) for keyword in indicator.split(':'))]
            if single_keyword_matches:
                available_indicators.append(single_keyword_matches[0])

        if not available_indicators:
            return None

        # 处理数据 - 改为更灵活的缺失数据策略
        complete_data_df = filtered_df[available_indicators].copy()

        # 转换所有列为数值型
        for col in complete_data_df.columns:
            complete_data_df[col] = pd.to_numeric(complete_data_df[col], errors='coerce')

        # 新策略：只要有至少3个指标有数据就显示（而不是要求全部4个）
        # 计算每行的非空值数量
        non_null_counts = complete_data_df.count(axis=1)

        # 保留至少有3个指标数据的行
        min_indicators = 3
        valid_rows = non_null_counts >= min_indicators
        complete_data_df = complete_data_df[valid_rows]

        if complete_data_df.empty:
            return None

        # 创建图表
        fig = go.Figure()

        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        def get_legend_name(indicator):
            """将完整的指标名称转换为简化的图例名称"""
            legend_mapping = {
                '规模以上工业企业:利润总额:累计同比': '利润总额累计同比',
                'PPI:累计同比': 'PPI累计同比',
                '规模以上工业增加值:累计同比': '工业增加值累计同比',
                '规模以上工业企业:营业收入利润率:累计值年同比': '营业收入利润率累计同比'
            }
            return legend_mapping.get(indicator, indicator)

        # 定义哪些指标用条形图（堆积），哪些用线图
        bar_indicators = [
            '规模以上工业增加值:累计同比',
            'PPI:累计同比',
            '规模以上工业企业:营业收入利润率:累计值年同比'
        ]
        line_indicators = [
            '规模以上工业企业:利润总额:累计同比'
        ]

        # 先添加线图指标
        for i, indicator in enumerate(available_indicators):
            if indicator in complete_data_df.columns:
                # 获取数据，保留NaN值用于正确处理缺失点
                y_data = complete_data_df[indicator].dropna()

                if not y_data.empty:
                    # 检查是否应该用线图
                    is_line_indicator = any(line_key in indicator for line_key in line_indicators)

                    if is_line_indicator:
                        # 添加线图，使用xperiod让数据点居中对齐到月份
                        fig.add_trace(go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode='lines+markers',
                            name=get_legend_name(indicator),
                            line=dict(width=2.5, color=colors[i % len(colors)]),
                            marker=dict(size=4),
                            connectgaps=False,  # 不连接缺失点
                            xperiod="M1",  # 周期为1个月
                            xperiodalignment="middle",  # 数据点居中对齐
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
                    is_bar_indicator = any(bar_key in indicator for bar_key in bar_indicators)

                    if is_bar_indicator:
                        # 计算渐变透明度：按原始指标顺序
                        opacity_value = 0.9 - (i * 0.15)  # 0.9, 0.75, 0.6, 0.45...
                        opacity_value = max(0.5, opacity_value)  # 最小透明度为0.5

                        # 添加堆积条形图，使用xperiod让柱子居中对齐到月份
                        fig.add_trace(go.Bar(
                            x=y_data.index,
                            y=y_data,
                            name=get_legend_name(indicator),
                            marker_color=colors[i % len(colors)],
                            opacity=opacity_value,
                            xperiod="M1",  # 周期为1个月
                            xperiodalignment="middle",  # 柱子居中对齐
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
            title="",  # No x-axis title
            type="date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="M3",  # 3-month intervals
            hoverformat="%Y-%m"  # 修复hover显示：只显示年月
        )

        # Set the range to actual data if available
        if min_date and max_date:
            xaxis_config['range'] = [min_date, max_date]

        # 更新布局 - 移除所有文字，设置堆积条形图
        fig.update_layout(
            title="",  # No chart title
            xaxis=xaxis_config,
            yaxis=dict(
                title="",  # No y-axis title
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            barmode='relative',  # 设置为相对堆积模式，正确处理正负值堆叠
            hovermode='x unified',
            height=500,
            margin=dict(l=50, r=50, t=30, b=80),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
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


def create_profit_chart_unified(df_grouped_profit: pd.DataFrame,
                               filter_prefix: str,
                               filter_suffix: str = '_利润累计同比',
                               time_range: str = "3年",
                               custom_start_date: Optional[str] = None,
                               custom_end_date: Optional[str] = None) -> Optional[go.Figure]:
    """
    统一的利润图表创建函数（消除重复代码）

    这个函数替代了：
    - create_upstream_downstream_profit_chart (132行)
    - create_export_dependency_profit_chart (132行)

    Args:
        df_grouped_profit: 包含分组利润累计同比数据的DataFrame
        filter_prefix: 列名前缀（如 '上中下游_' 或 '出口依赖_'）
        filter_suffix: 列名后缀（默认 '_利润累计同比'）
        time_range: 时间范围选择
        custom_start_date: 自定义开始日期
        custom_end_date: 自定义结束日期

    Returns:
        plotly Figure对象
    """
    try:
        if df_grouped_profit is None or df_grouped_profit.empty:
            return None

        # 应用时间过滤
        filtered_df = filter_data_by_time_range(df_grouped_profit, time_range, custom_start_date, custom_end_date)

        if filtered_df.empty:
            return None

        # 创建图表
        fig = go.Figure()

        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # 添加工业企业累计利润总额累计同比（线图）
        if '工业企业累计利润总额累计同比' in filtered_df.columns:
            total_profit_data = filtered_df['工业企业累计利润总额累计同比'].dropna()
            if not total_profit_data.empty:
                fig.add_trace(go.Scatter(
                    x=total_profit_data.index,
                    y=total_profit_data,
                    mode='lines+markers',
                    name='工业企业累计利润总额累计同比',
                    line=dict(width=2.5, color=colors[0]),
                    marker=dict(size=4),
                    connectgaps=False,
                    xperiod="M1",  # 周期为1个月
                    xperiodalignment="middle",  # 数据点居中对齐
                    hovertemplate='<b>工业企业累计利润总额累计同比</b><br>' +
                                  '时间: %{x|%Y年%m月}<br>' +
                                  '数值: %{y:.2f}%<extra></extra>'
                ))

        # 添加分组数据（条形图）
        # 收集所有符合条件的分组列
        group_columns = []
        for col in filtered_df.columns:
            if filter_prefix in col and filter_suffix in col:
                group_columns.append(col)

        # 按名称排序以保持一致性
        group_columns.sort()

        color_index = 1  # 从第二个颜色开始

        for col in group_columns:
            group_data = filtered_df[col].dropna()
            if not group_data.empty:
                # 清理列名用于显示
                if col.startswith(filter_prefix) and col.endswith(filter_suffix):
                    # 提取中间部分
                    display_name = col[len(filter_prefix):-len(filter_suffix)]
                else:
                    # 回退逻辑
                    display_name = col.replace(filter_prefix, '').replace(filter_suffix, '')

                # 计算透明度
                opacity_value = 0.9 - ((color_index - 1) * 0.15)
                opacity_value = max(0.5, opacity_value)

                fig.add_trace(go.Bar(
                    x=group_data.index,
                    y=group_data,
                    name=display_name,
                    marker_color=colors[color_index % len(colors)],
                    opacity=opacity_value,
                    xperiod="M1",  # 周期为1个月
                    xperiodalignment="middle",  # 柱子居中对齐
                    hovertemplate=f'<b>{display_name}</b><br>' +
                                  '时间: %{x|%Y年%m月}<br>' +
                                  '数值: %{y:.2f}%<extra></extra>'
                ))
                color_index += 1

        # 设置x轴配置
        min_date = filtered_df.index.min() if not filtered_df.empty else None
        max_date = filtered_df.index.max() if not filtered_df.empty else None

        xaxis_config = dict(
            title="",
            type="date",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="M3",
            hoverformat="%Y-%m"  # 修复hover显示：只显示年月
        )

        if min_date and max_date:
            xaxis_config['range'] = [min_date, max_date]

        # 更新布局
        fig.update_layout(
            title="",
            xaxis=xaxis_config,
            yaxis=dict(
                title="",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            barmode='relative',
            hovermode='x unified',
            height=500,
            margin=dict(l=50, r=50, t=30, b=100),  # 增加底部边距确保图例空间充足
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,  # 固定图例位置，避免自动调整
                xanchor="center",
                x=0.5,
                tracegroupgap=0  # 减少trace组间距
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    except KeyError as e:
        logger.warning(f"创建利润图表时列不存在: {e}")
        return None
    except ValueError as e:
        logger.warning(f"创建利润图表时数据值错误: {e}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning("利润数据为空")
        return None
    except Exception as e:
        logger.error(f"创建利润图表时发生未预期错误: {e}", exc_info=True)
        return None


# 便捷接口函数
def create_upstream_downstream_profit_chart(df_grouped_profit: pd.DataFrame, time_range: str = "3年",
                                          custom_start_date: Optional[str] = None,
                                          custom_end_date: Optional[str] = None) -> Optional[go.Figure]:
    """创建上中下游分组利润总额累计同比图表"""
    return create_profit_chart_unified(
        df_grouped_profit,
        filter_prefix='上中下游_',
        filter_suffix='_利润累计同比',
        time_range=time_range,
        custom_start_date=custom_start_date,
        custom_end_date=custom_end_date
    )


def create_export_dependency_profit_chart(df_grouped_profit: pd.DataFrame, time_range: str = "3年",
                                         custom_start_date: Optional[str] = None,
                                         custom_end_date: Optional[str] = None) -> Optional[go.Figure]:
    """创建出口依赖分组利润总额累计同比图表"""
    return create_profit_chart_unified(
        df_grouped_profit,
        filter_prefix='出口依赖_',
        filter_suffix='_利润累计同比',
        time_range=time_range,
        custom_start_date=custom_start_date,
        custom_end_date=custom_end_date
    )


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

    # Remove title text as requested

    if uploaded_file is None:
        # Remove info text as requested
        return

    # 读取企业利润拆解数据
    df_profit = load_enterprise_profit_data(uploaded_file)

    if df_profit is None:
        return

    # Remove success message and data overview as requested

    # 处理"规模以上工业企业:营业收入利润率:累计值"转换为年同比
    profit_margin_col = None
    for col in df_profit.columns:
        if '营业收入利润率' in str(col) and '累计值' in str(col):
            profit_margin_col = col
            break

    if profit_margin_col:
        # Remove processing info text as requested

        # 转换为年同比
        yoy_data = convert_cumulative_to_yoy(df_profit[profit_margin_col])

        # 创建新的列名
        yoy_col_name = profit_margin_col.replace('累计值', '累计值年同比')
        df_profit[yoy_col_name] = yoy_data

    # Remove section title as requested

    # 图表1：利润总额拆解（使用统一Fragment组件 - P1-4优化）
    time_range, custom_start, custom_end = _create_enterprise_chart_fragment(
        st_obj=st_obj,
        chart_title="#### 利润总额拆解",
        state_key='enterprise_time_range_chart1',
        chart_func=create_enterprise_indicators_chart,
        chart_data=df_profit,
        radio_key_base="industrial_enterprise_operations_fragment",
        chart_key="enterprise_operations_chart_1_fragment",
        initialize_states=True  # 第1个图表需要预初始化
    )

    # 添加第一个图表的数据下载功能 - 使用统一下载工具
    download_df1 = filter_data_by_time_range(df_profit, time_range, custom_start, custom_end)

    if not download_df1.empty:
        create_excel_download_button(
            st_obj=st_obj,
            data=download_df1,
            file_name=f"企业经营指标_{time_range}.xlsx",
            sheet_name='企业经营指标',
            button_key="industrial_enterprise_operations_download_data_button",
            column_ratio=(1, 3)
        )

    # 添加第二个和第三个图表：分组利润总额累计同比
    if df_weights is not None:
        # 读取分上中下游利润拆解数据
        df_profit_breakdown = load_profit_breakdown_data(uploaded_file)

        if df_profit_breakdown is not None:
            # 计算分组利润总额累计同比
            df_grouped_profit = calculate_grouped_profit_yoy(df_profit_breakdown, df_weights)

            if not df_grouped_profit.empty:
                # 添加一些间距
                st_obj.markdown("---")

                # 图表2：上中下游分组（使用统一Fragment组件 - P1-4优化）
                time_range_2, custom_start_2, custom_end_2 = _create_enterprise_chart_fragment(
                    st_obj=st_obj,
                    chart_title="#### 上中下游分组利润总额累计同比",
                    state_key='enterprise_time_range_chart2',
                    chart_func=create_upstream_downstream_profit_chart,
                    chart_data=df_grouped_profit,
                    radio_key_base="industrial_enterprise_operations_chart2_fragment",
                    chart_key="enterprise_operations_chart_2_fragment"
                )

                # 添加第二个图表的独立数据下载功能 - 使用统一下载工具
                download_df2 = filter_data_by_time_range(df_grouped_profit, time_range_2, custom_start_2, custom_end_2)

                if not download_df2.empty:
                    # 筛选上中下游相关列
                    upstream_downstream_cols = ['工业企业累计利润总额累计同比'] + [col for col in download_df2.columns if '上中下游_' in col and '利润累计同比' in col]
                    download_df2_filtered = download_df2[upstream_downstream_cols]

                    create_excel_download_button(
                        st_obj=st_obj,
                        data=download_df2_filtered,
                        file_name=f"上中下游分组利润累计同比_{time_range_2}.xlsx",
                        sheet_name='上中下游分组利润累计同比',
                        button_key="industrial_enterprise_operations_download_upstream_downstream_button",
                        column_ratio=(1, 3)
                    )

                # 添加一些间距
                st_obj.markdown("---")

                # 图表3：出口依赖分组（使用统一Fragment组件 - P1-4优化）
                time_range_3, custom_start_3, custom_end_3 = _create_enterprise_chart_fragment(
                    st_obj=st_obj,
                    chart_title="#### 出口依赖分组利润总额累计同比",
                    state_key='enterprise_time_range_chart3',
                    chart_func=create_export_dependency_profit_chart,
                    chart_data=df_grouped_profit,
                    radio_key_base="industrial_enterprise_operations_chart3_fragment",
                    chart_key="enterprise_operations_chart_3_fragment"
                )

                # 添加第三个图表的独立数据下载功能 - 使用统一下载工具
                download_df3 = filter_data_by_time_range(df_grouped_profit, time_range_3, custom_start_3, custom_end_3)

                if not download_df3.empty:
                    # 筛选出口依赖相关列
                    export_dependency_cols = ['工业企业累计利润总额累计同比'] + [col for col in download_df3.columns if '出口依赖_' in col and '利润累计同比' in col]
                    download_df3_filtered = download_df3[export_dependency_cols]

                    create_excel_download_button(
                        st_obj=st_obj,
                        data=download_df3_filtered,
                        file_name=f"出口依赖分组利润累计同比_{time_range_3}.xlsx",
                        sheet_name='出口依赖分组利润累计同比',
                        button_key="industrial_enterprise_operations_download_export_dependency_button",
                        column_ratio=(1, 3)
                    )


def render_enterprise_operations_tab(st_obj):
    """
    企业经营分析标签页

    Args:
        st_obj: Streamlit 对象
    """
    # 直接调用新的分析函数
    render_enterprise_operations_analysis_with_data(st_obj, None, None)
