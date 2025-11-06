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
            title="",
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
            margin=dict(l=80, r=50, t=30, b=120),
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
        logger.info(f"[调试] create_enterprise_indicators_chart开始: df形状={df_data.shape}")
        logger.info(f"[调试] df_data索引类型={type(df_data.index)}, 索引前5个值={df_data.index[:5].tolist() if len(df_data.index) > 0 else []}")
        logger.info(f"[调试] 时间范围={time_range}, 自定义开始={custom_start_date}, 自定义结束={custom_end_date}")

        # Apply time filtering first
        filtered_df = filter_data_by_time_range(df_data, time_range, custom_start_date, custom_end_date)
        logger.info(f"[调试] 过滤后df形状={filtered_df.shape}")
        logger.info(f"[调试] 过滤后索引类型={type(filtered_df.index)}, 索引前5个值={filtered_df.index[:5].tolist() if len(filtered_df.index) > 0 else []}")

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
        logger.info(f"[调试] 可用指标数量={len(available_indicators)}, 指标={available_indicators}")

        if not available_indicators:
            logger.warning("[调试] 没有可用指标，返回None")
            return None

        # 处理数据 - 改为更灵活的缺失数据策略
        complete_data_df = filtered_df[available_indicators].copy()
        logger.info(f"[调试] complete_data_df初始形状={complete_data_df.shape}")
        logger.info(f"[调试] complete_data_df前5行:\n{complete_data_df.head()}")

        # 转换所有列为数值型
        for col in complete_data_df.columns:
            complete_data_df[col] = pd.to_numeric(complete_data_df[col], errors='coerce')

        logger.info(f"[调试] 转换为数值型后，各列非空值数量:\n{complete_data_df.count()}")

        # 新策略：降低要求，只要有至少1个指标有数据就显示
        # 计算每行的非空值数量
        non_null_counts = complete_data_df.count(axis=1)
        logger.info(f"[调试] 每行非空值统计:\n{non_null_counts.describe()}")
        logger.info(f"[调试] 非空值数量分布:\n{non_null_counts.value_counts().sort_index()}")

        # 保留至少有1个指标数据的行（降低要求）
        min_indicators = 1
        valid_rows = non_null_counts >= min_indicators
        logger.info(f"[调试] 满足最少{min_indicators}个指标的行数: {valid_rows.sum()} / {len(valid_rows)}")

        complete_data_df = complete_data_df[valid_rows]
        logger.info(f"[调试] 过滤后complete_data_df形状={complete_data_df.shape}")

        if complete_data_df.empty:
            logger.warning("[调试] 过滤后数据为空，返回None")
            return None

        # 创建图表
        fig = go.Figure()

        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        def get_legend_name(indicator):
            """将完整的指标名称转换为简化的图例名称"""
            legend_mapping = {
                '中国:利润总额:规模以上工业企业:累计同比': '利润总额累计同比',
                'PPI:累计同比': 'PPI累计同比',
                '中国:工业增加值:规模以上工业企业:累计同比': '工业增加值累计同比',
                '规模以上工业企业:营业收入利润率:累计同比': '营业收入利润率累计同比'
            }
            return legend_mapping.get(indicator, indicator)

        # 定义哪些指标用条形图（堆积），哪些用线图
        bar_indicators = [
            '中国:工业增加值:规模以上工业企业:累计同比',
            'PPI:累计同比',
            '规模以上工业企业:营业收入利润率:累计同比'
        ]
        line_indicators = [
            '中国:利润总额:规模以上工业企业:累计同比'
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
                    is_bar_indicator = any(bar_key in indicator for bar_key in bar_indicators)

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
            title="",  # No chart title
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
            margin=dict(l=80, r=50, t=30, b=120),
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
        logger.error("[调试] load_enterprise_profit_data返回None")
        st_obj.error("错误：无法加载企业利润数据")
        return

    logger.info(f"[调试] 加载企业利润数据成功，形状: {df_profit.shape}")
    logger.info(f"[调试] df_profit列名: {list(df_profit.columns)}")
    logger.info(f"[调试] df_profit前3行:\n{df_profit.head(3)}")

    # 处理"规模以上工业企业:营业收入利润率:累计值"转换为年同比
    profit_margin_col = None
    for col in df_profit.columns:
        if '营业收入利润率' in str(col) and '累计值' in str(col):
            profit_margin_col = col
            break

    # 检查索引是否已经是DatetimeIndex（load_enterprise_profit_data已经处理过）
    if not isinstance(df_profit.index, pd.DatetimeIndex):
        # 如果不是，则手动处理日期列并设置为索引
        date_col = df_profit.columns[0]
        df_profit[date_col] = pd.to_datetime(df_profit[date_col], errors='coerce')
        df_profit_with_index = df_profit.set_index(date_col)
    else:
        # 如果已经是DatetimeIndex，直接使用
        df_profit_with_index = df_profit
        logger.info("[调试] 数据已包含DatetimeIndex，无需重新设置")

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

    # 调试日志：打印过滤前后的数据月份信息
    logger.info(f"[调试] 过滤1月2月前数据行数: {len(df_profit_with_index)}, 过滤后数据行数: {len(df_profit)}")
    logger.info(f"[调试] 过滤后数据月份分布: {df_profit.index.month.value_counts().sort_index().to_dict()}")
    if len(df_profit) > 0:
        logger.info(f"[调试] 过滤后数据日期范围: {df_profit.index.min()} 到 {df_profit.index.max()}")
        logger.info(f"[调试] 过滤后df_profit形状: {df_profit.shape}")
        logger.info(f"[调试] 过滤后df_profit列名: {list(df_profit.columns)}")
    else:
        logger.error("[调试] 过滤1月2月后，数据为空！")
        st_obj.error("错误：过滤1月2月后，数据为空！请检查上传的Excel文件。")
        return

    # 图表1：工业企业利润拆解（添加时间筛选器）
    st_obj.markdown("#### 工业企业利润分析")

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

    # 添加调试信息显示
    logger.info(f"[调试] df_profit形状: {df_profit.shape}")
    logger.info(f"[调试] df_profit列名: {list(df_profit.columns)}")
    logger.info(f"[调试] 可用变量: {available_vars}")

    if not available_vars:
        st_obj.error(f"错误：未找到任何必需的指标列！数据列名: {list(df_profit.columns[:5])}")
        return

    # 定义图表创建函数（包装create_enterprise_indicators_chart以符合Fragment接口）
    def create_chart1(df, variables, time_range, custom_start_date, custom_end_date):
        logger.info(f"[调试] create_chart1调用: df形状={df.shape}, variables={variables}")
        logger.info(f"[调试] df列名: {list(df.columns)}")
        fig = create_enterprise_indicators_chart(
            df_data=df,
            time_range=time_range,
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        logger.info(f"[调试] 图表创建结果: {fig is not None}")
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
    # 分割线 - 分行业利润拆解
    # ============================================================================
    st_obj.markdown("---")

    # 加载分行业利润数据
    from dashboard.analysis.industrial.utils import load_industry_profit_data
    from dashboard.analysis.industrial.utils.contribution_calculator import calculate_profit_contributions

    df_industry_profit = load_industry_profit_data(uploaded_file)

    if df_industry_profit is None:
        logger.error("[调试] 无法加载分行业利润数据")
        st_obj.error("错误：无法加载分行业利润数据")
        return

    logger.info(f"[调试] 分行业利润数据形状: {df_industry_profit.shape}")
    logger.info(f"[调试] 分行业利润数据列数: {len(df_industry_profit.columns)}")

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

        logger.info(f"[调试] 利润拉动率计算完成")
        logger.info(f"[调试] 上中下游分组数: {len(stream_contribution_df.columns)}")
        logger.info(f"[调试] 验证通过: {validation_result['passed']}")

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

