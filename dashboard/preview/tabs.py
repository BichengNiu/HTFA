# -*- coding: utf-8 -*-
"""
Preview模块统一Tab组件
通过配置驱动,一个函数支持所有频率的数据展示
包含时间序列Tab和数据概览Tab
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

from dashboard.preview.config import UNIFIED_FREQUENCY_CONFIGS, UI_TEXT
from dashboard.preview.calculators import calculate_summary
from dashboard.preview.plotting import plot_indicator
from dashboard.preview.components import create_filter_ui, display_summary_table
from dashboard.ui.utils.state_helpers import get_all_preview_data, set_preview_state, get_preview_state
from dashboard.preview.frequency_utils import get_indicator_frequencies, filter_indicators_by_frequency
from dashboard.preview.config import FREQUENCY_ORDER


@st.cache_data(show_spinner=False, ttl=1800, max_entries=10)
def calculate_indicator_statistics_cached(all_data_dict, indicator_maps):
    """缓存版本的统计计算（简化版：使用Streamlit自动缓存）

    优化：移除手动MD5缓存键，Streamlit会自动基于参数生成缓存键

    Args:
        all_data_dict: {频率名: DataFrame}字典
        indicator_maps: 包含industry、type、clean_industry的映射字典

    Returns:
        dict: 统计结果字典
    """
    # 调用原始的统计计算函数
    return calculate_indicator_statistics(all_data_dict, indicator_maps)


def _calculate_indicator_detail(indicator, frequency, series, industry, ind_type):
    """计算单个指标在某频率下的详细信息

    Args:
        indicator: 指标名称
        frequency: 频率名称（中文）
        series: 数据序列（已dropna）
        industry: 行业名称
        ind_type: 指标类型

    Returns:
        dict: 包含指标名称、行业、类型、频率、时间范围、有效值、缺失率
    """
    start_date = series.index.min()
    end_date = series.index.max()

    # 计算期望数据点数
    expected_points = calculate_expected_data_points(
        start_date, end_date, None, frequency
    )

    # 计算有效值和缺失率
    effective_values = len(series)
    missing_rate = (
        (expected_points - effective_values) / expected_points * 100
        if expected_points > 0 else 0
    )

    return {
        "指标名称": indicator,
        "行业": industry,
        "类型": ind_type,
        "频率": frequency,
        "时间范围": f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}",
        "有效值": effective_values,
        "缺失率": f"{missing_rate:.1f}%"
    }


def calculate_indicator_statistics(all_data_dict, indicator_maps):
    """优化版统计计算：反转循环顺序，降低复杂度

    优化前：O(indicators × frequencies²) - 先遍历指标，内部再调用get_indicator_frequencies遍历所有频率
    优化后：O(frequencies × indicators) - 先遍历频率，再遍历该频率的指标

    Args:
        all_data_dict: {频率名: DataFrame}字典
        indicator_maps: 包含industry、type、clean_industry的映射字典

    Returns:
        dict: {
            'all_indicators': set,
            'industry_stats': {行业名: 指标数量},
            'type_stats': {类型名: 指标数量},
            'frequency_stats': {频率名: 指标数量},
            'indicator_details': [{指标详情字典}]
        }
    """
    # 初始化统计字典
    stats = {
        'all_indicators': set(),
        'industry_stats': {},
        'type_stats': {},
        'frequency_stats': {},
        'indicator_details': []
    }

    # 已处理的指标集合（用于行业/类型统计去重）
    processed_for_industry_type = set()

    # 优化：先遍历频率，再遍历指标（降低复杂度）
    for freq_name, df in all_data_dict.items():
        if df is None or df.empty:
            continue

        # 统计该频率的指标数量
        freq_indicators = set(df.columns)
        stats['frequency_stats'][freq_name] = len(freq_indicators)

        # 遍历该频率的指标
        for indicator in freq_indicators:
            # 收集所有指标
            stats['all_indicators'].add(indicator)

            # 行业和类型统计（每个指标只统计一次）
            if indicator not in processed_for_industry_type:
                industry = indicator_maps['industry'].get(indicator, "未分类")
                stats['industry_stats'][industry] = stats['industry_stats'].get(industry, 0) + 1

                ind_type = indicator_maps['type'].get(indicator, "未分类")
                stats['type_stats'][ind_type] = stats['type_stats'].get(ind_type, 0) + 1

                processed_for_industry_type.add(indicator)
            else:
                # 已处理，直接获取
                industry = indicator_maps['industry'].get(indicator, "未分类")
                ind_type = indicator_maps['type'].get(indicator, "未分类")

            # 生成该频率的详细信息
            series = df[indicator].dropna()
            if not series.empty:
                detail = _calculate_indicator_detail(
                    indicator, freq_name, series, industry, ind_type
                )
                stats['indicator_details'].append(detail)

    return stats


def _create_summary_dataframe(stats_dict, column_names):
    """创建统计摘要DataFrame（私有辅助函数）"""
    df = pd.DataFrame(
        list(stats_dict.items()),
        columns=column_names
    )
    df = df.sort_values(column_names[1], ascending=False)

    # 添加合计行
    total_row = pd.DataFrame(
        [['合计', df[column_names[1]].sum()]],
        columns=column_names
    )
    df = pd.concat([df, total_row], ignore_index=True)

    return df


def render_statistics_summary(stats, st_module):
    """渲染统计摘要（行业、类型、频率统计）

    Args:
        stats: calculate_indicator_statistics返回的统计字典
        st_module: streamlit模块
    """
    col1, col2, col3 = st_module.columns(3)

    # 渲染行业统计
    with col1:
        st_module.markdown("**行业统计**")
        industry_df = _create_summary_dataframe(
            stats['industry_stats'],
            ['行业名称', '指标数量']
        )
        st_module.dataframe(industry_df, use_container_width=True, hide_index=True)

    # 渲染类型统计
    with col2:
        st_module.markdown("**类型统计**")
        type_df = _create_summary_dataframe(
            stats['type_stats'],
            ['指标类型', '指标数量']
        )
        st_module.dataframe(type_df, use_container_width=True, hide_index=True)

    # 渲染频率统计
    with col3:
        st_module.markdown("**频率统计**")
        if stats['frequency_stats']:
            frequency_df = _create_summary_dataframe(
                stats['frequency_stats'],
                ['数据频率', '指标数量']
            )
            st_module.dataframe(frequency_df, use_container_width=True, hide_index=True)
        else:
            st_module.warning("无频率数据")


def render_indicator_details(indicator_details, st_module):
    """渲染指标详情表格及下载功能

    Args:
        indicator_details: 指标详情列表
        st_module: streamlit模块
    """
    if not indicator_details:
        st_module.warning("未找到有效的指标数据")
        return

    # 转换为DataFrame
    detailed_df = pd.DataFrame(indicator_details)

    # 添加缺失率数值列用于排序（将"5.2%"转换为5.2）
    detailed_df['缺失率_数值'] = detailed_df['缺失率'].str.replace('%', '').astype(float)

    # 排序：按缺失率（由低到高）、行业、类型、频率
    detailed_df = detailed_df.sort_values(
        by=['缺失率_数值', '行业', '类型', '频率'],
        ascending=[True, True, True, True]
    )

    # 删除临时排序列
    detailed_df = detailed_df.drop(columns=['缺失率_数值'])

    # 重置索引并添加序号列（从1开始）
    detailed_df = detailed_df.reset_index(drop=True)
    detailed_df.insert(0, '序号', range(1, len(detailed_df) + 1))

    # 显示表格（隐藏默认index）
    st_module.dataframe(detailed_df, use_container_width=True, hide_index=True)

    # 添加间距
    st_module.markdown("")

    # 创建下载按钮区域
    col1, col2 = st_module.columns(2)

    with col1:
        # 下载指标详情统计表
        buffer1 = io.BytesIO()
        with pd.ExcelWriter(buffer1, engine='openpyxl') as writer:
            detailed_df.to_excel(writer, sheet_name='指标详情', index=False)

            # 设置列宽自适应
            worksheet = writer.sheets['指标详情']
            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) for cell in column if cell.value)
                worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

        st_module.download_button(
            label="下载指标详情统计表",
            data=buffer1.getvalue(),
            file_name=f"指标详情统计_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        # 下载全部原始数据
        # 收集所有频率的DataFrame（按顺序：日度、周度、旬度、月度、年度）
        frequency_data = {
            '日度': get_preview_state('daily_df'),
            '周度': get_preview_state('weekly_df'),
            '旬度': get_preview_state('ten_day_df'),
            '月度': get_preview_state('monthly_df'),
            '年度': get_preview_state('yearly_df')
        }

        buffer2 = io.BytesIO()
        with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
            for freq_name, df in frequency_data.items():
                if df is not None and not df.empty:
                    # 创建副本以避免修改原始数据
                    df_export = df.copy()

                    # 重置索引并重命名为"时间"
                    df_export = df_export.reset_index()
                    if len(df_export.columns) > 0 and df_export.columns[0] != '时间':
                        df_export.rename(columns={df_export.columns[0]: '时间'}, inplace=True)

                    # 格式化时间列为年-月-日格式
                    if '时间' in df_export.columns:
                        df_export['时间'] = pd.to_datetime(df_export['时间']).dt.strftime('%Y-%m-%d')

                    # 写入Excel（不包含index，因为已经转为时间列）
                    df_export.to_excel(writer, sheet_name=freq_name, index=False)

                    # 设置列宽自适应
                    worksheet = writer.sheets[freq_name]
                    for column in worksheet.columns:
                        max_length = max(len(str(cell.value)) for cell in column if cell.value)
                        worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

        st_module.download_button(
            label="下载全部原始数据",
            data=buffer2.getvalue(),
            file_name=f"全部原始数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def _get_available_types_for_industries(selected_industries, all_indicators, indicator_maps):
    """获取指定行业的可用类型（私有辅助函数）"""
    available_types = set()
    for indicator in all_indicators:
        industry = indicator_maps['industry'].get(indicator, "未分类")
        if industry in selected_industries:
            ind_type = indicator_maps['type'].get(indicator, "未分类")
            available_types.add(ind_type)
    return sorted(list(available_types))


def display_time_series_tab(st_module, frequency):
    """通用的时间序列数据Tab

    通过频率参数和配置驱动,支持周度/月度/日度/旬度/年度所有频率

    Args:
        st_module: Streamlit模块
        frequency: 数据频率 ('weekly'/'monthly'/'daily'/'ten_day'/'yearly')
    """
    # 1. 获取配置
    config = UNIFIED_FREQUENCY_CONFIGS[frequency]

    # 2. 获取数据（使用缓存）
    loaded_file = get_preview_state('data_loaded_files')
    preview_data = get_all_preview_data(cache_key=loaded_file)
    df = preview_data.get(config['df_key'], pd.DataFrame())

    if df is None or df.empty:
        st_module.info(config['empty_message'])
        return

    # 3. 获取行业和映射数据
    industries = preview_data.get(config['industries_key'], [])
    clean_industry_map = preview_data.get('clean_industry_map', {})
    source_map = preview_data.get('source_map', {})
    indicator_type_map = preview_data.get('indicator_type_map', {})

    # 4. 创建筛选UI
    selected_industry, selected_type, filtered_indicators, display_name = \
        create_filter_ui(
            st=st_module,
            industries=industries,
            df=df,
            indicator_type_map=indicator_type_map,
            clean_industry_map=clean_industry_map,
            source_map=source_map,
            key_prefix=config['key_prefix']
        )

    if not filtered_indicators:
        st_module.warning(UI_TEXT['no_data_warning'].format(display_name))
        return

    # 5. 显示摘要表
    st_module.markdown(f"**{display_name} - {config['display_name']}数据摘要**")

    # 计算摘要（使用Streamlit内置缓存，自动处理缓存逻辑）
    with st_module.spinner(f"正在计算 '{display_name}' 的{config['display_name']}摘要..."):
        filtered_df = df[filtered_indicators]
        try:
            # 获取单位和类型映射
            indicator_unit_map = preview_data.get('indicator_unit_map', {})
            # 调用calculate_summary时传入映射字典
            summary_table = calculate_summary(
                filtered_df,
                frequency,
                indicator_unit_map,
                indicator_type_map
            )
        except Exception as e:
            st_module.error(f"计算{config['display_name']}摘要时出错 ({display_name}): {e}")
            summary_table = pd.DataFrame()

    # 显示摘要表
    if not summary_table.empty:
        display_summary_table(
            st=st_module,
            summary_table=summary_table,
            sort_column=config['summary_config']['sort_column'],
            highlight_columns=config['summary_config']['highlight_columns'],
            percentage_columns=config['summary_config']['percentage_columns'],
            download_prefix=config['summary_config']['download_prefix']
        )

    # 6. 绘制图表
    current_year = datetime.now().year
    previous_year = current_year - 1
    indicator_unit_map = preview_data.get('indicator_unit_map', {})

    # 创建两列布局
    col1, col2 = st_module.columns(2)
    col_idx = 0

    for indicator in sorted(filtered_indicators):
        series = df[indicator].dropna()
        if not series.empty:
            current_col = col1 if col_idx % 2 == 0 else col2
            with current_col:
                with st_module.spinner(UI_TEXT['loading_message'].format(indicator)):
                    try:
                        unit = indicator_unit_map.get(indicator, None)
                        fig = plot_indicator(
                            series=series,
                            name=indicator,
                            frequency=frequency,
                            current_year=current_year,
                            previous_year=previous_year,
                            unit=unit
                        )
                        st_module.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st_module.error(f"为指标 '{indicator}' 生成图表时出错: {e}")
            col_idx += 1


def calculate_expected_data_points(start_date, end_date, full_df, frequency):
    """计算期望的数据点数

    用于计算缺失率时,应该相对于有效值在其时间范围内的理论数据点数

    Args:
        start_date: 开始日期
        end_date: 结束日期
        full_df: 完整的dataframe
        frequency: 数据频率(周度、月度、日度、旬度、年度)

    Returns:
        int: 应该存在的数据点数
    """
    if frequency == "周度":
        days_diff = (end_date - start_date).days
        theoretical_weeks = days_diff // 7 + 1
        return theoretical_weeks

    elif frequency == "月度":
        months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        return months_diff

    elif frequency == "日度":
        days_diff = (end_date - start_date).days + 1
        return days_diff

    elif frequency == "旬度":
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day

        start_ten = 1 if start_day <= 10 else (2 if start_day <= 20 else 3)
        end_ten = 1 if end_day <= 10 else (2 if end_day <= 20 else 3)

        if start_year == end_year and start_month == end_month:
            return end_ten - start_ten + 1

        start_month_tens = 3 - start_ten + 1
        end_month_tens = end_ten
        total_months = (end_year - start_year) * 12 + (end_month - start_month)
        middle_months = total_months - 1
        middle_tens = middle_months * 3 if middle_months > 0 else 0

        return start_month_tens + middle_tens + end_month_tens

    elif frequency == "年度":
        years_diff = end_date.year - start_date.year + 1
        return years_diff

    else:
        return 0


def display_overview_tab(st_module):
    """数据概览Tab（重构后）

    显示所有频率数据的统计概览、指标详情、数据下载功能
    重构后的版本：
    - 单次遍历替代5次遍历
    - 职责清晰的子函数
    - 减少约70%代码量

    Args:
        st_module: Streamlit模块
    """
    # 1. 获取数据并创建统一结构（使用工具函数，避免硬编码）
    loaded_file = get_preview_state('data_loaded_files')
    preview_data = get_all_preview_data(cache_key=loaded_file)

    # 使用工具函数创建频率到DataFrame的字典
    all_data_dict = {
        config['display_name']: preview_data.get(config['df_key'], pd.DataFrame())
        for config in UNIFIED_FREQUENCY_CONFIGS.values()
    }

    indicator_maps = {
        'industry': preview_data.get('indicator_industry_map', {}),
        'type': preview_data.get('indicator_type_map', {}),
        'clean_industry': preview_data.get('clean_industry_map', {})
    }

    # 2. 计算统计（一次遍历完成所有统计）- 使用缓存版本
    stats = calculate_indicator_statistics_cached(all_data_dict, indicator_maps)

    # 3. 第一部分：指标概览
    st_module.markdown("#### 指标概览")
    render_statistics_summary(stats, st_module)
    st_module.markdown("---")

    # 4. 第二部分：指标详情
    st_module.markdown("#### 指标详情")
    render_indicator_details(stats['indicator_details'], st_module)
