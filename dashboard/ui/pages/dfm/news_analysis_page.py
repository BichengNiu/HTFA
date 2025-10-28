# -*- coding: utf-8 -*-
"""
DFM新闻分析页面 - 纽约联储风格

完全重构版本，采用上下布局：图在上，表在下
"""

import streamlit as st
import pandas as pd
import os
import traceback
from datetime import datetime
from typing import Dict, Any

# 导入统一状态管理
from dashboard.core import get_global_dfm_manager
from dashboard.models.DFM.decomp import execute_news_analysis


def get_dfm_manager():
    """获取DFM模块管理器实例"""
    try:
        dfm_manager = get_global_dfm_manager()
        if dfm_manager is None:
            raise RuntimeError("全局DFM管理器不可用")
        return dfm_manager
    except Exception as e:
        print(f"[DFM News Analysis] Error getting DFM manager: {e}")
        raise RuntimeError(f"DFM管理器获取失败: {e}")


def get_dfm_state(key, default=None):
    """获取DFM状态值 - 仅从news_analysis命名空间读取"""
    dfm_manager = get_dfm_manager()
    # 所有键都从news_analysis命名空间获取（不再跨命名空间读取）
    return dfm_manager.get_dfm_state('news_analysis', key, default)


def set_dfm_state(key, value):
    """设置DFM状态值"""
    dfm_manager = get_dfm_manager()
    return dfm_manager.set_dfm_state('news_analysis', key, value)


def render_dfm_news_analysis_page(st_module: Any) -> Dict[str, Any]:
    """
    渲染纽约联储风格的新闻分析页面

    Args:
        st_module: Streamlit模块

    Returns:
        渲染结果字典
    """
    try:

        # 文件上传区域
        st_module.markdown("### 模型文件上传")
        st_module.info("请上传模型训练模块导出的模型文件和元数据文件")

        col_model, col_metadata = st_module.columns(2)

        with col_model:
            st_module.markdown("**模型文件 (.joblib)**")
            uploaded_model_file = st_module.file_uploader(
                "选择模型文件",
                type=['joblib'],
                key="news_model_upload",
                help="上传模型训练模块导出的DFM模型文件"
            )

            if uploaded_model_file:
                set_dfm_state("dfm_model_file_news", uploaded_model_file)
            else:
                existing_model = get_dfm_state('dfm_model_file_news', None)
                if existing_model is not None and hasattr(existing_model, 'name'):
                    st_module.info(f"当前文件: {existing_model.name}")

        with col_metadata:
            st_module.markdown("**元数据文件 (.pkl)**")
            uploaded_metadata_file = st_module.file_uploader(
                "选择元数据文件",
                type=['pkl'],
                key="news_metadata_upload",
                help="上传模型训练模块导出的元数据文件"
            )

            if uploaded_metadata_file:
                set_dfm_state("dfm_metadata_file_news", uploaded_metadata_file)
            else:
                existing_metadata = get_dfm_state('dfm_metadata_file_news', None)
                if existing_metadata is not None and hasattr(existing_metadata, 'name'):
                    st_module.info(f"当前文件: {existing_metadata.name}")

        st_module.markdown("---")

        # 检查文件是否已上传
        model_file = get_dfm_state('dfm_model_file_news', None)
        metadata_file = get_dfm_state('dfm_metadata_file_news', None)

        if model_file is None or metadata_file is None:
            missing = []
            if model_file is None:
                missing.append("模型文件")
            if metadata_file is None:
                missing.append("元数据文件")
            st_module.warning(f"缺少必要文件: {', '.join(missing)}。请上传后再继续。")
            return {'status': 'error', 'error': '缺少模型文件'}

        # 参数设置区域，获取用户选择的目标月份
        selected_target_month = _render_parameter_section(st_module)

        # 执行按钮和结果渲染
        if st_module.button("执行新闻分析", type="primary", use_container_width=True, key="execute_news_analysis_btn"):
            result = _execute_analysis(st_module, model_file, metadata_file, selected_target_month)
            # 如果分析成功，直接渲染结果，不使用rerun
            if result and result.get('returncode') == 0:
                _render_results_direct(st_module, result)
        else:
            # 非按钮点击时，检查是否有已完成的分析结果
            if get_dfm_state('news_analysis_completed', False):
                _render_results(st_module)

        return {'status': 'success', 'page': 'news_analysis'}

    except Exception as e:
        st_module.error(f"页面渲染失败: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def _render_parameter_section(st_module):
    """渲染参数设置区域，返回用户选择的目标月份"""

    # 从状态中获取已选择的目标月份，如果没有则使用当前月份
    current_target_month = get_dfm_state('news_target_month', None)
    if current_target_month:
        # 解析已保存的月份字符串
        try:
            default_value = datetime.strptime(current_target_month, '%Y-%m').replace(day=1)
        except:
            default_value = datetime.now().replace(day=1)
    else:
        default_value = datetime.now().replace(day=1)

    target_date = st_module.date_input(
        "**目标月份**",
        value=default_value,
        key="news_target_month_selector",
        help="选择要分析的目标月份"
    )

    # 更新状态并返回选择的月份
    selected_month = target_date.strftime('%Y-%m')
    set_dfm_state('news_target_month', selected_month)
    print(f"[UI DEBUG] 用户选择的目标月份: {selected_month}")

    return selected_month


def _execute_analysis(st_module, model_file, metadata_file, target_month):
    """执行新闻分析，返回结果"""
    with st_module.spinner("正在执行新闻分析，请稍候..."):
        try:
            print(f"[UI DEBUG] 执行分析使用的目标月份: {target_month}")

            # 调用后端API
            result = execute_news_analysis(
                dfm_model_file_content=model_file.getbuffer(),
                dfm_metadata_file_content=metadata_file.getbuffer(),
                target_month=target_month,
                plot_start_date=None,
                plot_end_date=None
            )

            if result['returncode'] == 0:
                # 确保result是可序列化的字典
                serializable_result = {
                    'returncode': result.get('returncode', 0),
                    'csv_paths': result.get('csv_paths', {}),
                    'csv_contents': result.get('csv_contents', {}),  # 保存CSV内容到状态
                    'plot_paths': result.get('plot_paths', {}),
                    'summary': result.get('summary', {}),
                    'data_flow': result.get('data_flow', []),
                    'workspace_dir': result.get('workspace_dir', '')
                }

                # 保存结果到状态（用于页面刷新后恢复）
                set_dfm_state('news_analysis_completed', True)
                set_dfm_state('news_analysis_result', serializable_result)
                set_dfm_state('news_target_month_executed', target_month)

                st_module.success("分析执行成功！")
                return serializable_result
            else:
                st_module.error(f"分析执行失败: {result.get('error_message', '未知错误')}")
                return None

        except Exception as e:
            st_module.error(f"执行过程中发生错误: {str(e)}")
            st_module.code(traceback.format_exc(), language="python")
            return None


def _render_results_direct(st_module, result):
    """直接渲染分析结果（不从状态读取）"""
    if not result or result.get('returncode') != 0:
        st_module.error("分析结果不可用")
        return

    st_module.markdown("---")
    st_module.markdown("### 分析结果")

    # 1. 统计摘要卡片（关键指标和行业分解）
    _render_summary_cards(st_module, result)

    st_module.markdown("---")

    # 2. 数据流表格区域
    _render_data_flow_table(st_module, result)

    st_module.markdown("---")

    # 3. 数据下载区域
    _render_download_section(st_module, result)


def _render_results(st_module):
    """渲染分析结果（从状态读取）"""
    result = get_dfm_state('news_analysis_result', {})

    if not result or result.get('returncode') != 0:
        st_module.error("分析结果不可用")
        return

    st_module.markdown("---")

    # 1. 统计摘要卡片（关键指标和行业分解）
    _render_summary_cards(st_module, result)

    st_module.markdown("---")

    # 2. 数据流表格区域
    _render_data_flow_table(st_module, result)

    st_module.markdown("---")

    # 3. 数据下载区域
    _render_download_section(st_module, result)


def _render_data_flow_table(st_module, result):
    """渲染数据流表格"""
    data_flow = result.get('data_flow', [])

    if not data_flow:
        st_module.info("没有数据流信息")
        return

    # 获取用户选择的目标月份
    target_month = get_dfm_state('news_target_month_executed', datetime.now().strftime('%Y-%m'))

    # 调试信息：显示data_flow的前几个日期
    if data_flow:
        print(f"[DEBUG] data_flow总数: {len(data_flow)}")
        print(f"[DEBUG] target_month: {target_month}")
        print(f"[DEBUG] 前5个date_entry的日期: {[entry['date'] for entry in data_flow[:5]]}")

    # 计算目标月份的日期范围
    try:
        target_date = pd.to_datetime(target_month + '-01')
        # 目标月份的第一天和最后一天
        month_start = target_date
        # 计算下个月的第一天，然后减一天得到当月最后一天
        if target_date.month == 12:
            month_end = pd.Timestamp(year=target_date.year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=target_date.year, month=target_date.month + 1, day=1) - pd.Timedelta(days=1)

        print(f"[DEBUG] 目标月份日期范围: {month_start.strftime('%Y-%m-%d')} 到 {month_end.strftime('%Y-%m-%d')}")
    except Exception as e:
        st_module.error(f"日期解析失败: {e}")
        return

    # 过滤目标月份的数据
    filtered_data_flow = []
    for date_entry in data_flow:
        date_str = date_entry['date']
        try:
            entry_date = pd.to_datetime(date_str)
            # 判断日期是否在目标月份范围内
            if month_start <= entry_date <= month_end:
                filtered_data_flow.append(date_entry)
                print(f"[DEBUG] 匹配日期: {date_str}")
        except Exception as e:
            print(f"[DEBUG] 日期解析失败: {date_str}, 错误: {e}")
            continue

    print(f"[DEBUG] 过滤后的数据流数量: {len(filtered_data_flow)}")

    if not filtered_data_flow:
        # data_flow是按降序排列的（最新在前），所以[-1]是最早日期，[0]是最新日期
        st_module.info(f"目标月份 {target_month} 没有数据发布（数据覆盖范围: {data_flow[-1]['date'] if data_flow else 'N/A'} - {data_flow[0]['date'] if data_flow else 'N/A'}）")
        return

    st_module.markdown(f"##### 数据发布影响明细 - {target_month} ({len(filtered_data_flow)} 个日期)")

    for date_entry in filtered_data_flow:
        date_str = date_entry['date']
        nowcast_value = date_entry.get('nowcast_value')
        releases = date_entry.get('releases', [])

        # 日期标题
        if nowcast_value is not None:
            st_module.markdown(f"**{date_str}** - Nowcast: {nowcast_value:.4f}")
        else:
            st_module.markdown(f"**{date_str}**")

        # 发布列表
        if releases:
            release_data = []
            for release in releases:
                release_data.append({
                    '变量名称': release['variable'],
                    '所属行业': release['industry'],
                    '观测值': f"{release['actual']:.4f}",
                    '贡献值(%)': f"{release['contribution']:.2f}",
                    '影响值': f"{release['impact']:+.4f}",
                    '方向': '↑' if release['is_positive'] else '↓'
                })

            df = pd.DataFrame(release_data)
            st_module.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st_module.caption("无发布数据")

    # 添加计算说明
    with st_module.expander("指标说明"):
        st_module.markdown("""
**贡献值（Contribution %）**：表示该数据发布对目标变量预测变动的相对贡献度。

- 计算公式：`贡献值(%) = |该变量的影响值| / Σ|所有变量的影响值| × 100`
- 所有变量的贡献值总和为100%
- 贡献值越大，说明该变量对预测变动的解释力越强

**影响值（Impact）**：表示该数据发布导致目标变量预测值的绝对变动量。

- 计算公式：`影响值 = λ_y' × K_t[:, i] × (观测值 - 期望值)`
  - `λ_y`：目标变量的因子载荷向量
  - `K_t`：第t期卡尔曼增益矩阵
  - `观测值 - 期望值`：新息（News）
- 影响值为正表示该数据发布提升了预测值，为负表示降低了预测值
- 所有变量的影响值总和等于预测值的总变动量

**示例**：如果GDP增长率预测从5.0%变为5.2%，某变量的影响值为+0.15，贡献值为75%，说明：
- 该变量使预测值提升了0.15个百分点
- 在总变动0.2个百分点中，该变量解释了75%的变化
        """)


def _render_summary_cards(st_module, result):
    """渲染统计摘要卡片"""
    summary = result.get('summary', {})

    st_module.markdown("##### 关键指标")

    col1, col2, col3, col4 = st_module.columns(4)

    with col1:
        st_module.metric(
            label="总影响",
            value=f"{summary.get('total_impact', 0):.4f}",
            help="所有数据发布对nowcast的累积影响"
        )

    with col2:
        st_module.metric(
            label="数据发布数",
            value=f"{summary.get('total_releases', 0)}",
            help="分析期内发布的数据点数量"
        )

    with col3:
        st_module.metric(
            label="正向影响",
            value=f"{summary.get('positive_impact_sum', 0):.4f}",
            delta="+",
            help="提升nowcast值的数据发布总影响"
        )

    with col4:
        st_module.metric(
            label="负向影响",
            value=f"{summary.get('negative_impact_sum', 0):.4f}",
            delta="-",
            delta_color="inverse",
            help="降低nowcast值的数据发布总影响"
        )

    # 行业分解
    industry_breakdown = summary.get('industry_breakdown', {})
    if industry_breakdown:
        st_module.markdown("##### 按行业分解")

        industry_data = []
        for industry, stats in industry_breakdown.items():
            industry_data.append({
                '行业': industry,
                '总影响': stats['impact'],
                '数据发布数': stats['count'],
                '正向影响': stats['positive_impact'],
                '负向影响': stats['negative_impact'],
                '贡献度(%)': stats['contribution_pct']
            })

        industry_df = pd.DataFrame(industry_data)
        industry_df = industry_df.sort_values(by='总影响', key=abs, ascending=False)
        st_module.dataframe(industry_df, use_container_width=True, hide_index=True)


def _render_download_section(st_module, result):
    """渲染下载区域"""
    csv_paths = result.get('csv_paths', {})
    csv_contents = result.get('csv_contents', {})

    st_module.markdown("##### 数据导出")

    col1, col2 = st_module.columns(2)

    with col1:
        impacts_path = csv_paths.get('impacts')
        impacts_data = csv_contents.get('impacts')
        if impacts_data:
            st_module.download_button(
                label="下载数据发布影响CSV",
                data=impacts_data,
                file_name=os.path.basename(impacts_path) if impacts_path else 'data_release_impacts.csv',
                mime="text/csv",
                use_container_width=True,
                key="download_impacts_csv"
            )
        else:
            st_module.info("数据发布影响CSV未生成")

    with col2:
        contributions_path = csv_paths.get('contributions')
        contributions_data = csv_contents.get('contributions')
        if contributions_data:
            st_module.download_button(
                label="下载贡献分解CSV",
                data=contributions_data,
                file_name=os.path.basename(contributions_path) if contributions_path else 'contributions_decomposition.csv',
                mime="text/csv",
                use_container_width=True,
                key="download_contributions_csv"
            )
        else:
            st_module.info("贡献分解CSV未生成")


# 兼容性接口
def render_dfm_news_analysis_tab(st_module: Any) -> Dict[str, Any]:
    """兼容性接口"""
    return render_dfm_news_analysis_page(st_module)
