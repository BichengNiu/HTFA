"""
Industrial Macro Operations Analysis Module
工业宏观运行分析模块 - 主入口文件
"""

# 导出本模块定义的函数
__all__ = ['render_macro_operations_tab', 'render_macro_operations_analysis_with_data', '_render_macro_operations_analysis']

# 导入必要的模块
import streamlit as st
import pandas as pd
from typing import Any, Optional, List
import plotly.graph_objects as go
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 导入统一的工具函数
from dashboard.analysis.industrial.utils import (
    get_weight_for_year,
    filter_data_from_2012,
    load_macro_data,
    load_weights_data,
    load_overall_industrial_data,
    filter_data_by_time_range,
    create_grouping_mappings,
    # 新增：优化的加权计算
    calculate_weighted_groups_optimized,
    # 新增：统一Fragment组件
    create_chart_with_time_selector_fragment,
    # 新增：统一下载工具
    create_excel_download_button,
    create_download_with_annotation,
    prepare_grouping_annotation_data,
    # 新增：统一图表创建器
    create_time_series_chart
)

# 导入拉动率计算模块
from dashboard.analysis.industrial.utils.contribution_calculator import calculate_all_contributions
from dashboard.ui.utils.debug_helpers import debug_log

def get_monitoring_state(key: str, default: Any = None):
    """获取监测分析状态"""
    try:
        import streamlit as st
        full_key = f'monitoring.industrial.macro.{key}'
        return st.session_state.get(full_key, default)
    except Exception as e:
        logger.error(f"获取监测分析状态失败: {e}")
        return default


def set_monitoring_state(key: str, value: Any, is_initialization: bool = False):
    """设置监测分析状态"""
    try:
        import streamlit as st
        full_key = f'monitoring.industrial.macro.{key}'
        st.session_state[full_key] = value
        return True
    except Exception as e:
        logger.error(f"设置监测分析状态失败: {e}")
        return False

def initialize_monitoring_states():
    """预初始化监测分析状态，避免第一次点击时刷新"""
    try:
        import streamlit as st
        # 静默初始化所有时间筛选相关的状态
        time_range_keys = [
            'macro_time_range_chart1',
            'macro_time_range_chart2',
            'macro_time_range_chart3'
        ]

        for key in time_range_keys:
            full_key = f'monitoring.industrial.macro.{key}'
            # 只有在状态不存在时才初始化
            if full_key not in st.session_state:
                st.session_state[full_key] = "3年"

        return True
    except (AttributeError, KeyError, TypeError) as e:
        logger.warning(f"初始化监测状态失败: {e}")
        return False
    except Exception as e:
        logger.error(f"初始化监测状态时发生未预期错误: {e}", exc_info=True)
        return False


# 数据加载函数已移至utils.data_loader模块，删除重复代码


def validate_data_format(df_macro: pd.DataFrame, df_weights: pd.DataFrame, target_columns: List[str]) -> tuple:
    """
    验证数据格式并返回详细的诊断信息

    Returns:
        (is_valid: bool, error_message: str, debug_info: dict)
    """
    debug_info = {}

    # 检查基本数据
    if df_macro.empty:
        return False, "分行业工业增加值同比增速数据为空", debug_info
    if df_weights.empty:
        return False, "权重数据为空", debug_info

    debug_info['macro_shape'] = df_macro.shape
    debug_info['weights_shape'] = df_weights.shape
    debug_info['target_columns_count'] = len(target_columns)

    # 检查权重数据必要列
    required_weight_columns = ['指标名称', '出口依赖', '上中下游']
    weight_year_columns = ['权重_2012', '权重_2018', '权重_2020']

    missing_columns = [col for col in required_weight_columns if col not in df_weights.columns]
    if missing_columns:
        return False, f"权重数据缺少必要列: {missing_columns}", debug_info

    # 检查是否至少有一个权重年份列
    available_weight_columns = [col for col in weight_year_columns if col in df_weights.columns]
    if not available_weight_columns:
        return False, f"权重数据缺少权重列，需要以下任一列: {weight_year_columns}", debug_info

    debug_info['available_weight_columns'] = available_weight_columns

    debug_info['weights_columns'] = list(df_weights.columns)

    # 检查目标列匹配情况
    available_columns = [col for col in target_columns if col in df_macro.columns]
    debug_info['available_columns_count'] = len(available_columns)
    debug_info['available_columns'] = available_columns[:5]  # 只显示前5个

    if not available_columns:
        return False, "目标列在分行业工业增加值同比增速数据中未找到匹配项", debug_info

    # 检查权重数据中的指标名称匹配
    weight_indicators = df_weights['指标名称'].dropna().tolist()
    matched_indicators = [ind for ind in weight_indicators if ind in available_columns]
    debug_info['matched_indicators_count'] = len(matched_indicators)
    debug_info['matched_indicators'] = matched_indicators[:5]  # 只显示前5个

    if not matched_indicators:
        return False, "权重数据中的指标名称与分行业工业增加值同比增速数据列名无匹配项", debug_info

    return True, "数据格式验证通过", debug_info


# 旧的calculate_weighted_groups函数已删除，请使用utils中的calculate_weighted_groups_optimized
# 该优化版本性能提升10-100倍，代码更简洁，参见：dashboard/analysis/industrial/utils/weighted_calculation.py

# 旧的图表创建函数已删除，请使用utils中的create_time_series_chart
# create_single_axis_chart (173行) 和 create_overall_industrial_chart (207行) 已替换为统一的图表创建器
# 参见：dashboard/analysis/industrial/utils/chart_creator_unified.py



def render_macro_operations_analysis_with_data(st_obj, df_macro: pd.DataFrame, df_weights: pd.DataFrame):
    """
    使用预加载数据渲染分行业工业增加值同比增速分析（用于统一模块）

    Args:
        st_obj: Streamlit对象
        df_macro: 分行业工业增加值同比增速数据
        df_weights: 权重数据
    """
    if df_macro is None or df_weights is None:
        st_obj.error("数据未正确加载，无法进行分行业工业增加值同比增速分析")
        return

    # 直接使用传入的数据进行分析
    _render_macro_operations_analysis(st_obj, df_macro, df_weights)


def _render_macro_operations_analysis(st_obj, df: pd.DataFrame, df_weights: pd.DataFrame):
    """
    内部函数：执行分行业工业增加值同比增速分析的核心逻辑
    """
    if df is not None and df_weights is not None:
        # 预初始化状态以避免第一次点击时刷新
        initialize_monitoring_states()

        # Store the data in state
        set_monitoring_state('uploaded_data', df)
        set_monitoring_state('weights_data', df_weights)
        set_monitoring_state('file_name', 'shared_data')

        # First chart - Overall industrial value-added data from "总体工业增加值同比增速" sheet
        try:
            # Get the uploaded file from unified state management with proper namespace
            from dashboard.analysis.industrial.industrial_analysis import get_industrial_state
            uploaded_file = get_industrial_state('macro.uploaded_file')
            if uploaded_file is not None:
                # Load overall industrial data
                df_overall = load_overall_industrial_data(uploaded_file)

                if df_overall is not None and not df_overall.empty:
                    # Extract the four variables from the overall industrial data
                    # Assuming the four variables are in columns 1-4 (after the date column)
                    overall_vars = [col for col in df_overall.columns if pd.notna(col)][:4]

                    if overall_vars:
                        # Create variable name mapping for display
                        var_name_mapping = {}
                        for var in overall_vars:
                            if var == "规模以上工业增加值:当月同比":
                                var_name_mapping[var] = "工业总体"
                            elif "制造业" in var and "当月同比" in var:
                                var_name_mapping[var] = "制造业"
                            elif "采矿业" in var and "当月同比" in var:
                                var_name_mapping[var] = "采矿业"
                            elif ("电力" in var or "热力" in var or "燃气" in var or "水生产" in var or "供应业" in var) and "当月同比" in var:
                                var_name_mapping[var] = "水电燃气供应业"
                            else:
                                # Fallback to original name if no mapping found
                                var_name_mapping[var] = var


                        # 添加第1个图的标题
                        st_obj.markdown("#### 工业增加值总体")

                        # 定义图表创建函数 - 使用统一的图表创建器
                        def create_chart1(df, variables, time_range, custom_start_date, custom_end_date, var_mapping=None):
                            return create_time_series_chart(
                                df=df,
                                variables=variables,
                                title="",
                                time_range=time_range,
                                custom_start_date=custom_start_date,
                                custom_end_date=custom_end_date,
                                var_name_mapping=var_mapping or {},
                                y_axis_title="同比增速(%)",
                                height=350,
                                bottom_margin=80
                            )

                        # 使用统一Fragment组件
                        current_time_range_1, custom_start_1, custom_end_1 = create_chart_with_time_selector_fragment(
                            st_obj=st_obj,
                            chart_id="macro_chart1",
                            state_namespace="monitoring.industrial.macro",
                            chart_title=None,
                            chart_creator_func=create_chart1,
                            chart_data=df_overall,
                            chart_variables=overall_vars,
                            get_state_func=get_monitoring_state,
                            set_state_func=set_monitoring_state,
                            additional_chart_kwargs={'var_mapping': var_name_mapping}
                        )

                        # Data download functionality for first chart
                        # Validate that all variables exist in the dataframe
                        valid_vars = [var for var in overall_vars if var in df_overall.columns]
                        if len(valid_vars) != len(overall_vars):
                            pass  # Some variables are missing

                        if valid_vars:
                            filtered_df1 = filter_data_by_time_range(df_overall[valid_vars], current_time_range_1, custom_start_1, custom_end_1)

                            if not filtered_df1.empty:
                                create_excel_download_button(
                                    st_obj=st_obj,
                                    data=filtered_df1,
                                    file_name=f"总体工业增加值_{current_time_range_1}.xlsx",
                                    button_key="download_chart1",
                                    column_ratio=(1, 4)
                                )
                else:
                    st_obj.warning("无法加载总体工业增加值同比增速数据，请检查文件是否包含'总体工业增加值同比增速'工作表")
            else:
                st_obj.warning("未找到上传的文件，无法加载总体工业增加值数据")
        except (KeyError, ValueError) as e:
            logger.warning(f"数据格式错误，无法加载总体工业增加值数据: {e}")
            st_obj.error(f"数据格式错误: {e}")
        except pd.errors.EmptyDataError:
            logger.warning("Excel文件中'总体工业增加值同比增速'工作表为空")
            st_obj.error("数据文件为空，请检查Excel文件内容")
        except Exception as e:
            logger.error(f"加载总体工业增加值数据时发生未预期错误: {e}", exc_info=True)
            st_obj.error(f"加载数据失败: {e}")

        # Define target columns for weighted analysis
        # 根据用户说明：分行业工业增加值数据在第二列到最后一列
        column_names = df.columns.tolist()

        # 使用第二列到最后一列作为目标列（索引从1开始）
        if len(column_names) > 1:
            target_columns = column_names[1:]  # 从第二列开始到最后一列
            target_columns = [col for col in target_columns if pd.notna(col)]  # 过滤掉空值

        if len(column_names) > 1 and target_columns:
            # Check if weighted groups are already calculated and cached
            cached_weighted_df = get_monitoring_state('cached_weighted_df')
            cached_data_hash = get_monitoring_state('cached_data_hash')

            # Create a simple hash of the input data to detect changes
            # 注意：2025-10-30修改为拉动率显示，增加版本号以清理旧缓存
            import hashlib
            current_data_str = f"v4_contrib_fixed_{df.shape}_{df_weights.shape}_{len(target_columns)}"
            current_data_hash = hashlib.md5(current_data_str.encode()).hexdigest()

            # Only recalculate if data has changed or cache is empty
            if cached_weighted_df is None or cached_data_hash != current_data_hash:
                # Calculate weighted groups - 使用优化版本
                with st_obj.spinner("正在计算加权分组数据..."):
                    weighted_df = calculate_weighted_groups_optimized(df, df_weights, target_columns)

                # Cache the results
                set_monitoring_state('cached_weighted_df', weighted_df)
                set_monitoring_state('cached_data_hash', current_data_hash)

                # 计算拉动率
                debug_log("开始计算拉动率", "INFO")
                try:
                    # 获取总体增速数据并合并
                    from dashboard.analysis.industrial.industrial_analysis import get_industrial_state
                    uploaded_file = get_industrial_state('macro.uploaded_file')

                    if uploaded_file is not None:
                        df_overall = load_overall_industrial_data(uploaded_file)
                        if df_overall is not None and '规模以上工业增加值:当月同比' in df_overall.columns:
                            # 合并总体增速到分行业数据
                            total_growth_series = df_overall['规模以上工业增加值:当月同比']
                            df_with_total = pd.concat([total_growth_series, df], axis=1)
                            df_with_total = df_with_total.dropna(how='all')

                            # 过滤2012年及以后
                            df_macro_filtered = filter_data_from_2012(df_with_total)

                            # 计算拉动率
                            contribution_results = calculate_all_contributions(
                                df_macro_filtered, df_weights
                            )

                            # 缓存拉动率结果
                            set_monitoring_state('cached_contribution_export', contribution_results['export_groups'])
                            set_monitoring_state('cached_contribution_stream', contribution_results['stream_groups'])
                            set_monitoring_state('cached_contribution_individual', contribution_results['individual'])
                            set_monitoring_state('cached_total_growth', contribution_results['total_growth'])

                            debug_log(
                                f"拉动率计算完成，验证结果: {contribution_results['validation']['passed']}",
                                "INFO"
                            )
                        else:
                            debug_log("总体增速数据加载失败", "WARNING")
                            set_monitoring_state('cached_contribution_export', None)
                            set_monitoring_state('cached_contribution_stream', None)
                    else:
                        debug_log("未找到上传文件，无法加载总体增速", "WARNING")
                        set_monitoring_state('cached_contribution_export', None)
                        set_monitoring_state('cached_contribution_stream', None)

                except Exception as e:
                    debug_log(f"拉动率计算失败: {e}", "ERROR")
                    import traceback
                    debug_log(f"错误详情: {traceback.format_exc()}", "ERROR")
                    # 即使拉动率计算失败，也不影响其他功能
                    set_monitoring_state('cached_contribution_export', None)
                    set_monitoring_state('cached_contribution_stream', None)
            else:
                # Use cached results
                weighted_df = cached_weighted_df

            if not weighted_df.empty:
                # 添加横线分隔符
                st_obj.markdown("---")

                # 添加第2个图的标题
                st_obj.markdown("#### 分出口依赖行业拉动率")

                # Second chart - Export dependency groups (contribution mode only)
                export_vars = [col for col in weighted_df.columns if col.startswith('出口依赖_')]
                if export_vars:
                    # 直接使用拉动率数据
                    contribution_export = get_monitoring_state('cached_contribution_export')
                    if contribution_export is not None:
                        chart2_data = contribution_export
                        chart2_vars = [col for col in contribution_export.columns if col.startswith('出口依赖_')]
                        y_axis_title_2 = "%"
                    else:
                        st_obj.warning("拉动率数据未计算，无法显示图表")
                        chart2_data = None
                        chart2_vars = []

                    if chart2_data is not None and chart2_vars:
                        # 定义图表创建函数 - 使用统一的图表创建器
                        def create_chart2(df, variables, time_range, custom_start_date, custom_end_date):
                            return create_time_series_chart(
                                df=df,
                                variables=variables,
                                title="",
                                time_range=time_range,
                                custom_start_date=custom_start_date,
                                custom_end_date=custom_end_date,
                                y_axis_title=y_axis_title_2,
                                height=350,
                                bottom_margin=80
                            )

                        # 使用统一Fragment组件
                        current_time_range_2, custom_start_2, custom_end_2 = create_chart_with_time_selector_fragment(
                            st_obj=st_obj,
                            chart_id="macro_chart2",
                            state_namespace="monitoring.industrial.macro",
                            chart_title=None,
                            chart_creator_func=create_chart2,
                            chart_data=chart2_data,
                            chart_variables=chart2_vars,
                            get_state_func=get_monitoring_state,
                            set_state_func=set_monitoring_state
                        )

                        # 数据下载功能 - 使用统一下载工具
                        filtered_df2 = filter_data_by_time_range(chart2_data[chart2_vars], current_time_range_2, custom_start_2, custom_end_2)
                        if not filtered_df2.empty:
                            export_groups, _ = create_grouping_mappings(df_weights)
                            annotation_df = prepare_grouping_annotation_data(df_weights, export_groups, '出口依赖')
                            create_download_with_annotation(
                                st_obj=st_obj,
                                data=filtered_df2,
                                file_name=f"出口依赖分组_拉动率_{current_time_range_2}",
                                annotation_data=annotation_df,
                                button_key="download_chart2"
                            )

                # 添加横线分隔符
                st_obj.markdown("---")

                # 添加第3个图的标题
                st_obj.markdown("#### 分上中下游行业拉动率")

                # Third chart - Upstream/downstream groups (contribution mode only)
                stream_vars = [col for col in weighted_df.columns if col.startswith('上中下游_')]

                # 按照指定顺序排序图例：上游XX、中游XX、下游XX
                def get_sort_key(col):
                    # 提取列名中的行业类型
                    industry_type = col.replace('上中下游_', '')

                    # 按照上游、中游、下游的顺序排序
                    if industry_type.startswith('上游'):
                        return (0, industry_type)  # 上游排在最前
                    elif industry_type.startswith('中游'):
                        return (1, industry_type)  # 中游排在中间
                    elif industry_type.startswith('下游'):
                        return (2, industry_type)  # 下游排在最后
                    else:
                        return (3, industry_type)  # 未知类型排在最后

                stream_vars = sorted(stream_vars, key=get_sort_key)
                if stream_vars:
                    # 直接使用拉动率数据
                    contribution_stream = get_monitoring_state('cached_contribution_stream')
                    if contribution_stream is not None:
                        chart3_data = contribution_stream
                        chart3_vars = [col for col in contribution_stream.columns if col.startswith('上中下游_')]
                        chart3_vars = sorted(chart3_vars, key=get_sort_key)
                        y_axis_title_3 = "%"
                    else:
                        st_obj.warning("拉动率数据未计算，无法显示图表")
                        chart3_data = None
                        chart3_vars = []

                    if chart3_data is not None and chart3_vars:
                        # 定义图表创建函数 - 使用统一的图表创建器
                        def create_chart3(df, variables, time_range, custom_start_date, custom_end_date):
                            return create_time_series_chart(
                                df=df,
                                variables=variables,
                                title="",
                                time_range=time_range,
                                custom_start_date=custom_start_date,
                                custom_end_date=custom_end_date,
                                y_axis_title=y_axis_title_3,
                                height=350,
                                bottom_margin=80
                            )

                        # 使用统一Fragment组件
                        current_time_range_3, custom_start_3, custom_end_3 = create_chart_with_time_selector_fragment(
                            st_obj=st_obj,
                            chart_id="macro_chart3",
                            state_namespace="monitoring.industrial.macro",
                            chart_title=None,
                            chart_creator_func=create_chart3,
                            chart_data=chart3_data,
                            chart_variables=chart3_vars,
                            get_state_func=get_monitoring_state,
                            set_state_func=set_monitoring_state
                        )

                        # 数据下载功能 - 使用统一下载工具
                        filtered_df3 = filter_data_by_time_range(chart3_data[chart3_vars], current_time_range_3, custom_start_3, custom_end_3)
                        if not filtered_df3.empty:
                            _, stream_groups = create_grouping_mappings(df_weights)
                            annotation_df = prepare_grouping_annotation_data(df_weights, stream_groups, '上中下游')
                            create_download_with_annotation(
                                st_obj=st_obj,
                                data=filtered_df3,
                                file_name=f"上中下游分组_拉动率_{current_time_range_3}",
                                annotation_data=annotation_df,
                                button_key="download_chart3"
                            )
            else:
                # 使用简化的验证逻辑，提供简洁的错误信息
                is_valid, error_message, debug_info = validate_data_format(df, df_weights, target_columns)

                st_obj.error(f"加权分组计算失败: {error_message}")

                # 显示简洁的诊断信息
                import os
                DEBUG_MODE = os.getenv('INDUSTRIAL_DEBUG_MODE') == 'true'

                if DEBUG_MODE:
                    # 调试模式：显示详细信息
                    with st_obj.expander("详细诊断信息（调试模式）"):
                        st_obj.json(debug_info)
                else:
                    # 正常模式：只显示关键信息
                    with st_obj.expander("快速诊断"):
                        if 'matched_indicators_count' in debug_info:
                            st_obj.info(f"匹配的指标数: {debug_info['matched_indicators_count']}")
                            if debug_info['matched_indicators_count'] > 0 and 'matched_indicators' in debug_info:
                                st_obj.text(f"示例: {debug_info['matched_indicators']}")

                        if 'available_columns_count' in debug_info:
                            st_obj.info(f"可用目标列数: {debug_info['available_columns_count']}")

                        st_obj.markdown("**数据格式要求：**")
                        st_obj.markdown("""
                        - 权重数据需包含：`指标名称`、`出口依赖`、`上中下游`、权重列
                        - 指标名称需与宏观数据列名完全匹配
                        - 设置环境变量 `INDUSTRIAL_DEBUG_MODE=true` 查看详细诊断
                        """)
        else:
            st_obj.error("未找到目标列范围，请检查Excel文件格式。")
    else:
        st_obj.error("数据加载失败，无法进行分行业工业增加值同比增速分析。")


def render_macro_operations_tab(st_obj):
    """
    分行业工业增加值同比增速分析标签页

    Args:
        st_obj: Streamlit对象
    """
    # File upload in sidebar
    with st_obj.sidebar:
        st_obj.markdown("#### [DATA] 数据文件上传")
        uploaded_file = st_obj.file_uploader(
            "上传监测分析Excel模板文件",
            type=['xlsx', 'xls'],
            key="macro_operations_file_uploader",
            help="请上传包含'分行业工业增加值同比增速'工作表的Excel文件"
        )

    if uploaded_file is not None:
        # Load and process the data
        with st_obj.spinner("正在处理数据..."):
            df = load_template_data(uploaded_file)
            df_weights = load_weights_data(uploaded_file)

        if df is not None and df_weights is not None:
            _render_macro_operations_analysis(st_obj, df, df_weights)
        else:
            if df is None:
                st_obj.error("分行业工业增加值同比增速数据加载失败，请检查文件格式。")
            if df_weights is None:
                st_obj.error("权重数据加载失败，请检查文件是否包含'工业增加值分行业指标权重'工作表。")
    else:
        st_obj.info("请在左侧上传Excel数据文件以开始分行业工业增加值同比增速分析。")

