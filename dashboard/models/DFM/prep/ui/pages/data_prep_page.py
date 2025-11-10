# -*- coding: utf-8 -*-
"""
DFM数据准备页面

在新架构下100%恢复旧版本功能：
- 日期范围自动检测
- 8个参数完整配置
- 自动映射加载
- 移除变量详细日志
- 行业映射文件导出
- 导出文件名自定义
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, date
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 状态管理辅助函数
# ============================================================================

def _get_state(key: str, default=None):
    """获取状态值（命名空间: data_prep）"""
    full_key = f'data_prep.{key}'
    return st.session_state.get(full_key, default)


def _set_state(key: str, value):
    """设置状态值（命名空间: data_prep）"""
    full_key = f'data_prep.{key}'
    st.session_state[full_key] = value


def _sync_dates_to_train_model():
    """将数据准备的日期设置同步到模型训练模块"""
    try:
        data_start = _get_state('param_data_start_date')
        data_end = _get_state('param_data_end_date')

        if data_start:
            st.session_state['train_model.training_start_date'] = data_start
        if data_end:
            st.session_state['train_model.validation_end_date'] = data_end

        return True
    except Exception as e:
        logger.error(f"日期同步失败: {e}")
        return False


# ============================================================================
# 核心功能函数
# ============================================================================

def _detect_data_date_range(uploaded_file) -> Tuple[Optional[date], Optional[date]]:
    """
    从上传的文件中检测所有数据工作表的真实日期范围

    Args:
        uploaded_file: 上传的文件对象

    Returns:
        (开始日期, 结束日期) 元组，如果检测失败返回 (None, None)
    """
    try:
        if uploaded_file is None or not hasattr(uploaded_file, 'getvalue'):
            return None, None

        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            return None, None

        excel_file = io.BytesIO(file_bytes)
        all_dates_found = []

        # 获取所有工作表名称
        try:
            xl_file = pd.ExcelFile(excel_file)
            sheet_names = xl_file.sheet_names
            print(f"检测到工作表: {sheet_names}")
        except:
            sheet_names = [0]

        # 检查所有数据工作表
        for sheet_name in sheet_names:
            try:
                excel_file.seek(0)

                # 跳过明显的元数据工作表
                if any(keyword in str(sheet_name).lower() for keyword in ['指标体系', 'mapping', 'meta', 'info']):
                    print(f"跳过元数据工作表: {sheet_name}")
                    continue

                # 读取工作表
                df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)

                if len(df_raw) < 5:
                    continue

                # 检测日期列
                date_values = []

                # 检查是否是Wind格式
                if 'Wind' in sheet_name or (len(df_raw) > 0 and df_raw.iloc[0, 0] == '指标名称'):
                    if len(df_raw) > 1:
                        date_values = pd.to_datetime(df_raw.iloc[1:, 0], errors='coerce')
                else:
                    # 尝试前两列作为日期
                    for col_idx in range(min(2, len(df_raw.columns))):
                        try:
                            test_dates = pd.to_datetime(df_raw.iloc[:, col_idx], errors='coerce')
                            valid_dates = test_dates[test_dates.notna()]

                            if len(valid_dates) > len(df_raw) * 0.5:
                                date_values = valid_dates
                                break
                        except:
                            continue

                # 收集有效日期
                if len(date_values) > 0:
                    valid_dates = date_values[date_values.notna()]
                    if len(valid_dates) > 5:
                        all_dates_found.extend(valid_dates.tolist())
                        print(f"  {sheet_name}: 找到 {len(valid_dates)} 个日期")

            except Exception as e:
                print(f"  处理 {sheet_name} 时出错: {str(e)}")
                continue

        # 汇总所有数据工作表的日期范围
        if all_dates_found:
            all_dates = pd.to_datetime(all_dates_found)

            # 过滤掉1990年之前的异常日期
            cutoff_date = pd.Timestamp('1990-01-01')
            valid_dates = all_dates[all_dates >= cutoff_date]

            if len(valid_dates) > 0:
                actual_start = valid_dates.min().date()
                actual_end = valid_dates.max().date()
                print(f"检测到的真实日期范围: {actual_start} 到 {actual_end}")

                if len(all_dates) > len(valid_dates):
                    filtered_count = len(all_dates) - len(valid_dates)
                    print(f"  (已过滤 {filtered_count} 个1990年之前的异常日期)")

                return actual_start, actual_end

        return None, None

    except Exception as e:
        print(f"日期检测异常: {e}")
        return None, None


def _auto_load_mapping_data(current_file, mapping_sheet_name: str = '指标体系'):
    """
    自动加载映射数据

    Args:
        current_file: 当前文件对象
        mapping_sheet_name: 映射表工作表名称
    """
    # 会话级缓存，避免重复加载
    cache_key = f"mapping_loaded_{current_file.name if current_file else 'none'}"
    if _get_state(cache_key, False):
        return

    try:
        # 检查是否已经加载了映射数据
        existing_industry_map = _get_state('industry_map_obj')
        existing_type_map = _get_state('var_type_map_obj')

        if existing_industry_map and existing_type_map:
            _set_state(cache_key, True)
            return

        if current_file is None:
            return

        # 加载映射数据
        from dashboard.models.DFM.prep.modules.mapping_manager import load_mappings

        var_type_map, var_industry_map, var_dfm_single_stage_map, var_dfm_two_stage_map, var_first_stage_target_map = load_mappings(
            excel_path=current_file,
            sheet_name=mapping_sheet_name,
            indicator_col='指标名称',
            type_col='类型',
            industry_col='行业',
            single_stage_col='一次估计',
            two_stage_col='二次估计',
            first_stage_target_col='一阶段目标'
        )

        # 保存映射数据
        _set_state("var_type_map_obj", var_type_map if var_type_map else {})
        _set_state("industry_map_obj", var_industry_map if var_industry_map else {})
        _set_state("dfm_default_single_stage_map", var_dfm_single_stage_map if var_dfm_single_stage_map else {})
        _set_state("dfm_default_two_stage_map", var_dfm_two_stage_map if var_dfm_two_stage_map else {})
        _set_state("dfm_first_stage_target_map", var_first_stage_target_map if var_first_stage_target_map else {})

        # 标记为已加载
        _set_state(cache_key, True)

        print(f"自动加载映射数据完成: {len(var_industry_map)} 个指标")
        print(f"一次估计默认变量: {len(var_dfm_single_stage_map)} 个")
        print(f"二次估计默认变量: {len(var_dfm_two_stage_map)} 个")
        print(f"一阶段目标映射: {len(var_first_stage_target_map)} 个")

    except Exception as e:
        print(f"自动加载映射数据失败: {e}")


# ============================================================================
# UI渲染函数
# ============================================================================

def _render_file_upload_section(st_obj):
    """渲染文件上传区域"""

    st_obj.markdown("### 数据文件上传")

    # 检查已有文件
    existing_file = _get_state('training_data_file')
    existing_file_path = _get_state('uploaded_file_path')

    # 显示当前文件状态
    if existing_file and existing_file_path:
        col1, col2 = st_obj.columns([3, 1])
        with col1:
            st_obj.success(f"已加载文件: {existing_file_path}")
        with col2:
            if st_obj.button("重新上传", key="dfm_reupload_btn", type="primary"):
                _set_state("training_data_file", None)
                _set_state("uploaded_file_path", None)
                _set_state("file_processed", False)
                _set_state("date_detection_needed", True)
                st_obj.rerun()

    # 文件上传组件
    uploaded_file_new = st_obj.file_uploader(
        "选择Excel数据文件",
        type=['xlsx', 'xls'],
        key="dfm_data_prep_file_uploader",
        help="请上传包含时间序列数据的Excel文件（支持.xlsx, .xls格式）"
    )

    # 处理新上传的文件
    uploaded_file = None
    if uploaded_file_new is not None:
        # 检查文件是否真的变更
        file_bytes = uploaded_file_new.getvalue()

        new_file_id = f"{uploaded_file_new.name}_{len(file_bytes)}"
        existing_file_id = f"{existing_file_path}_{len(_get_state('file_bytes', b''))}"

        # 只有文件真正变更时才更新状态
        if new_file_id != existing_file_id:
            _set_state("training_data_file", uploaded_file_new)
            _set_state("file_bytes", file_bytes)
            _set_state("uploaded_file_path", uploaded_file_new.name)
            _set_state("file_processed", False)
            _set_state("date_detection_needed", True)
            print(f"新文件上传: {uploaded_file_new.name}，字节大小: {len(file_bytes)}")

        uploaded_file = uploaded_file_new
    elif existing_file:
        uploaded_file = existing_file

    return uploaded_file


def _render_date_detection(st_obj, uploaded_file):
    """
    渲染日期检测逻辑

    Returns:
        (detected_start, detected_end, min_date, max_date) 元组
    """

    if uploaded_file is None:
        return None, None, date(1990, 1, 1), date(2050, 12, 31)

    # 使用文件名和大小作为缓存键
    try:
        file_bytes = uploaded_file.getvalue()
        cache_key = f"date_range_{uploaded_file.name}_{len(file_bytes)}"
    except:
        cache_key = "date_range_none"

    # 检查缓存是否有效
    cached_result = _get_state(cache_key)
    cache_valid = cached_result is not None and not _get_state('date_detection_needed')

    if not cache_valid:
        # 需要重新检测
        print(f"执行日期检测: {uploaded_file.name}")
        with st_obj.spinner("正在检测数据日期范围..."):
            detected_start, detected_end = _detect_data_date_range(uploaded_file)

        # 缓存结果
        _set_state(cache_key, (detected_start, detected_end))
        _set_state("detected_start_date", detected_start)
        _set_state("detected_end_date", detected_end)
        _set_state("date_detection_needed", False)

        # 清理旧缓存
        try:
            all_keys = [k for k in st.session_state.keys() if k.startswith("data_prep.date_range_")]
            old_keys = [k for k in all_keys if k != f"data_prep.{cache_key}"]
            for old_key in old_keys:
                del st.session_state[old_key]
        except Exception as e:
            print(f"清理旧缓存时出错: {e}")
    else:
        # 使用缓存的结果
        detected_start, detected_end = cached_result if cached_result else (None, None)
        _set_state("detected_start_date", detected_start)
        _set_state("detected_end_date", detected_end)
        print(f"使用缓存的日期范围: {detected_start} 到 {detected_end}")

    # 显示检测结果
    if detected_start and detected_end:
        st_obj.success(f"已检测数据文件的真实日期范围: {detected_start} 到 {detected_end}")
        # 设置日期选择器范围（宽松一年）
        min_date = detected_start - pd.Timedelta(days=365)
        max_date = detected_end + pd.Timedelta(days=365)
    else:
        st_obj.warning("无法自动检测文件日期范围，使用默认值。请手动调整日期设置。")
        min_date = date(1990, 1, 1)
        max_date = date(2050, 12, 31)

    return detected_start, detected_end, min_date, max_date


def _render_parameter_config(st_obj, detected_start, detected_end, min_date, max_date):
    """
    渲染参数配置区域（8个参数，4行2列布局）

    Returns:
        bool - 参数是否有效
    """

    # 设置默认值
    default_start_date = date(2020, 1, 1)
    default_end_date = detected_end if detected_end else date(2025, 4, 30)

    param_defaults = {
        'param_target_variable': '规模以上工业增加值:当月同比',
        'param_target_sheet_name': '工业增加值同比增速_月度_同花顺',
        'param_target_freq': 'W-FRI',
        'param_remove_consecutive_nans': True,
        'param_consecutive_nan_threshold': 10,
        'param_type_mapping_sheet': '指标体系',
        'param_data_start_date': default_start_date,
        'param_data_end_date': default_end_date
    }

    # 初始化默认值
    for key, default_value in param_defaults.items():
        current_value = _get_state(key)
        if current_value is None:
            _set_state(key, default_value)
        elif key == 'param_data_end_date' and detected_end:
            # 结束日期特殊处理：自动更新到最新检测日期
            _set_state(key, default_end_date)

    # 第1行：日期范围
    row1_col1, row1_col2 = st_obj.columns(2)
    with row1_col1:
        start_date_value = st_obj.date_input(
            "数据开始日期",
            value=_get_state('param_data_start_date'),
            min_value=min_date,
            max_value=max_date,
            key="ss_dfm_data_start",
            help=f"设置系统处理数据的最早日期边界。数据实际范围：{detected_start} 到 {detected_end}" if detected_start else "设置系统处理数据的最早日期边界。"
        )
        _set_state("param_data_start_date", start_date_value)

        # 自动同步到模型训练tab
        if start_date_value != _get_state('param_data_start_date'):
            _sync_dates_to_train_model()

    with row1_col2:
        end_date_value = st_obj.date_input(
            "数据结束日期",
            value=_get_state('param_data_end_date'),
            min_value=min_date,
            max_value=max_date,
            key="ss_dfm_data_end",
            help=f"设置系统处理数据的最晚日期边界。数据实际范围：{detected_start} 到 {detected_end}" if detected_end else "设置系统处理数据的最晚日期边界。"
        )
        _set_state("param_data_end_date", end_date_value)

    # 第2行：目标工作表和目标变量
    row2_col1, row2_col2 = st_obj.columns(2)
    with row2_col1:
        target_sheet = st_obj.text_input(
            "目标工作表名称",
            value=_get_state('param_target_sheet_name'),
            key="ss_dfm_target_sheet"
        )
        _set_state("param_target_sheet_name", target_sheet)

    with row2_col2:
        target_var = st_obj.text_input(
            "目标变量",
            value=_get_state('param_target_variable'),
            key="ss_dfm_target_var"
        )
        _set_state("param_target_variable", target_var)

    # 第3行：NaN阈值和移除选项
    row3_col1, row3_col2 = st_obj.columns(2)
    with row3_col1:
        nan_threshold = st_obj.number_input(
            "连续 NaN 阈值",
            min_value=0,
            value=_get_state('param_consecutive_nan_threshold'),
            step=1,
            key="ss_dfm_nan_thresh"
        )
        _set_state("param_consecutive_nan_threshold", nan_threshold)

    with row3_col2:
        remove_nans = st_obj.checkbox(
            "移除过多连续 NaN 的变量",
            value=_get_state('param_remove_consecutive_nans'),
            key="ss_dfm_remove_nans",
            help="移除列中连续缺失值数量超过阈值的变量"
        )
        _set_state("param_remove_consecutive_nans", remove_nans)

    # 第4行：目标频率和映射表名称
    row4_col1, row4_col2 = st_obj.columns(2)
    with row4_col1:
        target_freq = st_obj.text_input(
            "目标频率",
            value=_get_state('param_target_freq'),
            help="例如: W-FRI, D, M, Q",
            key="ss_dfm_target_freq"
        )
        _set_state("param_target_freq", target_freq)

    with row4_col2:
        mapping_sheet = st_obj.text_input(
            "指标映射表名称",
            value=_get_state('param_type_mapping_sheet'),
            key="ss_dfm_type_map_sheet"
        )
        _set_state("param_type_mapping_sheet", mapping_sheet)

    return True


def _render_processing_section(st_obj, uploaded_file):
    """
    渲染数据处理与导出区域

    Returns:
        bool - 是否点击了处理按钮
    """

    st_obj.markdown("---")

    # 使用列布局
    title_left_col, title_right_col = st_obj.columns([1, 2])
    with title_left_col:
        st_obj.markdown("#### 数据预处理与导出")
    with title_right_col:
        st_obj.markdown("#### 处理结果")

    left_col, right_col = st_obj.columns([1, 2])

    with left_col:
        # 导出文件基础名称
        export_base_name = st_obj.text_input(
            "导出文件基础名称",
            value=_get_state('export_base_name', 'dfm_prepared_output'),
            key="ss_dfm_export_basename"
        )
        _set_state("export_base_name", export_base_name)

        # 三个按钮并排
        btn_col1, btn_col2, btn_col3 = st_obj.columns(3)

        with btn_col1:
            run_button_clicked = st_obj.button(
                "开始预处理",
                key="ss_dfm_run_preprocessing",
                type="primary",
                use_container_width=True
            )

        with btn_col2:
            download_data_placeholder = st_obj.empty()

        with btn_col3:
            download_map_placeholder = st_obj.empty()

        # 保存占位符到状态，以便后续使用
        _set_state('download_data_placeholder', download_data_placeholder)
        _set_state('download_map_placeholder', download_map_placeholder)

    return run_button_clicked, left_col, right_col


def _execute_data_preparation(st_obj, uploaded_file):
    """执行数据准备"""

    # 清空旧结果
    _set_state("processed_outputs", None)
    _set_state("prepared_data_df", None)
    _set_state("transform_log_obj", None)
    _set_state("industry_map_obj", None)
    _set_state("removed_vars_log_obj", None)

    if uploaded_file is None:
        st_obj.error("错误：请先上传训练数据集")
        return

    export_base_name = _get_state('export_base_name')
    if not export_base_name:
        st_obj.error("错误：请指定有效的文件基础名称")
        return

    try:
        progress_bar = st_obj.progress(0)
        status_text = st_obj.empty()

        try:
            status_text.text("正在准备数据...")
            progress_bar.progress(10)

            uploaded_file_bytes = uploaded_file.getvalue()
            excel_file_like_object = io.BytesIO(uploaded_file_bytes)

            # 获取参数
            start_date = _get_state('param_data_start_date')
            end_date = _get_state('param_data_end_date')

            # 从状态中读取检测到的有效日期范围
            detected_start = _get_state("detected_start_date")
            detected_end = _get_state("detected_end_date")

            # 使用检测到的有效日期范围进行验证和修正
            cutoff_date = date(2000, 1, 1)
            future_date = date(2030, 12, 31)

            if detected_start:
                if start_date and start_date < cutoff_date:
                    print(f"日期修正: 开始日期 {start_date} 早于2000年，使用检测日期 {detected_start}")
                    start_date = detected_start
                elif not start_date:
                    start_date = detected_start

            if detected_end:
                if end_date and end_date > future_date:
                    print(f"日期修正: 结束日期 {end_date} 过于遥远，使用检测日期 {detected_end}")
                    end_date = detected_end
                elif not end_date:
                    end_date = detected_end

            start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

            status_text.text("正在读取数据文件...")
            progress_bar.progress(20)

            # 准备NaN阈值参数
            nan_threshold_int = None
            if _get_state('param_remove_consecutive_nans'):
                nan_threshold = _get_state('param_consecutive_nan_threshold')
                if not pd.isna(nan_threshold):
                    try:
                        nan_threshold_int = int(nan_threshold)
                    except ValueError:
                        st_obj.warning(f"连续NaN阈值 '{nan_threshold}' 不是有效整数，将忽略此阈值")

            status_text.text("正在执行数据预处理...")
            progress_bar.progress(30)

            # 调用数据准备API
            from dashboard.models.DFM.prep import prepare_dfm_data

            print(f"调用prepare_dfm_data参数:")
            print(f"  - target_freq: {_get_state('param_target_freq')}")
            print(f"  - target_sheet_name: {_get_state('param_target_sheet_name')}")
            print(f"  - target_variable_name: {_get_state('param_target_variable')}")
            print(f"  - consecutive_nan_threshold: {nan_threshold_int}")
            print(f"  - data_start_date: {start_date_str}")
            print(f"  - data_end_date: {end_date_str}")

            result = prepare_dfm_data(
                uploaded_file=excel_file_like_object,
                target_freq=_get_state('param_target_freq'),
                target_sheet_name=_get_state('param_target_sheet_name'),
                target_variable_name=_get_state('param_target_variable'),
                consecutive_nan_threshold=nan_threshold_int,
                data_start_date=start_date_str,
                data_end_date=end_date_str,
                reference_sheet_name=_get_state('param_type_mapping_sheet')
            )

            status_text.text("数据预处理完成，正在生成结果...")
            progress_bar.progress(70)

            # 处理返回结果
            if result['status'] == 'success':
                prepared_data = result['data']
                industry_map = result['metadata']['variable_mapping']
                transform_log = result['metadata']['transform_log']
                removed_variables_log = result['metadata']['removal_log']
                mapping_validation = result['metadata'].get('mapping_validation', {})
                print(f"准备数据形状: {prepared_data.shape if prepared_data is not None else 'None'}")
                print(f"移除日志长度: {len(removed_variables_log) if removed_variables_log else 0}")
            else:
                prepared_data = None
                industry_map = {}
                transform_log = {}
                removed_variables_log = []
                mapping_validation = {}
                st_obj.error(f"数据预处理失败: {result['message']}")

            if prepared_data is not None:
                status_text.text("正在处理结果数据...")
                progress_bar.progress(80)

                # 保存数据对象到状态管理
                _set_state("prepared_data_df", prepared_data)
                _set_state("transform_log_obj", transform_log)
                _set_state("industry_map_obj", industry_map)
                _set_state("removed_vars_log_obj", removed_variables_log)
                _set_state("mapping_validation_result", mapping_validation)

                st_obj.success("数据预处理完成！结果已准备就绪，可用于模型训练模块。")
                st_obj.info(f"预处理后数据形状: {prepared_data.shape}")

                # 准备导出数据
                processed_outputs = {
                    'base_name': export_base_name,
                    'data': None,
                    'industry_map': None
                }

                if prepared_data is not None:
                    processed_outputs['data'] = prepared_data.to_csv(
                        index=True,
                        index_label='Date',
                        encoding='utf-8-sig'
                    ).encode('utf-8-sig')

                if industry_map:
                    try:
                        # 重新加载映射以确保和industry_map使用同一份数据
                        from dashboard.models.DFM.prep.modules.mapping_manager import load_mappings

                        excel_file_like_object.seek(0)
                        _, _, dfm_single_stage_map, dfm_two_stage_map, dfm_first_stage_target_map = load_mappings(
                            excel_path=excel_file_like_object,
                            sheet_name=_get_state('param_type_mapping_sheet'),
                            indicator_col='指标名称',
                            type_col='类型',
                            industry_col='行业',
                            single_stage_col='一次估计',
                            two_stage_col='二次估计',
                            first_stage_target_col='一阶段目标'
                        )

                        # 创建统一映射数据
                        all_indicators = list(industry_map.keys())
                        unified_mapping_data = []

                        for indicator in all_indicators:
                            industry = industry_map.get(indicator, '')
                            single_stage_default = dfm_single_stage_map.get(indicator, '')
                            two_stage_default = dfm_two_stage_map.get(indicator, '')
                            first_stage_target = dfm_first_stage_target_map.get(indicator, '')
                            unified_mapping_data.append({
                                'Indicator': indicator,
                                'Industry': industry,
                                '一次估计': single_stage_default,
                                '二次估计': two_stage_default,
                                '一阶段目标': first_stage_target
                            })

                        # 创建统一映射DataFrame
                        df_unified_map = pd.DataFrame(
                            unified_mapping_data,
                            columns=['Indicator', 'Industry', '一次估计', '二次估计', '一阶段目标']
                        )
                        processed_outputs['industry_map'] = df_unified_map.to_csv(
                            index=False,
                            encoding='utf-8-sig'
                        ).encode('utf-8-sig')

                        print(f"导出统一映射文件: {len(df_unified_map)} 条记录")
                        single_yes_count = len(df_unified_map[df_unified_map['一次估计'] == '是'])
                        two_yes_count = len(df_unified_map[df_unified_map['二次估计'] == '是'])
                        first_stage_target_count = len(df_unified_map[df_unified_map['一阶段目标'] != ''])
                        print(f"其中一次估计默认变量: {single_yes_count} 个")
                        print(f"其中二次估计默认变量: {two_yes_count} 个")
                        print(f"其中一阶段目标变量: {first_stage_target_count} 个")

                    except Exception as e:
                        st_obj.warning(f"映射文件转换到CSV时出错: {e}")
                        processed_outputs['industry_map'] = None

                # 保存处理结果
                _set_state("processed_outputs", processed_outputs)

                # 同步到训练模块
                st.session_state['train_model.prepared_data'] = prepared_data
                st.session_state['train_model.transform_log'] = transform_log

            else:
                progress_bar.progress(100)
                status_text.text("处理失败")
                st_obj.error("数据预处理失败或未返回数据。请检查控制台日志获取更多信息。")
                _set_state("processed_outputs", None)

            if 'progress_bar' in locals():
                progress_bar.progress(100)
                status_text.text("处理完成！")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

        except Exception as e:
            st_obj.error(f"运行数据预处理时发生错误: {e}")
            import traceback
            st_obj.text_area("详细错误信息:", traceback.format_exc(), height=200)
            _set_state("processed_outputs", None)

    except Exception as outer_e:
        st_obj.error(f"数据预处理过程中发生未预期的错误: {outer_e}")
        import traceback
        st_obj.text_area("详细错误信息:", traceback.format_exc(), height=200)


def _render_mapping_warnings(st_obj):
    """渲染行业映射警告信息"""

    mapping_validation = _get_state("mapping_validation_result")

    if not mapping_validation:
        return

    # conflicts = mapping_validation.get('conflicts', [])  # 冲突检查已禁用
    undefined = mapping_validation.get('undefined_in_reference', [])

    # 只显示未定义变量的警告
    if not undefined:
        return

    # 显示警告信息
    with st_obj.expander("行业映射警告", expanded=True):
        if undefined:
            st_obj.warning(f"发现 {len(undefined)} 个变量在指标体系中未定义行业（已标记为Unknown）")
            st_obj.markdown("**建议**: 在Excel模板的\"指标体系\"sheet中为这些变量补充行业信息")

            # 最多显示20个
            undefined_to_show = undefined[:20]
            st_obj.write(undefined_to_show)

            if len(undefined) > 20:
                st_obj.caption(f"... 还有 {len(undefined) - 20} 个未定义变量")


def _render_removed_variables_log(st_obj):
    """渲染移除变量详细日志（按原因分组显示）"""

    removed_vars_log = _get_state("removed_vars_log_obj")

    if removed_vars_log and len(removed_vars_log) > 0:
        # 按原因分组
        reason_groups = {}
        for entry in removed_vars_log:
            reason = entry.get('Reason', '未知原因')
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append(entry)

        # 显示统计信息
        st_obj.info(f"共有 {len(removed_vars_log)} 个变量被移除")

        # 使用expander显示详细信息
        with st_obj.expander("查看详细信息", expanded=False):
            for reason, entries in reason_groups.items():
                st_obj.markdown(f"**{reason}** ({len(entries)}个变量)")
                for entry in entries:
                    variable = entry.get('Variable', '未知变量')
                    details = entry.get('Details', {})

                    # 显示缺失时间段信息
                    if details and 'nan_period' in details:
                        nan_period = details.get('nan_period', '未知')
                        max_consecutive = details.get('max_consecutive_nan', 'N/A')
                        st_obj.markdown(f"- {variable} ({nan_period}, {max_consecutive})")
                    else:
                        st_obj.markdown(f"- {variable}")
                st_obj.markdown("---")

    elif removed_vars_log is not None and len(removed_vars_log) == 0:
        st_obj.success("所有变量都通过了筛选，没有变量被移除")


def _render_download_buttons(st_obj):
    """渲染下载按钮"""

    processed_outputs = _get_state("processed_outputs")

    if processed_outputs:
        base_name = processed_outputs['base_name']

        # 获取占位符
        download_data_placeholder = _get_state('download_data_placeholder')
        download_map_placeholder = _get_state('download_map_placeholder')

        # 在占位符中显示下载按钮
        if download_data_placeholder and processed_outputs.get('data'):
            with download_data_placeholder.container():
                st_obj.download_button(
                    label="下载数据",
                    data=processed_outputs['data'],
                    file_name=f"{base_name}.csv",
                    mime='text/csv',
                    key='download_data_csv',
                    use_container_width=True,
                    type="primary"
                )

        if download_map_placeholder and processed_outputs.get('industry_map'):
            with download_map_placeholder.container():
                st_obj.download_button(
                    label="下载映射",
                    data=processed_outputs['industry_map'],
                    file_name=f"{base_name}_industry_map.csv",
                    mime='text/csv',
                    key='download_industry_map_csv',
                    use_container_width=True,
                    type="primary"
                )


# ============================================================================
# 主渲染函数
# ============================================================================

def render_dfm_data_prep_page(st_obj):
    """
    渲染DFM数据准备页面

    在新架构下100%恢复旧版本功能：
    1. 文件上传与变更检测
    2. 日期范围自动检测与缓存
    3. 8个参数完整配置（4行2列布局）
    4. 自动映射加载
    5. 数据预处理执行
    6. 移除变量详细日志（按原因分组）
    7. 数据和映射文件导出
    8. 与训练模块状态同步
    """

    # 初始化状态
    if _get_state('training_data_file') is None:
        _set_state("training_data_file", None)
    if _get_state('prepared_data_df') is None:
        _set_state("prepared_data_df", None)
    if _get_state('transform_log_obj') is None:
        _set_state("transform_log_obj", None)
    if _get_state('industry_map_obj') is None:
        _set_state("industry_map_obj", None)
    if _get_state('removed_vars_log_obj') is None:
        _set_state("removed_vars_log_obj", None)
    if _get_state('export_base_name') is None:
        _set_state("export_base_name", "dfm_prepared_output")
    if _get_state('processed_outputs') is None:
        _set_state("processed_outputs", None)

    # 1. 文件上传区域
    uploaded_file = _render_file_upload_section(st_obj)

    st_obj.markdown("---")

    if uploaded_file is None:
        st_obj.info("请上传训练数据集以开始数据准备。")
        return

    # 2. 日期检测
    detected_start, detected_end, min_date, max_date = _render_date_detection(st_obj, uploaded_file)

    # 3. 自动加载映射数据
    if uploaded_file:
        mapping_sheet_name = _get_state('param_type_mapping_sheet', '指标体系')
        _auto_load_mapping_data(uploaded_file, mapping_sheet_name)

    # 4. 参数配置区域（8个参数）
    _render_parameter_config(st_obj, detected_start, detected_end, min_date, max_date)

    # 5. 数据处理与导出区域
    run_button_clicked, left_col, right_col = _render_processing_section(st_obj, uploaded_file)

    # 6. 执行数据准备
    if run_button_clicked:
        _execute_data_preparation(st_obj, uploaded_file)

    # 7. 在右侧列显示处理结果和警告信息
    with right_col:
        _render_mapping_warnings(st_obj)
        _render_removed_variables_log(st_obj)

    # 8. 渲染下载按钮
    _render_download_buttons(st_obj)


__all__ = ['render_dfm_data_prep_page']
