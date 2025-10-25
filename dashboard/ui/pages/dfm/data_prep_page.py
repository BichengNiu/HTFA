# -*- coding: utf-8 -*-
"""
DFM数据预处理页面组件

完全重构版本，与dfm_old_ui/data_prep_ui.py保持完全一致
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import json
import time
from datetime import datetime, date
from typing import Optional, Dict, Any

# 添加路径以导入统一状态管理
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

# 导入统一状态管理
from dashboard.core import get_global_dfm_manager
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)


def get_dfm_manager():
    """获取DFM模块管理器实例（使用全局单例）"""
    try:
        dfm_manager = get_global_dfm_manager()
        if dfm_manager is None:
            raise RuntimeError("全局DFM管理器不可用")
        return dfm_manager
    except Exception as e:
        print(f"[DFM Data Prep] Error getting DFM manager: {e}")
        raise RuntimeError(f"DFM管理器获取失败: {e}")


def get_dfm_state(key, default=None):
    """获取DFM状态值（使用统一状态管理）"""
    try:
        dfm_manager = get_dfm_manager()
        if dfm_manager:
            # 只从data_prep命名空间获取，使用标准键名
            value = dfm_manager.get_dfm_state('data_prep', key, default)
            from dashboard.ui.utils.debug_helpers import debug_log
            if key in ['dfm_training_data_file', 'dfm_uploaded_excel_file_path']:
                if value is not None:
                    if key == 'dfm_training_data_file' and hasattr(value, 'name'):
                        debug_log(f"状态读取 - 键: {key}, 文件名: {value.name}", "DEBUG")
                    else:
                        debug_log(f"状态读取 - 键: {key}, 值: {value}", "DEBUG")
                else:
                    debug_log(f"状态读取 - 键: {key}, 值: None (使用默认值: {default})", "DEBUG")
            return value
        else:
            debug_log(f"警告 - DFM统一状态管理器不可用，键: {key}", "WARNING")
            return default

    except Exception as e:
        debug_log(f"错误 - 获取DFM状态失败，键: {key}, 错误: {e}", "ERROR")
        return default


def set_dfm_state(key, value):
    """设置DFM状态值（使用统一状态管理）"""
    try:
        dfm_manager = get_dfm_manager()
        if dfm_manager:
            success = dfm_manager.set_dfm_state('data_prep', key, value)
            from dashboard.ui.utils.debug_helpers import debug_log
            if key in ['dfm_training_data_file', 'dfm_uploaded_excel_file_path']:
                if key == 'dfm_training_data_file' and hasattr(value, 'name'):
                    debug_log(f"状态设置 - 键: {key}, 文件名: {value.name}, 成功: {success}", "DEBUG")
                else:
                    debug_log(f"状态设置 - 键: {key}, 值: {value}, 成功: {success}", "DEBUG")
            return success
        else:
            debug_log(f"警告 - DFM统一状态管理器不可用，无法设置键: {key}", "WARNING")
            return False
    except Exception as e:
        debug_log(f"错误 - 设置DFM状态失败，键: {key}, 错误: {e}", "ERROR")
        return False


def sync_dates_to_train_model():
    """将数据准备tab的日期设置同步到模型训练tab"""
    try:
        dfm_manager = get_dfm_manager()

        # 获取数据准备tab的日期设置
        data_start = dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_start_date', None)
        data_end = dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_end_date', None)

        # 同步到模型训练tab
        if data_start:
            dfm_manager.set_dfm_state('train_model', 'dfm_training_start_date', data_start)

        if data_end:
            dfm_manager.set_dfm_state('train_model', 'dfm_validation_end_date', data_end)

        return True
    except Exception as e:
        return False

def get_dfm_param(key, default=None):
    """获取DFM参数的便捷函数"""
    dfm_manager = get_dfm_manager()
    if dfm_manager:
        # 尝试从多个可能的位置获取
        value = dfm_manager.get_dfm_state('data_prep', key, None)
        if value is None:
            value = dfm_manager.get_state(key, None)
        if value is None:
            value = dfm_manager.get_state(f'dfm_{key}', default)
        return value
    else:
        return default


def set_dfm_param(key, value):
    """设置DFM参数的便捷函数"""
    dfm_manager = get_dfm_manager()
    if dfm_manager:
        success = dfm_manager.set_dfm_state('data_prep', key, value)
        return success
    else:
        return False


def render_dfm_data_prep_tab(st):
    """Renders the DFM Model Data Preparation tab."""

    # 初始化文件存储
    if get_dfm_state('dfm_training_data_file') is None:
        set_dfm_state("dfm_training_data_file", None)

    if get_dfm_state('dfm_prepared_data_df') is None:
        set_dfm_state("dfm_prepared_data_df", None)
    if get_dfm_state('dfm_transform_log_obj') is None:
        set_dfm_state("dfm_transform_log_obj", None)
    if get_dfm_state('dfm_industry_map_obj') is None:
        set_dfm_state("dfm_industry_map_obj", None)
    if get_dfm_state('dfm_removed_vars_log_obj') is None:
        set_dfm_state("dfm_removed_vars_log_obj", None)
    if get_dfm_state('dfm_var_type_map_obj') is None:
        set_dfm_state("dfm_var_type_map_obj", None)

    # 初始化导出相关的状态 - 使用统一状态管理器
    if get_dfm_state('dfm_export_base_name') is None:
        set_dfm_state("dfm_export_base_name", "dfm_prepared_output")
    if get_dfm_state('dfm_processed_outputs') is None:
        set_dfm_state("dfm_processed_outputs", None)

    # 检查侧边栏是否已经上传了文件
    print("[HOT] [文件检查] 开始检查侧边栏上传的文件状态...")
    existing_file = get_dfm_state('dfm_training_data_file')
    existing_file_bytes = get_dfm_state('dfm_training_data_bytes')
    existing_file_path = get_dfm_state('dfm_uploaded_excel_file_path')

    print(f"[HOT] [文件检查] existing_file: {existing_file is not None}")
    print(f"[HOT] [文件检查] existing_file_bytes: {existing_file_bytes is not None and len(existing_file_bytes) > 0 if existing_file_bytes else False}")
    print(f"[HOT] [文件检查] existing_file 类型: {type(existing_file)}")
    if existing_file:
        print(f"[HOT] [文件检查] existing_file.name: {getattr(existing_file, 'name', 'N/A')}")
    print(f"[HOT] [文件检查] existing_file_path: {existing_file_path}")

    if existing_file_bytes and existing_file_path:
        # 从字节内容重建文件对象
        print(f"[HOT] [文件检查] 从字节内容重建文件对象: {existing_file_path}")
        import io

        class UploadedFileFromBytes:
            """从字节内容模拟 Streamlit UploadedFile 对象"""
            def __init__(self, name: str, data: bytes):
                self.name = name
                self._data = data

            def getvalue(self):
                return self._data

            def read(self):
                return self._data

        uploaded_file = UploadedFileFromBytes(existing_file_path, existing_file_bytes)
        print(f"[HOT] [文件检查] 文件对象重建成功，字节大小: {len(existing_file_bytes)}")
    elif existing_file and existing_file_path:
        # 使用已存在的文件对象（用于当前会话）
        print(f"[HOT] [文件检查] 找到已上传文件对象: {existing_file_path}")
        uploaded_file = existing_file
    elif existing_file:
        # 如果有文件对象但没有路径，仍然使用文件对象
        print(f"[HOT] [文件检查] 找到文件对象但路径为空，仍然使用文件")
        uploaded_file = existing_file
    else:
        # 如果侧边栏没有上传文件，显示简洁提示
        print("[HOT] [文件检查] 未找到已上传文件，显示警告")
        st.warning("[WARNING] 请先在侧边栏上传数据文件")
        uploaded_file = None

    if uploaded_file is not None:
        # 检查是否是新文件上传（避免重复处理）
        current_file = get_dfm_state('dfm_training_data_file')
        file_changed = (
            current_file is None or
            current_file.name != uploaded_file.name or
            get_dfm_state('dfm_file_processed', False) == False
        )

        # 如果是从备用上传器上传的新文件，需要保存到状态管理
        if file_changed and uploaded_file != existing_file:
            set_dfm_state("dfm_training_data_file", uploaded_file)
            # 保存Excel文件路径用于训练模块
            set_dfm_state("dfm_uploaded_excel_file_path", uploaded_file.name)
            set_dfm_state("dfm_use_full_data_preparation", True)

            print(f"[UI] 检测到新文件上传: {uploaded_file.name}，标记需要重新检测...")
            set_dfm_state("dfm_file_processed", False)  # 重置处理标记
            set_dfm_state("dfm_date_detection_needed", True)  # 标记需要日期检测

            # 标记文件已处理
            set_dfm_state("dfm_file_processed", True)


    else:
        st.info("请上传训练数据集。")

    # 根据数据文件的实际日期范围进行检测
    def detect_data_date_range(uploaded_file):
        """从上传的文件中检测数据的真实日期范围 - 获取所有sheet的日期并集"""
        try:
            if uploaded_file is None:
                return None, None

            # 检查文件对象是否有效
            if not hasattr(uploaded_file, 'getvalue'):
                print(f"警告: 文件对象无效，缺少getvalue方法")
                return None, None

            # 读取文件
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                print(f"警告: 文件内容为空")
                return None, None

            excel_file = io.BytesIO(file_bytes)

            all_dates_found = []

            # 获取所有工作表名称
            try:
                xl_file = pd.ExcelFile(excel_file)
                sheet_names = xl_file.sheet_names
                print(f"检测到工作表: {sheet_names}")
            except:
                sheet_names = [0]  # 回退到第一个工作表

            # 检查每个工作表寻找真实的日期数据
            for sheet_name in sheet_names:
                try:
                    excel_file.seek(0)  # 重置文件指针

                    # 跳过明显的元数据工作表
                    if any(keyword in str(sheet_name).lower() for keyword in ['指标体系', 'mapping', 'meta', 'info']):
                        print(f"跳过元数据工作表: {sheet_name}")
                        continue

                    # 首先尝试不带index_col读取，以便处理Wind格式
                    df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    if len(df_raw) < 5:  # 跳过数据太少的工作表
                        continue

                    # 检测日期列
                    date_col_idx = None
                    date_values = []
                    
                    # 检查是否是Wind格式（第一行第一列是"指标名称"）
                    if 'Wind' in sheet_name or (len(df_raw) > 0 and df_raw.iloc[0, 0] == '指标名称'):
                        # Wind格式：跳过第一行，第一列是日期
                        if len(df_raw) > 1:
                            date_values = pd.to_datetime(df_raw.iloc[1:, 0], errors='coerce')
                    else:
                        # 尝试第一列作为日期
                        for col_idx in range(min(2, len(df_raw.columns))):  # 检查前两列
                            try:
                                # 尝试转换为日期
                                test_dates = pd.to_datetime(df_raw.iloc[:, col_idx], errors='coerce')
                                valid_dates = test_dates[test_dates.notna()]
                                
                                # 如果有足够的有效日期，认为这是日期列
                                if len(valid_dates) > len(df_raw) * 0.5:  # 至少50%是有效日期
                                    date_values = valid_dates
                                    break
                            except:
                                continue
                    
                    # 收集有效日期
                    if len(date_values) > 0:
                        valid_dates = date_values[date_values.notna()]
                        if len(valid_dates) > 5:  # 至少要有5个有效日期
                            all_dates_found.extend(valid_dates.tolist())
                            print(f"  {sheet_name}: 找到 {len(valid_dates)} 个日期")

                except Exception as e:
                    print(f"  处理 {sheet_name} 时出错: {str(e)}")
                    continue

            # 汇总所有真实日期，返回实际的数据范围（并集）
            if all_dates_found:
                all_dates = pd.to_datetime(all_dates_found)

                # 过滤掉异常早期的日期（如1970年代的默认值/缺失值标记）
                # 只保留2000年1月1日之后的日期
                cutoff_date = pd.Timestamp('2000-01-01')
                valid_dates = all_dates[all_dates >= cutoff_date]

                if len(valid_dates) > 0:
                    actual_start = valid_dates.min().date()
                    actual_end = valid_dates.max().date()
                    print(f"检测到的总体日期范围: {actual_start} 到 {actual_end}")

                    # 如果过滤掉了很多早期日期，给出提示
                    if len(all_dates) > len(valid_dates):
                        filtered_count = len(all_dates) - len(valid_dates)
                        print(f"  (已过滤 {filtered_count} 个2000年之前的异常日期)")

                    return actual_start, actual_end
                else:
                    print("未能检测到2000年之后的有效日期数据")
                    return None, None
            else:
                print("未能检测到有效的日期数据")
                return None, None

        except Exception as e:
            return None, None

    # 增强缓存机制，避免重复检测
    # 检测上传文件的日期范围（只在文件变化时执行）
    current_file = get_dfm_state('dfm_training_data_file')
    if current_file and hasattr(current_file, 'getvalue'):
        try:
            file_hash = hash(current_file.getvalue())
            cache_key = f"date_range_{current_file.name}_{file_hash}"
        except Exception as e:
            st.warning(f"文件哈希计算失败: {e}")
            cache_key = f"date_range_{current_file.name}_fallback"
    else:
        cache_key = "date_range_none"

    # 优化：使用dfm_manager实例进行缓存操作
    dfm_manager = get_dfm_manager()

    # 检查缓存是否存在且有效
    cached_result = dfm_manager.get_dfm_state('data_prep', cache_key, None)
    cache_valid = (
        cached_result is not None and
        not get_dfm_state('dfm_date_detection_needed')
    )

    if not cache_valid:
        # 需要重新检测
        if current_file:
            print(f"[UI] 执行日期检测: {current_file.name}")
            with st.spinner("[VIEW] 正在检测数据日期范围..."):
                detected_start, detected_end = detect_data_date_range(current_file)
            # 缓存结果
            dfm_manager.set_dfm_state('data_prep', cache_key, (detected_start, detected_end))
            # 同时保存到专门的状态键，供按钮处理时使用
            set_dfm_state("dfm_detected_start_date", detected_start)
            set_dfm_state("dfm_detected_end_date", detected_end)
            set_dfm_state("dfm_date_detection_needed", False)

            # 清理旧的缓存
            try:
                # 获取所有状态键，查找以date_range_开头的旧缓存
                all_keys = dfm_manager.get_all_keys() if hasattr(dfm_manager, 'get_all_keys') else []
                old_keys = [k for k in all_keys if k.startswith("date_range_") and k != cache_key]
                for old_key in old_keys:
                    dfm_manager.delete_state(old_key)
            except Exception as e:
                print(f"[缓存清理] 清理旧缓存时出错: {e}")
        else:
            detected_start, detected_end = None, None
            dfm_manager.set_dfm_state('data_prep', cache_key, (None, None))
    else:
        # 使用缓存的结果
        detected_start, detected_end = cached_result if cached_result else (None, None)
        if detected_start and detected_end:
            # 静默处理缓存的日期范围，避免重复日志
            # 同时保存到专门的状态键，供按钮处理时使用
            set_dfm_state("dfm_detected_start_date", detected_start)
            set_dfm_state("dfm_detected_end_date", detected_end)

    # 设置默认值：优先使用检测到的日期，否则使用硬编码默认值
    default_start_date = detected_start if detected_start else datetime(2020, 1, 1).date()
    default_end_date = detected_end if detected_end else datetime(2025, 4, 30).date()

    param_defaults = {
        'dfm_param_target_variable': '规模以上工业增加值:当月同比',
        'dfm_param_target_sheet_name': '工业增加值同比增速_月度_同花顺',
        'dfm_param_target_freq': 'W-FRI',
        'dfm_param_remove_consecutive_nans': True,
        'dfm_param_consecutive_nan_threshold': 10,
        'dfm_param_type_mapping_sheet': '指标体系',
        'dfm_param_data_start_date': default_start_date,
        'dfm_param_data_end_date': default_end_date
    }

    # 只在首次初始化或文件更新时设置默认值
    # 优化：批量获取参数以减少重复调用
    dfm_manager = get_dfm_manager()
    for key, default_value in param_defaults.items():
        current_value = dfm_manager.get_dfm_state('data_prep', key, None)
        if current_value is None:
            dfm_manager.set_dfm_state('data_prep', key, default_value)
        elif key in ['dfm_param_data_start_date', 'dfm_param_data_end_date'] and detected_start and detected_end:
            # 如果检测到新的日期范围，更新日期设置
            if key == 'dfm_param_data_start_date':
                dfm_manager.set_dfm_state('data_prep', key, default_start_date)
            elif key == 'dfm_param_data_end_date':
                dfm_manager.set_dfm_state('data_prep', key, default_end_date)

    # 显示检测结果
    if detected_start and detected_end:
        st.success(f"[SUCCESS] 已自动检测文件日期范围: {detected_start} 到 {detected_end}")
    elif current_file:
        st.warning("[WARNING] 无法自动检测文件日期范围，使用默认值。请手动调整日期设置。")

    if current_file:
        _auto_load_mapping_data_if_needed(current_file)

    # 优化：批量获取参数值以减少重复调用
    dfm_manager = get_dfm_manager()
    param_values = {
        'data_start_date': dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_start_date', None),
        'data_end_date': dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_end_date', None),
        'target_sheet_name': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_sheet_name', None),
        'target_variable': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_variable', None)
    }

    # 根据检测到的日期范围设置日期选择器的限制
    # 如果检测到了日期范围，使用检测到的范围；否则使用宽松的默认范围
    if detected_start and detected_end:
        # 为了用户友好，允许选择比实际数据范围稍宽的日期
        min_date = detected_start - pd.Timedelta(days=365)  # 提前一年
        max_date = detected_end + pd.Timedelta(days=365)    # 延后一年
    else:
        # 如果没有检测到日期，使用宽松的默认范围
        min_date = date(1990, 1, 1)
        max_date = date(2050, 12, 31)
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        start_date_value = st.date_input(
            "数据开始日期 (系统边界)",
            value=param_values['data_start_date'],
            min_value=min_date,  # 使用动态计算的最小日期
            max_value=max_date,  # 使用动态计算的最大日期
            key="ss_dfm_data_start",
            help=f"设置系统处理数据的最早日期边界。数据实际范围：{detected_start} 到 {detected_end}" if detected_start else "设置系统处理数据的最早日期边界。"
        )
        set_dfm_state("dfm_param_data_start_date", start_date_value)

        # 自动同步到模型训练tab
        if start_date_value != param_values['data_start_date']:
            sync_dates_to_train_model()
    with row1_col2:
        set_dfm_state("dfm_param_data_end_date", st.date_input(
            "数据结束日期 (系统边界)",
            value=param_values['data_end_date'],
            min_value=min_date,  # 使用动态计算的最小日期
            max_value=max_date,  # 使用动态计算的最大日期
            key="ss_dfm_data_end",
            help=f"设置系统处理数据的最晚日期边界。数据实际范围：{detected_start} 到 {detected_end}" if detected_end else "设置系统处理数据的最晚日期边界。"
        ))

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        set_dfm_state("dfm_param_target_sheet_name", st.text_input(
            "目标工作表名称 (Target Sheet Name)",
            value=param_values['target_sheet_name'],
            key="ss_dfm_target_sheet"
        ))
    with row2_col2:
        set_dfm_state("dfm_param_target_variable", st.text_input(
            "目标变量 (Target Variable)",
            value=param_values['target_variable'],
            key="ss_dfm_target_var"
        ))

    # 继续批量获取剩余参数值
    param_values.update({
        'consecutive_nan_threshold': dfm_manager.get_dfm_state('data_prep', 'dfm_param_consecutive_nan_threshold', None),
        'remove_consecutive_nans': dfm_manager.get_dfm_state('data_prep', 'dfm_param_remove_consecutive_nans', None),
        'target_freq': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_freq', None),
        'type_mapping_sheet': dfm_manager.get_dfm_state('data_prep', 'dfm_param_type_mapping_sheet', None)
    })

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        set_dfm_state("dfm_param_consecutive_nan_threshold", st.number_input(
            "连续 NaN 阈值 (Consecutive NaN Threshold)",
            min_value=0,
            value=param_values['consecutive_nan_threshold'],
            step=1,
            key="ss_dfm_nan_thresh"
        ))
    with row3_col2:
        set_dfm_state("dfm_param_remove_consecutive_nans", st.checkbox(
            "移除过多连续 NaN 的变量",
            value=param_values['remove_consecutive_nans'],
            key="ss_dfm_remove_nans",
            help="移除列中连续缺失值数量超过阈值的变量"
        ))

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        set_dfm_state("dfm_param_target_freq", st.text_input(
            "目标频率 (Target Frequency)",
            value=param_values['target_freq'],
            help="例如: W-FRI, D, M, Q",
            key="ss_dfm_target_freq"
        ))
    with row4_col2:
        set_dfm_state("dfm_param_type_mapping_sheet", st.text_input(
            "指标映射表名称 (Type Mapping Sheet)",
            value=param_values['type_mapping_sheet'],
            key="ss_dfm_type_map_sheet"
        ))

    st.markdown("--- ") # Separator before the new section

    # 使用列布局，让两个标题在同一水平线上
    title_left_col, title_right_col = st.columns([1, 2])
    with title_left_col:
        st.markdown("#### 数据预处理与导出")
    with title_right_col:
        st.markdown("#### 处理结果")

    left_col, right_col = st.columns([1, 2]) # Left col for inputs, Right col for outputs/messages

    with left_col:
        # 获取导出基础名称参数
        export_base_name = dfm_manager.get_dfm_state('data_prep', 'dfm_export_base_name', None)
        set_dfm_state("dfm_export_base_name", st.text_input(
            "导出文件基础名称 (Export Base Filename)",
            value=export_base_name,
            key="ss_dfm_export_basename"
        ))

        # 将按钮放在一行，使用列布局
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            run_button_clicked = st.button("开始预处理", key="ss_dfm_run_preprocessing", use_container_width=True)
        with btn_col2:
            # 下载按钮占位，实际按钮在后面根据数据可用性显示
            download_data_placeholder = st.empty()
        with btn_col3:
            # 下载行业映射按钮占位
            download_map_placeholder = st.empty()

    # 按钮点击处理逻辑
    if run_button_clicked:
        # 清空旧的处理结果
        set_dfm_state("dfm_processed_outputs", None)
        set_dfm_state("dfm_prepared_data_df", None)
        set_dfm_state("dfm_transform_log_obj", None)
        set_dfm_state("dfm_industry_map_obj", None)
        set_dfm_state("dfm_removed_vars_log_obj", None)
        set_dfm_state("dfm_var_type_map_obj", None)

        current_file = get_dfm_state('dfm_training_data_file')
        if current_file is None:
            st.error("错误：请先上传训练数据集！")
        # 优化：批量获取所有需要的参数
        processing_params = {
            'export_base_name': dfm_manager.get_dfm_state('data_prep', 'dfm_export_base_name', None),
            'data_start_date': dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_start_date', None),
            'data_end_date': dfm_manager.get_dfm_state('data_prep', 'dfm_param_data_end_date', None),
            'target_freq': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_freq', None),
            'target_sheet_name': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_sheet_name', None),
            'target_variable': dfm_manager.get_dfm_state('data_prep', 'dfm_param_target_variable', None),
            'remove_consecutive_nans': dfm_manager.get_dfm_state('data_prep', 'dfm_param_remove_consecutive_nans', None),
            'consecutive_nan_threshold': dfm_manager.get_dfm_state('data_prep', 'dfm_param_consecutive_nan_threshold', None),
            'type_mapping_sheet': dfm_manager.get_dfm_state('data_prep', 'dfm_param_type_mapping_sheet', None)
        }

        if not processing_params['export_base_name']:
            st.error("错误：请指定有效的文件基础名称！")
        else:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("[LOADING] 正在准备数据...")
                    progress_bar.progress(10)

                    # 检查文件是否存在
                    if current_file is None:
                        st.error("未找到上传的文件，请重新上传数据文件。")
                        return

                    uploaded_file_bytes = current_file.getvalue()
                    excel_file_like_object = io.BytesIO(uploaded_file_bytes)

                    # 使用批量获取的参数
                    start_date = processing_params['data_start_date']
                    end_date = processing_params['data_end_date']

                    # 从状态中读取检测到的有效日期范围
                    detected_start_from_state = get_dfm_state("dfm_detected_start_date")
                    detected_end_from_state = get_dfm_state("dfm_detected_end_date")

                    # 调试信息：打印所有日期值
                    print(f"[日期检查] start_date: {start_date}, 类型: {type(start_date)}")
                    print(f"[日期检查] end_date: {end_date}, 类型: {type(end_date)}")
                    print(f"[日期检查] detected_start_from_state: {detected_start_from_state}, 类型: {type(detected_start_from_state)}")
                    print(f"[日期检查] detected_end_from_state: {detected_end_from_state}, 类型: {type(detected_end_from_state)}")

                    # 使用检测到的有效日期范围进行验证和修正
                    cutoff_date = date(2000, 1, 1)
                    future_date = date(2030, 12, 31)

                    # 强制使用检测到的日期，如果用户选择的日期异常
                    if detected_start_from_state:
                        if start_date and start_date < cutoff_date:
                            print(f"[日期修正] 用户选择的开始日期 {start_date} 早于2000年，使用检测到的日期 {detected_start_from_state}")
                            start_date = detected_start_from_state
                        elif not start_date:
                            print(f"[日期修正] 开始日期为空，使用检测到的日期 {detected_start_from_state}")
                            start_date = detected_start_from_state

                    if detected_end_from_state:
                        if end_date and end_date > future_date:
                            print(f"[日期修正] 用户选择的结束日期 {end_date} 过于遥远，使用检测到的日期 {detected_end_from_state}")
                            end_date = detected_end_from_state
                        elif not end_date:
                            print(f"[日期修正] 结束日期为空，使用检测到的日期 {detected_end_from_state}")
                            end_date = detected_end_from_state

                    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
                    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

                    status_text.text("[DATA] 正在读取数据文件...")
                    progress_bar.progress(20)

                    # [CONFIG] 修复：只有在启用移除连续NaN功能时才传递阈值
                    nan_threshold_int = None
                    remove_consecutive_nans = processing_params['remove_consecutive_nans']
                    if remove_consecutive_nans:
                        nan_threshold = processing_params['consecutive_nan_threshold']
                        if not pd.isna(nan_threshold):
                            try:
                                nan_threshold_int = int(nan_threshold)
                            except ValueError:
                                st.warning(f"连续NaN阈值 '{nan_threshold}' 不是一个有效的整数。将忽略此阈值。")
                                nan_threshold_int = None

                    # 数据预处理参数
                    status_text.text("[CONFIG] 正在执行数据预处理...")
                    progress_bar.progress(30)

                    # 调用真正的数据预处理函数
                    from dashboard.models.DFM.prep.data_preparation import prepare_data

                    try:
                        # 打印调试信息
                        print(f"[DEBUG] 调用prepare_data参数:")
                        print(f"  - target_freq: {processing_params['target_freq']}")
                        print(f"  - target_sheet_name: {processing_params['target_sheet_name']}")
                        print(f"  - target_variable_name: {processing_params['target_variable']}")
                        print(f"  - consecutive_nan_threshold: {nan_threshold_int}")
                        print(f"  - data_start_date: {start_date_str}")
                        print(f"  - data_end_date: {end_date_str}")
                        print(f"  - reference_sheet_name: {processing_params['type_mapping_sheet']}")
                        
                        results = prepare_data(
                            excel_path=excel_file_like_object,
                            target_freq=processing_params['target_freq'],
                            target_sheet_name=processing_params['target_sheet_name'],
                            target_variable_name=processing_params['target_variable'],
                            consecutive_nan_threshold=nan_threshold_int,
                            data_start_date=start_date_str,
                            data_end_date=end_date_str,
                            reference_sheet_name=processing_params['type_mapping_sheet']
                        )

                        status_text.text("[SUCCESS] 数据预处理完成，正在生成结果...")
                        progress_bar.progress(70)

                        # 打印返回结果调试信息
                        print(f"[DEBUG] prepare_data返回结果:")
                        print(f"  - results类型: {type(results)}")
                        print(f"  - results长度: {len(results) if results else 'None'}")
                        
                        # 解包结果
                        if results and len(results) >= 4:
                            prepared_data, industry_map, transform_log, removed_variables_detailed_log = results
                            print(f"[DEBUG] 解包后的结果:")
                            print(f"  - prepared_data: {type(prepared_data)}, shape: {prepared_data.shape if prepared_data is not None else 'None'}")
                            print(f"  - industry_map: {type(industry_map)}")
                            print(f"  - transform_log: {type(transform_log)}")
                            print(f"  - removed_log长度: {len(removed_variables_detailed_log) if removed_variables_detailed_log else 0}")
                        else:
                            prepared_data = None
                            print(f"[DEBUG] results不满足条件，prepared_data设置为None")
                            industry_map = {}
                            transform_log = {}
                            removed_variables_detailed_log = []
                            st.warning("数据预处理返回的结果格式不正确")

                    except Exception as prep_e:
                        st.error(f"数据预处理失败: {prep_e}")
                        import traceback
                        st.text_area("预处理错误详情:", traceback.format_exc(), height=150)
                        prepared_data = None
                        industry_map = {}
                        transform_log = {}
                        removed_variables_detailed_log = []

                    if prepared_data is not None:
                        status_text.text("[INFO] 正在处理结果数据...")
                        progress_bar.progress(80)

                        # 保存数据对象到统一状态管理器
                        print(f"[DEBUG] 保存removed_variables_detailed_log到状态: {len(removed_variables_detailed_log) if removed_variables_detailed_log else 0} 条记录")
                        set_dfm_state("dfm_prepared_data_df", prepared_data)
                        set_dfm_state("dfm_transform_log_obj", transform_log)
                        set_dfm_state("dfm_industry_map_obj", industry_map)
                        set_dfm_state("dfm_removed_vars_log_obj", removed_variables_detailed_log)
                        set_dfm_state("dfm_var_type_map_obj", {})

                        # 验证保存是否成功
                        saved_log = get_dfm_state("dfm_removed_vars_log_obj")
                        print(f"[DEBUG] 验证保存后的removed_vars_log: {len(saved_log) if saved_log else 0} 条记录")

                        st.success("数据预处理完成！结果已准备就绪，可用于模型训练模块。")
                        st.info(f"[DATA] 预处理后数据形状: {prepared_data.shape}")

                        # Prepare for download (existing logic)
                        export_base_name = processing_params['export_base_name']
                        processed_outputs = {
                            'base_name': export_base_name,
                            'data': None, 'industry_map': None, 'transform_log': None, 'removed_vars_log': None
                        }

                        if prepared_data is not None:
                            processed_outputs['data'] = prepared_data.to_csv(index=True, index_label='Date', encoding='utf-8-sig').encode('utf-8-sig')

                        if industry_map:
                            try:
                                df_industry_map = pd.DataFrame(list(industry_map.items()), columns=['Indicator', 'Industry'])
                                processed_outputs['industry_map'] = df_industry_map.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                            except Exception as e_im:
                                st.warning(f"行业映射转换到CSV时出错: {e_im}")
                                processed_outputs['industry_map'] = None

                        # 保存处理结果到统一状态管理器
                        set_dfm_state("dfm_processed_outputs", processed_outputs)

                    else:
                        progress_bar.progress(100)
                        status_text.text("[ERROR] 处理失败")
                        st.error("数据预处理失败或未返回数据。请检查控制台日志获取更多信息。")
                        set_dfm_state("dfm_processed_outputs", None)

                    if 'progress_bar' in locals():
                        progress_bar.progress(100)
                        status_text.text("[SUCCESS] 处理完成！")
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()

                except Exception as e:
                    st.error(f"运行数据预处理时发生错误: {e}")
                    import traceback
                    st.text_area("详细错误信息:", traceback.format_exc(), height=200)
                    set_dfm_state("dfm_processed_outputs", None)

            except Exception as outer_e:
                st.error(f"数据预处理过程中发生未预期的错误: {outer_e}")
                import traceback
                st.text_area("详细错误信息:", traceback.format_exc(), height=200)

    # 在右侧列显示移除变量的信息
    with right_col:
        removed_vars_log = get_dfm_state("dfm_removed_vars_log_obj")
        if removed_vars_log and len(removed_vars_log) > 0:
            # 按原因分组统计，并保留完整的entry信息
            reason_groups = {}
            for entry in removed_vars_log:
                reason = entry.get('Reason', '未知原因')
                variable = entry.get('Variable', '未知变量')
                if reason not in reason_groups:
                    reason_groups[reason] = []
                reason_groups[reason].append(entry)

            # 显示统计信息
            st.info(f"共有 {len(removed_vars_log)} 个变量被移除")

            # 使用expander显示详细信息
            with st.expander("查看详细信息", expanded=True):
                for reason, entries in reason_groups.items():
                    st.markdown(f"**{reason}** ({len(entries)}个变量)")
                    for entry in entries:
                        variable = entry.get('Variable', '未知变量')
                        details = entry.get('Details', {})

                        # 显示缺失时间段信息（去掉汉字标签）
                        if details and 'nan_period' in details:
                            nan_period = details.get('nan_period', '未知')
                            max_consecutive = details.get('max_consecutive_nan', 'N/A')
                            st.markdown(f"- {variable} ({nan_period}, {max_consecutive})")
                        else:
                            st.markdown(f"- {variable}")
                    st.markdown("---")
        elif removed_vars_log is not None and len(removed_vars_log) == 0:
            st.success("所有变量都通过了筛选，没有变量被移除")

    # Render download buttons if data is available - 放在左侧列的占位符中
    processed_outputs = get_dfm_state("dfm_processed_outputs")
    if processed_outputs:
        outputs = processed_outputs
        base_name = outputs['base_name']

        # 在左侧列的占位符中显示下载按钮
        with download_data_placeholder.container():
            if outputs.get('data'):
                st.download_button(
                    label="下载数据",
                    data=outputs['data'],
                    file_name=f"{base_name}.csv",
                    mime='text/csv',
                    key='download_data_csv',
                    use_container_width=True
                )

        with download_map_placeholder.container():
            if outputs.get('industry_map'):
                st.download_button(
                    label="下载映射",
                    data=outputs['industry_map'],
                    file_name=f"{base_name}_industry_map.csv",
                    mime='text/csv',
                    key='download_industry_map_csv',
                    use_container_width=True
                )


def _auto_load_mapping_data_if_needed(current_file):
    """
    自动加载映射数据（如果需要）- 优化版本

    Args:
        current_file: 当前文件对象
    """
    # 会话级缓存，避免重复加载
    cache_key = f"mapping_loaded_{current_file.name if current_file else 'none'}"
    if get_dfm_state(cache_key, False):
        return

    try:
        # 检查是否已经加载了映射数据
        existing_industry_map = get_dfm_state('dfm_industry_map_obj', None)
        existing_type_map = get_dfm_state('dfm_var_type_map_obj', None)

        # 如果映射数据已存在且不为空，则不需要重新加载
        if (existing_industry_map and len(existing_industry_map) > 0 and
            existing_type_map and len(existing_type_map) > 0):
            set_dfm_state(cache_key, True)
            return

        if current_file is None:
            return

        # 获取映射表名称
        mapping_sheet_name = get_dfm_state('dfm_param_type_mapping_sheet', '指标体系')

        # 加载映射数据
        from dashboard.models.DFM.prep.data_preparation import load_mappings

        var_type_map, var_industry_map_loaded = load_mappings(
            excel_path=current_file,
            sheet_name=mapping_sheet_name,
            indicator_col='指标名称',
            type_col='类型',
            industry_col='行业'
        )

        # 保存映射数据
        final_industry_map = var_industry_map_loaded if var_industry_map_loaded else {}
        final_type_map = var_type_map if var_type_map else {}

        set_dfm_state("dfm_var_type_map_obj", final_type_map)
        set_dfm_state("dfm_industry_map_obj", final_industry_map)

        # 标记为已加载，避免重复加载
        set_dfm_state(cache_key, True)

    except Exception as e:
        # 静默处理映射数据加载失败
        pass


def render_dfm_data_prep_page(st_module: Any) -> Dict[str, Any]:
    """
    渲染DFM数据预处理页面

    Args:
        st_module: Streamlit模块

    Returns:
        Dict[str, Any]: 渲染结果
    """
    try:
        # 调用主要的UI渲染函数
        render_dfm_data_prep_tab(st_module)

        return {
            'status': 'success',
            'page': 'data_prep',
            'components': ['file_upload', 'parameter_config', 'data_processing']
        }

    except Exception as e:
        st_module.error(f"数据预处理页面渲染失败: {str(e)}")
        return {
            'status': 'error',
            'page': 'data_prep',
            'error': str(e)
        }


