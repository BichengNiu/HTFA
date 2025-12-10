# -*- coding: utf-8 -*-
"""
DFM模型训练页面组件

完全版本，与dfm_old_ui/train_model_ui.py保持完全一致
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta, date, time
from collections import defaultdict
import traceback
from typing import Dict, List, Optional, Union, Any

# 添加路径以导入状态管理辅助函数
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

import logging

# 导入文本标准化工具（从共享工具库）
from dashboard.models.DFM.utils.text_utils import normalize_text

# 导入组件化训练状态管理
from dashboard.models.DFM.train.ui.components.training_status import TrainingStatusComponent

# 导入新增工具和组件
from dashboard.models.DFM.train.utils import StateManager, filter_industries_by_target, get_non_target_indicators
from dashboard.models.DFM.train.ui.components.file_uploader_component import FileUploaderComponent
from dashboard.models.DFM.train.config import UIConfig
from dashboard.models.DFM.train.ui.utils.config_builder import TrainingConfigBuilder
from dashboard.models.DFM.train.ui.utils.text_helpers import (
    normalize_variable_name,
    normalize_variable_name_no_space,
    build_normalized_mapping,
    filter_exclude_targets
)

# 配置日志记录器
logger = logging.getLogger(__name__)

# 创建全局状态管理器实例
_state = StateManager('train_model')

def debug_training_state(message: str, show_in_ui: bool = False):
    """
    调试训练状态同步过程

    Args:
        message: 调试消息
        show_in_ui: 是否在UI中显示调试信息
    """
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    debug_msg = f"[{timestamp}] {message}"

    # 控制台输出
    print(debug_msg)

    # 日志记录
    logger.debug(debug_msg)

    # 可选的UI显示（用于开发调试）
    if show_in_ui and hasattr(st, 'sidebar'):
        with st.sidebar.expander("[CONFIG] 训练状态调试", expanded=False):
            st.text(debug_msg)

def check_current_training_state():
    """检查当前训练状态的详细信息"""
    try:
        training_status = _state.get('dfm_training_status')
        training_results = _state.get('dfm_model_results_paths')
        training_completed_refreshed = _state.get('training_completed_refreshed')
        polling_count = _state.get('training_completion_polling_count', 0)
        training_log = _state.get('dfm_training_log', [])

        debug_msg = f"""
当前训练状态检查:
- dfm_training_status: {training_status}
- dfm_model_results_paths: {training_results}
- training_completed_refreshed: {training_completed_refreshed}
- training_completion_polling_count: {polling_count}
- dfm_training_log 条数: {len(training_log)}
"""
        print(debug_msg)
        return {
            'training_status': training_status,
            'training_results': training_results,
            'training_completed_refreshed': training_completed_refreshed,
            'polling_count': polling_count,
            'log_count': len(training_log)
        }
    except Exception as e:
        print(f"[ERROR] 状态检查失败: {e}")
        return None

# 配置已移除，所有参数通过UI设置
CONFIG_AVAILABLE = False

# 配置已移除，使用硬编码默认值
class TrainModelConfig:
    # 基于项目结构的路径设置
    PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

    # UI默认配置值
    TYPE_MAPPING_SHEET = '指标体系'
    TARGET_VARIABLE = '规模以上工业增加值:当月同比'
    INDICATOR_COLUMN_NAME_IN_EXCEL = '指标名称'
    INDUSTRY_COLUMN_NAME_IN_EXCEL = '行业'
    TYPE_COLUMN_NAME_IN_EXCEL = '类型'

config = TrainModelConfig()

_TRAIN_UI_IMPORT_ERROR_MESSAGE = None # Stores combined error messages
_DATA_PREPARATION_MODULE = None

# 2. 导入数据预处理模块
# 从data_prep目录导入
data_prep_dir = os.path.join(dashboard_root, 'dashboard', 'DFM', 'data_prep')
if data_prep_dir not in sys.path:
    sys.path.insert(0, data_prep_dir)

# 直接实现load_mappings函数，避免复杂的导入问题
def load_mappings_direct(excel_path: str, sheet_name: str, indicator_col: str = '指标名称',
                       type_col: str = '类型', industry_col: str = '行业'):
    """直接实现的load_mappings函数"""
    import pandas as pd
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 移除类型映射，只保留行业映射
    var_industry_map = {}

    if indicator_col in df.columns:
        if industry_col in df.columns:
            # 标准化键名（转换为小写）以与训练过程保持一致
            raw_industry_map = dict(zip(df[indicator_col].fillna(''), df[industry_col].fillna('')))
            var_industry_map = {
                normalize_variable_name(k): str(v).strip()
                for k, v in raw_industry_map.items()
                if k and str(k).strip() and v and str(v).strip()
            }

    # 只返回行业映射（移除类型映射）
    return {}, var_industry_map

class MappingOnlyWrapper:
    @staticmethod
    def load_mappings(*args, **kwargs):
        return load_mappings_direct(*args, **kwargs)

_DATA_PREPARATION_MODULE = MappingOnlyWrapper()
print("[SUCCESS] 数据预处理模块（直接实现）加载成功")
# 清除任何之前的错误消息，因为导入成功了
_TRAIN_UI_IMPORT_ERROR_MESSAGE = None

# Make the data_preparation available for the rest of the module
data_preparation = _DATA_PREPARATION_MODULE

# 3. 导入DFM训练脚本
try:
    # 导入train模块
    from dashboard.models.DFM.train import DFMTrainer, TrainingConfig, TrainingResult
    print("[SUCCESS] 成功导入DFMTrainer和TrainingConfig from train")
    _TRAIN_UI_IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    print(f"[ERROR] 导入train模块失败: {e}")
    _TRAIN_UI_IMPORT_ERROR_MESSAGE = f"train module import error: {e}"
    raise

# 模拟的UIDefaults和TrainDefaults类
class UIDefaults:
    NUM_COLS_INDUSTRY = 3
    VARIABLE_SELECTION_OPTIONS = {
        'none': "无筛选 (使用全部已选变量)",
        'global_backward': "全局后向剔除 (在已选变量中筛选)"
    }
    FACTOR_SELECTION_STRATEGY_OPTIONS = {
        'information_criteria': "信息准则",
        'fixed_number': "固定因子数",
        'cumulative_variance': "累积方差贡献"
    }
    INFORMATION_CRITERION_OPTIONS = {
        'bic': "BIC",
        'aic': "AIC",
        'hqc': "HQC"
    }
    EM_CONVERGENCE_CRITERION_OPTIONS = {
        'params': "参数变化",
        'likelihood': "似然函数"
    }
    MISSING_VALUE_OPTIONS = {
        'interpolate': "线性插值",
        'forward_fill': "前向填充",
        'drop': "删除缺失"
    }
    IC_MAX_FACTORS_DEFAULT = 10
    MAX_ITERATIONS_DEFAULT = 30
    MAX_ITERATIONS_MIN = 1
    MAX_ITERATIONS_STEP = 10
    CUM_VARIANCE_MIN = 0.1

class TrainDefaults:
    VARIABLE_SELECTION_METHOD = 'none'
    FACTOR_SELECTION_STRATEGY = 'information_criteria'
    EM_MAX_ITER = 100
    FIXED_NUMBER_OF_FACTORS = 3
    CUM_VARIANCE_THRESHOLD = 0.8
    FACTOR_AR_ORDER = 1
    INFORMATION_CRITERION = 'bic'
    EM_CONVERGENCE_CRITERION = 'params'
    EM_TOLERANCE = 1e-6
    MISSING_VALUE_METHOD = 'interpolate'

    IC_MAX_FACTORS = 10
    K_FACTORS_RANGE_MIN = 1
    TRAINING_YEARS_BACK = 5
    VALIDATION_END_YEAR = 2024
    VALIDATION_END_MONTH = 12
    VALIDATION_END_DAY = 31




# 已删除 load_mappings_from_state() 函数 - 移除与数据准备模块的耦合
# 现在完全依赖文件上传器加载的映射数据


def _reset_training_state():
    """重置所有训练相关状态"""
    from dashboard.core.ui.utils.debug_helpers import debug_log

    training_keys = [
        'dfm_training_status',
        'dfm_training_log',
        'dfm_training_progress',
        'dfm_model_results_paths',
        'dfm_model_results',
        'dfm_training_error',
        'dfm_training_start_time',
        'dfm_training_end_time',
        'dfm_force_reset_training_state',
        'dfm_page_initialized',
        'dfm_training_completed_timestamp'
    ]

    for key in training_keys:
        _state.set(key, None)
        debug_log(f"状态重置 - 已清理状态键: {key}", "DEBUG")

    # 重置为初始状态
    _state.set('dfm_training_status', '等待开始')
    _state.set('dfm_training_log', [])
    _state.set('dfm_training_progress', 0)
    _state.set('dfm_model_results_paths', None)
    _state.set('dfm_model_results', None)

    debug_log("状态重置 - 已重置所有训练状态到初始值", "DEBUG")




def convert_to_datetime(date_input):
    """将日期输入转换为datetime对象"""
    if date_input is None:
        return None

    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, date):
        return datetime.combine(date_input, time.min)
    elif isinstance(date_input, str):
        try:
            return datetime.strptime(date_input, '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.strptime(date_input, '%Y/%m/%d')
            except ValueError:
                return None
    else:
        return None


def render_dfm_model_training_page(st_instance):

    # 确保datetime在函数开头就可用
    from datetime import datetime
    import time

    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        if "train" in _TRAIN_UI_IMPORT_ERROR_MESSAGE:
            st_instance.error(f"关键模块导入错误，模型训练功能不可用:\n{_TRAIN_UI_IMPORT_ERROR_MESSAGE}")
            return  # 如果训练模块不可用，直接返回
        else:
            # 如果只是数据准备模块的导入问题，显示警告但继续
            st_instance.warning("[WARNING] 数据准备模块导入警告，但映射数据传递已修复，功能应该正常")

    current_training_status = _state.get('dfm_training_status')
    current_model_results = _state.get('dfm_model_results_paths')

    # 如果页面刚加载且存在之前的训练完成状态，询问用户是否要重置
    page_just_loaded = _state.get('dfm_page_initialized') is None

    if page_just_loaded:
        _state.set('dfm_page_initialized', True)

        # 如果存在之前的训练结果，显示提示并自动重置（避免混淆）
        if current_training_status == '训练完成' and current_model_results:
            st_instance.info("[LOADING] 检测到之前的训练结果，已自动重置训练状态以开始新的训练")
            _reset_training_state()
            current_training_status = '等待开始'

    # 使用状态管理器初始化DFM状态
    if current_training_status is None:
        _state.set('dfm_training_status', "等待开始")
    if _state.get('dfm_model_results') is None:
        _state.set('dfm_model_results', None)
    if _state.get('dfm_training_log') is None:
        _state.set('dfm_training_log', [])
    if _state.get('dfm_model_results_paths') is None:
        _state.set('dfm_model_results_paths', None)



    training_status = _state.get('dfm_training_status', '等待开始')

    training_results = _state.get('dfm_model_results_paths')
    if training_results and training_status != '训练完成':
        print(f"[状态修复] 检测到训练结果存在但状态未更新: {training_status} -> 训练完成")
        _state.set('dfm_training_status', '训练完成')
        training_status = '训练完成'

    from dashboard.core.ui.utils.debug_helpers import debug_log
    debug_log(f"UI状态检查 - 当前训练状态: {training_status}", "DEBUG")

    # 使用组件化的文件上传器
    state_manager = StateManager('train_model')
    file_uploader = FileUploaderComponent(state_manager)
    input_df, var_industry_map, dfm_default_map, var_frequency_map, var_unit_map = file_uploader.render(st_instance)

    # 如果文件验证失败，提前返回
    if input_df is None or not var_industry_map:
        return

    available_target_vars = []
    if input_df is not None:
        # 从已加载数据中获取可选的目标变量
        available_target_vars = [col for col in input_df.columns if 'date' not in col.lower() and 'time' not in col.lower() and col not in getattr(config, 'EXCLUDE_COLS_FROM_TARGET', [])]

        default_target = None
        if hasattr(config, 'TARGET_VARIABLE'):
            default_target = config.TARGET_VARIABLE
        else:
            default_target = '规模以上工业增加值:当月同比'  # 硬编码的默认值

        # 如果默认目标变量在数据中存在但不在过滤列表中，则添加它
        if default_target and default_target in input_df.columns and default_target not in available_target_vars:
            available_target_vars.insert(0, default_target)  # 插入到开头作为首选


        # 如果过滤后的列表为空，但默认目标变量存在，则使用它
        if not available_target_vars and default_target and default_target in input_df.columns:
            available_target_vars = [default_target]


        if not available_target_vars:
            st_instance.warning("预处理数据中未找到合适的目标变量候选。")
            # 即使没找到，也提供一个默认选项避免selectbox为空
            if default_target:
                available_target_vars = [default_target]
                st_instance.info(f"使用默认目标变量: {default_target}")


    else:

        st_instance.warning("数据尚未准备，请先在\"数据准备\"选项卡中处理数据。变量选择功能将受限。")

        default_target = None
        if hasattr(config, 'TARGET_VARIABLE'):
            default_target = config.TARGET_VARIABLE
        else:
            default_target = '规模以上工业增加值:当月同比'  # 硬编码的默认值

        if default_target:
            available_target_vars = [default_target]
            st_instance.info(f"使用默认目标变量: {default_target}")


    # 直接使用文件上传器加载的映射数据（已由FileUploaderComponent处理）
    # 完全解耦，不从其他模块的session_state读取数据

    # 构建行业到指标的映射
    industry_to_indicators_temp = defaultdict(list)
    if var_industry_map:
        for indicator, industry in var_industry_map.items():
            if indicator and industry:
                industry_to_indicators_temp[str(industry).strip()].append(str(indicator).strip())

    unique_industries = sorted(list(industry_to_indicators_temp.keys()))
    var_to_indicators_map_by_industry = {k: sorted(v) for k, v in industry_to_indicators_temp.items()}

    # 主布局：现在是上下结构，不再使用列
    # REMOVED: var_selection_col, param_col = st_instance.columns([1, 1.5])

    # 添加变量选择大标题
    st_instance.subheader("变量选择")

    # 1. 目标变量选择/识别
    st_instance.markdown("**目标变量**")

    # 获取当前估计方法
    current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')

    if current_estimation_method == 'single_stage':
        # 一次估计法：从映射文件自动识别目标变量
        # 目标变量来自'二阶段目标'列（总量指标）
        second_stage_target_map = _state.get('dfm_second_stage_target_map', {})

        # 将标准化的键转换回原始列名
        single_stage_targets = []

        if input_df is not None:
            for col in input_df.columns:
                col_norm = normalize_text(col)
                if col_norm in second_stage_target_map:
                    single_stage_targets.append(col)


        # 验证唯一性
        if len(single_stage_targets) == 0:
            st_instance.error("[ERROR] 映射文件'二阶段目标'列中未标记任何目标变量，一次估计法需要目标变量")
            st_instance.info("提示：一次估计法的目标变量来自'二阶段目标'列（总量指标），预测变量默认来自'一次估计'列")
            _state.set('dfm_target_variable', None)
        elif len(single_stage_targets) > 1:
            st_instance.error(f"[ERROR] 映射文件'二阶段目标'列中标记了多个目标变量，请确保只有一个目标变量标记为'是'")
            st_instance.warning(f"当前识别到的目标变量：{', '.join(single_stage_targets)}")
            _state.set('dfm_target_variable', None)
        else:
            # 唯一目标变量
            target_var = single_stage_targets[0]
            _state.set('dfm_target_variable', target_var)
            st_instance.info(f"已从映射文件自动识别目标变量：{target_var}")
    else:
        # 二次估计法：从映射文件自动识别目标变量
        first_stage_target_map = _state.get('dfm_first_stage_target_map', {})
        second_stage_target_map = _state.get('dfm_second_stage_target_map', {})

        # 将标准化的键转换回原始列名
        first_stage_targets = []
        second_stage_targets = []

        if input_df is not None:
            for col in input_df.columns:
                col_norm = normalize_text(col)
                if col_norm in first_stage_target_map:
                    first_stage_targets.append(col)
                if col_norm in second_stage_target_map:
                    second_stage_targets.append(col)


        # 显示识别到的目标变量
        if first_stage_targets or second_stage_targets:
            st_instance.info(f"已从映射文件自动识别目标变量")

            if first_stage_targets:
                with st_instance.expander("一阶段目标变量（各行业）", expanded=False):
                    for idx, target in enumerate(first_stage_targets, 1):
                        st_instance.text(f"{idx}. {target}")
                # 显示自动全选提示
                st_instance.success(f"已自动选择 {len(first_stage_targets)} 个一阶段目标变量用于训练")
            else:
                st_instance.warning("未识别到一阶段目标变量（请检查映射文件'一阶段目标'列）")

            if second_stage_targets:
                # 二阶段目标只有一个，直接显示不使用expander
                st_instance.info(f"二阶段目标变量（总量）：**{second_stage_targets[0]}**")
                # 显示自动选择提示
                st_instance.success(f"已自动选择二阶段目标变量用于最终预测")

                # 将二阶段目标变量保存到全局状态（用于后续训练逻辑）
                _state.set('dfm_target_variable', second_stage_targets[0] if second_stage_targets else None)
                _state.set('dfm_first_stage_target_variables', first_stage_targets)
                _state.set('dfm_second_stage_target_variable', second_stage_targets[0] if second_stage_targets else None)
            else:
                st_instance.error("未识别到二阶段目标变量（请检查映射文件'二阶段目标'列）")
                _state.set('dfm_target_variable', None)
                _state.set('dfm_second_stage_target_variable', None)
        else:
            st_instance.error("[ERROR] 映射文件中未标记任何目标变量，请检查'一阶段目标'和'二阶段目标'列")
            _state.set('dfm_target_variable', None)

    # 根据估计方法选择正确的预测变量默认映射
    if current_estimation_method == 'single_stage':
        # 一次估计法：使用'一次估计'列作为预测变量默认选择
        dfm_default_map = _state.get('dfm_default_single_stage_map', {})
    else:
        # 二次估计法：使用'一阶段预测'列作为预测变量默认选择
        dfm_default_map = _state.get('dfm_first_stage_pred_map', {})

    # 更新到状态，供后续使用
    _state.set('dfm_default_variables_map', dfm_default_map)

    # 2. 过滤行业：移除仅包含目标变量的行业（使用工具函数）
    current_target_var = _state.get('dfm_target_variable', None)
    filtered_industries = filter_industries_by_target(
        unique_industries,
        var_to_indicators_map_by_industry,
        current_target_var
    )

    # 暂时存储过滤后的行业，以供后续步骤3中使用
    if not filtered_industries:
        st_instance.info("没有可用的行业数据。")
        _state.set('dfm_selected_industries', [])
    else:
        pass  # 继续进行到步骤3，用户将直接选择指标

    # 3. 根据选定行业选择预测指标 (每个行业一个多选下拉菜单，默认全选)
    with st_instance.expander("选择预测指标", expanded=False):
        # 获取当前估计方法，用于状态键命名
        current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')
        indicators_state_key = f'dfm_selected_indicators_per_industry_{current_estimation_method}'

        # 初始化指标选择状态（按估计方法分别存储）
        if _state.get(indicators_state_key, None) is None:
            _state.set(indicators_state_key, {})

        final_selected_indicators_flat = []
        current_selected_industries = filtered_industries  # 直接使用过滤后的所有行业

        # 全局控制：预先计算每个行业的可用指标
        industry_available_indicators = {}
        if current_selected_industries:
            current_target_var = _state.get('dfm_target_variable', None)
            first_stage_targets = _state.get('dfm_first_stage_target_variables', [])

            # 构建排除列表
            exclude_targets = []
            if current_target_var:
                exclude_targets.append(current_target_var)
            if first_stage_targets:
                exclude_targets.extend(first_stage_targets)

            for industry_name in current_selected_industries:
                all_indicators_for_industry = var_to_indicators_map_by_industry.get(industry_name, [])

                # 排除目标变量
                if exclude_targets:
                    indicators_for_this_industry = [
                        indicator for indicator in all_indicators_for_industry
                        if indicator not in exclude_targets
                    ]
                else:
                    indicators_for_this_industry = all_indicators_for_industry

                # 二次估计法：排除"综合"类变量
                if current_estimation_method == 'two_stage':
                    var_industry_map = _state.get('dfm_industry_map_filtered', None)
                    if var_industry_map is None:
                        var_industry_map = _state.get('dfm_industry_map_obj', {})

                    if var_industry_map:
                        indicators_for_this_industry = [
                            indicator for indicator in indicators_for_this_industry
                            if var_industry_map.get(
                                normalize_variable_name(indicator),
                                None
                            ) != '综合'
                        ]

                # 只保存有可用指标的行业
                if indicators_for_this_industry:
                    industry_available_indicators[industry_name] = indicators_for_this_industry

        # 添加全局控制按钮
        if current_selected_industries and industry_available_indicators:
            st_instance.markdown("**全局控制**")
            global_control_col1, global_control_col2 = st_instance.columns(2)

            with global_control_col1:
                global_select_all = st_instance.button(
                    "全选所有指标",
                    key=f"dfm_global_select_all_{current_estimation_method}",
                    use_container_width=True
                )

            with global_control_col2:
                global_deselect_all = st_instance.button(
                    "取消全选所有指标",
                    key=f"dfm_global_deselect_all_{current_estimation_method}",
                    use_container_width=True
                )

            # 处理全局全选
            if global_select_all:
                for industry_name, available_indicators in industry_available_indicators.items():
                    multiselect_key = f"dfm_indicators_multiselect_{industry_name}_{current_estimation_method}"
                    select_all_key = f"dfm_select_all_{industry_name}_{current_estimation_method}"

                    # 更新multiselect状态
                    st.session_state[multiselect_key] = available_indicators
                    # 删除checkbox状态，让它在下次渲染时重新初始化
                    if select_all_key in st.session_state:
                        del st.session_state[select_all_key]

                # 刷新页面
                st_instance.rerun()

            # 处理全局取消全选
            if global_deselect_all:
                for industry_name in industry_available_indicators.keys():
                    multiselect_key = f"dfm_indicators_multiselect_{industry_name}_{current_estimation_method}"
                    select_all_key = f"dfm_select_all_{industry_name}_{current_estimation_method}"

                    # 清空multiselect状态
                    st.session_state[multiselect_key] = []
                    # 删除checkbox状态，让它在下次渲染时重新初始化
                    if select_all_key in st.session_state:
                        del st.session_state[select_all_key]

                # 刷新页面
                st_instance.rerun()

            st_instance.markdown("---")

        if not current_selected_industries:
            st_instance.info("没有可用的行业数据。")
        else:
            current_selection = _state.get(indicators_state_key, {})

            num_cols = 3
            cols = st_instance.columns(num_cols)
            col_idx = 0

            for industry_name in current_selected_industries:
                all_indicators_for_industry = var_to_indicators_map_by_industry.get(industry_name, [])

                # 修复：排除目标变量，确保用户无法选择目标变量作为预测变量
                current_target_var = _state.get('dfm_target_variable', None)
                first_stage_targets = _state.get('dfm_first_stage_target_variables', [])
                current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')

                # 构建排除列表：包括全局目标变量和一阶段目标变量
                exclude_targets = []
                if current_target_var:
                    exclude_targets.append(current_target_var)
                if first_stage_targets:
                    exclude_targets.extend(first_stage_targets)

                # 排除所有目标变量
                if exclude_targets:
                    indicators_for_this_industry = [
                        indicator for indicator in all_indicators_for_industry
                        if indicator not in exclude_targets
                    ]
                else:
                    indicators_for_this_industry = all_indicators_for_industry

                # 二次估计法：第一阶段排除"综合"类变量（综合变量应该在第二阶段使用）
                if current_estimation_method == 'two_stage':
                    var_industry_map = _state.get('dfm_industry_map_filtered', None)
                    if var_industry_map is None:
                        var_industry_map = _state.get('dfm_industry_map_obj', {})

                    if var_industry_map:
                        indicators_before_filter = indicators_for_this_industry.copy()
                        indicators_for_this_industry = [
                            indicator for indicator in indicators_for_this_industry
                            if var_industry_map.get(
                                normalize_variable_name(indicator),
                                None
                            ) != '综合'
                        ]
                        excluded_general_count = len(indicators_before_filter) - len(indicators_for_this_industry)

                # 修复：完全跳过没有可用指标的行业，不显示任何内容
                if not indicators_for_this_industry:
                    current_selection[industry_name] = []
                    col_idx += 1
                    continue

                with cols[col_idx % num_cols]:
                    # 只有在有指标被排除且仍有可用指标时才显示提示
                    excluded_count = len(all_indicators_for_industry) - len(indicators_for_this_industry)

                    # 从状态读取已设置的默认变量映射（在目标变量识别后已根据估计方法设置）
                    dfm_default_map = _state.get('dfm_default_variables_map', {})
                    current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')
                    method_label = "一阶段预测" if current_estimation_method == 'two_stage' else "一次估计"


                    # 从状态管理器读取已选指标，或使用DFM默认选择
                    default_selection_for_industry = current_selection.get(industry_name, None)

                    # 如果状态管理器中没有选择，使用DFM变量列配置
                    if default_selection_for_industry is None:
                        dfm_default_indicators = [
                            indicator for indicator in indicators_for_this_industry
                            if normalize_text(indicator) in dfm_default_map
                        ]
                        default_selection_for_industry = dfm_default_indicators

                    # 确保默认值是实际可选列表的子集
                    valid_default = [item for item in default_selection_for_industry if item in indicators_for_this_industry]

                    # 初始化multiselect（key中包含估计方法，确保切换方法时重新初始化）
                    multiselect_key = f"dfm_indicators_multiselect_{industry_name}_{current_estimation_method}"
                    if multiselect_key not in st.session_state:
                        st.session_state[multiselect_key] = valid_default

                    # 判断是否应该勾选全选checkbox（根据当前multiselect的实际值判断）
                    current_multiselect_value = st.session_state.get(multiselect_key, [])
                    should_check_select_all = (
                        len(current_multiselect_value) > 0 and
                        set(current_multiselect_value) == set(indicators_for_this_industry)
                    )

                    # 行业名称与全选checkbox同行
                    header_col1, header_col2 = st_instance.columns([3, 1])
                    with header_col1:
                        st_instance.markdown(f"**{industry_name}**")
                    with header_col2:
                        # 创建checkbox（使用计算出的状态，不使用key避免状态冲突）
                        select_all_checked = st_instance.checkbox(
                            "全选",
                            value=should_check_select_all,
                            key=f"dfm_select_all_{industry_name}_{current_estimation_method}"
                        )

                    if excluded_count > 0:
                        st_instance.caption(f"排除目标变量: {excluded_count}个")

                    # 同步checkbox与multiselect
                    if select_all_checked:
                        st.session_state[multiselect_key] = indicators_for_this_industry
                    else:
                        if st.session_state[multiselect_key] == indicators_for_this_industry:
                            st.session_state[multiselect_key] = valid_default

                    selected_in_widget = st_instance.multiselect(
                        "选择指标",
                        options=indicators_for_this_industry,
                        key=multiselect_key,
                        label_visibility="collapsed"
                    )

                    current_selection[industry_name] = selected_in_widget
                    final_selected_indicators_flat.extend(selected_in_widget)

                col_idx += 1

            industries_to_remove_from_state = [
                ind for ind in current_selection
                if ind not in current_selected_industries
            ]
            for ind_to_remove in industries_to_remove_from_state:
                del current_selection[ind_to_remove]

            _state.set(indicators_state_key, current_selection)

        # 修复：如果循环没有执行（filtered_industries为空），但状态中有数据
        # 说明是旧数据，应该从状态重建指标列表
        if len(final_selected_indicators_flat) == 0 and len(current_selected_industries) == 0:
            saved_selection = _state.get(indicators_state_key, {})
            if saved_selection:
                for industry, indicators in saved_selection.items():
                    final_selected_indicators_flat.extend(indicators)

        # 更新最终的扁平化预测指标列表 (去重)
        final_indicators = sorted(list(set(final_selected_indicators_flat)))
        _state.set('dfm_selected_indicators', final_indicators)


        # 从选择的指标自动推断实际使用的行业（只有当该行业有指标被选中时）
        inferred_industries = []
        selected_indicators_per_industry = _state.get(indicators_state_key, {})
        for industry, indicators in selected_indicators_per_industry.items():
            if indicators and len(indicators) > 0:  # 如果该行业有选中的指标
                inferred_industries.append(industry)

        _state.set('dfm_selected_industries', inferred_industries)

    # 变量选择完成

    # 显示变量选择汇总信息
    current_target_var = _state.get('dfm_target_variable', None)
    current_selected_indicators = _state.get('dfm_selected_indicators', [])
    current_selected_industries_for_display = _state.get('dfm_selected_industries', [])
    current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')

    # 计算总预测变量数（包含第二阶段额外变量）
    total_predictor_count = len(current_selected_indicators)
    if current_estimation_method == 'two_stage':
        second_stage_extra_predictors = _state.get('dfm_second_stage_extra_predictors', [])
        total_predictor_count += len(second_stage_extra_predictors)

    st_instance.text(f" - 目标变量: {current_target_var if current_target_var else '未选择'}")
    st_instance.text(f" - 选定行业数: {len(current_selected_industries_for_display)}")
    st_instance.text(f" - 选定预测指标总数: {total_predictor_count}")

    st_instance.markdown("--- ")
    st_instance.subheader("模型参数")

    # 创建三列布局
    col1_time, col2_factor_core, col3_factor_specific = st_instance.columns(3)

    with col1_time:


        # 计算基于数据的智能默认值
        def get_data_based_date_defaults():
            """基于实际数据计算日期默认值，优先使用数据准备页面设置的日期边界"""
            from datetime import datetime, timedelta
            today = datetime.now().date()

            data_prep_start = _state.get('dfm_param_data_start_date')
            data_prep_end = _state.get('dfm_param_data_end_date')

            # 使用统一配置类
            static_defaults = UIConfig.get_date_defaults()

            try:
                # 仅从train_model模块获取数据（完全解耦）
                data_df = _state.get('dfm_prepared_data_df')

                if data_df is not None and isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                    # 从数据获取第一期和最后一期
                    data_first_date = data_df.index.min().date()  # 第一期数据
                    data_last_date = data_df.index.max().date()   # 最后一期数据

                    # 重要：确保数据的最后日期不是未来日期
                    if data_last_date > today:
                        print(f"[WARNING] 警告: 数据包含未来日期 {data_last_date}，将使用今天作为最后日期")
                        data_last_date = today

                    # 训练开始日期默认为数据的第一期
                    training_start_date = data_first_date
                    validation_start_date = UIConfig.DEFAULT_VALIDATION_START
                    validation_end_date = UIConfig.DEFAULT_VALIDATION_END

                    return {
                        'training_start': training_start_date,       # 训练开始日：默认为数据的开始日期
                        'validation_start': validation_start_date,   # 验证开始日：使用配置默认值
                        'validation_end': validation_end_date        # 验证结束日：使用配置默认值
                    }
                else:
                    return static_defaults
            except Exception as e:
                print(f"[WARNING] 计算数据默认日期失败: {e}，使用静态默认值")
                return static_defaults

        # 获取智能默认值
        date_defaults = get_data_based_date_defaults()

        has_data = False
        # 仅从train_model模块获取数据（完全解耦）
        data_df = _state.get('dfm_prepared_data_df')
        if data_df is not None:
            has_data = True

        if has_data:
            if isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                # 计算数据的实际日期范围用于比较
                actual_data_start = data_df.index.min().date()
                actual_data_end = data_df.index.max().date()

                # 初始化默认日期（只在状态为空时设置）
                current_training_start = _state.get('dfm_training_start_date')
                if current_training_start is None:
                    _state.set('dfm_training_start_date', date_defaults['training_start'])

                current_validation_start = _state.get('dfm_validation_start_date')
                if current_validation_start is None:
                    _state.set('dfm_validation_start_date', date_defaults['validation_start'])

                current_validation_end = _state.get('dfm_validation_end_date')
                if current_validation_end is None:
                    _state.set('dfm_validation_end_date', date_defaults['validation_end'])

                # 简化数据范围信息
                data_start = data_df.index.min().strftime('%Y-%m-%d')
                data_end = data_df.index.max().strftime('%Y-%m-%d')
                data_count = len(data_df.index)
                # st_instance.info(f"[DATA] 数据: {data_start} 至 {data_end} ({data_count}点)")
        # 1. 训练期开始日期
        training_start_value = st_instance.date_input(
            "训练期开始日期 (Training Start Date)",
            value=_state.get('dfm_training_start_date', date_defaults['training_start']),
            key='dfm_training_start_date_input',
            help="选择模型训练数据的起始日期。默认为上传数据的开始日期。"
        )
        _state.set('dfm_training_start_date', training_start_value)

        # 2. 验证期开始日期
        validation_start_value = st_instance.date_input(
            "验证期开始日期 (Validation Start Date)",
            value=_state.get('dfm_validation_start_date', date_defaults['validation_start']),
            key='dfm_validation_start_date_input',
            help="选择验证期开始日期。默认为最后一期数据前3个月。"
        )
        _state.set('dfm_validation_start_date', validation_start_value)

        # 3. 验证期结束日期
        validation_end_value = st_instance.date_input(
            "验证期结束日期 (Validation End Date)",
            value=_state.get('dfm_validation_end_date', date_defaults['validation_end']),
            key='dfm_validation_end_date_input',
            help="选择验证期结束日期。默认为数据的最后一期。"
        )
        _state.set('dfm_validation_end_date', validation_end_value)

    with col2_factor_core:
        # 估计方法选择
        estimation_methods = {
            'single_stage': '一次估计法',
            'two_stage': '二次估计法'
        }

        # 定义回调函数：切换估计方法时同步状态
        # 注意：回调执行后Streamlit会自动重新运行脚本，无需手动调用rerun()
        def on_estimation_method_change():
            # 从临时key读取新值并同步到dfm状态
            new_method = st_instance.session_state.get('temp_estimation_method_selector', 'single_stage')
            _state.set('dfm_estimation_method', new_method)

        # 获取当前选择的估计方法（用于确定默认index）
        current_method = _state.get('dfm_estimation_method', 'single_stage')
        current_index = 0 if current_method == 'single_stage' else 1

        estimation_method = st_instance.selectbox(
            "估计方法",
            options=list(estimation_methods.keys()),
            format_func=lambda x: estimation_methods[x],
            index=current_index,
            key='temp_estimation_method_selector',  # 使用临时key
            on_change=on_estimation_method_change,  # 回调函数处理状态同步和重载
            help="一次估计法：直接预测总量指标；二次估计法：先预测各行业，再加总"
        )

        # 手动同步状态（确保即使回调未触发也能保持一致）
        _state.set('dfm_estimation_method', estimation_method)

        variable_selection_options = {
            'none': "无筛选 (使用全部已选变量)",
            'global_backward': "全局后向剔除 (在已选变量中筛选)"
        }
        default_var_method = 'global_backward'

        # 获取当前变量选择方法
        current_var_method = _state.get('dfm_variable_selection_method', default_var_method)

        var_method_value = st_instance.selectbox(
            "变量选择方法",
            options=list(variable_selection_options.keys()),
            format_func=lambda x: variable_selection_options[x],
            index=list(variable_selection_options.keys()).index(current_var_method),
            key='dfm_variable_selection_method_input',
            help=(
                "选择在已选变量基础上的筛选方法：\n"
                "- 无筛选: 直接使用所有已选择的变量\n"
                "- 全局后向剔除: 从已选变量开始，逐个剔除不重要的变量"
            )
        )
        _state.set('dfm_variable_selection_method', var_method_value)

        enable_var_selection = (var_method_value != 'none')
        _state.set('dfm_enable_variable_selection', enable_var_selection)

        # 目标变量配对模式（2025-12新增）
        target_alignment_options = {
            'next_month': "下月值 (m月nowcast预测m+1月目标)",
            'current_month': "本月值 (m月nowcast预测m月目标)"
        }
        default_alignment = 'next_month'

        current_alignment = _state.get('dfm_target_alignment_mode', default_alignment)

        alignment_value = st_instance.selectbox(
            "目标变量配对方式",
            options=list(target_alignment_options.keys()),
            format_func=lambda x: target_alignment_options[x],
            index=list(target_alignment_options.keys()).index(current_alignment),
            key='dfm_target_alignment_mode_input',
            help=(
                "选择nowcast预测值与目标变量实际值的配对方式：\n"
                "- 下月值: m月的nowcast与m+1月的target配对（默认）\n"
                "- 本月值: m月的nowcast与m月的target配对"
            )
        )
        _state.set('dfm_target_alignment_mode', alignment_value)

        # 后向剔除基于性能比较（HR和RMSE），不使用统计显著性阈值

    with col3_factor_specific:
        # 因子选择策略
        factor_strategy_options = {
            'fixed_number': "固定因子数",
            'cumulative_variance': "累积方差贡献",
            'kaiser': "Kaiser准则(特征值>1)"
        }
        default_strategy = 'fixed_number'

        current_strategy = _state.get('dfm_factor_selection_strategy', default_strategy)

        strategy_value = st_instance.selectbox(
            "因子选择策略",
            options=list(factor_strategy_options.keys()),
            format_func=lambda x: factor_strategy_options[x],
            index=list(factor_strategy_options.keys()).index(current_strategy),
            key='dfm_factor_selection_strategy',
            help="选择确定因子数量的方法"
        )
        
        # [CRITICAL FIX] 添加策略选择的状态保存
        _state.set('dfm_factor_selection_strategy', strategy_value)

        # 根据策略显示相应参数
        if strategy_value == 'fixed_number':
            # 固定因子数
            default_fixed_factors = 4

            # 获取当前估计方法以调整help文本
            current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')
            if current_estimation_method == 'two_stage':
                k_factors_help = "第二阶段总量模型使用的因子数量（第一阶段各行业因子数在下方单独设置）"
            else:
                k_factors_help = "指定使用的因子数量"

            fixed_factors_value = st_instance.number_input(
                "固定因子数",
                min_value=1,
                max_value=15,
                value=_state.get('dfm_fixed_number_of_factors', default_fixed_factors),
                step=1,
                key='dfm_fixed_number_of_factors',
                help=k_factors_help
            )
            # [CRITICAL FIX] 添加缺失的状态保存
            _state.set('dfm_fixed_number_of_factors', fixed_factors_value)

        elif strategy_value == 'cumulative_variance':
            # 累积方差贡献
            default_cum_var = 0.8

            cum_var_value = st_instance.number_input(
                "累积方差贡献阈值",
                min_value=0.5,
                max_value=0.99,
                value=_state.get('dfm_cumulative_variance_threshold', default_cum_var),
                step=0.01,
                format="%.2f",
                key='dfm_cumulative_variance_threshold_input',
                help="因子累积解释方差的阈值"
            )
            _state.set('dfm_cumulative_variance_threshold', cum_var_value)

        elif strategy_value == 'kaiser':
            # Kaiser准则特征值阈值
            default_kaiser_threshold = 1.0

            kaiser_threshold_value = st_instance.number_input(
                "特征值阈值",
                min_value=0.5,
                max_value=2.0,
                value=_state.get('dfm_kaiser_threshold', default_kaiser_threshold),
                step=0.1,
                format="%.1f",
                key='dfm_kaiser_threshold_input',
                help="选择特征值大于此阈值的因子（经典Kaiser准则使用1.0）"
            )
            _state.set('dfm_kaiser_threshold', kaiser_threshold_value)

        # 因子自回归阶数
        default_ar_order = 1

        ar_order_value = st_instance.number_input(
            "因子自回归阶数",
            min_value=0,
            max_value=5,
            value=_state.get('dfm_factor_ar_order', default_ar_order),
            step=1,
            key='dfm_factor_ar_order_input',
            help="因子的自回归阶数，通常设为1"
        )
        _state.set('dfm_factor_ar_order', ar_order_value)

    # 二次估计法配置
    if estimation_method == 'two_stage':
        st_instance.info("二次估计法说明：上方'固定因子数'为第二阶段总量模型的因子数，下方设置各行业模型的因子数")
        with st_instance.expander("第一阶段：分行业因子数设置", expanded=False):
            # 使用训练模块上传的数据（input_df从第614行获得）
            if input_df is not None:
                # 提取行业列表
                from dashboard.models.DFM.train.utils.industry_data_processor import extract_industry_list

                industry_list = extract_industry_list(input_df)

                if industry_list:
                    st_instance.info(f"共识别到 {len(industry_list)} 个行业，请设定各行业因子数（默认值为1）")

                    # 使用列布局优化显示（每行3个）
                    cols_per_row = 3
                    industry_k_factors = {}

                    for i in range(0, len(industry_list), cols_per_row):
                        cols = st_instance.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(industry_list):
                                industry = industry_list[idx]
                                with col:
                                    k_val = st_instance.number_input(
                                        f"{industry}",
                                        min_value=1,
                                        max_value=5,
                                        value=_state.get(f'industry_k_{industry}', 1),
                                        step=1,
                                        key=f'industry_k_{industry}'
                                    )
                                    industry_k_factors[industry] = k_val

                    # 存储到session_state
                    _state.set('dfm_industry_k_factors', industry_k_factors)
                else:
                    st_instance.warning("未能从数据中识别到任何行业信息，请检查数据格式")

            else:
                st_instance.warning("请先上传预处理后的数据文件")

        # 第二阶段额外指标选择
        with st_instance.expander("第二阶段：变量选择", expanded=False):
            if input_df is not None:
                # 获取当前估计方法
                current_estimation_method = _state.get('dfm_estimation_method', 'single_stage')

                # 获取需要排除的变量
                target_variable = _state.get('dfm_target_variable', None)
                first_stage_selected_indicators = _state.get('dfm_selected_indicators', [])
                var_industry_map = _state.get('dfm_industry_map_filtered', None)
                if var_industry_map is None:
                    var_industry_map = _state.get('dfm_industry_map_obj', {})

                # 构建排除集合（使用标准化名称）
                excluded_vars = set()
                excluded_vars_normalized = set()  # 标准化后的排除集合

                # 排除第一阶段已选择的所有预测变量（关键修复：使用标准化名称）
                if first_stage_selected_indicators:
                    excluded_vars.update(first_stage_selected_indicators)
                    # 构建标准化版本的排除集合（去除所有空格）
                    for var in first_stage_selected_indicators:
                        normalized_var = normalize_variable_name_no_space(var)
                        excluded_vars_normalized.add(normalized_var)

                # 遍历所有列，根据映射关系判断是否排除
                for col in input_df.columns:
                    # 标准化列名（用于映射查找）
                    normalized_col = normalize_variable_name(col)
                    # 标准化列名（去除所有空格，用于第一阶段变量匹配）
                    normalized_col_no_space = normalize_variable_name_no_space(col)

                    # 检查变量的行业归属
                    industry = None
                    if var_industry_map and normalized_col in var_industry_map:
                        industry = var_industry_map[normalized_col]

                    # 排除逻辑：
                    # 1. 排除所有非"综合"的行业级变量（无论是否在第一阶段使用）
                    if industry and industry != '综合':
                        excluded_vars.add(col)
                        continue

                    # 2. 如果是第一阶段选择的变量
                    if normalized_col_no_space in excluded_vars_normalized:
                        # 如果是"综合"类变量，保留（不排除）
                        if industry == '综合':
                            continue
                        # 否则排除
                        else:
                            excluded_vars.add(col)
                            continue

                    # 3. 排除包含"工业增加值"的变量（未映射的行业目标变量）
                    if '工业增加值' in col:
                        excluded_vars.add(col)

                # 排除第二阶段目标变量
                if target_variable:
                    excluded_vars.add(target_variable)

                # 筛选可用的额外预测变量
                available_extra_vars = [
                    col for col in input_df.columns
                    if col not in excluded_vars
                ]

                # 获取二阶段目标的默认变量
                dfm_second_stage_target_map = _state.get('dfm_second_stage_target_map', {})

                # 计算默认选中的变量
                default_extra_vars = [
                    col for col in available_extra_vars
                    if normalize_variable_name(col) in dfm_second_stage_target_map
                ]

                # multiselect key包含估计方法，确保切换时重新初始化
                stage2_key = f'dfm_second_stage_extra_predictors_{current_estimation_method}'

                # 初始化默认值
                if stage2_key not in st.session_state:
                    st.session_state[stage2_key] = default_extra_vars

                extra_predictors = st_instance.multiselect(
                    "第二阶段额外预测变量（可选）",
                    options=available_extra_vars,
                    key=stage2_key,
                    help="在分行业nowcasting之外添加的宏观指标"
                )

                _state.set('dfm_second_stage_extra_predictors', extra_predictors)

    # 第四行：开始训练按钮（左对齐）
    # 重新获取变量选择状态（用于训练条件检查）
    current_target_var = _state.get('dfm_target_variable', None)
    current_selected_indicators = _state.get('dfm_selected_indicators', [])

    # 日期验证
    training_start_value = _state.get('dfm_training_start_date')
    validation_start_value = _state.get('dfm_validation_start_date')
    validation_end_value = _state.get('dfm_validation_end_date')

    date_validation_passed = True
    if training_start_value and validation_start_value and validation_end_value:
        if training_start_value >= validation_start_value:
            st_instance.error("[ERROR] 训练期开始日期必须早于验证期开始日期")
            date_validation_passed = False
        elif validation_start_value >= validation_end_value:
            st_instance.error("[ERROR] 验证期开始日期必须早于验证期结束日期")
            date_validation_passed = False
        else:
            pass
    else:
        st_instance.warning("[WARNING] 请设置完整的日期范围")
        date_validation_passed = False

    # 检查训练准备状态
    training_ready = (
        current_target_var is not None and
        len(current_selected_indicators) > 0 and
        date_validation_passed and
        input_df is not None
    )

    if not training_ready:
        st_instance.warning("[WARNING] 训练条件未满足，请检查上述设置")

    # 开始训练按钮
    if training_ready:
        train_btn_clicked = st_instance.button("开始训练",
                            key="dfm_start_training",
                            help="开始DFM模型训练",
                            type="primary")
    else:
        train_btn_clicked = st_instance.button("开始训练",
                         disabled=True,
                         key="dfm_start_training_disabled",
                         help="请先满足所有训练条件",
                         type="primary")

    # 训练逻辑
    if training_ready and train_btn_clicked:
        current_status = _state.get('dfm_training_status', '等待开始')
        if current_status in ['正在训练...', '准备启动训练...']:
            st_instance.warning("[WARNING] 训练已在进行中，请勿重复启动")
        else:
            try:
                # 使用TrainingConfigBuilder构建配置
                state_manager = StateManager('train_model')
                config_builder = TrainingConfigBuilder(state_manager)

                training_config = config_builder.build(
                    input_df=input_df,
                    var_industry_map=_state.get('dfm_industry_map_obj', {})
                )

                print(f"[TRAIN_CONFIG] 因子选择策略: {training_config.factor_selection_method}")
                print(f"[TRAIN_CONFIG] 最大迭代次数: {training_config.max_iterations}")
                print(f"[TRAIN_CONFIG] AR阶数: {training_config.max_lags}")
                print(f"[TRAIN_CONFIG] 训练期: {training_config.training_start} 至 {training_config.train_end}")
                print(f"[TRAIN_CONFIG] 验证期: {training_config.validation_start} 至 {training_config.validation_end}")
                print(f"[TRAIN_CONFIG] 选择的指标数: {len(training_config.selected_indicators)}")
                print(f"[TRAIN_CONFIG] 变量选择: {training_config.enable_variable_selection}")

                # 创建进度回调函数
                def progress_callback(message: str):
                    """进度回调函数"""
                    print(f"[TRAIN_REF] {message}")
                    # 更新训练日志
                    training_log = _state.get('dfm_training_log', [])
                    training_log.append(message)
                    _state.set('dfm_training_log', training_log)

                # 设置训练状态
                _state.set('dfm_training_status', '正在训练...')
                _state.set('dfm_training_log', ['[TRAIN_REF] 开始训练...'])

                # 根据估计方法选择训练器并训练（同步执行）
                if estimation_method == 'single_stage':
                    trainer = DFMTrainer(training_config)
                    result = trainer.train(
                        progress_callback=progress_callback,
                        enable_export=True,
                        export_dir=None
                    )
                else:  # two_stage
                    from dashboard.models.DFM.train.training.two_stage_trainer import TwoStageTrainer
                    st_instance.info("[LOADING] 正在训练模型（二次估计法），请稍候...")
                    trainer = TwoStageTrainer(training_config)
                    result = trainer.train(
                        progress_callback=progress_callback,
                        enable_export=True,
                        export_dir=None
                    )

                # 处理训练结果并保存
                from dashboard.models.DFM.train.core.models import TwoStageTrainingResult

                # 判断结果类型并提取关键信息
                if isinstance(result, TwoStageTrainingResult):
                    # 二次估计法结果：从second_stage_result中提取
                    final_result = result.second_stage_result
                    result_summary = {
                        'estimation_method': 'two_stage',
                        'selected_variables': final_result.selected_variables,
                        'k_factors': final_result.k_factors,
                        'metrics': {
                            'is_rmse': final_result.metrics.is_rmse if final_result.metrics else None,
                            'oos_rmse': final_result.metrics.oos_rmse if final_result.metrics else None,
                            'is_hit_rate': final_result.metrics.is_hit_rate if final_result.metrics else None,
                            'oos_hit_rate': final_result.metrics.oos_hit_rate if final_result.metrics else None
                        },
                        'training_time': result.total_training_time,
                        'first_stage_count': len(result.first_stage_results),
                        'industry_k_factors': result.industry_k_factors_used
                    }

                    training_time_display = result.total_training_time
                    selected_variables = final_result.selected_variables
                    k_factors_display = final_result.k_factors
                    metrics_obj = final_result.metrics

                else:
                    # 一次估计法结果
                    result_summary = {
                        'estimation_method': 'single_stage',
                        'selected_variables': result.selected_variables,
                        'k_factors': result.k_factors,
                        'metrics': {
                            'is_rmse': result.metrics.is_rmse if result.metrics else None,
                            'oos_rmse': result.metrics.oos_rmse if result.metrics else None,
                            'is_hit_rate': result.metrics.is_hit_rate if result.metrics else None,
                            'oos_hit_rate': result.metrics.oos_hit_rate if result.metrics else None
                        },
                        'training_time': result.training_time
                    }

                    training_time_display = result.training_time
                    selected_variables = result.selected_variables
                    k_factors_display = result.k_factors
                    metrics_obj = result.metrics

                # 保存到状态管理器
                _state.set('dfm_training_result', result_summary)
                _state.set('dfm_model_results_paths', result.export_files)
                _state.set('dfm_training_status', '训练完成')
                _state.set('dfm_training_completed_timestamp', time.time())

                # 添加完成日志
                training_log = _state.get('dfm_training_log', [])

                if isinstance(result, TwoStageTrainingResult):
                    training_log.append(f"[SUCCESS] 二次估计法训练完成！总耗时: {training_time_display:.2f}秒")
                    training_log.append(f"[STAGE1] 成功训练 {len(result.first_stage_results)} 个行业模型")
                    training_log.append(f"[STAGE2] 总量模型训练完成")

                training_log.append(f"[RESULT] 选中变量数: {len(selected_variables)}")
                training_log.append(f"[RESULT] 因子数: {k_factors_display}")

                if metrics_obj:
                    training_log.append(f"[METRICS] 样本外RMSE: {metrics_obj.oos_rmse:.4f}")
                    # 检查Hit Rate是否有效
                    import numpy as np
                    hit_rate_value = metrics_obj.oos_hit_rate
                    if np.isnan(hit_rate_value) or np.isinf(hit_rate_value):
                        hit_rate_str = "N/A (数据不足)"
                    else:
                        hit_rate_str = f"{hit_rate_value:.2f}%"
                    training_log.append(f"[METRICS] 样本外Hit Rate: {hit_rate_str}")

                _state.set('dfm_training_log', training_log)

                st_instance.success("[SUCCESS] 训练完成！")

            except Exception as e:
                import traceback
                error_msg = f"启动训练失败: {str(e)}\n{traceback.format_exc()}"
                print(f"[ERROR] {error_msg}")
                _state.set('dfm_training_status', f'训练失败: {str(e)}')
                _state.set('dfm_training_error', error_msg)
                st_instance.error(f"[ERROR] {error_msg}")

    st_instance.markdown("---")

    # 训练日志
    st_instance.markdown("**训练日志**")

    training_log = _state.get('dfm_training_log', [])
    current_training_status = _state.get('dfm_training_status', '等待开始')

    if current_training_status == '正在训练...':
        if training_log:
            log_text = "\n".join(training_log[-20:])
            st_instance.text_area(
                "训练日志",
                value=log_text,
                height=300,
                key="dfm_training_log_display",
                help="显示最近20条训练日志",
                label_visibility="hidden"
            )
            st_instance.info("[LOADING] 训练正在进行中，日志实时更新...")
        else:
            st_instance.info("[LOADING] 训练正在启动，请稍候...")
    elif training_log:
        log_text = "\n".join(training_log[-20:])
        st_instance.text_area(
            "训练日志",
            value=log_text,
            height=300,
            key="dfm_training_log_display",
            help="显示最近20条训练日志",
            label_visibility="hidden"
        )
    else:
        st_instance.info("[NONE] 无日志")

    # 文件下载按钮
    training_status = _state.get('dfm_training_status') or '等待开始'
    training_results = _state.get('dfm_model_results_paths')

    from dashboard.core.ui.utils.debug_helpers import debug_log
    debug_log(f"UI状态检查 - 当前训练状态: {training_status}", "DEBUG")
    debug_log(f"UI状态检查 - 结果文件数量: {len(training_results) if training_results else 0}", "DEBUG")

    if training_status == '训练完成':
        print(f"[UI状态检查] 检测到训练完成状态")
        debug_training_state("训练完成，显示最终结果", show_in_ui=False)

        if training_results:
            print(f"[UI状态检查] 开始处理训练结果，类型: {type(training_results)}")
            print(f"[UI状态检查] 训练结果内容: {training_results}")

            if isinstance(training_results, dict) and training_results:
                print(f"[UI状态检查] 处理字典格式结果，包含 {len(training_results)} 个条目")

                target_files = ['final_model_joblib', 'metadata']
                available_files = []

                for file_key in target_files:
                    file_path = training_results.get(file_key)
                    if file_path and os.path.exists(file_path):
                        file_name = os.path.basename(file_path)
                        available_files.append((file_key, file_path, file_name))
                        print(f"[UI状态检查] 文件存在: {file_name}")
                    else:
                        print(f"[UI状态检查] 文件不存在或路径为空: {file_key}")

                if available_files:
                    # 创建ZIP压缩包
                    import zipfile
                    import io
                    from datetime import datetime

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_key, file_path, file_name in available_files:
                            zip_file.write(file_path, file_name)

                    zip_buffer.seek(0)
                    zip_data = zip_buffer.getvalue()

                    # 生成压缩包文件名（包含时间戳）
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    zip_filename = f"dfm_model_{timestamp}.zip"

                    # 单一下载按钮
                    st_instance.download_button(
                        label="文件下载",
                        data=zip_data,
                        file_name=zip_filename,
                        mime="application/zip",
                        key="dfm_download_zip",
                        type="primary"
                    )
                else:
                    st_instance.warning("[WARNING] 未找到可用的结果文件")
            else:
                st_instance.warning("[WARNING] 训练完成但未找到结果文件")

    elif training_status.startswith('训练失败'):
        training_error = _state.get('dfm_training_error')
        st_instance.error(f"[ERROR] {training_status}")
        if training_error:
            st_instance.error(f"错误详情: {training_error}")


def _get_file_size(file_path: str) -> str:
    """获取文件大小的可读格式"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        return "未知大小"
