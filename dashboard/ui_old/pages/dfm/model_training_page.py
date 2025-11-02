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
import unicodedata
from datetime import datetime, timedelta, date, time
from collections import defaultdict
import traceback
import threading
from typing import Dict, List, Optional, Union, Any

# 添加路径以导入状态管理辅助函数
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

import logging

# 导入文本标准化工具
from dashboard.models.DFM.prep.utils.text_utils import normalize_text

# 导入组件化训练状态管理
from dashboard.ui.components.dfm.train_model.training_status import TrainingStatusComponent

# 配置日志记录器
logger = logging.getLogger(__name__)

_training_status_component = None

def get_training_status_component():
    """获取训练状态组件实例（单例模式）"""
    global _training_status_component
    if _training_status_component is None:
        _training_status_component = TrainingStatusComponent()
    return _training_status_component

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
        training_status = get_dfm_state('dfm_training_status')
        training_results = get_dfm_state('dfm_model_results_paths')
        training_completed_refreshed = get_dfm_state('training_completed_refreshed')
        polling_count = get_dfm_state('training_completion_polling_count', 0)
        training_log = get_dfm_state('dfm_training_log', [])

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
    import unicodedata
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 移除类型映射，只保留行业映射
    var_industry_map = {}

    if indicator_col in df.columns:
        if industry_col in df.columns:
            # 标准化键名（转换为小写）以与训练过程保持一致
            raw_industry_map = dict(zip(df[indicator_col].fillna(''), df[industry_col].fillna('')))
            var_industry_map = {
                unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
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
    from dashboard.models.DFM.train.training import DFMTrainer, TrainingConfig
    from dashboard.models.DFM.train.training.trainer import TrainingResult
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

# 全局DFM管理器实例
_dfm_manager = None


def manage_download_state(action, session_id=None):
    """统一管理下载相关状态"""
    import uuid
    import time

    if action == 'start':
        # 生成新的下载会话ID
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        # 设置下载状态
        set_dfm_state('dfm_downloading_files', True)
        set_dfm_state('dfm_download_start_time', time.time())
        set_dfm_state('dfm_download_session_id', session_id)
        return session_id

    elif action == 'check':
        # 检查下载状态
        is_downloading = get_dfm_state('dfm_downloading_files', False)
        start_time = get_dfm_state('dfm_download_start_time', 0)
        session_id = get_dfm_state('dfm_download_session_id', '')

        return {
            'is_downloading': is_downloading,
            'start_time': start_time,
            'session_id': session_id,
            'duration': time.time() - start_time if start_time else 0
        }

    elif action == 'clear':
        # 清理下载状态
        set_dfm_state('dfm_downloading_files', False)
        set_dfm_state('dfm_download_start_time', None)
        set_dfm_state('dfm_download_session_id', None)

        return True

    return False


def is_download_protected():
    """检查当前是否处于下载保护状态"""
    download_info = manage_download_state('check')

    # 检查是否正在下载
    if not download_info['is_downloading']:
        return False

    # 检查下载是否超时（5分钟）
    if download_info['duration'] > 300:
        print(f"[WARNING] [下载保护] 下载会话超时，自动清理: {download_info['session_id']}")
        manage_download_state('clear')
        return False

    return True


def cleanup_expired_downloads(timeout_seconds=300):
    """清理超时的下载状态"""
    download_info = manage_download_state('check')

    if download_info['is_downloading'] and download_info['duration'] > timeout_seconds:
        print(f"[CLEAN] [下载清理] 清理超时下载会话: {download_info['session_id']}")
        manage_download_state('clear')
        return True

    return False


def validate_date_consistency():
    """验证数据准备tab和模型训练tab的日期参数一致性"""
    data_prep_start = get_dfm_state('dfm_param_data_start_date')
    data_prep_end = get_dfm_state('dfm_param_data_end_date')
    training_start = get_dfm_state('dfm_training_start_date')
    validation_end = get_dfm_state('dfm_validation_end_date')

    # 如果数据准备页面设置了边界，确保训练参数在合理范围内
    if data_prep_start and training_start:
        if training_start < data_prep_start:
            print(f"[WARNING] [日期一致性] 训练开始日期 {training_start} 早于数据边界 {data_prep_start}")

    if data_prep_end and validation_end:
        if validation_end > data_prep_end:
            print(f"[WARNING] [日期一致性] 验证结束日期 {validation_end} 晚于数据边界 {data_prep_end}")

    return True


def load_mappings_from_state(available_data_columns=None):
    """
    从会话状态中获取映射数据构建行业到指标的映射
    优先使用数据准备模块已经加载好的映射，避免重复读取Excel文件

    Args:
        available_data_columns: 实际数据中可用的列名列表（用于过滤）

    Returns:
        tuple: (unique_industries, industry_to_indicators_map, all_indicators_flat, dfm_default_map)
               如果没有找到映射数据，返回 (None, None, None, None)
    """
    try:
        var_industry_map = get_dfm_state('dfm_industry_map_obj', None)
        dfm_default_map = get_dfm_state('dfm_default_variables_map', {})

        if var_industry_map is None:
            print("[WARNING] [映射加载] 会话状态中未找到行业映射数据")
            return None, None, None, None

        if not var_industry_map:
            print("[WARNING] [映射加载] 行业映射数据为空")
            return None, None, None, None

        # 如果提供了实际数据列名，进行过滤
        if available_data_columns is not None:
            # 标准化实际数据的列名
            normalized_data_columns = {}
            for col in available_data_columns:
                if col and pd.notna(col):
                    norm_col = unicodedata.normalize('NFKC', str(col)).strip().lower()
                    if norm_col:
                        normalized_data_columns[norm_col] = col

            # 过滤映射，只保留实际存在的变量
            filtered_var_industry_map = {}
            for indicator_norm, industry in var_industry_map.items():
                if indicator_norm in normalized_data_columns:
                    filtered_var_industry_map[indicator_norm] = industry

            var_industry_map = filtered_var_industry_map
            
            # 修复：将过滤后的映射同步到状态管理器
            set_dfm_state('dfm_industry_map_filtered', var_industry_map)

            # 添加缓存机制避免重复打印
            original_count = len(get_dfm_state('dfm_industry_map_obj', {}))
            filtered_count = len(var_industry_map)
            cache_key = f"mapping_filter_log_{original_count}_{filtered_count}"

            if not get_dfm_state(cache_key, False):
                print(f"[DATA] [映射过滤] 原始映射 {original_count} 个变量，过滤后 {filtered_count} 个变量")
                set_dfm_state(cache_key, True)

        # 构建行业到指标的映射
        industry_to_indicators_temp = defaultdict(list)
        for indicator, industry in var_industry_map.items():
            if indicator and industry:
                industry_to_indicators_temp[str(industry).strip()].append(str(indicator).strip())

        # 排序并返回结果
        unique_industries = sorted(list(industry_to_indicators_temp.keys()))
        industry_to_indicators_map = {k: sorted(v) for k, v in industry_to_indicators_temp.items()}
        all_indicators_flat = sorted(list(var_industry_map.keys()))

        return unique_industries, industry_to_indicators_map, all_indicators_flat, dfm_default_map

    except Exception as e:
        print(f"[ERROR] [映射加载] 从状态管理加载映射数据失败: {e}")
        return None, None, None, None


def _reset_training_state():
    """重置所有训练相关状态（使用组件化方法）"""
    try:
        training_component = get_training_status_component()
        training_component._reset_training_state()

        from dashboard.ui.utils.debug_helpers import debug_log
        debug_log("状态重置 - 使用组件化方法重置训练状态", "DEBUG")

        import streamlit as st
        training_keys = [
            'dfm_training_status',
            'dfm_training_log',
            'dfm_training_progress',
            'dfm_model_results_paths',
            'dfm_model_results',
            'dfm_training_error',
            'dfm_training_start_time',
            'dfm_training_end_time',
            'training_completed_refreshed',
            'training_completion_polling_count',
            'dfm_force_reset_training_state',
            'dfm_page_initialized'
        ]

        for key in training_keys:
            # 从状态管理中删除
            set_dfm_state(key, None)
            debug_log(f"状态重置 - 已清理状态键: {key}", "DEBUG")

        set_dfm_state('dfm_training_status', '等待开始')
        set_dfm_state('dfm_training_log', [])
        set_dfm_state('dfm_training_progress', 0)
        set_dfm_state('dfm_model_results_paths', None)
        set_dfm_state('dfm_model_results', None)

        debug_log("状态重置 - 已重置所有训练状态到初始值", "DEBUG")

    except Exception as e:
        debug_log(f"状态重置 - 重置失败: {str(e)}", "ERROR")
        # 如果组件化方法失败，使用备用方法
        keys_to_clear = [
            'dfm_training_status',
            'dfm_model_results_paths',
            'dfm_training_log',
            'dfm_training_error',
            'dfm_training_polling_count',
            'dfm_force_reset_training_state',
            'training_completed_refreshed',
            'training_completion_polling_count'
        ]
        for key in keys_to_clear:
            set_dfm_state(key, None)

        # 重新设置初始状态
        set_dfm_state('dfm_training_status', '等待开始')
        set_dfm_state('dfm_training_log', [])


    keys_to_clear = [
        'dfm_training_status',
        'dfm_model_results_paths',
        'dfm_training_log',
        'dfm_training_error',
        'existing_results_checked',
        'training_completed_refreshed',
        'training_completion_polling_count',  # [HOT] 新增：轮询计数器状态键
        'dfm_force_reset_training_state',
        'dfm_training_completed_timestamp',
        'dfm_force_ui_refresh',
        'dfm_training_in_progress',
        'dfm_training_completion_confirmed'
    ]

    for key in keys_to_clear:
        set_dfm_state(key, None)

    # 重置为初始状态
    set_dfm_state('dfm_training_status', '等待开始')
    set_dfm_state('dfm_training_log', [])




def get_dfm_state(key, default=None):
    """获取DFM状态值 - 仅从train_model命名空间读取"""
    try:
        # 导入调试工具
        from dashboard.ui.utils.debug_helpers import debug_log
        import streamlit as st

        # 直接使用st.session_state，不再通过get_global_dfm_manager
        full_key = f'train_model.{key}'
        value = st.session_state.get(full_key, default)

        # 详细的状态读取日志（仅在调试模式下输出）
        debug_log(f"前端状态读取 - 键: {key}, 值类型: {type(value).__name__}, 来源: train_model", "DEBUG")

        return value
    except Exception as e:
        from dashboard.ui.utils.debug_helpers import debug_log
        debug_log(f"状态读取 - 异常 - 键: {key}, 错误: {str(e)}", "ERROR")
        return default


def set_dfm_state(key, value):
    """设置DFM状态值（直接使用st.session_state）"""
    try:
        import streamlit as st
        from dashboard.ui.utils.debug_helpers import debug_log

        # 训练相关键列表
        training_keys = [
            'dfm_training_status',
            'dfm_training_log',
            'dfm_training_progress',
            'dfm_model_results_paths',
            'dfm_training_error',
            'dfm_training_start_time',
            'dfm_training_end_time',
            'training_completed_refreshed',
            'training_completion_polling_count'
        ]

        # 直接使用st.session_state
        full_key = f'train_model.{key}'
        st.session_state[full_key] = value

        # 调试日志（仅在调试模式下输出）
        debug_log(f"状态设置 - 键: {key}, 值类型: {type(value)}", "DEBUG")
        return True
    except Exception as e:
        from dashboard.ui.utils.debug_helpers import debug_log
        debug_log(f"状态设置 - 异常 - 键: {key}, 错误: {str(e)}", "ERROR")
        return False


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


    cleanup_expired_downloads()

    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        if "train" in _TRAIN_UI_IMPORT_ERROR_MESSAGE:
            st_instance.error(f"关键模块导入错误，模型训练功能不可用:\n{_TRAIN_UI_IMPORT_ERROR_MESSAGE}")
            return  # 如果训练模块不可用，直接返回
        else:
            # 如果只是数据准备模块的导入问题，显示警告但继续
            st_instance.warning("[WARNING] 数据准备模块导入警告，但映射数据传递已修复，功能应该正常")
    else:
        # 如果没有错误消息，显示成功信息
        # st_instance.success("[SUCCESS] 所有必需模块已成功加载(使用train)，模型训练功能可用")
        pass  # 已禁用模块加载成功提示

    current_training_status = get_dfm_state('dfm_training_status')
    current_model_results = get_dfm_state('dfm_model_results_paths')

    # 如果页面刚加载且存在之前的训练完成状态，询问用户是否要重置
    page_just_loaded = get_dfm_state('dfm_page_initialized') is None

    if page_just_loaded:
        set_dfm_state('dfm_page_initialized', True)

        # 如果存在之前的训练结果，显示提示并自动重置（避免混淆）
        if current_training_status == '训练完成' and current_model_results:
            st_instance.info("[LOADING] 检测到之前的训练结果，已自动重置训练状态以开始新的训练")
            _reset_training_state()
            current_training_status = '等待开始'

    # 使用状态管理器初始化DFM状态
    if current_training_status is None:
        set_dfm_state('dfm_training_status', "等待开始")
    if get_dfm_state('dfm_model_results') is None:
        set_dfm_state('dfm_model_results', None)
    if get_dfm_state('dfm_training_log') is None:
        set_dfm_state('dfm_training_log', [])
    if get_dfm_state('dfm_model_results_paths') is None:
        set_dfm_state('dfm_model_results_paths', None)


    # 状态重置现在仅通过用户点击按钮触发

    # === 自动检测功能已禁用 - 用户不希望自动恢复训练状态 ===

    # 移除已有训练结果检测功能，不再检查dym_estimate目录
    def _detect_existing_results():
        """不再检测已存在的训练结果文件，所有结果通过UI下载获得"""
        return None

    # 检测已有结果并更新状态
    if (get_dfm_state('dfm_training_status') == '等待开始' and
        get_dfm_state('existing_results_checked') is None):

        set_dfm_state('existing_results_checked', True)
        existing_results = _detect_existing_results()

        if existing_results:
            # 更新全局状态和状态管理器
            set_dfm_state('dfm_training_status', '训练完成')
            set_dfm_state('dfm_model_results_paths', existing_results)
            set_dfm_state('dfm_training_log', ['[自动检测] 发现已有训练结果，已自动加载'])

            # 刷新UI显示
            st_instance.rerun()


    training_status = get_dfm_state('dfm_training_status', '等待开始')

    training_results = get_dfm_state('dfm_model_results_paths')
    if training_results and training_status != '训练完成':
        print(f"[HOT] [状态修复] 检测到训练结果存在但状态未更新: {training_status} -> 训练完成")
        set_dfm_state('dfm_training_status', '训练完成')
        training_status = '训练完成'

    ui_refresh_needed = get_dfm_state('ui_refresh_needed', False)
    training_completion_timestamp = get_dfm_state('training_completion_timestamp')

    # 检查是否有新的训练完成（基于时间戳）
    last_processed_timestamp = get_dfm_state('last_processed_completion_timestamp')
    training_just_completed = (training_completion_timestamp and
                              training_completion_timestamp != last_processed_timestamp)

    # 合并刷新标志
    force_ui_refresh = ui_refresh_needed or training_just_completed

    from dashboard.ui.utils.debug_helpers import debug_log
    debug_log(f"UI状态检查 - 当前训练状态: {training_status}", "DEBUG")
    debug_log(f"UI状态检查 - UI刷新需要标志: {ui_refresh_needed}", "DEBUG")
    debug_log(f"UI状态检查 - 强制刷新标志: {force_ui_refresh}", "DEBUG")
    debug_log(f"UI状态检查 - 训练刚完成标志: {training_just_completed}", "DEBUG")

    if force_ui_refresh or training_just_completed:
        # 防止连续刷新的保护机制
        last_forced_refresh = get_dfm_state('last_forced_refresh_time', 0)
        import time
        current_time = time.time()
        
        # 至少间隔3秒才允许强制刷新
        if current_time - last_forced_refresh > 3:
            print("[HOT] [UI更新] 检测到强制刷新标志，清除标志并刷新UI")
            set_dfm_state('last_forced_refresh_time', current_time)

            # 清除刷新标志
            set_dfm_state('ui_refresh_needed', False)
            if training_completion_timestamp:
                set_dfm_state('last_processed_completion_timestamp', training_completion_timestamp)

            if training_status == '训练完成':
                set_dfm_state('training_completed_refreshed', True)

            st_instance.rerun()
        else:
            print(f"[HOT] [UI更新] 强制刷新被跳过，距离上次刷新仅 {current_time - last_forced_refresh:.1f} 秒")

    elif training_status == '正在训练...':
        # 检查是否需要定期刷新以显示训练进度
        import time
        current_time = time.time()
        last_refresh_time = get_dfm_state('last_training_refresh_time', 0)
        
        # 增加防止循环的检查：只在真正需要时刷新，并增加最小间隔
        min_refresh_interval = 10  # 增加到10秒，减少刷新频率
        refresh_count = get_dfm_state('training_refresh_count', 0)
        max_refresh_count = 10  # 最大刷新次数限制
        
        if (current_time - last_refresh_time > min_refresh_interval and 
            refresh_count < max_refresh_count):
            set_dfm_state('last_training_refresh_time', current_time)
            set_dfm_state('training_refresh_count', refresh_count + 1)
            print(f"[HOT] [定期检查] 训练中，刷新UI以显示最新状态 (第{refresh_count + 1}次)")
            st_instance.rerun()

    elif training_status == '训练完成':
        training_completed_refreshed = get_dfm_state('training_completed_refreshed')
        print(f"[HOT] [备用检查] 训练完成状态检查 - 已刷新标志: {training_completed_refreshed}")

        if training_completed_refreshed is None:
            print("[HOT] [备用检查] 检测到训练完成但未刷新，执行备用刷新")
            set_dfm_state('training_completed_refreshed', True)
            
            # 清除刷新计数器，防止后续循环
            set_dfm_state('training_refresh_count', 0)
            set_dfm_state('last_training_refresh_time', 0)

            try:
                st_instance.cache_resource.clear()
                print("[HOT] [缓存清除] Streamlit资源缓存已清除")
            except Exception as e:
                print(f"[HOT] [缓存清除] 清除Streamlit缓存失败: {e}")

            debug_training_state("备用检查触发UI刷新", show_in_ui=True)
            st_instance.rerun()
        else:
            print("[HOT] [备用检查] 训练完成状态已处理，继续显示结果")
            # 确保刷新计数器已清除
            set_dfm_state('training_refresh_count', 0)

    # [REMOVED] 重复的训练完成检查代码块已移除，避免渲染循环

    # else:
    #     pass  # 跳过页面刷新

    # 文件上传区域 - 替代从data_prep命名空间读取数据
    st_instance.markdown("### 数据文件上传")
    st_instance.info("请上传数据准备模块导出的预处理数据和行业映射文件")

    col_upload1, col_upload2 = st_instance.columns(2)

    with col_upload1:
        st_instance.markdown("**预处理数据文件 (.csv)**")
        uploaded_data_file = st_instance.file_uploader(
            "选择预处理数据文件",
            type=['csv'],
            key="train_data_upload",
            help="上传数据准备模块导出的预处理数据CSV文件（包含日期索引和所有变量列）"
        )

        if uploaded_data_file:
            set_dfm_state("train_uploaded_data_file", uploaded_data_file)
        else:
            existing_data_file = get_dfm_state('train_uploaded_data_file', None)
            if existing_data_file is not None and hasattr(existing_data_file, 'name'):
                st_instance.info(f"当前文件: {existing_data_file.name}")

    with col_upload2:
        st_instance.markdown("**行业映射文件 (.csv)**")
        uploaded_industry_map_file = st_instance.file_uploader(
            "选择行业映射文件",
            type=['csv'],
            key="train_industry_map_upload",
            help="上传数据准备模块导出的行业映射CSV文件（包含Indicator和Industry两列）"
        )

        if uploaded_industry_map_file:
            set_dfm_state("train_uploaded_industry_map_file", uploaded_industry_map_file)
        else:
            existing_map_file = get_dfm_state('train_uploaded_industry_map_file', None)
            if existing_map_file is not None and hasattr(existing_map_file, 'name'):
                st_instance.info(f"当前文件: {existing_map_file.name}")

    # 加载上传的文件
    data_file = get_dfm_state('train_uploaded_data_file', None)
    industry_map_file = get_dfm_state('train_uploaded_industry_map_file', None)

    # 生成文件标识（用于检测文件变更）
    def get_file_id(file_obj):
        if file_obj is None:
            return None
        file_obj.seek(0, 2)  # 移动到文件末尾
        size = file_obj.tell()
        file_obj.seek(0)  # 重置到开头
        name = getattr(file_obj, 'name', 'unknown')
        return f"{name}_{size}"

    # 加载预处理数据（带缓存）
    input_df = None
    if data_file is not None:
        current_data_file_id = get_file_id(data_file)
        cached_data_file_id = get_dfm_state('cached_data_file_id', None)
        cached_df = get_dfm_state('dfm_prepared_data_df', None)

        # 检查缓存是否有效
        if cached_df is not None and current_data_file_id == cached_data_file_id:
            input_df = cached_df
            print(f"[模型训练] 使用缓存的预处理数据: {input_df.shape}")
        else:
            # 缓存无效，重新加载
            try:
                data_file.seek(0)
                input_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                set_dfm_state('dfm_prepared_data_df', input_df)
                set_dfm_state('cached_data_file_id', current_data_file_id)
                print(f"[模型训练] 重新加载预处理数据: {input_df.shape}")
            except Exception as e:
                st_instance.error(f"加载预处理数据失败: {e}")
                input_df = None

    # 加载行业映射（带缓存）
    var_industry_map = {}
    if industry_map_file is not None:
        current_map_file_id = get_file_id(industry_map_file)
        cached_map_file_id = get_dfm_state('cached_map_file_id', None)
        cached_industry_map = get_dfm_state('dfm_industry_map_obj', None)
        cached_dfm_default_map = get_dfm_state('dfm_default_variables_map', None)

        # 检查缓存是否有效
        if cached_industry_map is not None and current_map_file_id == cached_map_file_id:
            var_industry_map = cached_industry_map
            print(f"[模型训练] 使用缓存的行业映射: {len(var_industry_map)} 个变量")
            if cached_dfm_default_map is not None:
                print(f"[模型训练] 使用缓存的DFM变量配置: {len(cached_dfm_default_map)} 个标记为'是'的变量")
        else:
            # 缓存无效，重新加载
            try:
                import unicodedata as udata
                industry_map_file.seek(0)
                industry_map_df = pd.read_csv(industry_map_file)
                if 'Indicator' in industry_map_df.columns and 'Industry' in industry_map_df.columns:
                    var_industry_map = {
                        udata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                        for k, v in zip(industry_map_df['Indicator'], industry_map_df['Industry'])
                        if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip()
                    }
                    set_dfm_state('dfm_industry_map_obj', var_industry_map)
                    set_dfm_state('cached_map_file_id', current_map_file_id)
                    print(f"[模型训练] 重新加载行业映射: {len(var_industry_map)} 个变量")

                    # 读取DFM变量列（如果存在），只保留标记为"是"的条目
                    if 'DFM_Default' in industry_map_df.columns:
                        var_dfm_default_map = {
                            udata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                            for k, v in zip(industry_map_df['Indicator'], industry_map_df['DFM_Default'])
                            if pd.notna(k) and pd.notna(v) and str(v).strip() == '是'
                        }
                        set_dfm_state('dfm_default_variables_map', var_dfm_default_map)
                        print(f"[模型训练] 从CSV文件加载了 {len(var_dfm_default_map)} 个标记为'是'的变量")
                    else:
                        print(f"[模型训练] 映射文件未包含DFM_Default列（旧格式），跳过DFM变量配置加载")
                else:
                    st_instance.error("映射文件格式错误：必须包含 'Indicator' 和 'Industry' 列")
            except Exception as e:
                st_instance.error(f"加载映射文件失败: {e}")
                import traceback
                st_instance.code(traceback.format_exc(), language="python")

    # 检查文件是否都已上传
    if input_df is None or not var_industry_map:
        missing = []
        if input_df is None:
            missing.append("预处理数据文件")
        if not var_industry_map:
            missing.append("行业映射文件")
        st_instance.warning(f"缺少必要文件: {', '.join(missing)}。请上传后再继续。")
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


    # 1. 首先尝试从会话状态中获取已经准备好的映射数据
    available_data_columns = list(input_df.columns) if input_df is not None else None

    # 优先使用会话状态中的映射数据
    map_data = load_mappings_from_state(available_data_columns)

    if map_data and all(x is not None for x in map_data[:3]):
        unique_industries, var_to_indicators_map_by_industry, _, dfm_default_map = map_data

        # 修复：显示实际可用的指标数量
        actual_indicator_count = sum(len(v) for v in var_to_indicators_map_by_industry.values())
        # st_instance.success(f"[SUCCESS] 已加载映射数据：{len(unique_industries)} 个行业，{actual_indicator_count} 个可用指标")
        pass  # 已禁用映射数据加载成功提示
    else:
        # 不再回退到Excel文件，使用空映射继续（数据准备模块应该正确保存映射）

        st_instance.warning("[WARNING] 未找到映射数据，请确保已在'数据准备'模块正确处理数据")
        unique_industries = []
        var_to_indicators_map_by_industry = {}
        dfm_default_map = {}

    # 主布局：现在是上下结构，不再使用列
    # REMOVED: var_selection_col, param_col = st_instance.columns([1, 1.5])

    # 添加变量选择大标题
    st_instance.subheader("变量选择")

    # 1. 选择目标变量
    st_instance.markdown("**选择目标变量**")
    if available_target_vars:
        # 初始化目标变量状态
        if get_dfm_state('dfm_target_variable') is None:
            set_dfm_state('dfm_target_variable', available_target_vars[0])

        current_target_var = get_dfm_state('dfm_target_variable')

        # 确保当前目标变量在可选列表中
        if current_target_var not in available_target_vars:
            current_target_var = available_target_vars[0]
            set_dfm_state('dfm_target_variable', current_target_var)

        selected_target_var = st_instance.selectbox(
            "目标变量",
            options=available_target_vars,
            index=available_target_vars.index(current_target_var),
            key="ss_dfm_target_variable",
            help="选择您希望模型预测的目标序列。"
        )
        set_dfm_state('dfm_target_variable', selected_target_var)
    else:
        st_instance.error("[ERROR] 无法找到任何可用的目标变量")
        set_dfm_state('dfm_target_variable', None)

    # 2. 过滤行业：移除仅包含目标变量的行业
    current_target_var = get_dfm_state('dfm_target_variable', None)
    filtered_industries = []

    for industry_name in unique_industries:
        industry_indicators = var_to_indicators_map_by_industry.get(industry_name, [])
        if current_target_var and current_target_var in industry_indicators:
            non_target_indicators = [ind for ind in industry_indicators if ind != current_target_var]
            if non_target_indicators:
                filtered_industries.append(industry_name)
        else:
            filtered_industries.append(industry_name)

    # 暂时存储过滤后的行业，以供后续步骤3中使用
    if not filtered_industries:
        st_instance.info("没有可用的行业数据。")
        set_dfm_state('dfm_selected_industries', [])
    else:
        pass  # 继续进行到步骤3，用户将直接选择指标

    # 3. 根据选定行业选择预测指标 (每个行业一个多选下拉菜单，默认全选)
    st_instance.markdown("**选择预测指标**")
    # 初始化指标选择状态
    if get_dfm_state('dfm_selected_indicators_per_industry', None) is None:
        set_dfm_state('dfm_selected_indicators_per_industry', {})

    final_selected_indicators_flat = []
    current_selected_industries = filtered_industries  # 直接使用过滤后的所有行业

    if not current_selected_industries:
        st_instance.info("没有可用的行业数据。")
    else:
        current_selection = get_dfm_state('dfm_selected_indicators_per_industry', {})

        for industry_name in current_selected_industries:
            all_indicators_for_industry = var_to_indicators_map_by_industry.get(industry_name, [])
            
            # 修复：排除目标变量，确保用户无法选择目标变量作为预测变量
            current_target_var = get_dfm_state('dfm_target_variable', None)
            if current_target_var:
                indicators_for_this_industry = [
                    indicator for indicator in all_indicators_for_industry 
                    if indicator != current_target_var
                ]
            else:
                indicators_for_this_industry = all_indicators_for_industry

            # 修复：完全跳过没有可用指标的行业，不显示任何内容
            if not indicators_for_this_industry:
                current_selection[industry_name] = []
                continue

            # 只有当行业有可用指标时才显示
            st_instance.markdown(f"**行业: {industry_name}**")
            
            # 只有在有指标被排除且仍有可用指标时才显示提示
            excluded_count = len(all_indicators_for_industry) - len(indicators_for_this_industry)
            if excluded_count > 0:
                st_instance.text(f"  已自动排除目标变量 '{current_target_var}' (共排除 {excluded_count} 个)")

            # 从状态管理器读取已选指标，或使用DFM默认选择
            default_selection_for_industry = current_selection.get(industry_name, None)

            # 如果状态管理器中没有选择，使用DFM变量列配置
            if default_selection_for_industry is None:
                # dfm_default_map中已经只包含标记为"是"的变量，直接筛选该行业的指标
                dfm_default_indicators = [
                    indicator for indicator in indicators_for_this_industry
                    if normalize_text(indicator) in dfm_default_map
                ]
                default_selection_for_industry = dfm_default_indicators

            # 确保默认值是实际可选列表的子集
            valid_default = [item for item in default_selection_for_industry if item in indicators_for_this_industry]

            # 全选复选框，使用key确保状态追踪
            select_all_key = f"dfm_select_all_indicators_{industry_name}"
            select_all_prev_key = f"dfm_select_all_prev_{industry_name}"
            multiselect_key = f"dfm_indicators_multiselect_{industry_name}"

            # 判断是否应该默认勾选全选复选框（所有指标都已选中）
            should_check_select_all = (
                len(valid_default) > 0 and
                len(valid_default) == len(indicators_for_this_industry)
            )

            # 确保 session_state 中一定有初始值（避免 Streamlit 警告）
            if multiselect_key not in st.session_state:
                # 使用默认选择初始化
                st.session_state[multiselect_key] = valid_default
                print(f"[DEBUG] 初始化multiselect状态: {industry_name}, {len(valid_default)}个指标")

            select_all_checked = st_instance.checkbox(
                f"全选 {industry_name} 指标",
                value=should_check_select_all,
                key=select_all_key,
                help=f"勾选此框将选中所有 '{industry_name}' 的指标。"
            )

            # 检测全选复选框状态变化
            prev_select_all = get_dfm_state(select_all_prev_key, should_check_select_all)

            # 如果全选状态发生变化，更新multiselect的值
            if select_all_checked != prev_select_all:
                if select_all_checked:
                    # 全选被勾选：更新session_state中multiselect的值为所有指标
                    print(f"[DEBUG] 全选复选框被勾选: {industry_name}, 设置 {len(indicators_for_this_industry)} 个指标")
                    st.session_state[multiselect_key] = indicators_for_this_industry
                else:
                    # 全选被取消：清空multiselect的值
                    print(f"[DEBUG] 全选复选框被取消: {industry_name}, 清空指标选择")
                    st.session_state[multiselect_key] = []
                # 保存当前全选状态
                set_dfm_state(select_all_prev_key, select_all_checked)

            selected_in_widget = st_instance.multiselect(
                f"为 '{industry_name}' 选择指标",
                options=indicators_for_this_industry,
                key=multiselect_key,
                help=f"从 {industry_name} 行业中选择预测指标。"
            )

            # 同步全选复选框状态：如果用户手动修改了multiselect
            # 检查是否所有指标都被选中，如果是则同步全选状态
            if len(selected_in_widget) == len(indicators_for_this_industry) and len(selected_in_widget) > 0:
                # 所有指标都被选中，更新全选状态
                if not select_all_checked:
                    set_dfm_state(select_all_prev_key, True)
            elif len(selected_in_widget) == 0 or len(selected_in_widget) < len(indicators_for_this_industry):
                # 未全选，更新全选状态
                if select_all_checked:
                    set_dfm_state(select_all_prev_key, False)

            current_selection[industry_name] = selected_in_widget
            final_selected_indicators_flat.extend(selected_in_widget)

        industries_to_remove_from_state = [
            ind for ind in current_selection
            if ind not in current_selected_industries
        ]
        for ind_to_remove in industries_to_remove_from_state:
            del current_selection[ind_to_remove]

        set_dfm_state('dfm_selected_indicators_per_industry', current_selection)

    # 修复：如果循环没有执行（filtered_industries为空），但dfm_selected_indicators_per_industry中有数据
    # 说明这是旧数据，应该从dfm_selected_indicators_per_industry重建指标列表
    if len(final_selected_indicators_flat) == 0 and len(current_selected_industries) == 0:
        saved_selection = get_dfm_state('dfm_selected_indicators_per_industry', {})
        if saved_selection:
            for industry, indicators in saved_selection.items():
                final_selected_indicators_flat.extend(indicators)

    # 更新最终的扁平化预测指标列表 (去重)
    final_indicators = sorted(list(set(final_selected_indicators_flat)))
    set_dfm_state('dfm_selected_indicators', final_indicators)

    # 调试：打印选择的指标
    print(f"[UI] 用户选择的指标数量: {len(final_indicators)}")
    if final_indicators:
        print(f"[UI] 选择的指标列表:")
        for idx, var in enumerate(final_indicators, 1):
            print(f"  {idx}. {var}")

    # 从选择的指标自动推断实际使用的行业（只有当该行业有指标被选中时）
    inferred_industries = []
    selected_indicators_per_industry = get_dfm_state('dfm_selected_indicators_per_industry', {})
    for industry, indicators in selected_indicators_per_industry.items():
        if indicators and len(indicators) > 0:  # 如果该行业有选中的指标
            inferred_industries.append(industry)

    set_dfm_state('dfm_selected_industries', inferred_industries)

    # 变量选择完成

    # 显示汇总信息 (可选)
    st_instance.markdown("--- ")
    current_target_var = get_dfm_state('dfm_target_variable', None)
    current_selected_indicators = get_dfm_state('dfm_selected_indicators', [])
    current_selected_industries_for_display = get_dfm_state('dfm_selected_industries', [])

    st_instance.text(f" - 目标变量: {current_target_var if current_target_var else '未选择'}")
    st_instance.text(f" - 选定行业数: {len(current_selected_industries_for_display)}")
    st_instance.text(f" - 选定预测指标总数: {len(current_selected_indicators)}")

    st_instance.markdown("--- ") # 分隔线，将变量选择与参数配置分开
    st_instance.subheader("模型参数")

    # 创建三列布局
    col1_time, col2_factor_core, col3_factor_specific = st_instance.columns(3)

    with col1_time:


        # 计算基于数据的智能默认值
        def get_data_based_date_defaults():
            """基于实际数据计算日期默认值，优先使用数据准备页面设置的日期边界"""
            from datetime import datetime, timedelta
            today = datetime.now().date()

            data_prep_start = get_dfm_state('dfm_param_data_start_date')
            data_prep_end = get_dfm_state('dfm_param_data_end_date')

            static_defaults = {
                'training_start': datetime(2020, 1, 1).date(),  # 训练期开始：2020年1月1日
                'validation_start': datetime(2025, 1, 1).date(),  # 验证期开始：2025年1月1日
                'validation_end': datetime(2025, 12, 31).date()  # 验证期结束：2025年12月31日
            }

            try:
                data_df = None
                for key in ['dfm_prepared_data_df', 'data_prep.dfm_prepared_data_df', 'dfm.data_prep.dfm_prepared_data_df']:
                    df_value = get_dfm_state(key)
                    if df_value is not None:
                        data_df = df_value
                        break
                if data_df is not None and isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                    # 从数据获取第一期和最后一期
                    data_first_date = data_df.index.min().date()  # 第一期数据
                    data_last_date = data_df.index.max().date()   # 最后一期数据

                    # 重要：确保数据的最后日期不是未来日期
                    if data_last_date > today:
                        print(f"[WARNING] 警告: 数据包含未来日期 {data_last_date}，将使用今天作为最后日期")
                        data_last_date = today

                    # 训练期开始日期固定为2020年1月1日
                    training_start_date = datetime(2020, 1, 1).date()

                    # 验证期开始和结束日期固定
                    validation_start_date = datetime(2025, 1, 1).date()  # 验证期开始：2025年1月1日
                    validation_end_date = datetime(2025, 12, 31).date()  # 验证期结束：2025年12月31日

                    return {
                        'training_start': training_start_date,       # [HOT] 训练开始日：优先使用数据准备页面设置
                        'validation_start': validation_start_date,   # 验证开始日：计算得出
                        'validation_end': validation_end_date        # [HOT] 验证结束日：优先使用数据准备页面设置
                    }
                else:
                    return static_defaults
            except Exception as e:
                print(f"[WARNING] 计算数据默认日期失败: {e}，使用静态默认值")
                return static_defaults

        # 获取智能默认值
        date_defaults = get_data_based_date_defaults()

        has_data = False
        data_df = None
        for key in ['dfm_prepared_data_df', 'data_prep.dfm_prepared_data_df', 'dfm.data_prep.dfm_prepared_data_df']:
            df_value = get_dfm_state(key)
            if df_value is not None:
                has_data = True
                data_df = df_value
                break
        if has_data:
            if isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                # 计算数据的实际日期范围用于比较
                actual_data_start = data_df.index.min().date()
                actual_data_end = data_df.index.max().date()

                # 初始化默认日期（只在状态为空时设置）
                current_training_start = get_dfm_state('dfm_training_start_date')
                if current_training_start is None:
                    set_dfm_state('dfm_training_start_date', date_defaults['training_start'])

                current_validation_start = get_dfm_state('dfm_validation_start_date')
                if current_validation_start is None:
                    set_dfm_state('dfm_validation_start_date', date_defaults['validation_start'])

                current_validation_end = get_dfm_state('dfm_validation_end_date')
                if current_validation_end is None:
                    set_dfm_state('dfm_validation_end_date', date_defaults['validation_end'])

                # 简化数据范围信息
                data_start = data_df.index.min().strftime('%Y-%m-%d')
                data_end = data_df.index.max().strftime('%Y-%m-%d')
                data_count = len(data_df.index)
                # st_instance.info(f"[DATA] 数据: {data_start} 至 {data_end} ({data_count}点)")


        # 执行日期一致性验证
        validate_date_consistency()

        # 1. 训练期开始日期
        training_start_value = st_instance.date_input(
            "训练期开始日期 (Training Start Date)",
            value=get_dfm_state('dfm_training_start_date', date_defaults['training_start']),
            key='dfm_training_start_date_input',
            help="选择模型训练数据的起始日期。默认为数据的第一期。"
        )
        set_dfm_state('dfm_training_start_date', training_start_value)

        # 2. 验证期开始日期
        validation_start_value = st_instance.date_input(
            "验证期开始日期 (Validation Start Date)",
            value=get_dfm_state('dfm_validation_start_date', date_defaults['validation_start']),
            key='dfm_validation_start_date_input',
            help="选择验证期开始日期。默认为最后一期数据前3个月。"
        )
        set_dfm_state('dfm_validation_start_date', validation_start_value)

        # 3. 验证期结束日期
        validation_end_value = st_instance.date_input(
            "验证期结束日期 (Validation End Date)",
            value=get_dfm_state('dfm_validation_end_date', date_defaults['validation_end']),
            key='dfm_validation_end_date_input',
            help="选择验证期结束日期。默认为数据的最后一期。"
        )
        set_dfm_state('dfm_validation_end_date', validation_end_value)

    with col2_factor_core:


        if CONFIG_AVAILABLE:
            variable_selection_options = UIDefaults.VARIABLE_SELECTION_OPTIONS
            default_var_method = TrainDefaults.VARIABLE_SELECTION_METHOD
        else:
            variable_selection_options = {
                'none': "无筛选 (使用全部已选变量)",
                'global_backward': "全局后向剔除 (在已选变量中筛选)"
            }
            default_var_method = 'none'  # [HOT] 紧急修复：强制默认为none

        # 获取当前变量选择方法
        current_var_method = get_dfm_state('dfm_variable_selection_method', default_var_method)

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
        set_dfm_state('dfm_variable_selection_method', var_method_value)

        enable_var_selection = (var_method_value != 'none')
        set_dfm_state('dfm_enable_variable_selection', enable_var_selection)

        # 后向剔除基于性能比较（HR和RMSE），不使用统计显著性阈值

        col_left, col_right = st_instance.columns(2)

        with col_left:
            # 最大迭代次数
            if CONFIG_AVAILABLE:
                default_max_iter = TrainDefaults.EM_MAX_ITER
            else:
                default_max_iter = 30

            max_iter_value = st_instance.number_input(
                "最大迭代次数",
                min_value=10,
                max_value=1000,
                value=get_dfm_state('dfm_max_iter', default_max_iter),
                step=10,
                key='dfm_max_iter_input',
                help="EM算法的最大迭代次数"
            )
            set_dfm_state('dfm_max_iter', max_iter_value)

            # 因子自回归阶数
            if CONFIG_AVAILABLE:
                default_ar_order = TrainDefaults.FACTOR_AR_ORDER
            else:
                default_ar_order = 1

            ar_order_value = st_instance.number_input(
                "因子自回归阶数",
                min_value=0,
                max_value=5,
                value=get_dfm_state('dfm_factor_ar_order', default_ar_order),
                step=1,
                key='dfm_factor_ar_order_input',
                help="因子的自回归阶数，通常设为1"
            )
            set_dfm_state('dfm_factor_ar_order', ar_order_value)

        with col_right:
            # 预留空间，可以添加其他参数
            pass

    with col3_factor_specific:
        # 因子选择策略
        if CONFIG_AVAILABLE:
            factor_strategy_options = UIDefaults.FACTOR_SELECTION_STRATEGY_OPTIONS
            default_strategy = TrainDefaults.FACTOR_SELECTION_STRATEGY
        else:
            factor_strategy_options = {
                'fixed_number': "固定因子数",
                'cumulative_variance': "累积方差贡献"
            }
            default_strategy = 'fixed_number'

        current_strategy = get_dfm_state('dfm_factor_selection_strategy', default_strategy)

        strategy_value = st_instance.selectbox(
            "因子选择策略",
            options=list(factor_strategy_options.keys()),
            format_func=lambda x: factor_strategy_options[x],
            index=list(factor_strategy_options.keys()).index(current_strategy),
            key='dfm_factor_selection_strategy',
            help="选择确定因子数量的方法"
        )
        
        # [CRITICAL FIX] 添加策略选择的状态保存
        set_dfm_state('dfm_factor_selection_strategy', strategy_value)

        # 根据策略显示相应参数
        if strategy_value == 'fixed_number':
            # 固定因子数
            if CONFIG_AVAILABLE:
                default_fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
            else:
                default_fixed_factors = 3

            fixed_factors_value = st_instance.number_input(
                "固定因子数",
                min_value=1,
                max_value=15,
                value=get_dfm_state('dfm_fixed_number_of_factors', default_fixed_factors),
                step=1,
                key='dfm_fixed_number_of_factors',
                help="指定使用的因子数量"
            )
            # [CRITICAL FIX] 添加缺失的状态保存
            set_dfm_state('dfm_fixed_number_of_factors', fixed_factors_value)

        elif strategy_value == 'cumulative_variance':
            # 累积方差贡献
            if CONFIG_AVAILABLE:
                default_cum_var = TrainDefaults.CUM_VARIANCE_THRESHOLD
            else:
                default_cum_var = 0.8

            cum_var_value = st_instance.number_input(
                "累积方差贡献阈值",
                min_value=0.5,
                max_value=0.99,
                value=get_dfm_state('dfm_cumulative_variance_threshold', default_cum_var),
                step=0.01,
                format="%.2f",
                key='dfm_cumulative_variance_threshold_input',
                help="因子累积解释方差的阈值"
            )
            set_dfm_state('dfm_cumulative_variance_threshold', cum_var_value)

    st_instance.markdown("--- ")
    st_instance.subheader("模型训练")

    current_target_var = get_dfm_state('dfm_target_variable', None)
    current_selected_indicators = get_dfm_state('dfm_selected_indicators', [])

    # 日期验证
    training_start_value = get_dfm_state('dfm_training_start_date')
    validation_start_value = get_dfm_state('dfm_validation_start_date')
    validation_end_value = get_dfm_state('dfm_validation_end_date')

    date_validation_passed = True
    if training_start_value and validation_start_value and validation_end_value:
        if training_start_value >= validation_start_value:
            st_instance.error("[ERROR] 训练期开始日期必须早于验证期开始日期")
            date_validation_passed = False
        elif validation_start_value >= validation_end_value:
            st_instance.error("[ERROR] 验证期开始日期必须早于验证期结束日期")
            date_validation_passed = False
        else:
            st_instance.success("[SUCCESS] 日期设置验证通过")
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

    # 训练按钮
    col_train_btn, col_reset_btn = st_instance.columns([1, 1])

    with col_train_btn:
        # 开始训练按钮
        if training_ready:
            if st_instance.button("[START] 开始训练",
                                key="dfm_start_training",
                                help="开始DFM模型训练",
                                use_container_width=True):


                current_status = get_dfm_state('dfm_training_status', '等待开始')
                if current_status in ['正在训练...', '准备启动训练...']:
                    st_instance.warning("[WARNING] 训练已在进行中，请勿重复启动")
                    return

                try:
                    # 使用train模块进行训练
                    import tempfile
                    from datetime import timedelta

                    # 获取行业映射数据（用于后续分析）
                    var_industry_map = get_dfm_state('dfm_industry_map_obj', {})
                    if not var_industry_map:
                        print("[INFO] 行业映射数据为空，Factor-Industry R² 分析功能将不可用")
                        # 不在UI上显示警告，因为这不影响训练核心功能

                    # 计算train_end_date
                    train_end_date = None
                    if validation_start_value:
                        try:
                            train_end_date = validation_start_value - timedelta(days=1)
                        except Exception:
                            train_end_date = None

                    # 获取变量选择方法
                    var_selection_method = get_dfm_state('dfm_variable_selection_method', 'none')
                    enable_var_selection = (var_selection_method != 'none')

                    # 映射UI的变量选择方法到train模块的变量选择方法
                    # UI使用 'global_backward'，train模块使用 'backward'
                    var_selection_method_map = {
                        'none': 'none',
                        'global_backward': 'backward',
                        'global_forward': 'forward',
                        'backward': 'backward',
                        'forward': 'forward'
                    }
                    mapped_var_selection_method = var_selection_method_map.get(var_selection_method, 'none')

                    # 获取因子选择策略（严格验证，不使用默认值）
                    factor_strategy = get_dfm_state('dfm_factor_selection_strategy')

                    if factor_strategy is None:
                        st_instance.error("因子选择策略未设置，请先在'模型配置'中设置因子选择策略")
                        print("[ERROR] 因子选择策略未设置")
                        return

                    # 映射factor_selection_strategy到train模块的factor_selection_method
                    if factor_strategy == 'fixed_number':
                        factor_selection_method = 'fixed'
                        k_factors = get_dfm_state('dfm_fixed_number_of_factors')

                        if k_factors is None:
                            st_instance.error("固定因子数未设置，请先在'模型配置'中设置因子数")
                            print("[ERROR] 固定因子数未设置")
                            return

                        st.info(f"使用固定因子数策略，因子数：{k_factors}")

                    elif factor_strategy == 'cumulative_variance':
                        factor_selection_method = 'cumulative'
                        pca_threshold = get_dfm_state('dfm_cumulative_variance_threshold')

                        if pca_threshold is None:
                            st_instance.error("累积方差阈值未设置，请先在'模型配置'中设置阈值")
                            print("[ERROR] 累积方差阈值未设置")
                            return

                        # 将由PCA确定，传入None作为占位符
                        k_factors = 1  # 占位符，实际值由PCA确定
                        st.info(f"使用累积方差策略，阈值：{pca_threshold}")

                    else:
                        st_instance.error(f"未知的因子选择策略: {factor_strategy}")
                        print(f"[ERROR] 未知的因子选择策略: {factor_strategy}")
                        return

                    # 保存DataFrame到临时文件（TrainingConfig需要文件路径）
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
                    temp_data_path = temp_file.name
                    temp_file.close()
                    input_df.to_csv(temp_data_path)
                    print(f"[TRAIN_REF] 临时数据文件: {temp_data_path}")

                    # 调试：检查选择的变量是否都在数据中（支持不区分大小写匹配）
                    csv_columns = set(input_df.columns)

                    # 构建不区分大小写的列名映射
                    import unicodedata
                    column_mapping = {}
                    for col in csv_columns:
                        normalized_col = unicodedata.normalize('NFKC', str(col)).strip().lower()
                        column_mapping[normalized_col] = col

                    # 检查并修正变量名
                    corrected_indicators = []
                    case_mismatches = []

                    for var in current_selected_indicators:
                        if var in csv_columns:
                            corrected_indicators.append(var)
                        else:
                            # 尝试不区分大小写匹配
                            normalized_var = unicodedata.normalize('NFKC', str(var)).strip().lower()
                            if normalized_var in column_mapping:
                                actual_col = column_mapping[normalized_var]
                                corrected_indicators.append(actual_col)
                                case_mismatches.append((var, actual_col))
                            # 如果还是找不到，corrected_indicators不包含这个变量

                    if case_mismatches:
                        print(f"[INFO] 检测到{len(case_mismatches)}个变量名大小写不匹配，已自动修正:")
                        for original, corrected in case_mismatches:
                            print(f"  '{original}' -> '{corrected}'")

                    if len(corrected_indicators) < len(current_selected_indicators):
                        missing_count = len(current_selected_indicators) - len(corrected_indicators)
                        print(f"[WARNING] {missing_count}个变量在DataFrame中找不到")
                    else:
                        print(f"[INFO] 所有选择的变量({len(current_selected_indicators)}个)都已找到")

                    # 使用修正后的变量名
                    current_selected_indicators = corrected_indicators

                    # 从状态中读取行业映射
                    current_industry_map = get_dfm_state('dfm_industry_map_obj', {})
                    if not current_industry_map:
                        current_industry_map = {}
                    print(f"[INFO] 行业映射包含 {len(current_industry_map)} 个变量")

                    # 构建TrainingConfig
                    training_config = TrainingConfig(
                        # 核心配置
                        data_path=temp_data_path,
                        target_variable=current_target_var,
                        selected_indicators=current_selected_indicators,

                        # 训练/验证期配置
                        training_start=training_start_value.strftime('%Y-%m-%d') if training_start_value else None,
                        train_end=train_end_date.strftime('%Y-%m-%d') if train_end_date else None,
                        validation_start=validation_start_value.strftime('%Y-%m-%d') if validation_start_value else None,
                        validation_end=validation_end_value.strftime('%Y-%m-%d') if validation_end_value else None,
                        target_freq='W-FRI',

                        # 模型参数
                        k_factors=k_factors,
                        max_iterations=get_dfm_state('dfm_max_iter') or 30,  # 允许30作为合理默认值
                        max_lags=get_dfm_state('dfm_factor_ar_order') or 1,  # 允许1作为合理默认值
                        tolerance=1e-6,

                        # 变量选择配置
                        enable_variable_selection=enable_var_selection,
                        variable_selection_method=mapped_var_selection_method if enable_var_selection else 'none',

                        # 因子数选择配置
                        factor_selection_method=factor_selection_method,
                        pca_threshold=pca_threshold if factor_strategy == 'cumulative_variance' else 0.9,

                        # 并行计算配置
                        enable_parallel=True,
                        n_jobs=-1,
                        parallel_backend='loky',
                        min_variables_for_parallel=5,

                        # 行业映射
                        industry_map=current_industry_map
                    )

                    print(f"[TRAIN_REF] 训练配置: {training_config}")
                    print(f"[TRAIN_CONFIG] 因子选择策略: {factor_selection_method}")
                    print(f"[TRAIN_CONFIG] 因子数: {k_factors}")
                    print(f"[TRAIN_CONFIG] PCA阈值: {training_config.pca_threshold}")
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
                        training_log = get_dfm_state('dfm_training_log', [])
                        training_log.append(message)
                        set_dfm_state('dfm_training_log', training_log)

                    # 设置训练状态
                    set_dfm_state('dfm_training_status', '正在训练...')
                    set_dfm_state('dfm_training_log', ['[TRAIN_REF] 开始训练...'])

                    # 创建训练器并训练（同步执行）
                    st_instance.info("[LOADING] 正在训练模型，请稍候...")
                    trainer = DFMTrainer(training_config)
                    result: TrainingResult = trainer.train(
                        progress_callback=progress_callback,
                        enable_export=True,
                        export_dir=None
                    )

                    # 处理训练结果并保存
                    # 保存训练结果摘要
                    result_summary = {
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

                    # 保存到状态管理器
                    set_dfm_state('dfm_training_result', result_summary)
                    set_dfm_state('dfm_model_results_paths', result.export_files)
                    set_dfm_state('dfm_training_status', '训练完成')
                    set_dfm_state('dfm_training_completed_timestamp', time.time())

                    # 添加完成日志
                    training_log = get_dfm_state('dfm_training_log', [])
                    training_log.append(f"[SUCCESS] 训练完成！耗时: {result.training_time:.2f}秒")
                    training_log.append(f"[RESULT] 选中变量数: {len(result.selected_variables)}")
                    training_log.append(f"[RESULT] 因子数: {result.k_factors}")
                    if result.metrics:
                        training_log.append(f"[METRICS] 样本外RMSE: {result.metrics.oos_rmse:.4f}")
                        # 检查Hit Rate是否有效
                        import numpy as np
                        hit_rate_value = result.metrics.oos_hit_rate
                        if np.isnan(hit_rate_value) or np.isinf(hit_rate_value):
                            hit_rate_str = "N/A (数据不足)"
                        else:
                            hit_rate_str = f"{hit_rate_value:.2f}%"
                        training_log.append(f"[METRICS] 样本外Hit Rate: {hit_rate_str}")
                    set_dfm_state('dfm_training_log', training_log)

                    # 清理临时文件
                    try:
                        os.unlink(temp_data_path)
                    except:
                        pass

                    st_instance.success("[SUCCESS] 训练完成！")
                    st_instance.rerun()

                except Exception as e:
                    import traceback
                    error_msg = f"启动训练失败: {str(e)}\n{traceback.format_exc()}"
                    print(f"[ERROR] {error_msg}")
                    set_dfm_state('dfm_training_status', f'训练失败: {str(e)}')
                    set_dfm_state('dfm_training_error', error_msg)
                    st_instance.error(f"[ERROR] {error_msg}")
        else:
            st_instance.button("[START] 开始训练",
                             disabled=True,
                             key="dfm_start_training_disabled",
                             help="请先满足所有训练条件",
                             use_container_width=True)

    with col_reset_btn:
        # 重置训练按钮
        if st_instance.button("[LOADING] 重置训练",
                            key="dfm_reset_training_state",
                            help="重置所有训练状态",
                            use_container_width=True):
            set_dfm_state('dfm_force_reset_training_state', True)
            _reset_training_state()

    # 训练日志和结果显示（左右布局）
    col_log_left, col_result_right = st_instance.columns([2, 1])

    with col_log_left:
        st_instance.markdown("**训练日志**")

        training_log = get_dfm_state('dfm_training_log', [])
        current_training_status = get_dfm_state('dfm_training_status', '等待开始')

        if current_training_status == '正在训练...':
            if training_log:
                # 显示最近的日志条目
                log_text = "\n".join(training_log[-20:])  # 只显示最近20条
                st_instance.text_area(
                    "训练日志",
                    value=log_text,
                    height=300,
                    key="dfm_training_log_display",
                    help="显示最近20条训练日志",
                    label_visibility="hidden"
                )
                # 显示训练进度提示
                st_instance.info("[LOADING] 训练正在进行中，日志实时更新...")
            else:
                st_instance.info("[LOADING] 训练正在启动，请稍候...")
        elif training_log:
            # 显示最近的日志条目
            log_text = "\n".join(training_log[-20:])  # 只显示最近20条
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

    with col_result_right:
        st_instance.markdown("**文件下载**")

        training_status = get_dfm_state('dfm_training_status') or '等待开始'
        training_results = get_dfm_state('dfm_model_results_paths')
        
        if not training_results and training_status == '训练完成':
            # 尝试从session_state获取
            training_results = st_instance.session_state.get('dfm_model_results_paths')
            if training_results:
                debug_log("UI状态检查 - 从session_state获取到训练结果", "DEBUG")
                # 同步回状态管理器
                set_dfm_state('dfm_model_results_paths', training_results)

        from dashboard.ui.utils.debug_helpers import debug_log
        debug_log(f"UI状态检查 - 当前训练状态: {training_status}", "DEBUG")
        debug_log(f"UI状态检查 - 结果文件数量: {len(training_results) if training_results else 0}", "DEBUG")

        if training_status == '正在训练...':
            st_instance.info("[LOADING] 模型正在训练中，请耐心等待...")

        elif training_status == '训练完成':
            print(f"[HOT] [UI状态检查] 检测到训练完成状态")
            debug_training_state("训练完成，显示最终结果", show_in_ui=False)

            # 显示训练结果文件（如果有）
            if training_results:
                print(f"[HOT] [UI状态检查] 开始处理训练结果，类型: {type(training_results)}")
                print(f"[HOT] [UI状态检查] 训练结果内容: {training_results}")

                if isinstance(training_results, dict) and training_results:
                    print(f"[HOT] [UI状态检查] 处理字典格式结果，包含 {len(training_results)} 个条目")

                    # 只处理joblib和pkl(metadata)文件
                    target_files = ['final_model_joblib', 'metadata']
                    available_files = []

                    for file_key in target_files:
                        file_path = training_results.get(file_key)
                        if file_path and os.path.exists(file_path):
                            file_name = os.path.basename(file_path)
                            available_files.append((file_key, file_path, file_name))
                            print(f"[HOT] [UI状态检查] 文件存在: {file_name}")
                        else:
                            print(f"[HOT] [UI状态检查] 文件不存在或路径为空: {file_key}")

                    if available_files:
                        # 为每个文件创建下载按钮
                        for file_key, file_path, file_name in available_files:
                            try:
                                # 读取文件数据
                                with open(file_path, 'rb') as f:
                                    file_data = f.read()

                                # 确定显示名称
                                if file_key == 'final_model_joblib':
                                    display_name = "模型文件 (joblib)"
                                elif file_key == 'metadata':
                                    display_name = "元数据文件 (pkl)"
                                else:
                                    display_name = file_name

                                # 创建下载按钮
                                st_instance.download_button(
                                    label=display_name,
                                    data=file_data,
                                    file_name=file_name,
                                    mime="application/octet-stream",
                                    key=f"dfm_download_{file_key}"
                                )

                            except Exception as e:
                                st_instance.warning(f"[WARNING] {file_name} 文件读取失败: {e}")
                    else:
                        st_instance.warning("[WARNING] 未找到可用的结果文件")
                else:
                    st_instance.warning("[WARNING] 训练完成但未找到结果文件")

        elif training_status.startswith('训练失败'):
            training_error = get_dfm_state('dfm_training_error')
            st_instance.error(f"[ERROR] {training_status}")
            if training_error:
                st_instance.error(f"错误详情: {training_error}")

        elif training_status == '等待开始':
            st_instance.info("[NONE] 无结果")

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
