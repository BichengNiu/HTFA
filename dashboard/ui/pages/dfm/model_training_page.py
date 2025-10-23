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

# 添加路径以导入统一状态管理
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if dashboard_root not in sys.path:
    sys.path.insert(0, dashboard_root)

# 导入统一状态管理
from dashboard.core import get_global_dfm_manager
import logging

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

# 3. 导入DFM训练脚本 - 使用train_ref重构版本
# 添加DFM train_ref模块路径
dfm_train_ref_dir = os.path.join(dashboard_root, 'dashboard', 'DFM', 'train_ref')
if dfm_train_ref_dir not in sys.path:
    sys.path.insert(0, dfm_train_ref_dir)

try:
    # 导入train_ref模块
    from dashboard.DFM.train_ref.training import DFMTrainer, TrainingConfig
    from dashboard.DFM.train_ref.training.trainer import TrainingResult
    print("[SUCCESS] 成功导入DFMTrainer和TrainingConfig from train_ref")
    _TRAIN_UI_IMPORT_ERROR_MESSAGE = None
except ImportError as e:
    print(f"[ERROR] 导入train_ref模块失败: {e}")
    _TRAIN_UI_IMPORT_ERROR_MESSAGE = f"train_ref module import error: {e}"
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


def load_mappings_from_unified_state(available_data_columns=None):
    """
    从统一状态管理器中获取映射数据构建行业到指标的映射
    优先使用数据准备模块已经加载好的映射，避免重复读取Excel文件

    Args:
        available_data_columns: 实际数据中可用的列名列表（用于过滤）

    Returns:
        tuple: (unique_industries, industry_to_indicators_map, all_indicators_flat)
               如果没有找到映射数据，返回 (None, None, None)
    """
    try:
        var_industry_map = get_dfm_state('dfm_industry_map_obj', None)

        if var_industry_map is None:
            print("[WARNING] [映射加载] 统一状态管理器中未找到行业映射数据")
            return None, None, None

        if not var_industry_map:
            print("[WARNING] [映射加载] 行业映射数据为空")
            return None, None, None

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

        return unique_industries, industry_to_indicators_map, all_indicators_flat

    except Exception as e:
        print(f"[ERROR] [映射加载] 从统一状态管理器加载映射数据失败: {e}")
        return None, None, None


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
            # 从统一状态管理中删除
            set_dfm_state(key, None)
            debug_log(f"状态重置 - 已清理统一状态管理器键: {key}", "DEBUG")

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


def get_dfm_manager():
    """获取DFM模块管理器实例（使用全局单例）"""
    try:
        dfm_manager = get_global_dfm_manager()
        if dfm_manager is None:
            raise RuntimeError("全局DFM管理器不可用")

        return dfm_manager
    except Exception as e:
        raise RuntimeError(f"DFM管理器初始化失败: {e}")


def get_dfm_state(key, default=None):
    """获取DFM状态值（使用统一状态管理）"""
    try:
        # 导入调试工具
        from dashboard.ui.utils.debug_helpers import debug_log

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

        if key in training_keys:
            from dashboard.core import get_global_dfm_manager
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                # 验证使用的是正确的UnifiedStateManager（仅在调试模式下输出）
                if hasattr(dfm_manager, 'unified_manager'):
                    debug_log(f"前端状态读取 - UnifiedStateManager类型: {type(dfm_manager.unified_manager)}", "DEBUG")

                value = dfm_manager.get_dfm_state('train_model', key, default)

                # 详细的状态读取日志（仅在调试模式下输出）
                debug_log(f"前端状态读取 - 键: {key}, 值: {value}, 类型: {type(value).__name__}", "DEBUG")
                debug_log(f"前端状态读取 - DFM管理器类型: {type(dfm_manager)}", "DEBUG")

                return value
            else:
                debug_log(f"警告 - DFM管理器不可用，键: {key}", "WARNING")
                return default

        # 数据相关的键从data_prep命名空间获取
        dfm_manager = get_dfm_manager()
        if dfm_manager:
            data_keys = [
                'dfm_prepared_data_df',
                'dfm_transform_log_obj',
                'dfm_industry_map_obj',
                'dfm_removed_vars_log_obj',
                # 移除 'dfm_var_type_map_obj'
                'dfm_param_data_start_date',
                'dfm_param_data_end_date'
            ]

            if key in data_keys:
                return dfm_manager.get_dfm_state('data_prep', key, default)

            # 其他键从train_model命名空间获取
            return dfm_manager.get_dfm_state('train_model', key, default)
        else:
            return default
    except Exception as e:
        from dashboard.ui.utils.debug_helpers import debug_log
        debug_log(f"状态读取 - 异常 - 键: {key}, 错误: {str(e)}", "ERROR")
        return default


def set_dfm_state(key, value):
    """设置DFM状态值（修复缓存不一致问题）"""
    try:
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

        if key in training_keys:
            from dashboard.core import get_global_dfm_manager
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                success = dfm_manager.set_dfm_state('train_model', key, value)

                from dashboard.ui.utils.debug_helpers import debug_log
                debug_mode = dfm_manager.get_dfm_state('train_model', 'dfm_debug_mode', False)
                if debug_mode:
                    debug_log(f"统一状态设置 - 键: {key}, 值类型: {type(value)}, 成功: {success}", "DEBUG")

                # 同时尝试组件化方法，确保双重写入
                try:
                    training_component = get_training_status_component()
                    training_component._set_state(key, value)
                    if debug_mode:
                        debug_log(f"组件状态设置 - 键: {key}, 值类型: {type(value)}, 成功: True", "DEBUG")
                except Exception as e:
                    if debug_mode:
                        debug_log(f"组件状态设置失败 - 键: {key}, 错误: {str(e)}", "ERROR")

                return success
            else:
                debug_log(f"警告 - DFM统一状态管理器不可用，无法设置键: {key}", "WARNING")
                return False

        # 其他状态使用原有方法
        from dashboard.ui.utils.debug_helpers import debug_log
        dfm_manager = get_dfm_manager()
        if dfm_manager:
            success = dfm_manager.set_dfm_state('train_model', key, value)
            debug_log(f"状态设置 - 键: {key}, 值类型: {type(value)}, 成功: {success}", "DEBUG")
            return success
        else:
            debug_log(f"状态设置 - 失败 - DFM管理器不可用, 键: {key}", "WARNING")
            return False
    except Exception as e:
        debug_log(f"状态设置 - 异常 - 键: {key}, 错误: {str(e)}", "ERROR")
        import traceback
        debug_log(f"状态设置 - 异常堆栈: {traceback.format_exc()}", "ERROR")
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


def render_dfm_train_model_tab(st_instance):

    # 确保datetime在函数开头就可用
    from datetime import datetime
    import time


    cleanup_expired_downloads()

    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        if "train_ref" in _TRAIN_UI_IMPORT_ERROR_MESSAGE:
            st_instance.error(f"关键模块导入错误，模型训练功能不可用:\n{_TRAIN_UI_IMPORT_ERROR_MESSAGE}")
            return  # 如果训练模块不可用，直接返回
        else:
            # 如果只是数据准备模块的导入问题，显示警告但继续
            st_instance.warning("[WARNING] 数据准备模块导入警告，但映射数据传递已修复，功能应该正常")
    else:
        # 如果没有错误消息，显示成功信息
        st_instance.success("[SUCCESS] 所有必需模块已成功加载(使用train_ref)，模型训练功能可用")

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

    # 检测已有结果并更新状态（兼容新旧状态管理）
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

            # 清除刷新标志（使用统一状态管理）
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

    input_df = get_dfm_state('dfm_prepared_data_df')

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


    # 1. 首先尝试从统一状态管理器中获取已经准备好的映射数据
    available_data_columns = list(input_df.columns) if input_df is not None else None


    # 优先使用统一状态管理器中的映射数据
    map_data = load_mappings_from_unified_state(available_data_columns)

    if map_data and all(x is not None for x in map_data):
        unique_industries, var_to_indicators_map_by_industry, _ = map_data

        # 修复：显示实际可用的指标数量
        actual_indicator_count = sum(len(v) for v in var_to_indicators_map_by_industry.values())
        st_instance.success(f"[SUCCESS] 已加载映射数据：{len(unique_industries)} 个行业，{actual_indicator_count} 个可用指标")
    else:
        # 不再回退到Excel文件，使用空映射继续（数据准备模块应该正确保存映射）

        st_instance.warning("[WARNING] 未找到映射数据，请确保已在'数据准备'模块正确处理数据")
        unique_industries = []
        var_to_indicators_map_by_industry = {}

    # 主布局：现在是上下结构，不再使用列
    # REMOVED: var_selection_col, param_col = st_instance.columns([1, 1.5])


    # 1. 选择目标变量（兼容新旧状态管理）
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
            "**选择目标变量**",
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

            # 默认选中该行业下的所有指标
            default_selection_for_industry = current_selection.get(
                industry_name,
                indicators_for_this_industry # 默认全选
            )
            # 确保默认值是实际可选列表的子集
            valid_default = [item for item in default_selection_for_industry if item in indicators_for_this_industry]
            if not valid_default and indicators_for_this_industry: # 如果之前存的默认值无效了，且当前有可选指标，则全选
                valid_default = indicators_for_this_industry

            # 移除回调函数，改为直接逻辑处理

            # 取消全选复选框，使用key确保状态追踪
            deselect_all_checked = st_instance.checkbox(
                f"取消全选 {industry_name} 指标",
                key=f"dfm_deselect_all_indicators_{industry_name}",
                help=f"勾选此框将取消所有已为 '{industry_name}' 选中的指标。"
            )

            # 如果取消全选被勾选，清空该行业的选择
            if deselect_all_checked:
                valid_default = []

            selected_in_widget = st_instance.multiselect(
                f"为 '{industry_name}' 选择指标",
                options=indicators_for_this_industry,
                default=valid_default,
                help=f"从 {industry_name} 行业中选择预测指标。"
            )

            current_selection[industry_name] = selected_in_widget
            final_selected_indicators_flat.extend(selected_in_widget)

        industries_to_remove_from_state = [
            ind for ind in current_selection
            if ind not in current_selected_industries
        ]
        for ind_to_remove in industries_to_remove_from_state:
            del current_selection[ind_to_remove]

        set_dfm_state('dfm_selected_indicators_per_industry', current_selection)

    # 更新最终的扁平化预测指标列表 (去重)
    final_indicators = sorted(list(set(final_selected_indicators_flat)))
    set_dfm_state('dfm_selected_indicators', final_indicators)

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
    st_instance.text(f" - 目标变量: {current_target_var if current_target_var else '未选择'}")
    st_instance.text(f" - 选定行业数: {len(current_selected_industries)}")
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
                'training_start': data_prep_start if data_prep_start else datetime(today.year - 5, 1, 1).date(),
                'validation_start': datetime(2024, 7, 1).date(),  # 2024年7月1日
                'validation_end': datetime(2024, 12, 31).date()  # [HOT] 修复：验证期结束于2024年12月31日
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

                    if data_prep_start:
                        training_start_date = data_prep_start
                    elif data_first_date:
                        # 确保不早于合理的历史范围（2020年）
                        reasonable_start = datetime(2020, 1, 1).date()
                        training_start_date = max(data_first_date, reasonable_start)
                        print(f"[WARNING] [日期回退] 数据准备页面未设置，使用数据文件日期（限制在2020年后）: {training_start_date}")
                        print(f"   原始数据开始日期: {data_first_date}")
                    else:
                        training_start_date = datetime(2020, 1, 1).date()
                        print(f"[WARNING] [日期默认] 使用硬编码默认值: {training_start_date}")

                    # 计算验证期开始日期：使用数据时间范围的80%作为训练期
                    if data_prep_start and data_prep_end:
                        # 如果数据准备页面设置了边界，基于边界计算
                        total_days = (data_prep_end - data_prep_start).days
                        training_days = int(total_days * 0.8)
                        validation_start_date = data_prep_start + timedelta(days=training_days)
                    else:
                        # 否则基于实际数据计算
                        total_days = (data_last_date - data_first_date).days
                        training_days = int(total_days * 0.8)
                        validation_start_date = data_first_date + timedelta(days=training_days)

                    # 确保验证期开始日期不是未来日期
                    if validation_start_date > today:
                        validation_start_date = today - timedelta(days=30)  # 1个月前

                    # 验证期用于测试模型性能，必须使用历史数据
                    validation_end_date = datetime(2024, 12, 31).date()  # [HOT] 强制使用2024年底作为验证期结束

                    # 验证日期逻辑的合理性
                    if validation_start_date >= validation_end_date:
                        # 如果验证期开始晚于或等于结束，重新计算
                        validation_end_date = datetime(2024, 12, 31).date()  # [HOT] 强制使用2024年底
                        validation_start_date = validation_end_date - timedelta(days=90)  # 验证期3个月

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

                # 强制更新统一状态管理器中的日期默认值（检查是否为静态默认值或与数据不匹配）
                current_training_start = get_dfm_state('dfm_training_start_date')
                if (current_training_start == datetime(2010, 1, 1).date() or
                    current_training_start is None or
                    current_training_start != actual_data_start):
                    set_dfm_state('dfm_training_start_date', date_defaults['training_start'])

                current_validation_start = get_dfm_state('dfm_validation_start_date')
                if (current_validation_start == datetime(2020, 12, 31).date() or
                    current_validation_start is None):
                    set_dfm_state('dfm_validation_start_date', date_defaults['validation_start'])

                current_validation_end = get_dfm_state('dfm_validation_end_date')
                if (current_validation_end == datetime(2022, 12, 31).date() or
                    current_validation_end is None):
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
                'information_criteria': "信息准则",
                'fixed_number': "固定因子数",
                'cumulative_variance': "累积方差贡献"
            }
            default_strategy = 'information_criteria'

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
        if strategy_value == 'information_criteria':
            # 信息准则方法
            if CONFIG_AVAILABLE:
                ic_options = UIDefaults.INFORMATION_CRITERION_OPTIONS
                default_ic = TrainDefaults.INFORMATION_CRITERION
            else:
                ic_options = {
                    'bic': "BIC",
                    'aic': "AIC",
                    'hqc': "HQC"
                }
                default_ic = 'bic'

            current_ic = get_dfm_state('dfm_information_criterion', default_ic)

            ic_value = st_instance.selectbox(
                "信息准则方法",
                options=list(ic_options.keys()),
                format_func=lambda x: ic_options[x],
                index=list(ic_options.keys()).index(current_ic),
                key='dfm_information_criterion_input',
                help="选择信息准则类型"
            )
            set_dfm_state('dfm_information_criterion', ic_value)

            # IC最大因子数
            if CONFIG_AVAILABLE:
                default_ic_max = UIDefaults.IC_MAX_FACTORS_DEFAULT
            else:
                default_ic_max = 10

            ic_max_value = st_instance.number_input(
                "IC最大因子数",
                min_value=1,
                max_value=20,
                value=get_dfm_state('dfm_ic_max_factors', default_ic_max),
                step=1,
                key='dfm_ic_max_factors_input',
                help="信息准则搜索的最大因子数"
            )
            set_dfm_state('dfm_ic_max_factors', ic_max_value)

        elif strategy_value == 'fixed_number':
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
                    # 使用train_ref进行训练
                    import tempfile
                    from datetime import timedelta

                    # 获取行业映射数据（用于后续分析）
                    var_industry_map = get_dfm_state('dfm_industry_map_obj', {})
                    if not var_industry_map:
                        st.warning("[WARNING] 行业映射数据为空，Factor-Industry R² 将无法计算")

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

                    # 映射UI的变量选择方法到train_ref的变量选择方法
                    # UI使用 'global_backward'，train_ref使用 'backward'
                    var_selection_method_map = {
                        'none': 'none',
                        'global_backward': 'backward',
                        'global_forward': 'forward',
                        'backward': 'backward',
                        'forward': 'forward'
                    }
                    mapped_var_selection_method = var_selection_method_map.get(var_selection_method, 'none')

                    # 获取因子选择策略
                    factor_strategy = get_dfm_state('dfm_factor_selection_strategy', 'information_criteria')

                    # 映射factor_selection_strategy到train_ref的factor_selection_method
                    if factor_strategy == 'information_criteria':
                        factor_selection_method = 'fixed'  # IC方法最终也是确定固定因子数
                        k_factors = get_dfm_state('dfm_ic_max_factors', 10)  # 先用最大值，后续可优化
                    elif factor_strategy == 'fixed_number':
                        factor_selection_method = 'fixed'
                        k_factors = get_dfm_state('dfm_fixed_number_of_factors', 3)
                        st.info(f"使用固定因子数策略，因子数：{k_factors}")
                    elif factor_strategy == 'cumulative_variance':
                        factor_selection_method = 'cumulative'
                        k_factors = 4  # 默认值，实际会通过PCA确定
                    else:
                        factor_selection_method = 'fixed'
                        k_factors = 3

                    # 保存DataFrame到临时文件（TrainingConfig需要文件路径）
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
                    temp_data_path = temp_file.name
                    temp_file.close()
                    input_df.to_csv(temp_data_path)
                    print(f"[TRAIN_REF] 临时数据文件: {temp_data_path}")

                    # 构建TrainingConfig
                    training_config = TrainingConfig(
                        # 核心配置
                        data_path=temp_data_path,
                        target_variable=current_target_var,
                        selected_indicators=current_selected_indicators,

                        # 训练/验证期配置
                        train_start=training_start_value.strftime('%Y-%m-%d') if training_start_value else None,
                        train_end=train_end_date.strftime('%Y-%m-%d') if train_end_date else None,
                        validation_start=validation_start_value.strftime('%Y-%m-%d') if validation_start_value else None,
                        validation_end=validation_end_value.strftime('%Y-%m-%d') if validation_end_value else None,
                        target_freq='W-FRI',

                        # 模型参数
                        k_factors=k_factors,
                        max_iterations=get_dfm_state('dfm_max_iter', 30),
                        max_lags=get_dfm_state('dfm_factor_ar_order', 1),
                        tolerance=1e-6,

                        # 变量选择配置
                        enable_variable_selection=enable_var_selection,
                        variable_selection_method=mapped_var_selection_method if enable_var_selection else 'none',

                        # 因子数选择配置
                        factor_selection_method=factor_selection_method,
                        pca_threshold=get_dfm_state('dfm_cumulative_variance_threshold', 0.9) if factor_strategy == 'cumulative_variance' else 0.9,

                        # 优化配置
                        use_cache=False,
                        use_precompute=False
                    )

                    print(f"[TRAIN_REF] 训练配置: {training_config}")

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
                    result: TrainingResult = trainer.train(progress_callback=progress_callback)

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
                    set_dfm_state('dfm_training_status', '训练完成')
                    set_dfm_state('dfm_training_completed_timestamp', time.time())

                    # 添加完成日志
                    training_log = get_dfm_state('dfm_training_log', [])
                    training_log.append(f"[SUCCESS] 训练完成！耗时: {result.training_time:.2f}秒")
                    training_log.append(f"[RESULT] 选中变量数: {len(result.selected_variables)}")
                    training_log.append(f"[RESULT] 因子数: {result.k_factors}")
                    if result.metrics:
                        training_log.append(f"[METRICS] 样本外RMSE: {result.metrics.oos_rmse:.4f}")
                        training_log.append(f"[METRICS] 样本外Hit Rate: {result.metrics.oos_hit_rate:.2f}%")
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

            # 显示训练结果摘要（train_ref版本）
            training_result_summary = get_dfm_state('dfm_training_result')
            if training_result_summary:
                st_instance.success("[SUCCESS] 训练已完成")
                st_instance.markdown("**训练结果摘要:**")

                # 显示关键指标
                col1, col2 = st_instance.columns(2)
                with col1:
                    st_instance.metric("选中变量数", len(training_result_summary.get('selected_variables', [])))
                    st_instance.metric("因子数", training_result_summary.get('k_factors', 'N/A'))
                with col2:
                    metrics = training_result_summary.get('metrics', {})
                    if metrics.get('oos_rmse'):
                        st_instance.metric("样本外RMSE", f"{metrics['oos_rmse']:.4f}")
                    if metrics.get('oos_hit_rate'):
                        st_instance.metric("样本外Hit Rate", f"{metrics['oos_hit_rate']:.2f}%")

                st_instance.metric("训练耗时", f"{training_result_summary.get('training_time', 0):.2f}秒")

                # 显示选中的变量列表
                with st_instance.expander("查看选中的变量"):
                    selected_vars = training_result_summary.get('selected_variables', [])
                    if selected_vars:
                        for i, var in enumerate(selected_vars, 1):
                            st_instance.text(f"{i}. {var}")
                    else:
                        st_instance.info("无变量选择")

            # 显示训练结果文件（如果有）
            elif training_results:
                st_instance.success("[SUCCESS] 训练已完成")
                print(f"[HOT] [UI状态检查] 开始处理训练结果，类型: {type(training_results)}")
                print(f"[HOT] [UI状态检查] 训练结果内容: {training_results}")

                if isinstance(training_results, dict) and training_results:
                    st_instance.markdown("**生成的文件:**")
                    print(f"[HOT] [UI状态检查] 处理字典格式结果，包含 {len(training_results)} 个条目")

                    # 显示文件信息
                    file_count = 0
                    available_files = []
                    for file_key, file_path in training_results.items():
                        print(f"[HOT] [UI状态检查] 检查文件: {file_key} -> {file_path}")
                        if file_path and os.path.exists(file_path):
                            file_count += 1
                            file_name = os.path.basename(file_path)
                            file_size = _get_file_size(file_path)
                            st_instance.write(f"{file_count}. {file_name} ({file_size})")
                            available_files.append((file_key, file_path, file_name))
                            print(f"[HOT] [UI状态检查] 文件存在: {file_name}")
                        else:
                            print(f"[HOT] [UI状态检查] 文件不存在或路径为空: {file_path}")

                    if available_files:
                        st_instance.info(f"[DATA] 共生成 {len(available_files)} 个文件")

                        # 为每个文件创建下载按钮
                        for file_key, file_path, file_name in available_files:
                            try:
                                # 读取文件数据
                                with open(file_path, 'rb') as f:
                                    file_data = f.read()

                                # 确定MIME类型和显示名称
                                if file_key == 'final_model_joblib':
                                    display_name = "[PACKAGE] 模型文件"
                                    mime_type = "application/octet-stream"
                                elif file_key == 'metadata':
                                    display_name = "元数据文件"
                                    mime_type = "application/octet-stream"
                                elif file_key == 'excel_report':
                                    display_name = "[DATA] Excel报告"
                                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                else:
                                    display_name = f"{file_key}"
                                    mime_type = "application/octet-stream"

                                # 创建下载按钮
                                st_instance.download_button(
                                    label=display_name,
                                    data=file_data,
                                    file_name=file_name,
                                    mime=mime_type,
                                    key=f"dfm_download_{file_key}",
                                    use_container_width=True
                                )

                            except Exception as e:
                                st_instance.warning(f"[WARNING] {file_name} 文件读取失败: {e}")
                    else:
                        st_instance.warning("[WARNING] 未找到可用的结果文件")

                elif isinstance(training_results, list) and training_results:
                    # 兼容旧的列表格式
                    st_instance.markdown("**生成的文件:**")
                    for i, file_path in enumerate(training_results, 1):
                        st_instance.write(f"{i}. {file_path}")
                    st_instance.info("[INFO] 文件路径已显示，请手动复制")
                else:
                    st_instance.warning("[WARNING] 训练完成但未找到结果文件")

        elif training_status.startswith('训练失败'):
            training_error = get_dfm_state('dfm_training_error')
            st_instance.error(f"[ERROR] {training_status}")
            if training_error:
                st_instance.error(f"错误详情: {training_error}")

        elif training_status == '等待开始':
            st_instance.info("[NONE] 无结果")


def render_dfm_model_training_page(st_module: Any) -> Dict[str, Any]:
    """
    渲染DFM模型训练页面

    Args:
        st_module: Streamlit模块

    Returns:
        Dict[str, Any]: 渲染结果
    """
    try:
        render_dfm_train_model_tab(st_module)

        return {
            'status': 'success',
            'page': 'model_training',
            'components': ['variable_selection', 'date_range', 'model_parameters', 'training_status']
        }

    except Exception as e:
        st_module.error(f"模型训练页面渲染失败: {str(e)}")
        return {
            'status': 'error',
            'page': 'model_training',
            'error': str(e)
        }


# 原来的render_dfm_train_model_tab函数已经在第122行定义，这里不需要重复定义

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
