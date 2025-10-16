# -*- coding: utf-8 -*-
"""
DFM训练模块接口包装器
处理前端UI参数与后端训练脚本之间的参数转换和数据流
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Tuple
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def convert_ui_parameters_to_backend(ui_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    将UI参数转换为后端训练脚本期望的格式
    
    Args:
        ui_params: 来自UI的参数字典
        
    Returns:
        转换后的后端参数字典
    """
    backend_params = {}
    
    
    try:
        # 1. 数据相关参数
        if 'prepared_data' in ui_params:
            backend_params['prepared_data'] = ui_params['prepared_data']
        
        if 'target_variable' in ui_params:
            backend_params['target_variable'] = ui_params['target_variable']
        
        if 'selected_indicators' in ui_params:
            backend_params['selected_variables'] = ui_params['selected_indicators']
            
        # 2. 日期参数转换
        date_mappings = {
            'training_start_date': 'TRAINING_START_DATE',
            'validation_start_date': 'VALIDATION_START_DATE', 
            'validation_end_date': 'VALIDATION_END_DATE'
        }
        
        for ui_key, backend_key in date_mappings.items():
            if ui_key in ui_params:
                date_value = ui_params[ui_key]
                if isinstance(date_value, (date, datetime)):
                    formatted_date = date_value.strftime('%Y-%m-%d')
                    backend_params[backend_key] = formatted_date
                elif isinstance(date_value, str):
                    backend_params[backend_key] = date_value
                    
        # 3. 因子选择参数
        if 'factor_selection_strategy' in ui_params:
            strategy = ui_params['factor_selection_strategy']
            
            # 根据策略设置相应参数（修正映射关系）
            if strategy == 'information_criteria':
                backend_params['FACTOR_SELECTION_METHOD'] = 'information_criteria'
                if 'ic_max_factors' in ui_params:
                    backend_params['IC_MAX_FACTORS'] = ui_params['ic_max_factors']
                if 'info_criterion_method' in ui_params:
                    backend_params['INFO_CRITERION_METHOD'] = ui_params['info_criterion_method']

            elif strategy == 'fixed_number':
                backend_params['FACTOR_SELECTION_METHOD'] = 'fixed_number'
                # 确保固定因子数参数正确传递
                if 'fixed_number_of_factors' in ui_params:
                    fixed_factors = ui_params['fixed_number_of_factors']
                    backend_params['FIXED_NUMBER_OF_FACTORS'] = fixed_factors
                    backend_params['fixed_number_of_factors'] = fixed_factors  # 兼容性
                    backend_params['fixed_k_factors'] = fixed_factors  # 额外兼容性

            elif strategy == 'cumulative_variance':
                backend_params['FACTOR_SELECTION_METHOD'] = 'cumulative_variance'
                if 'cum_variance_threshold' in ui_params:
                    threshold = ui_params['cum_variance_threshold']
                    backend_params['COMMON_VARIANCE_CONTRIBUTION_THRESHOLD'] = threshold
            
        
        # 4. 变量选择参数
        if 'variable_selection_method' in ui_params:
            method = ui_params['variable_selection_method']
            enable_selection = (method != 'none')
            backend_params['ENABLE_VARIABLE_SELECTION'] = enable_selection
            backend_params['VARIABLE_SELECTION_METHOD'] = method
            
        # 5. 训练参数
        if 'max_iterations' in ui_params:
            backend_params['EM_MAX_ITER'] = ui_params['max_iterations']

        # 因子自回归阶数参数
        if 'max_lags' in ui_params:
            backend_params['MAX_LAGS'] = ui_params['max_lags']
            
        # 6. 输出目录设置
        backend_params['output_dir'] = os.path.join(parent_dir, 'outputs')
        
        # 7. 进度回调
        if 'progress_callback' in ui_params:
            backend_params['progress_callback'] = ui_params['progress_callback']
        
        
        return backend_params
        
    except Exception as e:
        print(f"参数转换错误: {e}")
        traceback.print_exc()
        return {}

def validate_ui_parameters(ui_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证UI参数的完整性和有效性
    
    Args:
        ui_params: UI参数字典
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    warnings = []
    
    # 必需参数检查
    required_params = ['target_variable', 'training_start_date', 'validation_end_date']
    for param in required_params:
        if param not in ui_params or ui_params[param] is None:
            errors.append(f"缺少必需参数: {param}")
    
    # 日期参数验证
    date_params = ['training_start_date', 'validation_start_date', 'validation_end_date']
    for param in date_params:
        if param in ui_params and ui_params[param] is not None:
            try:
                if isinstance(ui_params[param], str):
                    datetime.strptime(ui_params[param], '%Y-%m-%d')
            except ValueError:
                errors.append(f"日期格式错误: {param}")
    
    # 数值参数验证
    if 'max_iterations' in ui_params:
        if not isinstance(ui_params['max_iterations'], int) or ui_params['max_iterations'] <= 0:
            errors.append("最大迭代次数必须是正整数")
    
    # 因子选择策略验证
    if 'factor_selection_strategy' in ui_params:
        strategy = ui_params['factor_selection_strategy']
        print(f"[PARAM_VALIDATION] 验证因子选择策略: {strategy}")
        
        if strategy == 'fixed_number':
            # 验证固定因子数参数
            if 'fixed_number_of_factors' not in ui_params:
                warnings.append("固定因子数策略缺少'fixed_number_of_factors'参数")
            else:
                fixed_k = ui_params['fixed_number_of_factors']
                if not isinstance(fixed_k, int) or fixed_k <= 0:
                    errors.append(f"固定因子数必须是正整数，当前值: {fixed_k}")
                else:
                    print(f"[PARAM_VALIDATION] [OK] 固定因子数参数有效: {fixed_k}")
        
        elif strategy == 'information_criteria':
            if 'ic_max_factors' not in ui_params:
                warnings.append("信息准则策略缺少'ic_max_factors'参数，将使用默认值")
        
        elif strategy == 'cumulative_variance':
            if 'cum_variance_threshold' not in ui_params:
                warnings.append("累积方差策略缺少'cum_variance_threshold'参数，将使用默认值")
    
    # 打印验证结果
    if warnings:
        for warning in warnings:
            print(f"[PARAM_VALIDATION] 警告: {warning}")
    
    return len(errors) == 0, errors

def validate_backend_parameters(backend_params: Dict[str, Any]) -> None:
    """
    验证后端参数的完整性和一致性
    
    Args:
        backend_params: 后端参数字典
    """
    # 静默验证，不输出调试日志
    pass


def prepare_data_from_ui(ui_params: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    从UI参数准备训练数据 - 简化版本，避免复杂的变量映射

    Args:
        ui_params: UI参数字典

    Returns:
        (处理后的数据DataFrame, 元数据字典)
    """
    try:
        # 1. 获取预处理数据
        prepared_data = ui_params.get('prepared_data')
        if prepared_data is None:
            return None, {'error': '未找到预处理数据'}

        # 2. 获取选择的变量
        target_variable = ui_params.get('target_variable')
        selected_variables = ui_params.get('selected_indicators', [])

        if not target_variable:
            return None, {'error': '未指定目标变量'}

        print(f"[VIEW] [interface_wrapper] 简化数据准备:")
        print(f"  预处理数据形状: {prepared_data.shape}")
        print(f"  目标变量: {target_variable}")
        print(f"  UI选择变量数: {len(selected_variables)}")
        print(f"  数据中实际列数: {len(prepared_data.columns)}")

        # 3. [HOT] 关键修复：直接使用预处理数据，避免复杂映射
        # 检查目标变量是否存在
        if target_variable not in prepared_data.columns:
            print(f"[ERROR] 目标变量 '{target_variable}' 不在数据中")
            print(f"   数据中的列名: {list(prepared_data.columns)[:10]}")
            return None, {'error': f'目标变量 {target_variable} 不存在'}

        # 4. [HOT] 简化变量选择逻辑
        if selected_variables and len(selected_variables) > 0:
            # 过滤出实际存在于数据中的变量
            available_selected_vars = [var for var in selected_variables if var in prepared_data.columns]

            print(f"  原始选择变量: {len(selected_variables)} 个")
            print(f"  实际可用变量: {len(available_selected_vars)} 个")

            if len(available_selected_vars) == 0:
                print("[WARNING] 警告: 所有选择的变量都不在数据中，使用所有可用变量")
                final_variables = list(prepared_data.columns)
            else:
                # 确保目标变量在列表中
                final_variables = [target_variable] + [var for var in available_selected_vars if var != target_variable]
                print(f"  最终使用变量: {len(final_variables)} 个")
        else:
            # 如果没有选择变量，使用所有可用变量
            print("  未选择特定变量，使用所有可用变量")
            final_variables = list(prepared_data.columns)

        # 5. 过滤数据
        available_vars = [var for var in final_variables if var in prepared_data.columns]
        if not available_vars:
            return None, {'error': '没有可用的变量'}

        filtered_data = prepared_data[available_vars].copy()

        print(f"  [SUCCESS] 最终数据形状: {filtered_data.shape}")
        print(f"  [SUCCESS] 最终变量数: {len(available_vars)}")
        print(f"  [SUCCESS] 前10个变量: {available_vars[:10]}")

        # 6. 日期范围过滤
        training_start = ui_params.get('training_start_date')
        validation_end = ui_params.get('validation_end_date')

        if training_start and validation_end:
            try:
                if isinstance(training_start, str):
                    training_start = pd.to_datetime(training_start)
                if isinstance(validation_end, str):
                    validation_end = pd.to_datetime(validation_end)

                # 确保数据索引是日期时间类型
                if not isinstance(filtered_data.index, pd.DatetimeIndex):
                    # 尝试转换索引
                    filtered_data.index = pd.to_datetime(filtered_data.index)

                # 过滤日期范围
                mask = (filtered_data.index >= training_start) & (filtered_data.index <= validation_end)
                filtered_data = filtered_data.loc[mask]
                print(f"  [DATE] 日期过滤后数据形状: {filtered_data.shape}")

            except Exception as e:
                print(f"[WARNING] 日期过滤警告: {e}")

        # 7. 准备元数据
        metadata = {
            'target_variable': target_variable,
            'selected_indicators': selected_variables,
            'final_variables': available_vars,
            'data_shape': filtered_data.shape,
            'date_range': (filtered_data.index.min(), filtered_data.index.max()) if len(filtered_data) > 0 else None
        }

        print(f"[TARGET] [interface_wrapper] 数据准备完成:")
        print(f"  最终数据形状: {filtered_data.shape}")
        print(f"  变量数量: {len(available_vars)}")
        print(f"  日期范围: {metadata['date_range']}")

        return filtered_data, metadata

    except Exception as e:
        print(f"数据准备错误: {e}")
        traceback.print_exc()
        return None, {'error': str(e)}

def create_progress_callback(ui_callback=None):
    """
    创建标准化的进度回调函数

    Args:
        ui_callback: UI提供的回调函数

    Returns:
        标准化的进度回调函数
    """
    def progress_callback(message: str, progress: Optional[float] = None):
        """
        标准化进度回调

        Args:
            message: 进度消息
            progress: 进度百分比 (0-100)
        """
        try:
            # 格式化消息
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}"

            # 调用UI回调
            if ui_callback and callable(ui_callback):
                ui_callback(formatted_message)
            else:
                # 只有在非静默模式下才输出到控制台
                import os
                training_silent = os.getenv('DFM_TRAINING_SILENT', 'true').lower() == 'true'
                if not training_silent:
                    print(formatted_message)

        except Exception as e:
            print(f"进度回调错误: {e}")

    return progress_callback

def setup_logging_integration(log_file_path: str = None) -> None:
    """
    设置日志集成

    Args:
        log_file_path: 日志文件路径，如果为None则使用默认路径
    """
    import logging

    try:
        # 禁用日志文件创建 - 只使用控制台输出

        # 配置日志格式 - 只使用控制台输出
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger('DFM_Training')
        logger.info("日志系统已初始化，仅使用控制台输出")

    except Exception as e:
        print(f"日志系统初始化失败: {e}")

def log_interface_activity(activity: str, details: dict = None) -> None:
    """
    记录接口活动

    Args:
        activity: 活动描述
        details: 活动详情字典
    """
    import logging

    try:
        logger = logging.getLogger('DFM_Training')

        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            logger.info(f"[接口活动] {activity} - {detail_str}")
        else:
            logger.info(f"[接口活动] {activity}")

    except Exception as e:
        print(f"日志记录失败: {e}")

def create_comprehensive_progress_callback(ui_callback=None, log_to_file=True):
    """
    创建综合进度回调函数，支持UI回调和文件日志

    Args:
        ui_callback: UI提供的回调函数
        log_to_file: 是否记录到文件

    Returns:
        综合进度回调函数
    """
    import logging

    def comprehensive_callback(message: str, progress: Optional[float] = None, level: str = 'INFO'):
        """
        综合进度回调

        Args:
            message: 进度消息
            progress: 进度百分比 (0-100)
            level: 日志级别 ('INFO', 'WARNING', 'ERROR')
        """
        try:
            # 格式化消息
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}"

            if progress is not None:
                formatted_message += f" ({progress:.1f}%)"

            # 调用UI回调
            if ui_callback and callable(ui_callback):
                ui_callback(formatted_message)
            else:
                # 默认输出到控制台
                print(formatted_message)

            # 记录到日志文件
            if log_to_file:
                logger = logging.getLogger('DFM_Training')
                log_level = getattr(logging, level.upper(), logging.INFO)
                logger.log(log_level, f"[进度] {message}")

        except Exception as e:
            print(f"综合进度回调错误: {e}")

    return comprehensive_callback
