# -*- coding: utf-8 -*-
"""
DFM数据准备模块 - 统一API接口

这个模块提供严格的前后端分离接口，所有UI层必须通过这些API与业务逻辑交互。

设计原则：
1. 所有API函数返回标准化的dict格式
2. 完整的类型注解和文档字符串
3. 统一的错误处理和日志记录
4. 不包含任何UI逻辑
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
import tempfile
from datetime import datetime

from dashboard.models.DFM.prep.modules.main_data_processor import prepare_data as _prepare_data_internal
from dashboard.models.DFM.prep.modules.mapping_manager import load_mappings as _load_mappings_internal
from dashboard.models.DFM.prep.modules.stationarity_processor import apply_stationarity_transforms as _apply_transforms_internal

logger = logging.getLogger(__name__)


def prepare_dfm_data(
    uploaded_file: Union[str, Any],
    data_start_date: str = "2010-01-31",
    data_end_date: str = "2025-07-03",
    target_freq: str = "W-FRI",
    target_sheet_name: str = "工业增加值同比增速_月度_同花顺",
    target_variable_name: str = "规模以上工业增加值:当月同比",
    consecutive_nan_threshold: int = 10,
    reference_sheet_name: str = "指标体系",
    reference_column_name: str = "指标名称"
) -> Dict[str, Any]:
    """
    DFM数据准备API - 处理上传的Excel文件并准备用于DFM训练的数据

    这是DFM数据准备的主要API接口，负责：
    1. 加载和验证Excel文件
    2. 数据清理和预处理
    3. 频率对齐（转换为周度数据）
    4. 平稳性处理
    5. 生成变量映射和转换日志

    Args:
        uploaded_file: Excel文件路径（str）或文件对象
        data_start_date: 数据起始日期，格式："YYYY-MM-DD"
        data_end_date: 数据结束日期，格式："YYYY-MM-DD"
        target_freq: 目标频率，默认"W-FRI"（周五结尾的周度数据）
        target_sheet_name: Excel中目标变量所在的工作表名称
        target_variable_name: 目标变量名称
        consecutive_nan_threshold: 允许的最大连续NaN值数量
        reference_sheet_name: 指标映射表的工作表名称
        reference_column_name: 映射表中的参考列名

    Returns:
        dict: {
            'status': str,              # 'success' 或 'error'
            'message': str,             # 处理结果消息
            'data': pd.DataFrame,       # 处理后的周度数据（仅成功时）
            'metadata': {               # 元数据（仅成功时）
                'variable_mapping': Dict,      # 变量到行业的映射
                'transform_log': Dict,         # 转换操作日志
                'removal_log': List[Dict],     # 移除变量日志
                'data_shape': tuple,           # 数据形状 (rows, cols)
                'time_range': tuple,           # 时间范围 (start, end)
                'processing_time': str         # 处理耗时
            }
        }

    Example:
        >>> result = prepare_dfm_data(
        ...     uploaded_file=file_object,
        ...     data_start_date="2015-01-01",
        ...     data_end_date="2024-12-31"
        ... )
        >>> if result['status'] == 'success':
        ...     prepared_data = result['data']
        ...     print(f"数据形状: {result['metadata']['data_shape']}")
    """
    start_time = datetime.now()

    try:
        logger.info("开始DFM数据准备流程")
        logger.info(f"参数: 起始日期={data_start_date}, 结束日期={data_end_date}, 目标频率={target_freq}")

        # 步骤1: 处理文件输入
        excel_path = _handle_file_input(uploaded_file)

        # 步骤2: 调用内部数据准备函数
        logger.info("调用数据处理引擎...")
        processed_data, variable_mapping, transform_log, removal_log = _prepare_data_internal(
            excel_path=excel_path,
            target_freq=target_freq,
            target_sheet_name=target_sheet_name,
            target_variable_name=target_variable_name,
            consecutive_nan_threshold=consecutive_nan_threshold,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            reference_sheet_name=reference_sheet_name,
            reference_column_name=reference_column_name
        )

        # 步骤3: 验证结果
        if processed_data is None or processed_data.empty:
            return {
                'status': 'error',
                'message': '数据处理失败：返回的数据为空',
                'data': None,
                'metadata': None
            }

        # 步骤4: 构建元数据
        processing_time = (datetime.now() - start_time).total_seconds()

        metadata = {
            'variable_mapping': variable_mapping or {},
            'transform_log': transform_log or {},
            'removal_log': removal_log or [],
            'data_shape': processed_data.shape,
            'time_range': (
                str(processed_data.index.min()) if not processed_data.empty else None,
                str(processed_data.index.max()) if not processed_data.empty else None
            ),
            'processing_time': f"{processing_time:.2f}秒",
            'parameters': {
                'data_start_date': data_start_date,
                'data_end_date': data_end_date,
                'target_freq': target_freq,
                'target_variable': target_variable_name,
                'nan_threshold': consecutive_nan_threshold
            }
        }

        logger.info(f"数据准备成功! 形状: {processed_data.shape}, 耗时: {processing_time:.2f}秒")

        return {
            'status': 'success',
            'message': f'数据准备成功! 处理了 {processed_data.shape[0]} 行 × {processed_data.shape[1]} 列数据',
            'data': processed_data,
            'metadata': metadata
        }

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return {
            'status': 'error',
            'message': f'文件未找到: {str(e)}',
            'data': None,
            'metadata': None
        }

    except ValueError as e:
        logger.error(f"参数错误: {e}")
        return {
            'status': 'error',
            'message': f'参数错误: {str(e)}',
            'data': None,
            'metadata': None
        }

    except Exception as e:
        logger.error(f"数据准备过程中发生错误: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'数据准备失败: {str(e)}',
            'data': None,
            'metadata': None
        }


def load_variable_mappings(
    excel_path: Union[str, Any],
    reference_sheet_name: str = "指标体系",
    reference_column_name: str = "指标名称"
) -> Dict[str, Any]:
    """
    加载变量映射配置

    从Excel文件中加载变量到行业的映射关系。

    Args:
        excel_path: Excel文件路径或文件对象
        reference_sheet_name: 映射表的工作表名称
        reference_column_name: 参考列名

    Returns:
        dict: {
            'status': str,
            'message': str,
            'mappings': Dict[str, str]  # 变量名 -> 行业名
        }
    """
    try:
        logger.info("加载变量映射配置...")

        # 处理文件输入
        file_path = _handle_file_input(excel_path)

        # 加载映射
        mappings = _load_mappings_internal(
            excel_path=file_path,
            reference_sheet_name=reference_sheet_name,
            reference_column_name=reference_column_name
        )

        if not mappings:
            return {
                'status': 'warning',
                'message': '未找到有效的变量映射',
                'mappings': {}
            }

        logger.info(f"成功加载 {len(mappings)} 个变量映射")

        return {
            'status': 'success',
            'message': f'成功加载 {len(mappings)} 个变量映射',
            'mappings': mappings
        }

    except Exception as e:
        logger.error(f"加载映射失败: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'加载映射失败: {str(e)}',
            'mappings': {}
        }


def apply_stationarity_transforms(
    data: pd.DataFrame,
    transform_rules: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    对数据应用平稳性转换

    根据提供的规则对DataFrame中的变量应用平稳性转换。
    如果某个变量在规则中找不到，则保留其原始值。

    Args:
        data: 输入数据（DataFrame）
        transform_rules: 转换规则字典 {列名: {'status': '转换类型', ...}}
                        转换类型可以是: 'diff', 'log_diff', 'level'
                        示例: {'GDP': {'status': 'diff'}, 'CPI': {'status': 'level'}}

    Returns:
        dict: {
            'status': str,              # 'success' 或 'error'
            'message': str,             # 处理结果消息
            'data': pd.DataFrame,       # 转换后的数据
            'metadata': {               # 元数据
                'original_shape': tuple,
                'transformed_shape': tuple,
                'applied_rules': int,   # 应用的规则数量
                'transform_summary': Dict  # 转换类型统计
            }
        }

    Example:
        >>> rules = {
        ...     'GDP': {'status': 'diff'},
        ...     'CPI': {'status': 'log_diff'}
        ... }
        >>> result = apply_stationarity_transforms(data, rules)
        >>> if result['status'] == 'success':
        ...     transformed_data = result['data']
    """
    try:
        logger.info("应用平稳性转换...")

        if data is None or data.empty:
            return {
                'status': 'error',
                'message': '输入数据为空',
                'data': None,
                'metadata': None
            }

        original_shape = data.shape

        # 如果没有提供转换规则，保持原始数据
        if not transform_rules:
            logger.warning("未提供转换规则，返回原始数据")
            return {
                'status': 'success',
                'message': '未提供转换规则，保持原始数据',
                'data': data,
                'metadata': {
                    'original_shape': original_shape,
                    'transformed_shape': data.shape,
                    'applied_rules': 0,
                    'transform_summary': {}
                }
            }

        # 应用转换
        transformed_data = _apply_transforms_internal(
            data=data,
            transform_rules=transform_rules
        )

        # 统计转换类型
        transform_summary = {}
        for col, rule in transform_rules.items():
            status = rule.get('status', 'level')
            transform_summary[status] = transform_summary.get(status, 0) + 1

        logger.info(f"平稳性转换完成，应用了 {len(transform_rules)} 个规则")

        return {
            'status': 'success',
            'message': f'平稳性转换完成，应用了 {len(transform_rules)} 个规则',
            'data': transformed_data,
            'metadata': {
                'original_shape': original_shape,
                'transformed_shape': transformed_data.shape,
                'applied_rules': len(transform_rules),
                'transform_summary': transform_summary
            }
        }

    except ValueError as e:
        logger.error(f"平稳性转换参数错误: {e}")
        return {
            'status': 'error',
            'message': f'转换参数错误: {str(e)}',
            'data': data,  # 返回原始数据
            'metadata': None
        }

    except Exception as e:
        logger.error(f"平稳性转换失败: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'平稳性转换失败: {str(e)}',
            'data': data,  # 返回原始数据
            'metadata': None
        }


def validate_preparation_parameters(
    data_start_date: str,
    data_end_date: str,
    target_freq: str
) -> Dict[str, Any]:
    """
    验证数据准备参数

    Args:
        data_start_date: 起始日期
        data_end_date: 结束日期
        target_freq: 目标频率

    Returns:
        dict: {
            'status': str,
            'message': str,
            'is_valid': bool,
            'errors': List[str]
        }
    """
    errors = []

    try:
        # 验证日期格式
        start_dt = pd.to_datetime(data_start_date)
        end_dt = pd.to_datetime(data_end_date)

        # 验证日期逻辑
        if start_dt >= end_dt:
            errors.append(f"起始日期 ({data_start_date}) 必须早于结束日期 ({data_end_date})")

        # 验证频率
        valid_freqs = ['W-FRI', 'W', 'D', 'M', 'Q', 'Y']
        if target_freq not in valid_freqs:
            errors.append(f"不支持的频率 '{target_freq}'，支持的频率: {', '.join(valid_freqs)}")

        if errors:
            return {
                'status': 'error',
                'message': '参数验证失败',
                'is_valid': False,
                'errors': errors
            }

        return {
            'status': 'success',
            'message': '参数验证通过',
            'is_valid': True,
            'errors': []
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'参数验证异常: {str(e)}',
            'is_valid': False,
            'errors': [str(e)]
        }


# 私有辅助函数

def _handle_file_input(file_input: Union[str, Any]) -> str:
    """
    处理文件输入，统一转换为文件路径

    Args:
        file_input: 文件路径（str）或文件对象

    Returns:
        str: 文件路径
    """
    if isinstance(file_input, str):
        # 已经是路径
        return file_input

    elif hasattr(file_input, 'read'):
        # 是文件对象，保存到临时文件
        file_input.seek(0)  # 重置文件指针
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file_input.read())
            tmp_path = tmp_file.name
            logger.debug(f"文件对象已保存到临时文件: {tmp_path}")
            return tmp_path

    elif isinstance(file_input, Path):
        # Path对象
        return str(file_input)

    else:
        raise TypeError(f"不支持的文件输入类型: {type(file_input)}")


# 导出的API函数
__all__ = [
    'prepare_dfm_data',
    'load_variable_mappings',
    'apply_stationarity_transforms',
    'validate_preparation_parameters'
]
