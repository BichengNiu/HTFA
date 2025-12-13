"""
UI后端服务模块

为UI层提供所有必需的业务逻辑接口，UI层不应包含任何业务逻辑，只负责显示
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd

from dashboard.models.DFM.prep.modules.variable_transformer import VariableTransformer
from dashboard.models.DFM.prep.utils.stationarity_checker import StationarityChecker
from dashboard.models.DFM.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class UIBackendService:
    """UI后端服务 - 提供UI需要的所有业务逻辑"""

    @staticmethod
    def transform_and_check_stationarity(
        data: pd.DataFrame,
        transform_config: List[Dict[str, Any]],
        target_freq: str = 'W-FRI',
        var_frequency_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        执行变量转换并检验转换后数据的平稳性

        这是UI层应该调用的唯一业务逻辑接口

        Args:
            data: 原始数据DataFrame
            transform_config: 转换配置列表，每项为：
                {
                    'variable': str,        # 变量名
                    'operation': str,       # 操作: 'none', 'log', 'diff_1', 'diff_yoy'
                    'zero_handling': str,   # 零值处理
                    'negative_handling': str # 负值处理
                }
            target_freq: 目标频率
            var_frequency_map: 变量频率映射（用于同比差分）

        Returns:
            {
                'status': 'success' | 'error',
                'message': str,
                'data': pd.DataFrame,                          # 转换后的数据
                'transform_details': Dict,                     # 转换详情
                'stationarity_results': Dict[str, Dict],       # 平稳性检验结果
                'problem_variables': List[Dict],               # 非平稳变量列表
                'errors': List[str]                           # 转换过程中的错误
            }
        """
        try:
            logger.info("UI后端: 开始变量转换和平稳性检验")
            logger.info(f"  输入数据形状: {data.shape}")
            logger.info(f"  转换配置数量: {len(transform_config)}")

            # 初始化转换器
            transformer = VariableTransformer(freq=target_freq)
            errors = []

            # 执行转换
            transformed_data = data.copy()
            transform_details = {}

            for config in transform_config:
                var_name = config.get('variable', '')
                operation = config.get('operation', 'none')

                if not var_name or var_name not in transformed_data.columns:
                    logger.warning(f"变量 '{var_name}' 不在数据中，跳过")
                    errors.append(f"变量 '{var_name}' 不在数据中")
                    continue

                try:
                    logger.debug(f"  转换: {var_name} <- {operation}")

                    # 提取单个变量的 Series
                    series = transformed_data[var_name].copy()

                    # 构建操作列表
                    operations = [operation] if operation and operation != 'none' else []

                    # 执行转换
                    transformed_series = transformer.transform_variable(
                        series=series,
                        operations=operations,
                        zero_method=config.get('zero_handling', 'none'),
                        neg_method=config.get('negative_handling', 'none')
                    )

                    # 更新 DataFrame
                    transformed_data[var_name] = transformed_series

                    # 记录转换详情（从 transformer 的内部记录中获取）
                    if var_name in transformer._transform_details:
                        detail = transformer._transform_details[var_name]
                        transform_details[var_name] = {
                            'operations': detail.get('operations', []),
                            'status': 'success'
                        }

                except Exception as e:
                    logger.error(f"变换变量 '{var_name}' 失败: {e}")
                    errors.append(f"变量 '{var_name}': {str(e)}")

            logger.info(f"转换完成: {len(transform_details)}个变量成功")

            # 执行平稳性检验
            logger.info("执行平稳性检验...")
            stationarity_results = StationarityChecker.batch_check_variables(
                transformed_data,
                variables=transformed_data.columns.tolist(),
                alpha=0.05,
                parallel=False
            )

            logger.info(f"平稳性检验完成: {len(stationarity_results)}个变量")

            # 提取非平稳变量
            problem_variables = UIBackendService._extract_problem_variables(
                stationarity_results,
                transformed_data,
                transform_details,
                var_frequency_map or {}
            )

            logger.info(f"非平稳变量: {len(problem_variables)}个")

            return {
                'status': 'success',
                'message': f'成功转换 {len(transform_details)} 个变量，检验 {len(stationarity_results)} 个变量',
                'data': transformed_data,
                'transform_details': transform_details,
                'stationarity_results': stationarity_results,
                'problem_variables': problem_variables,
                'errors': errors if errors else None
            }

        except Exception as e:
            logger.error(f"UI后端服务失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'处理失败: {str(e)}',
                'data': None,
                'transform_details': {},
                'stationarity_results': {},
                'problem_variables': [],
                'errors': [str(e)]
            }

    @staticmethod
    def _extract_problem_variables(
        stationarity_results: Dict[str, Dict],
        transformed_data: pd.DataFrame,
        transform_details: Dict,
        var_frequency_map: Dict[str, str]
    ) -> List[Dict]:
        """
        提取非平稳和数据不足的变量

        Returns:
            List of dicts with keys: variable, frequency, processing, p_value, stationarity
        """
        problem_vars = []

        for col, result in stationarity_results.items():
            status = result.get('status', '')
            p_value = result.get('p_value')

            # 跳过平稳的变量
            if status == '是':
                continue

            # 获取频率和处理操作
            col_normalized = normalize_text(col)
            freq = var_frequency_map.get(col_normalized, '-')
            ops = transform_details.get(col, {}).get('operations', [])
            ops_str = ' -> '.join(ops) if ops else '不处理'
            p_str = f"{p_value:.4f}" if p_value is not None else '-'

            # 转换平稳性状态显示
            stationarity_display = '数据不足' if status == '数据不足' else '非平稳'

            problem_vars.append({
                'variable': col,
                'frequency': freq,
                'processing': ops_str,
                'p_value': p_str,
                'stationarity': stationarity_display
            })

        return problem_vars


__all__ = ['UIBackendService']
