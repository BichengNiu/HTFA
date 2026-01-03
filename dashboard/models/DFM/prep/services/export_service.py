"""
导出服务模块

提供数据导出和处理日志构建功能
"""

import pandas as pd
import io
from typing import Dict, Any, Optional, List
import logging
from dashboard.models.DFM.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


class ExportService:
    """导出服务类"""

    @staticmethod
    def clear_non_stationary_markers(
        mappings: Dict[str, Any],
        stationarity_check_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        清除不平稳变量的"一次估计"和"一阶段预测"标记

        对于标记为"一次估计=是"或"一阶段预测=是"但不平稳的变量，
        将其标记替换为空值。

        Args:
            mappings: 映射字典（包含single_stage_map和first_stage_pred_map）
            stationarity_check_results: 平稳性检验结果

        Returns:
            更新后的mappings字典（深拷贝）
        """
        import copy
        updated_mappings = copy.deepcopy(mappings)

        single_stage_map = updated_mappings.get('single_stage_map', {})
        first_stage_pred_map = updated_mappings.get('first_stage_pred_map', {})

        cleared_vars = []

        # 检查所有标记为"是"的变量
        all_marked_vars = set(single_stage_map.keys()) | set(first_stage_pred_map.keys())

        for var in all_marked_vars:
            stat_result = stationarity_check_results.get(var, {})
            is_stationary = stat_result.get('is_stationary', True)

            if not is_stationary:
                # 清除标记
                if var in single_stage_map:
                    single_stage_map[var] = ''
                if var in first_stage_pred_map:
                    first_stage_pred_map[var] = ''
                cleared_vars.append(var)

        if cleared_vars:
            logger.info(f"平稳性过滤: 清除 {len(cleared_vars)} 个不平稳变量的标记: {cleared_vars}")

        updated_mappings['single_stage_map'] = single_stage_map
        updated_mappings['first_stage_pred_map'] = first_stage_pred_map

        return updated_mappings

    @staticmethod
    def build_processing_log(
        removed_vars_log: Optional[List[Dict]] = None,
        prepared_data: Optional[pd.DataFrame] = None,
        transform_details: Optional[Dict] = None,
        replacement_history: Optional[List[Dict]] = None,
        stationarity_check_results: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        构建处理日志DataFrame

        Args:
            removed_vars_log: 被删除变量的日志列表
            prepared_data: 处理后的数据
            transform_details: 变量转换详情
            replacement_history: 值替换历史记录
            stationarity_check_results: 平稳性检验结果

        Returns:
            DataFrame: [变量名, 状态, 处理详情, P值, 平稳性检验（0.05）]
        """
        log_data = []
        removed_vars_log = removed_vars_log or []
        transform_details = transform_details or {}
        replacement_history = replacement_history or []
        stationarity_check_results = stationarity_check_results or {}

        # 添加被删除的变量
        for entry in removed_vars_log:
            var_name = entry.get('Variable', '')
            var_name_norm = normalize_text(var_name)
            reason = entry.get('Reason', '')
            details = entry.get('Details', {})

            if 'nan_period' in details:
                detail_str = f"{reason} ({details['nan_period']}, 最大连续{details.get('max_consecutive_nan', 'N/A')}期)"
            else:
                detail_str = reason

            log_data.append({
                '变量名': var_name_norm,
                '状态': '删除',
                '处理详情': detail_str,
                'P值': '-',
                '平稳性检验（0.05）': '-'
            })

        # 添加值替换记录
        for h in replacement_history:
            var_name = h.get('variable', '')
            var_name_norm = normalize_text(var_name)

            # 优先使用检验结果，如果没有检验结果则显示"未检验"
            if var_name_norm in stationarity_check_results:
                stat_result = stationarity_check_results[var_name_norm]
                p_value = stat_result.get('p_value')
                status = stat_result.get('status', '未检验')

                # P值列：显示具体数值
                p_value_str = f"{p_value:.3f}" if p_value is not None else '-'

                # 平稳性检验列：显示平稳/非平稳/数据不足
                if status == '是':
                    stationarity_str = '平稳'
                elif status == '数据不足':
                    stationarity_str = '数据不足'
                elif status.startswith('计算失败'):
                    stationarity_str = status
                else:
                    stationarity_str = '非平稳'
            else:
                logger.warning(f"值替换变量 '{var_name_norm}' 没有检验结果，可能未在prepared_data中")
                p_value_str = '-'
                stationarity_str = '未检验'

            log_data.append({
                '变量名': var_name_norm,
                '状态': '值替换',
                '处理详情': f"规则: {h.get('rule', '')}, 替换为: {h.get('new_value', '')}, 影响{h.get('affected_count', 0)}行",
                'P值': p_value_str,
                '平稳性检验（0.05）': stationarity_str
            })

        # 添加保留的变量
        if prepared_data is not None:
            for col in prepared_data.columns:
                col_norm = normalize_text(col)

                # 获取转换详情（使用标准化名称）
                if col_norm in transform_details:
                    ops = transform_details[col_norm].get('operations', [])
                    ops_str = ' -> '.join(ops) if ops else '不处理'
                else:
                    ops_str = '不处理'

                # 获取平稳性检验结果（多级fallback查找）
                stat_result = None

                # 尝试1: 规范化键精确匹配
                if col_norm in stationarity_check_results:
                    stat_result = stationarity_check_results[col_norm]

                # 尝试2: 原始列名匹配
                if stat_result is None and col in stationarity_check_results:
                    stat_result = stationarity_check_results[col]

                # 尝试3: 模糊匹配（规范化后比较）
                if stat_result is None:
                    for key in stationarity_check_results.keys():
                        if normalize_text(key) == col_norm:
                            stat_result = stationarity_check_results[key]
                            break

                if stat_result is not None:
                    p_value = stat_result.get('p_value')
                    status = stat_result.get('status', '未检验')

                    # P值列：显示具体数值
                    p_value_str = f"{p_value:.3f}" if p_value is not None else '-'

                    # 平稳性检验列：显示平稳/非平稳/数据不足
                    if status == '是':
                        stationarity_str = '平稳'
                    elif status == '数据不足':
                        stationarity_str = '数据不足'
                    elif status.startswith('计算失败'):
                        stationarity_str = status
                    else:
                        stationarity_str = '非平稳'
                else:
                    logger.warning(f"保留变量 '{col}' 没有检验结果，可能检验失败或被跳过")
                    p_value_str = '-'
                    stationarity_str = '未检验'

                log_data.append({
                    '变量名': col_norm,
                    '状态': '保留',
                    '处理详情': ops_str,
                    'P值': p_value_str,
                    '平稳性检验（0.05）': stationarity_str
                })

        return pd.DataFrame(log_data, columns=['变量名', '状态', '处理详情', 'P值', '平稳性检验（0.05）'])

    @staticmethod
    def generate_excel(
        prepared_data: pd.DataFrame,
        industry_map: Dict[str, str],
        mappings: Dict[str, Any],
        removed_vars_log: Optional[List[Dict]] = None,
        transform_details: Optional[Dict] = None,
        replacement_history: Optional[List[Dict]] = None,
        stationarity_check_results: Optional[Dict[str, Dict]] = None
    ) -> bytes:
        """
        生成导出Excel文件

        Args:
            prepared_data: 处理后的数据
            industry_map: 行业映射
            mappings: 完整映射字典
            removed_vars_log: 被删除变量日志
            transform_details: 变量转换详情
            replacement_history: 值替换历史记录
            stationarity_check_results: 平稳性检验结果

        Returns:
            bytes: Excel文件字节内容
        """
        # 提取各类映射
        dfm_single_stage_map = mappings.get('single_stage_map', {})
        dfm_first_stage_pred_map = mappings.get('first_stage_pred_map', {})
        dfm_first_stage_target_map = mappings.get('first_stage_target_map', {})
        dfm_second_stage_target_map = mappings.get('second_stage_target_map', {})
        var_frequency_map = mappings.get('var_frequency_map', {})
        var_unit_map = mappings.get('var_unit_map', {})
        var_nature_map = mappings.get('var_nature_map', {})

        # 构建处理日志
        df_processing_log = ExportService.build_processing_log(
            removed_vars_log, prepared_data, transform_details, replacement_history,
            stationarity_check_results
        )

        # 创建处理日志查找字典
        log_lookup = {}
        if df_processing_log is not None and not df_processing_log.empty:
            for idx, row in df_processing_log.iterrows():
                var_name = row['变量名']
                log_lookup[var_name] = {
                    '状态': row.get('状态', ''),
                    '处理详情': row.get('处理详情', ''),
                    'P值': row.get('P值', ''),
                    '平稳性检验（0.05）': row.get('平稳性检验（0.05）', '')
                }

        # 创建统一映射数据
        all_indicators = list(industry_map.keys())
        unified_mapping_data = []

        for indicator in all_indicators:
            # 从处理日志查找该变量的信息
            log_info = log_lookup.get(indicator, {})

            unified_mapping_data.append({
                '指标名称': indicator,
                '行业': industry_map.get(indicator, ''),
                '频率': var_frequency_map.get(indicator, ''),
                '单位': var_unit_map.get(indicator, ''),
                '性质': var_nature_map.get(indicator, ''),
                '一次估计': dfm_single_stage_map.get(indicator, ''),
                '一阶段预测': dfm_first_stage_pred_map.get(indicator, ''),
                '一阶段目标': dfm_first_stage_target_map.get(indicator, ''),
                '二阶段目标': dfm_second_stage_target_map.get(indicator, ''),
                '处理详情': log_info.get('处理详情', ''),
                'P值': log_info.get('P值', ''),
                '平稳性检验（0.05）': log_info.get('平稳性检验（0.05）', '')
            })

        df_unified_map = pd.DataFrame(
            unified_mapping_data,
            columns=['指标名称', '行业', '频率', '单位', '性质',
                     '一次估计', '一阶段预测', '一阶段目标', '二阶段目标',
                     '处理详情', 'P值', '平稳性检验（0.05）']
        )

        # 写入Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Sheet1: 数据（按时间由近及远排列）
            export_data = prepared_data.copy()
            export_data = export_data.sort_index(ascending=False)
            if isinstance(export_data.index, pd.DatetimeIndex):
                export_data.index = export_data.index.strftime('%Y-%m-%d')
            export_data.to_excel(writer, sheet_name='数据', index=True, index_label='Date')

            # Sheet2: 映射表
            df_unified_map.to_excel(writer, sheet_name='映射', index=False)

        return excel_buffer.getvalue()


__all__ = ['ExportService']
