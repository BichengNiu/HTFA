# -*- coding: utf-8 -*-
"""
DFM变量选择组件

提供目标变量选择、行业分组、预测指标选择等功能
"""

import streamlit as st
import pandas as pd
import unicodedata
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager
from dashboard.models.DFM.prep.utils.text_utils import normalize_text


logger = logging.getLogger(__name__)


class VariableSelectionComponent(DFMComponent):
    """DFM变量选择组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化变量选择组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._default_num_cols = 3  # 默认行业选择列数
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "variable_selection"
    
    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键
        
        Returns:
            List[str]: 状态键列表
        """
        return [
            'dfm_target_variable',
            'dfm_selected_industries',
            'dfm_industry_checkbox_states',
            'dfm_selected_indicators_per_industry',
            'dfm_selected_indicators',
            'dfm_variable_selection_method'
            # 注：global_backward不需要threshold参数，它基于性能提升自动停止
        ]
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含training_data, industry_mapping, type_mapping
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查训练数据
            training_data = data.get('training_data')
            if training_data is None:
                logger.warning("缺少训练数据")
                return False
            
            if not isinstance(training_data, pd.DataFrame):
                logger.warning("训练数据不是DataFrame格式")
                return False
            
            if training_data.empty:
                logger.warning("训练数据为空")
                return False
            
            # 检查是否有可用的变量（除了日期列）
            available_vars = self._extract_available_variables(training_data)
            if len(available_vars) == 0:
                logger.warning("没有可用的变量")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"输入验证失败: {e}")
            return False
    
    def handle_service_error(self, error: Exception) -> None:
        """
        处理服务错误
        
        Args:
            error: 异常对象
        """
        error_msg = f"变量选择服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
    
    def render(self, st_obj, training_data: pd.DataFrame,
               industry_mapping: Dict[str, str] = None,
               type_mapping: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        渲染变量选择组件 - 与老代码UI完全一致

        Args:
            st_obj: Streamlit对象
            training_data: 训练数据
            industry_mapping: 行业映射字典
            type_mapping: 类型映射字典

        Returns:
            变量选择结果字典或None
        """
        try:
            # === 与老代码完全一致的UI布局 ===

            # 1. 目标变量选择 - 与老代码第895行完全一致
            available_target_vars = self._get_available_target_variables(training_data)
            selected_target_var = self._render_target_variable_selection_legacy(
                st_obj, available_target_vars
            )

            # 2. 获取行业映射数据
            industry_to_vars = self._get_industry_mapping_from_state()
            all_industries = list(industry_to_vars.keys()) if industry_to_vars else []

            # 2.5. 获取DFM变量选择配置（从Excel的"DFM变量"列读取）
            dfm_default_map = self._get_state('dfm_default_variables_map', {})
            print(f"[DEBUG] 从DFM变量列读取配置: {len(dfm_default_map)}个标记为'是'的变量")
            if not dfm_default_map:
                print(f"[DEBUG] DFM变量列为空或全部为非'是'值，将不选择任何变量")

            # 3. 预测指标选择 - 直接显示所有行业的指标供用户选择
            selected_indicators = self._render_indicator_selection_legacy(
                st_obj, all_industries, industry_to_vars, dfm_default_map
            )

            # 4. 从选择的指标直接统计涉及的唯一行业数
            # 获取变量到行业的映射
            var_industry_map = self._get_state('dfm_industry_map_obj', {})

            # 统计选中指标对应的唯一行业
            actual_industries_set = set()
            from dashboard.models.DFM.prep.utils.text_utils import normalize_text
            for indicator in selected_indicators:
                indicator_norm = normalize_text(indicator)
                if indicator_norm in var_industry_map:
                    industry = var_industry_map[indicator_norm]
                    actual_industries_set.add(industry)

            selected_industries = sorted(list(actual_industries_set))

            # 5. 显示汇总信息 - 与老代码第1108行完全一致
            self._render_selection_summary_legacy(st_obj, selected_target_var,
                                                selected_industries, selected_indicators)

            # 返回选择结果
            return {
                'target_variable': selected_target_var,
                'selected_industries': selected_industries,
                'selected_indicators': selected_indicators,
                'industry_to_vars': industry_to_vars
            }

        except Exception as e:
            self.handle_service_error(e)
            return None
    
    def _extract_available_variables(self, data: pd.DataFrame) -> List[str]:
        """
        提取可用变量（排除日期列和非数值列）
        
        Args:
            data: 训练数据
            
        Returns:
            可用变量列表
        """
        try:
            # 获取数值列
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            # 排除可能的日期相关列
            date_keywords = ['date', 'time', 'datetime', '日期', '时间']
            available_vars = []
            
            for col in numeric_cols:
                col_lower = col.lower()
                is_date_col = any(keyword in col_lower for keyword in date_keywords)
                if not is_date_col:
                    available_vars.append(col)
            
            return available_vars
            
        except Exception as e:
            logger.error(f"提取可用变量失败: {e}")
            return []
    
    def _group_variables_by_industry(self, variables: List[str], 
                                   industry_mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """
        按行业分组变量
        
        Args:
            variables: 变量列表
            industry_mapping: 行业映射字典
            
        Returns:
            按行业分组的变量字典
        """
        try:
            # 规范化行业映射的键
            normalized_industry_map = {}
            for k, v in industry_mapping.items():
                if pd.notna(k) and pd.notna(v):
                    normalized_key = unicodedata.normalize('NFKC', str(k)).strip().lower()
                    normalized_industry_map[normalized_key] = str(v).strip()
            
            # 按行业分组变量
            industry_to_vars = defaultdict(list)
            
            for var in variables:
                # 规范化变量名进行匹配
                lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                industry = normalized_industry_map.get(lookup_key, "_未知行业_")
                industry_to_vars[industry].append(var)
            
            # 转换为普通字典
            return dict(industry_to_vars)
            
        except Exception as e:
            logger.error(f"按行业分组变量失败: {e}")
            # 如果分组失败，将所有变量归为未知行业
            return {"_未知行业_": variables}
    
    def _initialize_target_variable(self, available_vars: List[str]) -> str:
        """
        初始化目标变量
        
        Args:
            available_vars: 可用变量列表
            
        Returns:
            初始化的目标变量
        """
        current_target = self._get_state('dfm_target_variable')
        
        if current_target is None or current_target not in available_vars:
            # 选择第一个可用变量作为默认目标变量
            default_target = available_vars[0] if available_vars else None
            self._set_state('dfm_target_variable', default_target)
            return default_target
        
        return current_target
    
    def _initialize_industry_selection(self, industries: List[str]) -> None:
        """
        初始化行业选择状态
        
        Args:
            industries: 行业列表
        """
        current_states = self._get_state('dfm_industry_checkbox_states', {})
        
        # 如果状态为空，默认全选所有行业
        if not current_states and industries:
            default_states = {industry: True for industry in industries}
            self._set_state('dfm_industry_checkbox_states', default_states)
    
    def _render_target_variable_selection(self, st_obj, available_vars: List[str]) -> str:
        """
        渲染目标变量选择
        
        Args:
            st_obj: Streamlit对象
            available_vars: 可用变量列表
            
        Returns:
            选择的目标变量
        """
        # 初始化目标变量
        current_target = self._initialize_target_variable(available_vars)
        
        # 确保当前目标变量在可选列表中
        if current_target not in available_vars:
            current_target = available_vars[0]
            self._set_state('dfm_target_variable', current_target)
        
        selected_target_var = st_obj.selectbox(
            "**选择目标变量**",
            options=available_vars,
            index=available_vars.index(current_target),
            key=f"{self.get_state_key_prefix()}_target_variable",
            help="选择您希望模型预测的目标序列。"
        )
        
        # 更新状态
        self._set_state('dfm_target_variable', selected_target_var)
        
        return selected_target_var
    
    def _render_industry_selection(self, st_obj, industries: List[str]) -> List[str]:
        """
        渲染行业选择
        
        Args:
            st_obj: Streamlit对象
            industries: 行业列表
            
        Returns:
            选择的行业列表
        """
        st_obj.markdown("**选择行业**")
        
        if not industries:
            st_obj.info("没有可用的行业数据。")
            return []
        
        
        # 初始化行业选择状态
        self._initialize_industry_selection(industries)
        
        # 批量操作按钮
        col1, col2, col3 = st_obj.columns(3)
        
        with col1:
            if st_obj.button("[LOADING] 重置", key=f"{self.get_state_key_prefix()}_reset_industries"):
                default_states = {industry: True for industry in industries}
                self._set_state('dfm_industry_checkbox_states', default_states)
                st_obj.rerun()
        
        with col2:
            if st_obj.button("取消全行业", key=f"{self.get_state_key_prefix()}_deselect_all"):
                deselect_states = {industry: False for industry in industries}
                self._set_state('dfm_industry_checkbox_states', deselect_states)
                st_obj.rerun()
        
        with col3:
            if st_obj.button("选择全行业", key=f"{self.get_state_key_prefix()}_select_all"):
                select_states = {industry: True for industry in industries}
                self._set_state('dfm_industry_checkbox_states', select_states)
                st_obj.rerun()
        
        # 创建列布局显示复选框
        industry_cols = st_obj.columns(self._default_num_cols)
        current_checkbox_states = self._get_state('dfm_industry_checkbox_states', {})
        
        # 渲染行业复选框
        for idx, industry_name in enumerate(industries):
            with industry_cols[idx % self._default_num_cols]:
                current_value = current_checkbox_states.get(industry_name, True)
                
                new_state = st_obj.checkbox(
                    industry_name,
                    value=current_value,
                    key=f"{self.get_state_key_prefix()}_industry_{idx}"
                )
                current_checkbox_states[industry_name] = new_state
        
        # 更新状态
        self._set_state('dfm_industry_checkbox_states', current_checkbox_states)
        
        # 获取选中的行业
        selected_industries = [
            industry for industry, checked in current_checkbox_states.items() if checked
        ]
        
        # 显示选择统计
        st_obj.text(f"已选择 {len(selected_industries)} 个行业")
        
        return selected_industries

    def _render_indicator_selection(self, st_obj, selected_industries: List[str],
                                   industry_to_vars: Dict[str, List[str]]) -> List[str]:
        """
        渲染预测指标选择

        Args:
            st_obj: Streamlit对象
            selected_industries: 选择的行业列表
            industry_to_vars: 行业到变量的映射

        Returns:
            选择的指标列表
        """
        st_obj.markdown("**选择预测指标**")

        if not selected_industries:
            st_obj.info("请先在上方选择至少一个行业。")
            return []

        final_selected_indicators = []

        # 为每个选中的行业渲染指标选择
        for industry_name in selected_industries:
            all_indicators_for_industry = industry_to_vars.get(industry_name, [])
            
            # 修复：排除目标变量，确保用户无法选择目标变量作为预测变量
            current_target_var = self._get_state('dfm_target_variable', None)
            if current_target_var:
                indicators_for_industry = [
                    indicator for indicator in all_indicators_for_industry 
                    if indicator != current_target_var
                ]
            else:
                indicators_for_industry = all_indicators_for_industry

            # 修复：完全跳过没有可用指标的行业，不显示任何内容
            if not indicators_for_industry:
                # 清空该行业的选择状态
                current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
                current_selection[industry_name] = []
                self._set_state('dfm_selected_indicators_per_industry', current_selection)
                continue

            # 只有当行业有可用指标时才显示行业标题
            st_obj.markdown(f"**行业: {industry_name}**")
            
            # 只有在有指标被排除且仍有可用指标时才显示提示
            excluded_count = len(all_indicators_for_industry) - len(indicators_for_industry)
            if excluded_count > 0:
                st_obj.text(f"  已自动排除目标变量 '{current_target_var}' (共排除 {excluded_count} 个)")

            # 获取该行业的当前选择状态
            current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
            default_selection = current_selection.get(industry_name, indicators_for_industry)

            # 确保默认值是实际可选列表的子集
            valid_default = [item for item in default_selection if item in indicators_for_industry]
            if not valid_default and indicators_for_industry:
                valid_default = indicators_for_industry  # 默认全选

            # 取消全选复选框
            deselect_all_checked = st_obj.checkbox(
                f"取消全选 {industry_name} 指标",
                key=f"{self.get_state_key_prefix()}_deselect_{industry_name}",
                help=f"勾选此框将取消所有已为 '{industry_name}' 选中的指标。"
            )

            # 如果取消全选被勾选，清空该行业的选择
            if deselect_all_checked:
                valid_default = []

            # 渲染多选框
            selected_in_widget = st_obj.multiselect(
                f"为 '{industry_name}' 选择指标",
                options=indicators_for_industry,
                default=valid_default,
                key=f"{self.get_state_key_prefix()}_indicators_{industry_name}",
                help=f"从 {industry_name} 行业中选择预测指标。"
            )

            # 更新该行业的选择状态
            current_selection[industry_name] = selected_in_widget
            self._set_state('dfm_selected_indicators_per_industry', current_selection)

            # 添加到最终选择列表
            final_selected_indicators.extend(selected_in_widget)

        # 清理不再被选中的行业条目
        self._clean_unused_industry_states(selected_industries)

        # 去重并排序
        final_indicators = sorted(list(set(final_selected_indicators)))

        return final_indicators

    def _render_variable_selection_method(self, st_obj) -> str:
        """
        渲染变量选择方法

        Args:
            st_obj: Streamlit对象

        Returns:
            选择的变量选择方法
        """
        # 变量选择方法选项
        variable_selection_options = {
            'none': "无筛选 (使用全部已选变量)",
            'global_backward': "全局后向剔除 (在已选变量中筛选)"
        }

        # 获取当前方法
        current_method = self._get_state('dfm_variable_selection_method', 'none')

        selected_method = st_obj.selectbox(
            "变量选择方法",
            options=list(variable_selection_options.keys()),
            format_func=lambda x: variable_selection_options[x],
            index=list(variable_selection_options.keys()).index(current_method),
            key=f"{self.get_state_key_prefix()}_selection_method",
            help=(
                "选择在已选变量基础上的筛选方法：\n"
                "- 无筛选: 直接使用所有已选择的变量\n"
                "- 全局后向剔除: 从已选变量开始，逐个剔除不重要的变量"
            )
        )

        # 更新状态
        self._set_state('dfm_variable_selection_method', selected_method)
        
        # 注：global_backward方法基于性能提升自动决定剔除，不需要阈值参数

        return selected_method

    # === 与老代码完全一致的辅助方法 ===

    def _get_available_target_variables(self, training_data: pd.DataFrame) -> List[str]:
        """获取可用的目标变量 - 与老代码第810-854行逻辑一致"""
        available_target_vars = []

        if training_data is not None:
            # 从已加载数据中获取可选的目标变量 - 与老代码第813行一致
            available_target_vars = [
                col for col in training_data.columns
                if 'date' not in col.lower() and 'time' not in col.lower()
            ]

            # 确保默认目标变量始终包含在选项中 - 与老代码第815-837行一致
            default_target = '规模以上工业增加值:当月同比'

            if default_target and default_target in training_data.columns and default_target not in available_target_vars:
                available_target_vars.insert(0, default_target)

            if not available_target_vars and default_target and default_target in training_data.columns:
                available_target_vars = [default_target]

            if not available_target_vars:
                if default_target:
                    available_target_vars = [default_target]
        else:
            # 即使没有数据，也提供默认目标变量选项 - 与老代码第844-853行一致
            default_target = '规模以上工业增加值:当月同比'
            if default_target:
                available_target_vars = [default_target]

        return available_target_vars

    def _get_industry_mapping_from_state(self) -> Dict[str, List[str]]:
        """从状态管理器获取行业映射 - 与老代码第856-874行逻辑一致"""
        try:
            # 修复：优先使用过滤后的映射
            var_industry_map = self._get_state('dfm_industry_map_filtered', None)
            if var_industry_map is None:
                # 如果没有过滤后的映射，使用原始映射
                var_industry_map = self._get_state('dfm_industry_map_obj', {})

            if not var_industry_map:
                return {}

            # 构建行业到指标的映射
            from collections import defaultdict
            industry_to_indicators_temp = defaultdict(list)
            for indicator, industry in var_industry_map.items():
                if indicator and industry:
                    industry_to_indicators_temp[str(industry).strip()].append(str(indicator).strip())

            # 排序并返回结果
            industry_to_indicators_map = {k: sorted(v) for k, v in industry_to_indicators_temp.items()}
            return industry_to_indicators_map

        except Exception as e:
            logger.error(f"获取行业映射失败: {e}")
            return {}

    def _render_target_variable_selection_legacy(self, st_obj, available_target_vars: List[str]) -> str:
        """渲染目标变量选择 - 与老代码第881-906行完全一致"""
        if available_target_vars:
            # 初始化目标变量状态
            if self._get_state('dfm_target_variable') is None:
                self._set_state('dfm_target_variable', available_target_vars[0])

            current_target_var = self._get_state('dfm_target_variable')

            # 确保当前目标变量在可选列表中
            if current_target_var not in available_target_vars:
                current_target_var = available_target_vars[0]
                self._set_state('dfm_target_variable', current_target_var)

            selected_target_var = st_obj.selectbox(
                "**选择目标变量**",
                options=available_target_vars,
                index=available_target_vars.index(current_target_var),
                key="new_ss_dfm_target_variable",
                help="选择您希望模型预测的目标序列。"
            )
            self._set_state('dfm_target_variable', selected_target_var)
            return selected_target_var
        else:
            # 紧急情况：如果仍然没有可选变量，显示错误并设置为None
            st_obj.error("[ERROR] 无法找到任何可用的目标变量")
            self._set_state('dfm_target_variable', None)
            return None

    def _render_industry_selection_legacy(self, st_obj, unique_industries: List[str],
                                        industry_to_vars: Dict[str, List[str]]) -> List[str]:
        """渲染行业选择 - 与老代码第934-1030行完全一致"""
        # 检查复选框状态是否需要初始化
        current_checkbox_states = self._get_state('dfm_industry_checkbox_states', None)

        # 如果状态为None、空字典，或者行业列表发生变化，则重新初始化
        needs_initialization = (
            current_checkbox_states is None or
            not current_checkbox_states or  # 空字典
            (unique_industries and set(current_checkbox_states.keys()) != set(unique_industries))  # 行业列表变化
        )

        if needs_initialization and unique_industries:
            initial_states = {industry: True for industry in unique_industries}
            self._set_state('dfm_industry_checkbox_states', initial_states)
        elif not unique_industries:
            self._set_state('dfm_industry_checkbox_states', {})

        # 为了避免在没有行业时出错，检查 unique_industries
        if not unique_industries:
            st_obj.info("没有可用的行业数据。")
            return []
        else:
            # 创建列以更好地布局复选框 - 与老代码第954-958行一致
            num_cols_industry = 3

            industry_cols = st_obj.columns(num_cols_industry)
            col_idx = 0
            current_checkbox_states = self._get_state('dfm_industry_checkbox_states', {})

            # 修复：过滤掉包含目标变量的行业
            current_target_var = self._get_state('dfm_target_variable', None)
            filtered_industries = []
            
            for industry_name in unique_industries:
                # 检查这个行业是否包含目标变量
                industry_indicators = industry_to_vars.get(industry_name, [])
                if current_target_var and current_target_var in industry_indicators:
                    # 如果该行业包含目标变量，检查是否还有其他指标
                    non_target_indicators = [ind for ind in industry_indicators if ind != current_target_var]
                    if non_target_indicators:
                        # 该行业除了目标变量还有其他指标，保留该行业
                        filtered_industries.append(industry_name)
                    # 如果该行业只有目标变量，则排除该行业
                else:
                    # 该行业不包含目标变量，保留
                    filtered_industries.append(industry_name)

            # 创建复选框并收集状态 - 与老代码第964-976行一致
            for industry_name in filtered_industries:
                with industry_cols[col_idx % num_cols_industry]:
                    # 获取当前状态：从统一状态管理器获取，默认为True
                    current_value = current_checkbox_states.get(industry_name, True)

                    # 使用唯一的key参数，确保Streamlit能正确追踪复选框状态
                    new_state = st_obj.checkbox(
                        industry_name,
                        value=current_value,
                        key=f"dfm_legacy_industry_checkbox_{industry_name}"
                    )
                    current_checkbox_states[industry_name] = new_state
                col_idx += 1

            # 更新状态管理器
            self._set_state('dfm_industry_checkbox_states', current_checkbox_states)

            # 使用按钮控制行业选择 - 与老代码第981-1012行完全一致
            col_deselect, col_select, col_reset = st_obj.columns(3)
            with col_deselect:
                if st_obj.button("取消全行业",
                                key='btn_dfm_deselect_all_industries',
                                help="点击取消所有已选中的行业",
                                use_container_width=True):
                    # 记录旧状态
                    old_states = self._get_state('dfm_industry_checkbox_states', {})
                    new_states = {industry: False for industry in filtered_industries}

                    # 更新统一状态管理器中的状态
                    self._set_state('dfm_industry_checkbox_states', new_states)

                    # 调试信息
                    logger.info(f"取消全行业按钮点击 - 旧状态: {sum(old_states.values())} 个行业已选")
                    logger.info(f"取消全行业按钮点击 - 新状态: 0 个行业已选")
                    logger.info(f"行业复选框状态已更新: {len(filtered_industries)} 个行业全部设为 False")
                    logger.info(f"过滤后的行业列表: {filtered_industries}")
                    logger.info(f"industry_to_vars内容: {list(industry_to_vars.keys())}")

                    # 强制刷新页面以更新UI
                    st_obj.rerun()

            with col_select:
                if st_obj.button("选择全行业",
                                key='btn_dfm_select_all_industries',
                                help="点击选择所有行业",
                                use_container_width=True):
                    # 记录旧状态
                    old_states = self._get_state('dfm_industry_checkbox_states', {})
                    new_states = {industry: True for industry in filtered_industries}

                    # 更新统一状态管理器中的状态
                    self._set_state('dfm_industry_checkbox_states', new_states)

                    # 调试信息
                    logger.info(f"选择全行业按钮点击 - 旧状态: {sum(old_states.values())} 个行业已选")
                    logger.info(f"选择全行业按钮点击 - 新状态: {len(filtered_industries)} 个行业已选")
                    logger.info(f"行业复选框状态已更新: {len(filtered_industries)} 个行业全部设为 True")

                    # 强制刷新页面以更新UI
                    st_obj.rerun()

            with col_reset:
                if st_obj.button("[LOADING] 重置",
                                key='btn_dfm_reset_industries',
                                help="重置为默认状态（全选）",
                                use_container_width=True):
                    # 记录旧状态
                    old_states = self._get_state('dfm_industry_checkbox_states', {})
                    reset_states = {industry: True for industry in filtered_industries}

                    # 直接设置为全选状态，而不是清空
                    self._set_state('dfm_industry_checkbox_states', reset_states)

                    # 调试信息
                    logger.info(f"重置行业按钮点击 - 旧状态: {sum(old_states.values())} 个行业已选")
                    logger.info(f"重置行业按钮点击 - 新状态: {len(filtered_industries)} 个行业已选")
                    logger.info(f"行业复选框状态已重置: {len(filtered_industries)} 个行业全部设为 True")

                    # 强制刷新页面以更新UI
                    st_obj.rerun()

            # 显示当前选择状态 - 与老代码第1014-1016行一致
            selected_count = sum(1 for industry, checked in current_checkbox_states.items() if checked and industry in filtered_industries)
            st_obj.info(f"[DATA] 当前状态：已选择 {selected_count} 个行业（共 {len(filtered_industries)} 个可选）")
            
            if len(filtered_industries) < len(unique_industries):
                excluded_count = len(unique_industries) - len(filtered_industries)
                st_obj.text(f"已自动排除 {excluded_count} 个仅包含目标变量的行业")

        # 更新当前选中的行业列表 - 与老代码第1018-1030行一致
        current_checkbox_states = self._get_state('dfm_industry_checkbox_states', {})

        # 重新计算过滤后的行业列表 (为了在这个作用域中访问)
        current_target_var = self._get_state('dfm_target_variable', None)
        filtered_industries_for_state = []
        for industry_name in unique_industries:
            industry_indicators = industry_to_vars.get(industry_name, [])
            if current_target_var and current_target_var in industry_indicators:
                non_target_indicators = [ind for ind in industry_indicators if ind != current_target_var]
                if non_target_indicators:
                    filtered_industries_for_state.append(industry_name)
            else:
                filtered_industries_for_state.append(industry_name)
        
        # 如果复选框状态为空但有行业数据，使用默认全选状态
        if not current_checkbox_states and filtered_industries_for_state:
            current_checkbox_states = {industry: True for industry in filtered_industries_for_state}
            self._set_state('dfm_industry_checkbox_states', current_checkbox_states)

        selected_industries = [
            industry for industry, checked in current_checkbox_states.items() if checked
        ]

        self._set_state('dfm_selected_industries', selected_industries)
        return selected_industries

    def _render_indicator_selection_legacy(self, st_obj, selected_industries: List[str],
                                         industry_to_vars: Dict[str, List[str]],
                                         dfm_default_map: Dict[str, str] = None) -> List[str]:
        """渲染预测指标选择 - 与老代码第1032-1104行完全一致"""
        if dfm_default_map is None:
            dfm_default_map = {}
        st_obj.markdown("**选择预测指标**")

        # 初始化指标选择状态
        if self._get_state('dfm_selected_indicators_per_industry', None) is None:
            self._set_state('dfm_selected_indicators_per_industry', {})

        final_selected_indicators_flat = []
        current_selected_industries = self._get_state('dfm_selected_industries', [])

        if not current_selected_industries:
            st_obj.info("请先在上方选择至少一个行业。")
        else:
            for industry_name in current_selected_industries:
                all_indicators_for_industry = industry_to_vars.get(industry_name, [])
                
                # 修复：排除目标变量，确保用户无法选择目标变量作为预测变量
                current_target_var = self._get_state('dfm_target_variable', None)
                if current_target_var:
                    indicators_for_this_industry = [
                        indicator for indicator in all_indicators_for_industry 
                        if indicator != current_target_var
                    ]
                else:
                    indicators_for_this_industry = all_indicators_for_industry

                # 修复：完全跳过没有可用指标的行业，不显示任何内容
                if not indicators_for_this_industry:
                    current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
                    current_selection[industry_name] = []
                    self._set_state('dfm_selected_indicators_per_industry', current_selection)
                    continue

                # 只有当行业有可用指标时才显示
                st_obj.markdown(f"**行业: {industry_name}**")
                
                # 只有在有指标被排除且仍有可用指标时才显示提示
                excluded_count = len(all_indicators_for_industry) - len(indicators_for_this_industry)
                if excluded_count > 0:
                    st_obj.text(f"  已自动排除目标变量 '{current_target_var}' (共排除 {excluded_count} 个)")

                # 从状态管理器读取已选指标，或使用DFM默认选择
                current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
                default_selection_for_industry = current_selection.get(industry_name, None)

                # 如果状态管理器中没有选择，使用DFM变量列配置
                if default_selection_for_industry is None:
                    if dfm_default_map:
                        # dfm_default_map中已经只包含标记为"是"的变量，直接筛选该行业的指标
                        dfm_selected_indicators = [
                            indicator for indicator in indicators_for_this_industry
                            if normalize_text(indicator) in dfm_default_map
                        ]
                        default_selection_for_industry = dfm_selected_indicators
                        print(f"[DEBUG] {industry_name}: 根据DFM变量列配置选择了{len(dfm_selected_indicators)}个指标")
                    else:
                        # DFM变量列全为空，全不选
                        default_selection_for_industry = []
                        print(f"[DEBUG] {industry_name}: DFM变量列为空，全不选")

                # 确保默认值是实际可选列表的子集
                valid_default = [item for item in default_selection_for_industry if item in indicators_for_this_industry]

                # 取消全选复选框，使用key确保状态追踪
                deselect_all_checked = st_obj.checkbox(
                    f"取消全选 {industry_name} 指标",
                    key=f"dfm_legacy_deselect_all_{industry_name}",
                    help=f"勾选此框将取消所有已为 '{industry_name}' 选中的指标。"
                )

                # 如果取消全选被勾选，清空该行业的选择
                if deselect_all_checked:
                    valid_default = []

                selected_in_widget = st_obj.multiselect(
                    f"为 '{industry_name}' 选择指标",
                    options=indicators_for_this_industry,
                    default=valid_default,
                    help=f"从 {industry_name} 行业中选择预测指标。"
                )

                # 更新统一状态管理器中的选择状态
                current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
                current_selection[industry_name] = selected_in_widget
                self._set_state('dfm_selected_indicators_per_industry', current_selection)

                final_selected_indicators_flat.extend(selected_in_widget)

            # 清理统一状态管理器中不再被选中的行业条目
            current_selection = self._get_state('dfm_selected_indicators_per_industry', {})
            industries_to_remove_from_state = [
                ind for ind in current_selection
                if ind not in current_selected_industries
            ]
            for ind_to_remove in industries_to_remove_from_state:
                del current_selection[ind_to_remove]
            self._set_state('dfm_selected_indicators_per_industry', current_selection)

        # 更新最终的扁平化预测指标列表 (去重)
        final_indicators = sorted(list(set(final_selected_indicators_flat)))
        self._set_state('dfm_selected_indicators', final_indicators)
        return final_indicators

    def _render_selection_summary_legacy(self, st_obj, selected_target_var: str,
                                       selected_industries: List[str], selected_indicators: List[str]):
        """渲染选择摘要 - 与老代码第1108-1114行完全一致"""
        st_obj.markdown("--- ")
        current_target_var = self._get_state('dfm_target_variable', None)
        current_selected_indicators = self._get_state('dfm_selected_indicators', [])
        st_obj.text(f" - 目标变量: {current_target_var if current_target_var else '未选择'}")
        st_obj.text(f" - 选定行业数: {len(selected_industries)}")
        st_obj.text(f" - 选定预测指标总数: {len(current_selected_indicators)}")

    def _update_selection_states(self, selected_industries: List[str],
                               selected_indicators: List[str]) -> None:
        """
        更新选择状态

        Args:
            selected_industries: 选择的行业列表
            selected_indicators: 选择的指标列表
        """
        try:
            # 更新选择的行业
            self._set_state('dfm_selected_industries', selected_industries)

            # 更新选择的指标
            self._set_state('dfm_selected_indicators', selected_indicators)

        except Exception as e:
            logger.error(f"更新选择状态失败: {e}")

    def _render_selection_summary(self, st_obj) -> None:
        """
        渲染选择摘要

        Args:
            st_obj: Streamlit对象
        """
        try:
            st_obj.markdown("---")

            # 获取当前选择状态
            current_target_var = self._get_state('dfm_target_variable', None)
            current_selected_industries = self._get_state('dfm_selected_industries', [])
            current_selected_indicators = self._get_state('dfm_selected_indicators', [])

            # 显示摘要信息
            st_obj.text(f" - 目标变量: {current_target_var if current_target_var else '未选择'}")
            st_obj.text(f" - 选定行业数: {len(current_selected_industries)}")
            st_obj.text(f" - 选定预测指标总数: {len(current_selected_indicators)}")

            # 可选：显示详细的指标列表
            if current_selected_indicators:
                with st_obj.expander("查看已选指标列表", expanded=False):
                    for indicator in current_selected_indicators:
                        st_obj.text(f"  • {indicator}")

        except Exception as e:
            logger.error(f"渲染选择摘要失败: {e}")

    def _clean_unused_industry_states(self, current_industries: List[str]) -> None:
        """
        清理不再被选中的行业状态

        Args:
            current_industries: 当前选择的行业列表
        """
        try:
            current_selection = self._get_state('dfm_selected_indicators_per_industry', {})

            # 找出需要删除的行业
            industries_to_remove = [
                industry for industry in current_selection
                if industry not in current_industries
            ]

            # 删除不再被选中的行业条目
            for industry_to_remove in industries_to_remove:
                del current_selection[industry_to_remove]

            # 更新状态
            self._set_state('dfm_selected_indicators_per_industry', current_selection)

        except Exception as e:
            logger.error(f"清理未使用的行业状态失败: {e}")

    def _get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                value = dfm_manager.get_dfm_state('train_model', key, default)
                logger.debug(f"获取DFM状态成功: {key} = {type(value).__name__}")
                return value
            else:
                logger.error(f"DFM状态管理器不可用，无法获取状态: {key}")
                return default

        except Exception as e:
            logger.error(f"获取状态失败: {key} - {str(e)}")
            return default

    def _set_state(self, key: str, value: Any) -> bool:
        """设置状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                success = dfm_manager.set_dfm_state('train_model', key, value)
                if success:
                    logger.debug(f"设置DFM状态成功: {key}")
                    return True
                else:
                    logger.error(f"设置DFM状态失败: {key}")
                    return False
            else:
                logger.error(f"DFM状态管理器不可用，无法设置状态: {key}")
                return False

        except Exception as e:
            logger.error(f"设置状态失败: {key} - {str(e)}")
            return False
