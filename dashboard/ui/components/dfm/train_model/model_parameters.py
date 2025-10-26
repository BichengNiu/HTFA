# -*- coding: utf-8 -*-
"""
DFM模型参数组件

提供模型参数配置、验证和管理功能
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager
from dashboard.models.DFM.config import TrainDefaults, UIDefaults


logger = logging.getLogger(__name__)


class ModelParametersComponent(DFMComponent):
    """DFM模型参数组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化模型参数组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._parameter_constraints = self._define_parameter_constraints()
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "model_parameters"
    
    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键
        
        Returns:
            List[str]: 状态键列表
        """
        return [
            'dfm_variable_selection_method',
            'dfm_enable_variable_selection',
            'dfm_max_iter',  # 修正：与model_training_page.py一致
            'dfm_factor_ar_order',  # 修正：与model_training_page.py一致
            'dfm_factor_selection_strategy',
            'dfm_fixed_number_of_factors',
            'dfm_cumulative_variance_threshold'  # 修正：与model_training_page.py一致
        ]
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含模型参数配置
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 验证参数约束
            is_valid, errors = self._validate_parameter_constraints(data)
            
            if not is_valid:
                for error in errors:
                    logger.warning(f"参数验证失败: {error}")
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
        error_msg = f"模型参数服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
    
    def render(self, st_obj) -> Optional[Dict[str, Any]]:
        """
        渲染模型参数组件
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            模型参数配置结果字典或None
        """
        try:
            # === 严格按照老代码真实存在的参数 + 新代码后端支持的参数 ===

            # 1. 变量选择方法 - 与老代码第1300-1333行一致
            variable_selection_method = self._render_variable_selection_method_legacy(st_obj)

            # 2. 最大迭代次数 (EM算法) - 与老代码第1335-1353行一致
            max_iterations = self._render_max_iterations_legacy(st_obj)

            # 3. 因子数量选择策略 - 与老代码第1359-1463行一致
            factor_selection_strategy = self._render_factor_selection_strategy_legacy(st_obj)

            # 4. 因子自回归阶数 - 新代码后端支持的参数
            max_lags = self._render_max_lags_legacy(st_obj)

            # 返回所有参数
            return {
                'variable_selection_method': variable_selection_method,
                'max_iter': max_iterations,  # 修正返回字典的键名
                'factor_selection_strategy': factor_selection_strategy,
                'factor_ar_order': max_lags  # 修正返回字典的键名
            }
                
        except Exception as e:
            self.handle_service_error(e)
            return None

    # === 与老代码完全一致的辅助方法 ===

    def _render_variable_selection_method_legacy(self, st_obj) -> str:
        """渲染变量选择方法 - 与老代码第1300-1320行一致"""
        variable_selection_options = [
            "无变量选择",
            "基于信息准则的变量选择",
            "基于交叉验证的变量选择",
            "基于正则化的变量选择"
        ]

        # 获取当前状态
        current_method = self._get_state('dfm_variable_selection_method', variable_selection_options[0])

        # 确保当前方法在选项中
        if current_method not in variable_selection_options:
            current_method = variable_selection_options[0]

        selected_method = st_obj.selectbox(
            "**变量选择方法**",
            options=variable_selection_options,
            index=variable_selection_options.index(current_method),
            key="new_ss_dfm_variable_selection_method",
            help="选择用于变量选择的方法。"
        )

        self._set_state('dfm_variable_selection_method', selected_method)
        return selected_method

    def _render_max_iterations_legacy(self, st_obj) -> int:
        """渲染最大迭代次数 - 与老代码第1335-1353行完全一致"""
        # 使用老代码中的默认值
        max_iter_default = 30
        max_iter_min = 1
        max_iter_step = 10

        max_iter_value = st_obj.number_input(
            "最大迭代次数 (Max Iterations for EM)",
            min_value=max_iter_min,
            value=self._get_state('dfm_max_iter', max_iter_default),  # 修正键名
            step=max_iter_step,
            key='new_dfm_max_iterations_input',
            help="EM估计算法允许的最大迭代次数。"
        )
        self._set_state('dfm_max_iter', max_iter_value)  # 修正键名
        return max_iter_value

    def _render_factor_selection_strategy_legacy(self, st_obj) -> str:
        """渲染因子选择策略 - 与老代码第1359-1463行完全一致"""
        # 与老代码完全一致的选项
        factor_selection_strategy_options = {
            'fixed_number': "固定因子数量 (Fixed Number of Factors)",
            'cumulative_variance': "累积共同方差 (Cumulative Common Variance)"
        }
        default_strategy = 'fixed_number'

        factor_strategy_value = st_obj.selectbox(
            "因子数量选择策略",
            options=list(factor_selection_strategy_options.keys()),
            format_func=lambda x: factor_selection_strategy_options[x],
            index=list(factor_selection_strategy_options.keys()).index(
                self._get_state('dfm_factor_selection_strategy', default_strategy)
            ),
            key='new_dfm_factor_selection_strategy_input',
            help=(
                "选择确定模型中因子数量的方法：\n"
                "- 固定因子数量: 直接指定因子数量。\n"
                "- 累积共同方差: 根据解释的方差比例确定因子数。"
            )
        )
        self._set_state('dfm_factor_selection_strategy', factor_strategy_value)

        # 根据策略显示对应参数 - 与老代码第1386-1463行一致
        self._render_strategy_specific_parameters(st_obj, factor_strategy_value)

        return factor_strategy_value

    def _render_strategy_specific_parameters(self, st_obj, factor_strategy_value: str):
        """根据策略显示对应参数 - 与老代码第1386-1463行完全一致"""
        if factor_strategy_value == 'fixed_number':
            default_fixed_factors = 3
            fixed_factors_value = st_obj.number_input(
                "固定因子数量 (Fixed Number of Factors)",
                min_value=1,
                value=self._get_state('dfm_fixed_number_of_factors', default_fixed_factors),
                step=1,
                key='new_dfm_fixed_number_of_factors_input',
                help="直接指定模型中要使用的因子数量。"
            )
            self._set_state('dfm_fixed_number_of_factors', fixed_factors_value)

        elif factor_strategy_value == 'cumulative_variance':
            # a. 累积共同方差阈值
            cum_variance_value = st_obj.slider(
                "累积共同方差阈值 (Cumulative Variance Threshold)",
                min_value=0.1, max_value=1.0,
                value=self._get_state('dfm_cumulative_variance_threshold', 0.8),  # 修正键名
                step=0.05,
                key='new_dfm_cum_variance_threshold_input',
                help="选择因子以解释至少此比例的共同方差。值在0到1之间。"
            )
            self._set_state('dfm_cumulative_variance_threshold', cum_variance_value)  # 修正键名

    def _render_max_lags_legacy(self, st_obj) -> int:
        """渲染因子自回归阶数 - 新代码后端支持的参数"""
        max_lags_value = st_obj.number_input(
            "因子自回归阶数 (Factor AR Order)",
            min_value=1,
            max_value=6,
            value=self._get_state('dfm_factor_ar_order', 1),  # 修正键名
            step=1,
            key='new_dfm_max_lags_input',
            help="因子的自回归阶数。1表示AR(1)，2表示AR(2)，以此类推。较高的阶数可以捕捉更复杂的动态特征，但会增加计算复杂度。"
        )
        self._set_state('dfm_factor_ar_order', max_lags_value)  # 修正键名
        return max_lags_value

    def _render_advanced_parameters_legacy(self, st_obj) -> Dict[str, Any]:
        """渲染高级参数设置 - 严格按照老代码真实存在的参数"""
        # 注意：老代码中没有用户可调的高级参数UI
        # EM收敛准则等参数都是在代码中预设的，不是用户界面参数
        # 因此这里返回空字典，不渲染任何UI
        return {}

    def _render_parameters_summary_legacy(self, st_obj, variable_selection_method: str,
                                        factor_selection_strategy: str, advanced_params: Dict[str, Any]):
        """显示参数摘要 - 严格按照老代码真实存在的参数"""
        # 老代码中没有参数摘要显示，因此这里不渲染任何内容
        pass
    
    def _define_parameter_constraints(self) -> Dict[str, Dict]:
        """
        定义参数约束
        
        Returns:
            参数约束字典
        """
        return {
            'max_iter': {  # 修正键名
                'min': 1,
                'max': 1000,
                'type': int
            },
            'factor_ar_order': {  # 修正键名
                'min': 1,
                'max': 6,
                'type': int
            },
            'fixed_number_of_factors': {
                'min': 1,
                'max': 20,
                'type': int
            },
            'cumulative_variance_threshold': {  # 修正键名
                'min': 0.1,
                'max': 1.0,
                'type': float
            }
        }
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        获取默认参数
        
        Returns:
            默认参数字典
        """
        try:
            # 尝试从配置获取默认值
            config_available = True
            
            if config_available:
                return {
                    'variable_selection_method': 'global_backward',
                    'enable_variable_selection': True,
                    'max_iter': getattr(UIDefaults, 'MAX_ITERATIONS_DEFAULT', 30),  # 修正键名
                    'factor_ar_order': 1,  # 修正键名
                    'factor_selection_strategy': getattr(TrainDefaults, 'FACTOR_SELECTION_STRATEGY', 'fixed_number'),
                    'fixed_number_of_factors': getattr(TrainDefaults, 'FIXED_NUMBER_OF_FACTORS', 3),
                    'cumulative_variance_threshold': getattr(TrainDefaults, 'CUM_VARIANCE_THRESHOLD', 0.8)  # 修正键名
                }
            else:
                # 硬编码默认值
                return {
                    'variable_selection_method': 'global_backward',
                    'enable_variable_selection': True,
                    'max_iter': 30,  # 修正键名
                    'factor_ar_order': 1,  # 修正键名
                    'factor_selection_strategy': 'fixed_number',
                    'fixed_number_of_factors': 3,
                    'cumulative_variance_threshold': 0.8  # 修正键名
                }
                
        except Exception as e:
            logger.error(f"获取默认参数失败: {e}")
            # 最基本的默认值
            return {
                'variable_selection_method': 'global_backward',
                'enable_variable_selection': True,
                'max_iter': 30,  # 修正键名
                'factor_ar_order': 1,  # 修正键名
                'factor_selection_strategy': 'fixed_number',
                'fixed_number_of_factors': 3,
                'cumulative_variance_threshold': 0.8  # 修正键名
            }
    
    def _validate_parameter_constraints(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证参数约束
        
        Args:
            parameters: 参数字典
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        try:
            for param_name, constraints in self._parameter_constraints.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    
                    # 类型检查
                    expected_type = constraints.get('type', type(value))
                    if not isinstance(value, expected_type):
                        try:
                            value = expected_type(value)
                        except (ValueError, TypeError):
                            errors.append(f"{param_name} 必须是 {expected_type.__name__} 类型")
                            continue
                    
                    # 范围检查
                    if 'min' in constraints and value < constraints['min']:
                        errors.append(f"{param_name} 不能小于 {constraints['min']}")
                    
                    if 'max' in constraints and value > constraints['max']:
                        errors.append(f"{param_name} 不能大于 {constraints['max']}")

            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"参数约束验证失败: {e}")
            return False, [f"参数验证过程中发生错误: {e}"]
    
    def _render_variable_selection_method(self, st_obj) -> str:
        """
        渲染变量选择方法
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            选择的变量选择方法
        """
        variable_selection_options = {
            'none': "无筛选 (使用全部已选变量)",
            'global_backward': "全局后向剔除 (在已选变量中筛选)"
        }
        
        current_method = self._get_state('dfm_variable_selection_method', 'global_backward')
        
        selected_method = st_obj.selectbox(
            "变量选择方法",
            options=list(variable_selection_options.keys()),
            format_func=lambda x: variable_selection_options[x],
            index=list(variable_selection_options.keys()).index(current_method),
            key=f"{self.get_state_key_prefix()}_variable_selection_method",
            help=(
                "选择在已选变量基础上的筛选方法：\n"
                "- 无筛选: 直接使用所有已选择的变量\n"
                "- 全局后向剔除: 从已选变量开始，逐个剔除不重要的变量"
            )
        )
        
        # 更新状态
        self._set_state('dfm_variable_selection_method', selected_method)
        self._set_state('dfm_enable_variable_selection', selected_method != 'none')
        
        return selected_method

    def _render_max_iterations(self, st_obj) -> int:
        """
        渲染最大迭代次数

        Args:
            st_obj: Streamlit对象

        Returns:
            最大迭代次数
        """
        current_value = self._get_state('dfm_max_iter', 30)  # 修正键名

        max_iterations = st_obj.number_input(
            "最大迭代次数 (Max Iterations for EM)",
            min_value=1,
            max_value=1000,
            value=current_value,
            step=10,
            key=f"{self.get_state_key_prefix()}_max_iterations",
            help="EM估计算法允许的最大迭代次数。"
        )

        # 更新状态
        self._set_state('dfm_max_iter', max_iterations)  # 修正键名

        return max_iterations

    def _render_max_lags(self, st_obj) -> int:
        """
        渲染因子自回归阶数

        Args:
            st_obj: Streamlit对象

        Returns:
            因子自回归阶数
        """
        current_value = self._get_state('dfm_factor_ar_order', 1)  # 修正键名

        max_lags = st_obj.number_input(
            "因子自回归阶数 (Factor AR Order)",
            min_value=1,
            max_value=6,
            value=current_value,
            step=1,
            key=f"{self.get_state_key_prefix()}_max_lags",
            help="因子的自回归阶数。1表示AR(1)，2表示AR(2)，以此类推。较高的阶数可以捕捉更复杂的动态特征，但会增加计算复杂度。"
        )

        # 更新状态
        self._set_state('dfm_factor_ar_order', max_lags)  # 修正键名

        return max_lags

    def _render_factor_selection_strategy(self, st_obj) -> str:
        """
        渲染因子选择策略

        Args:
            st_obj: Streamlit对象

        Returns:
            选择的因子选择策略
        """
        factor_selection_options = {
            'fixed_number': "固定因子数量 (Fixed Number of Factors)",
            'cumulative_variance': "累积共同方差 (Cumulative Common Variance)"
        }

        current_strategy = self._get_state('dfm_factor_selection_strategy', 'fixed_number')

        selected_strategy = st_obj.selectbox(
            "因子选择策略",
            options=list(factor_selection_options.keys()),
            format_func=lambda x: factor_selection_options[x],
            index=list(factor_selection_options.keys()).index(current_strategy),
            key=f"{self.get_state_key_prefix()}_factor_selection_strategy",
            help=(
                "选择确定因子数量的方法：\n"
                "- 固定因子数量: 手动指定固定的因子数量\n"
                "- 累积共同方差: 基于解释的累积方差比例选择因子数"
            )
        )

        # 更新状态
        self._set_state('dfm_factor_selection_strategy', selected_strategy)

        return selected_strategy

    def _render_strategy_specific_parameters(self, st_obj, strategy: str) -> Dict[str, Any]:
        """
        渲染策略特定参数

        Args:
            st_obj: Streamlit对象
            strategy: 因子选择策略

        Returns:
            策略特定参数字典
        """
        params = {}

        if strategy == 'fixed_number':
            # 固定因子数量
            fixed_factors = st_obj.number_input(
                "固定因子数量 (Fixed Number of Factors)",
                min_value=1,
                max_value=20,
                value=self._get_state('dfm_fixed_number_of_factors', 3),
                step=1,
                key=f"{self.get_state_key_prefix()}_fixed_number_of_factors",
                help="直接指定模型中要使用的因子数量。"
            )

            # 更新状态
            self._set_state('dfm_fixed_number_of_factors', fixed_factors)

            params = {
                'fixed_number_of_factors': fixed_factors
            }

        elif strategy == 'cumulative_variance':
            # 累积共同方差
            cum_variance_threshold = st_obj.slider(
                "累积共同方差阈值 (Cumulative Variance Threshold)",
                min_value=0.1,
                max_value=1.0,
                value=self._get_state('dfm_cumulative_variance_threshold', 0.8),  # 修正键名
                step=0.05,
                key=f"{self.get_state_key_prefix()}_cum_variance_threshold",
                help="选择因子以解释至少此比例的共同方差。值在0到1之间。"
            )

            # 更新状态
            self._set_state('dfm_cumulative_variance_threshold', cum_variance_threshold)  # 修正键名

            params = {
                'cumulative_variance_threshold': cum_variance_threshold  # 修正返回的键名
            }

        return params

    def _render_parameter_summary(self, st_obj, parameters: Dict[str, Any]) -> None:
        """
        渲染参数摘要

        Args:
            st_obj: Streamlit对象
            parameters: 参数字典
        """
        try:
            st_obj.markdown("---")
            st_obj.markdown("**[INFO] 参数配置摘要**")

            # 基本参数
            st_obj.text(f"变量选择方法: {parameters.get('variable_selection_method', '未设置')}")
            st_obj.text(f"最大迭代次数: {parameters.get('max_iter', '未设置')}")
            st_obj.text(f"因子自回归阶数: {parameters.get('factor_ar_order', '未设置')}")
            st_obj.text(f"因子选择策略: {parameters.get('factor_selection_strategy', '未设置')}")

            # 策略特定参数
            strategy = parameters.get('factor_selection_strategy')
            if strategy == 'fixed_number':
                st_obj.text(f"固定因子数量: {parameters.get('fixed_number_of_factors', '未设置')}")
            elif strategy == 'cumulative_variance':
                st_obj.text(f"累积方差阈值: {parameters.get('cumulative_variance_threshold', '未设置')}")

        except Exception as e:
            logger.error(f"渲染参数摘要失败: {e}")

    def _render_action_buttons(self, st_obj, parameters: Dict[str, Any]) -> None:
        """
        渲染操作按钮

        Args:
            st_obj: Streamlit对象
            parameters: 当前参数
        """
        st_obj.markdown("---")

        col1, col2, col3 = st_obj.columns(3)

        with col1:
            if st_obj.button("[LOADING] 重置为默认值", key=f"{self.get_state_key_prefix()}_reset_defaults"):
                self._reset_to_defaults()
                st_obj.rerun()

        with col2:
            if st_obj.button("导出参数", key=f"{self.get_state_key_prefix()}_export_params"):
                exported = self._export_parameters(parameters)
                st_obj.download_button(
                    label="下载参数配置",
                    data=exported,
                    file_name="dfm_model_parameters.json",
                    mime="application/json",
                    key=f"{self.get_state_key_prefix()}_download_params"
                )

        with col3:
            uploaded_file = st_obj.file_uploader(
                "导入参数",
                type=['json'],
                key=f"{self.get_state_key_prefix()}_import_params"
            )

            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    if self._import_parameters(content):
                        st_obj.success("[SUCCESS] 参数导入成功")
                        st_obj.rerun()
                    else:
                        st_obj.error("[ERROR] 参数导入失败")
                except Exception as e:
                    st_obj.error(f"[ERROR] 导入错误: {e}")

    def _reset_to_defaults(self) -> None:
        """重置为默认值"""
        try:
            defaults = self._get_default_parameters()

            for key, value in defaults.items():
                state_key = f'dfm_{key}' if not key.startswith('dfm_') else key
                self._set_state(state_key, value)

            logger.info("参数已重置为默认值")

        except Exception as e:
            logger.error(f"重置参数失败: {e}")

    def _export_parameters(self, parameters: Dict[str, Any]) -> str:
        """
        导出参数配置

        Args:
            parameters: 参数字典

        Returns:
            JSON格式的参数配置
        """
        try:
            export_data = {
                'dfm_model_parameters': parameters,
                'export_timestamp': str(datetime.now()),
                'version': '1.0'
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"导出参数失败: {e}")
            return "{}"

    def _import_parameters(self, json_content: str) -> bool:
        """
        导入参数配置

        Args:
            json_content: JSON格式的参数配置

        Returns:
            导入是否成功
        """
        try:
            data = json.loads(json_content)

            if 'dfm_model_parameters' in data:
                parameters = data['dfm_model_parameters']

                # 验证参数
                is_valid, errors = self._validate_parameter_constraints(parameters)
                if not is_valid:
                    logger.warning(f"导入的参数验证失败: {errors}")
                    return False

                # 设置参数
                for key, value in parameters.items():
                    state_key = f'dfm_{key}' if not key.startswith('dfm_') else key
                    self._set_state(state_key, value)

                logger.info("参数导入成功")
                return True
            else:
                logger.warning("导入文件格式不正确")
                return False

        except Exception as e:
            logger.error(f"导入参数失败: {e}")
            return False

    def _get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                value = dfm_manager.get_dfm_state('train_model', key, None)
                if value is not None:
                    return value
                return default
            else:
                # 如果DFM状态管理器不可用，抛出明确错误
                raise RuntimeError(f"DFM状态管理器不可用，无法获取状态: {key}")

        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            raise RuntimeError(f"状态获取失败: {key} - {str(e)}")

    def _set_state(self, key: str, value: Any) -> None:
        """设置状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                dfm_manager.set_dfm_state('train_model', key, value)
            else:
                # 如果DFM状态管理器不可用，抛出明确错误
                raise RuntimeError(f"DFM状态管理器不可用，无法设置状态: {key}")

        except Exception as e:
            logger.error(f"设置状态失败: {e}")
            raise RuntimeError(f"状态设置失败: {key} - {str(e)}")
