# -*- coding: utf-8 -*-
"""
DFM参数配置组件

提供数据预处理参数配置功能
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, date

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager


logger = logging.getLogger(__name__)


class ParameterConfigComponent(DFMComponent):
    """DFM参数配置组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化参数配置组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._parameter_defaults = {
            'dfm_param_target_variable': '规模以上工业增加值:当月同比',
            'dfm_param_target_sheet_name': '工业增加值同比增速_月度_同花顺',
            'dfm_param_target_freq': 'W-FRI',
            'dfm_param_remove_consecutive_nans': True,
            'dfm_param_consecutive_nan_threshold': 10,
            'dfm_param_type_mapping_sheet': '指标体系',
            'dfm_param_data_start_date': date(2020, 1, 1),
            'dfm_param_data_end_date': date(2025, 12, 31)
        }
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "parameter_config"
    
    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键
        
        Returns:
            List[str]: 状态键列表
        """
        return list(self._parameter_defaults.keys())
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含参数配置
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查必需参数
            required_params = [
                'dfm_param_target_variable',
                'dfm_param_target_sheet_name',
                'dfm_param_target_freq',
                'dfm_param_data_start_date',
                'dfm_param_data_end_date'
            ]
            
            for param in required_params:
                if param not in data or data[param] is None:
                    logger.warning(f"缺少必需参数: {param}")
                    return False
            
            # 验证日期范围
            start_date = data.get('dfm_param_data_start_date')
            end_date = data.get('dfm_param_data_end_date')
            if not self._validate_date_range(start_date, end_date):
                return False
            
            # 验证阈值
            threshold = data.get('dfm_param_consecutive_nan_threshold', 0)
            if not self._validate_threshold(threshold):
                return False
            
            # 验证字符串参数不为空
            string_params = [
                'dfm_param_target_variable',
                'dfm_param_target_sheet_name',
                'dfm_param_target_freq',
                'dfm_param_type_mapping_sheet'
            ]
            
            for param in string_params:
                value = data.get(param, '')
                if not isinstance(value, str) or not value.strip():
                    logger.warning(f"参数不能为空: {param}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"参数验证失败: {e}")
            return False
    
    def handle_service_error(self, error: Exception) -> None:
        """
        处理服务错误
        
        Args:
            error: 异常对象
        """
        error_msg = f"参数配置服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
    
    def render(self, st_obj) -> Optional[Dict[str, Any]]:
        """
        渲染参数配置组件
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            配置的参数字典或None
        """
        try:
            st_obj.markdown("### 预处理参数配置")
            
            # 初始化默认参数
            self._initialize_default_parameters()
            
            # 检测并更新日期范围
            self._update_date_range_from_file()
            
            # 渲染参数配置界面
            parameters = self._render_parameter_inputs(st_obj)
            
            # 验证参数
            if self.validate_input(parameters):
                # 保存参数到状态管理器
                self._save_parameters(parameters)
                return parameters
            else:
                self._show_validation_error("参数配置验证失败，请检查输入值。")
                return None
                
        except Exception as e:
            self.handle_service_error(e)
            return None
    
    def _initialize_default_parameters(self) -> None:
        """初始化默认参数"""
        try:
            for key, default_value in self._parameter_defaults.items():
                current_value = self._get_state(key)
                if current_value is None:
                    self._set_state(key, default_value)
                    logger.debug(f"初始化默认参数: {key} = {default_value}")
        except Exception as e:
            logger.error(f"初始化默认参数失败: {e}")
    
    def _update_date_range_from_file(self) -> None:
        """从上传的文件更新日期范围"""
        try:
            # 检查是否需要日期检测
            if not self._get_state('dfm_date_detection_needed', False):
                return
            
            # 获取上传的文件
            uploaded_file = self._get_state('dfm_training_data_file')
            if uploaded_file is None:
                return
            
            # 检测日期范围
            start_date, end_date = self._detect_date_range_from_file(uploaded_file)
            
            if start_date and end_date:
                # 更新日期参数
                self._set_state('dfm_param_data_start_date', start_date)
                self._set_state('dfm_param_data_end_date', end_date)
                
                # 标记检测完成
                self._set_state('dfm_date_detection_needed', False)
                
                logger.info(f"从文件检测到日期范围: {start_date} 到 {end_date}")
                
        except Exception as e:
            logger.error(f"从文件更新日期范围失败: {e}")
    
    def _render_parameter_inputs(self, st_obj) -> Dict[str, Any]:
        """
        渲染参数输入界面
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            参数字典
        """
        parameters = {}
        
        # 第一行：日期范围
        row1_col1, row1_col2 = st_obj.columns(2)
        
        with row1_col1:
            start_date = st_obj.date_input(
                "数据开始日期",
                value=self._get_state('dfm_param_data_start_date', date(2020, 1, 1)),
                key=f"{self.get_state_key_prefix()}_start_date",
                help="设置系统处理数据的最早日期边界。训练期、验证期必须在此日期之后。"
            )
            parameters['dfm_param_data_start_date'] = start_date
        
        with row1_col2:
            end_date = st_obj.date_input(
                "数据结束日期",
                value=self._get_state('dfm_param_data_end_date', date(2025, 12, 31)),
                key=f"{self.get_state_key_prefix()}_end_date",
                help="设置系统处理数据的最晚日期边界。训练期、验证期必须在此日期之前。"
            )
            parameters['dfm_param_data_end_date'] = end_date
        
        # 第二行：目标变量配置
        row2_col1, row2_col2 = st_obj.columns(2)
        
        with row2_col1:
            target_sheet = st_obj.text_input(
                "目标工作表名称 (Target Sheet Name)",
                value=self._get_state('dfm_param_target_sheet_name', ''),
                key=f"{self.get_state_key_prefix()}_target_sheet"
            )
            parameters['dfm_param_target_sheet_name'] = target_sheet
        
        with row2_col2:
            target_variable = st_obj.text_input(
                "目标变量 (Target Variable)",
                value=self._get_state('dfm_param_target_variable', ''),
                key=f"{self.get_state_key_prefix()}_target_variable"
            )
            parameters['dfm_param_target_variable'] = target_variable
        
        # 第三行：数据处理参数
        row3_col1, row3_col2 = st_obj.columns(2)
        
        with row3_col1:
            nan_threshold = st_obj.number_input(
                "连续 NaN 阈值 (Consecutive NaN Threshold)",
                min_value=0,
                value=self._get_state('dfm_param_consecutive_nan_threshold', 10),
                step=1,
                key=f"{self.get_state_key_prefix()}_nan_threshold"
            )
            parameters['dfm_param_consecutive_nan_threshold'] = nan_threshold
        
        with row3_col2:
            remove_nans = st_obj.checkbox(
                "移除过多连续 NaN 的变量",
                value=self._get_state('dfm_param_remove_consecutive_nans', True),
                key=f"{self.get_state_key_prefix()}_remove_nans",
                help="移除列中连续缺失值数量超过阈值的变量"
            )
            parameters['dfm_param_remove_consecutive_nans'] = remove_nans
        
        # 第四行：频率和映射表
        row4_col1, row4_col2 = st_obj.columns(2)
        
        with row4_col1:
            target_freq = st_obj.text_input(
                "目标频率 (Target Frequency)",
                value=self._get_state('dfm_param_target_freq', 'W-FRI'),
                help="例如: W-FRI, D, M, Q",
                key=f"{self.get_state_key_prefix()}_target_freq"
            )
            parameters['dfm_param_target_freq'] = target_freq
        
        with row4_col2:
            type_mapping_sheet = st_obj.text_input(
                "指标映射表名称 (Type Mapping Sheet)",
                value=self._get_state('dfm_param_type_mapping_sheet', '指标体系'),
                key=f"{self.get_state_key_prefix()}_type_mapping"
            )
            parameters['dfm_param_type_mapping_sheet'] = type_mapping_sheet
        
        return parameters
    
    def _save_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        保存参数到状态管理器
        
        Args:
            parameters: 参数字典
        """
        try:
            for key, value in parameters.items():
                self._set_state(key, value)
            logger.debug(f"保存参数: {len(parameters)} 个")
        except Exception as e:
            logger.error(f"保存参数失败: {e}")
            raise
    
    def _validate_date_range(self, start_date: date, end_date: date) -> bool:
        """
        验证日期范围
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 日期范围是否有效
        """
        try:
            if not isinstance(start_date, date) or not isinstance(end_date, date):
                return False
            
            if start_date >= end_date:
                logger.warning(f"无效日期范围: {start_date} >= {end_date}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"日期范围验证失败: {e}")
            return False
    
    def _validate_threshold(self, threshold: int) -> bool:
        """
        验证阈值
        
        Args:
            threshold: 阈值
            
        Returns:
            bool: 阈值是否有效
        """
        try:
            if not isinstance(threshold, (int, float)):
                return False
            
            if threshold < 0:
                logger.warning(f"阈值不能为负数: {threshold}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"阈值验证失败: {e}")
            return False
    
    def _detect_date_range_from_file(self, uploaded_file) -> Tuple[Optional[date], Optional[date]]:
        """
        从文件检测日期范围

        Args:
            uploaded_file: 上传的文件对象

        Returns:
            Tuple[开始日期, 结束日期]
        """
        try:
            # 这里应该实现实际的日期检测逻辑
            # 暂时返回默认值
            return date(2020, 1, 1), date(2025, 12, 31)
        except Exception as e:
            logger.error(f"日期范围检测失败: {e}")
            return None, None
    
    def _show_validation_error(self, message: str) -> None:
        """显示验证错误消息"""
        st.error(message)
    
    def _show_validation_warning(self, message: str) -> None:
        """显示验证警告消息"""
        st.warning(message)
    
    def _show_validation_success(self, message: str) -> None:
        """显示验证成功消息"""
        st.success(message)
    
    def _sync_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        同步参数到其他模块
        
        Args:
            parameters: 参数字典
        """
        try:
            # 这里可以实现参数同步逻辑
            # 例如同步到模型训练模块
            pass
        except Exception as e:
            logger.error(f"参数同步失败: {e}")
    
    def _get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                value = dfm_manager.get_dfm_state('data_prep', key, None)
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
                success = dfm_manager.set_dfm_state('data_prep', key, value)
                if success:
                    return

            # 如果DFM管理器不可用，记录错误
            logger.error(f"DFM状态管理器不可用，无法设置状态: {key}")

        except Exception as e:
            logger.error(f"设置状态失败: {e}")
            raise RuntimeError(f"状态设置失败: {key} - {str(e)}")
