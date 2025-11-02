# -*- coding: utf-8 -*-
"""
DFM模型加载组件

提供模型文件上传、验证、加载和信息展示功能
"""

import streamlit as st
import joblib
import pickle
import logging
from typing import Dict, Any, Optional, Tuple

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager


logger = logging.getLogger(__name__)


class ModelLoadComponent(DFMComponent):
    """DFM模型加载组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化模型加载组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._supported_model_formats = ['joblib']
        self._supported_metadata_formats = ['pkl']
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "model_load"
    
    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键
        
        Returns:
            List[str]: 状态键列表
        """
        return [
            'dfm_model_file_indep',
            'dfm_metadata_file_indep',
            'dfm_model_loaded',
            'dfm_metadata_loaded',
            'dfm_load_status',
            'dfm_load_errors'
        ]
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含模型文件和元数据文件
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查模型文件
            model_file = data.get('model_file')
            if not self._is_valid_file_object(model_file):
                logger.warning("模型文件无效或缺失")
                return False
            
            # 检查元数据文件
            metadata_file = data.get('metadata_file')
            if not self._is_valid_file_object(metadata_file):
                logger.warning("元数据文件无效或缺失")
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
        error_msg = f"模型加载服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
        
        # 更新错误状态
        self._update_load_status("加载失败", [str(error)])
    
    def render(self, st_obj) -> Optional[Dict[str, Any]]:
        """
        渲染模型加载组件
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            加载结果字典或None
        """
        try:
            st_obj.markdown("### [DATA] 模型文件加载")
            
            # 清理无效的文件状态
            self._cleanup_invalid_file_states()
            
            # 渲染文件上传区域
            files_ready = self._render_file_upload_section(st_obj)
            
            # 渲染文件状态
            status_ready = self._render_file_status(st_obj)
            
            if not files_ready or not status_ready:
                st_obj.info("[INFO] 请上传模型文件和元数据文件以继续分析。")
                return None
            
            # 加载文件
            model_file = self._get_state('dfm_model_file_indep')
            metadata_file = self._get_state('dfm_metadata_file_indep')
            
            model_data, metadata = self._load_files(model_file, metadata_file)
            
            if model_data is None or metadata is None:
                st_obj.error("[ERROR] 无法加载模型数据，请检查文件格式和内容。")
                return None
            
            # 显示加载成功信息
            st_obj.success("[SUCCESS] 模型和元数据加载成功！")
            
            # 渲染模型信息
            self._render_model_info(st_obj, metadata)
            
            # 返回加载结果
            return {
                'model_data': model_data,
                'metadata': metadata,
                'load_status': self._get_load_status(),
                'load_errors': self._get_state('dfm_load_errors', [])
            }
                
        except Exception as e:
            self.handle_service_error(e)
            return None
    
    def _render_file_upload_section(self, st_obj) -> bool:
        """
        渲染文件上传区域
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            bool: 文件是否准备就绪
        """
        # 创建两列布局
        col_model, col_metadata = st_obj.columns(2)
        
        files_uploaded = False
        
        # 模型文件上传
        with col_model:
            st_obj.markdown("**DFM 模型文件 (.joblib)**")
            uploaded_model_file = st_obj.file_uploader(
                "选择模型文件",
                type=self._supported_model_formats,
                key=f"{self.get_state_key_prefix()}_model_upload",
                help="上传训练好的DFM模型文件，通常为.joblib格式"
            )
            
            if uploaded_model_file:
                self._set_state("dfm_model_file_indep", uploaded_model_file)
                st_obj.success(f"[SUCCESS] 已上传: {uploaded_model_file.name}")
                files_uploaded = True
            else:
                existing_model_file = self._get_state('dfm_model_file_indep', None)
                if self._is_valid_file_object(existing_model_file):
                    file_name = getattr(existing_model_file, 'name', '未知文件')
                    st_obj.info(f"当前文件: {file_name}")
                    files_uploaded = True
        
        # 元数据文件上传
        with col_metadata:
            st_obj.markdown("**元数据文件 (.pkl)**")
            uploaded_metadata_file = st_obj.file_uploader(
                "选择元数据文件",
                type=self._supported_metadata_formats,
                key=f"{self.get_state_key_prefix()}_metadata_upload",
                help="上传包含训练元数据的.pkl文件"
            )
            
            if uploaded_metadata_file:
                self._set_state("dfm_metadata_file_indep", uploaded_metadata_file)
                st_obj.success(f"[SUCCESS] 已上传: {uploaded_metadata_file.name}")
                files_uploaded = True
            else:
                existing_metadata_file = self._get_state('dfm_metadata_file_indep', None)
                if self._is_valid_file_object(existing_metadata_file):
                    file_name = getattr(existing_metadata_file, 'name', '未知文件')
                    st_obj.info(f"当前文件: {file_name}")
                    files_uploaded = True
        
        return files_uploaded
    
    def _render_file_status(self, st_obj) -> bool:
        """
        渲染文件状态
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            bool: 文件状态是否就绪
        """
        model_file = self._get_state('dfm_model_file_indep', None)
        metadata_file = self._get_state('dfm_metadata_file_indep', None)
        
        model_file_valid = self._is_valid_file_object(model_file)
        metadata_file_valid = self._is_valid_file_object(metadata_file)
        
        if model_file_valid and metadata_file_valid:
            st_obj.success("[SUCCESS] 所有必需文件已上传完成，可以进行模型分析。")
            return True
        else:
            missing_files = []
            if not model_file_valid:
                missing_files.append("模型文件")
            if not metadata_file_valid:
                missing_files.append("元数据文件")
            
            st_obj.warning(f"[WARNING] 缺少文件: {', '.join(missing_files)}。请上传所有文件后再进行分析。")
            return False
    
    def _load_files(self, model_file, metadata_file) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        加载模型和元数据文件
        
        Args:
            model_file: 模型文件对象
            metadata_file: 元数据文件对象
            
        Returns:
            (模型数据, 元数据)元组
        """
        try:
            # 更新加载状态
            self._update_load_status("正在加载...", [])
            
            # 加载模型文件
            model_data = self._load_model_file(model_file)
            if model_data is not None:
                self._set_state('dfm_model_loaded', True)
            
            # 加载元数据文件
            metadata = self._load_metadata_file(metadata_file)
            if metadata is not None:
                self._set_state('dfm_metadata_loaded', True)
            
            # 更新最终状态
            if model_data is not None and metadata is not None:
                self._update_load_status("加载完成", [])
            else:
                self._update_load_status("加载失败", ["部分文件加载失败"])
            
            return model_data, metadata
            
        except Exception as e:
            logger.error(f"文件加载失败: {e}")
            self._update_load_status("加载失败", [str(e)])
            return None, None
    
    def _load_model_file(self, model_file) -> Optional[Any]:
        """
        加载模型文件
        
        Args:
            model_file: 模型文件对象
            
        Returns:
            模型数据或None
        """
        try:
            if not self._is_valid_file_object(model_file):
                return None
            
            model_file.seek(0)  # 重置文件指针
            model_data = joblib.load(model_file)
            logger.info("模型文件加载成功")
            return model_data
            
        except Exception as e:
            file_name = getattr(model_file, 'name', '未知文件')
            logger.error(f"加载模型文件 ('{file_name}') 时出错: {e}")
            return None

    def _render_model_info(self, st_obj, metadata: Optional[Dict]) -> None:
        """
        渲染模型信息

        Args:
            st_obj: Streamlit对象
            metadata: 元数据字典
        """
        if metadata is None:
            st_obj.warning("[WARNING] 无法显示模型信息：元数据缺失")
            return

        st_obj.markdown("---")
        st_obj.markdown("**[DATA] 模型信息摘要**")

        try:
            # 基本信息
            target_variable = metadata.get('target_variable', 'N/A')
            st_obj.write(f"- **目标变量:** {target_variable}")

            # 训练期信息
            train_start = metadata.get('training_start_date', 'N/A')
            train_end = metadata.get('training_end_date', metadata.get('train_end_date', 'N/A'))
            st_obj.write(f"- **训练期:** {train_start} 至 {train_end}")

            # 验证期信息
            val_start = metadata.get('validation_start_date', 'N/A')
            val_end = metadata.get('validation_end_date', 'N/A')
            st_obj.write(f"- **验证期:** {val_start} 至 {val_end}")

            # 选择的指标
            selected_indicators = metadata.get('selected_indicators', [])
            if selected_indicators:
                st_obj.write(f"- **选择指标数量:** {len(selected_indicators)}")
                with st_obj.expander("查看选择的指标", expanded=False):
                    for indicator in selected_indicators:
                        st_obj.write(f"  • {indicator}")
            else:
                st_obj.write("- **选择指标数量:** 0")

            # 模型参数
            model_params = metadata.get('model_parameters', {})
            if model_params:
                st_obj.write("- **模型参数:**")
                for param, value in model_params.items():
                    st_obj.write(f"  • {param}: {value}")

        except Exception as e:
            logger.error(f"渲染模型信息失败: {e}")
            st_obj.error(f"显示模型信息时出错: {e}")

    def _is_valid_file_object(self, file_obj) -> bool:
        """
        检查是否为有效的文件对象

        Args:
            file_obj: 文件对象

        Returns:
            bool: 是否为有效文件对象
        """
        if file_obj is None:
            return False

        # 检查是否具有文件对象的必要方法和属性
        return (hasattr(file_obj, 'seek') and
                hasattr(file_obj, 'read') and
                hasattr(file_obj, 'name') and
                getattr(file_obj, 'name', '未知文件') != '未知文件')

    def _cleanup_invalid_file_states(self) -> None:
        """清理可能存在的无效文件状态"""
        try:
            model_file = self._get_state('dfm_model_file_indep', None)
            metadata_file = self._get_state('dfm_metadata_file_indep', None)

            # 检查模型文件状态
            if model_file is not None and not self._is_valid_file_object(model_file):
                self._set_state('dfm_model_file_indep', None)
                logger.info("清理了无效的模型文件状态")

            # 检查元数据文件状态
            if metadata_file is not None and not self._is_valid_file_object(metadata_file):
                self._set_state('dfm_metadata_file_indep', None)
                logger.info("清理了无效的元数据文件状态")

        except Exception as e:
            logger.error(f"清理文件状态失败: {e}")

    def _get_load_status(self) -> str:
        """
        获取当前加载状态

        Returns:
            加载状态字符串
        """
        return self._get_state('dfm_load_status', '等待加载')

    def _update_load_status(self, status: str, errors: list = None) -> None:
        """
        更新加载状态

        Args:
            status: 新的加载状态
            errors: 错误列表
        """
        self._set_state('dfm_load_status', status)

        if errors is not None:
            self._set_state('dfm_load_errors', errors)

    def _get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        try:
            dfm_manager = get_global_dfm_manager()
            if dfm_manager:
                value = dfm_manager.get_dfm_state('model_analysis', key, default)
                logger.debug(f"获取DFM模型分析状态成功: {key} = {type(value).__name__}")
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
                success = dfm_manager.set_dfm_state('model_analysis', key, value)
                if success:
                    logger.debug(f"设置DFM模型分析状态成功: {key}")
                    return True
                else:
                    logger.error(f"设置DFM模型分析状态失败: {key}")
                    return False
            else:
                logger.error(f"DFM状态管理器不可用，无法设置状态: {key}")
                return False

        except Exception as e:
            logger.error(f"设置状态失败: {key} - {str(e)}")
            return False
    
    def _load_metadata_file(self, metadata_file) -> Optional[Dict]:
        """
        加载元数据文件
        
        Args:
            metadata_file: 元数据文件对象
            
        Returns:
            元数据字典或None
        """
        try:
            if not self._is_valid_file_object(metadata_file):
                return None
            
            metadata_file.seek(0)  # 重置文件指针
            metadata = pickle.load(metadata_file)
            logger.info("元数据文件加载成功")
            return metadata
            
        except Exception as e:
            file_name = getattr(metadata_file, 'name', '未知文件')
            logger.error(f"加载元数据文件 ('{file_name}') 时出错: {e}")
            return None
