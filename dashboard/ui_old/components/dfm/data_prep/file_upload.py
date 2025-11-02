# -*- coding: utf-8 -*-
"""
DFM文件上传组件

提供文件上传、验证、格式检查和结构诊断功能
"""

import streamlit as st
import pandas as pd
import io
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from dashboard.ui.components.dfm.base import DFMComponent, DFMServiceManager
from dashboard.core import get_global_dfm_manager


logger = logging.getLogger(__name__)


class FileUploadComponent(DFMComponent):
    """DFM文件上传组件"""
    
    def __init__(self, service_manager: Optional[DFMServiceManager] = None):
        """
        初始化文件上传组件
        
        Args:
            service_manager: DFM服务管理器
        """
        super().__init__(service_manager)
        self._supported_formats = ['xlsx', 'xls']
        self._max_file_size = 100 * 1024 * 1024  # 100MB
    
    def get_component_id(self) -> str:
        """获取组件ID"""
        return "file_upload"

    def get_state_keys(self) -> list:
        """
        获取组件相关的状态键

        Returns:
            List[str]: 状态键列表
        """
        return [
            'dfm_training_data_file',
            'dfm_uploaded_excel_file_path',
            'dfm_use_full_data_preparation',
            'dfm_file_processed',
            'dfm_date_detection_needed'
        ]
    
    def validate_input(self, data: Dict) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据字典，包含uploaded_file键
            
        Returns:
            bool: 验证是否通过
        """
        try:
            uploaded_file = data.get('uploaded_file')
            
            if uploaded_file is None:
                return False
            
            # 检查文件格式
            if not self._is_valid_format(uploaded_file.name):
                logger.warning(f"不支持的文件格式: {uploaded_file.name}")
                return False
            
            # 检查文件大小
            if hasattr(uploaded_file, 'size'):
                try:
                    if uploaded_file.size > self._max_file_size:
                        logger.warning(f"文件过大: {uploaded_file.size} bytes")
                        return False
                except (TypeError, AttributeError):
                    # 如果size属性不是数字类型，跳过大小检查
                    pass
            
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
        error_msg = f"文件上传服务错误: {str(error)}"
        logger.error(error_msg)
        st.error(error_msg)
    
    def render(self, st_obj) -> Optional[Any]:
        """
        渲染文件上传组件
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            上传的文件对象或None
        """
        try:
            st_obj.markdown("### [DATA] 数据文件上传")
            
            # 渲染文件上传区域
            uploaded_file = self._render_upload_section(st_obj)
            
            if uploaded_file is not None:
                # 验证输入
                if not self.validate_input({'uploaded_file': uploaded_file}):
                    st_obj.error("文件验证失败，请检查文件格式和大小。")
                    return None
                
                # 检查文件是否发生变化
                file_changed = self._is_file_changed(uploaded_file)
                
                # 保存文件到状态管理器
                self._save_file_to_state(uploaded_file, file_changed)
                
                # 显示上传成功信息
                st_obj.success(f"文件 '{uploaded_file.name}' 上传成功！")
                
                if file_changed:
                    st_obj.info("[DATE] 文件已加载，将自动检测日期范围。")
                
                return uploaded_file
                
            elif self._has_existing_file():
                # 显示已存在文件的信息和诊断工具
                self._render_existing_file_info(st_obj)
                return self._get_existing_file()
            else:
                st_obj.info("请上传Excel数据文件以开始数据预处理。")
                return None
                
        except Exception as e:
            self.handle_service_error(e)
            return None
    
    def _render_upload_section(self, st_obj) -> Optional[Any]:
        """
        渲染文件上传区域
        
        Args:
            st_obj: Streamlit对象
            
        Returns:
            上传的文件对象或None
        """
        return st_obj.file_uploader(
            "选择Excel数据文件",
            type=self._supported_formats,
            key=f"{self.get_state_key_prefix()}_uploader",
            help="请上传包含时间序列数据的Excel文件"
        )
    
    def _is_valid_format(self, filename: str) -> bool:
        """
        检查文件格式是否有效
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 格式是否有效
        """
        if not filename:
            return False
        
        file_extension = filename.lower().split('.')[-1]
        return file_extension in self._supported_formats
    
    def _is_file_changed(self, uploaded_file) -> bool:
        """
        检查文件是否发生变化
        
        Args:
            uploaded_file: 上传的文件对象
            
        Returns:
            bool: 文件是否发生变化
        """
        try:
            current_file = self._get_state('dfm_training_data_file')
            
            if current_file is None:
                return True
            
            if current_file.name != uploaded_file.name:
                return True
            
            # 检查处理标记
            if not self._get_state('dfm_file_processed', False):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"文件变化检测失败: {e}")
            return True
    
    def _save_file_to_state(self, uploaded_file, file_changed: bool) -> None:
        """
        保存文件到状态管理器
        
        Args:
            uploaded_file: 上传的文件对象
            file_changed: 文件是否发生变化
        """
        try:
            # 保存文件对象
            self._set_state("dfm_training_data_file", uploaded_file)
            self._set_state("dfm_uploaded_excel_file_path", uploaded_file.name)
            self._set_state("dfm_use_full_data_preparation", True)
            
            if file_changed:
                # 重置相关状态
                self._set_state("dfm_file_processed", False)
                self._set_state("dfm_date_detection_needed", True)
                
                logger.info(f"新文件上传: {uploaded_file.name}")
                
                # 标记文件已处理
                self._set_state("dfm_file_processed", True)
            
        except Exception as e:
            logger.error(f"保存文件状态失败: {e}")
            raise
    
    def _has_existing_file(self) -> bool:
        """
        检查是否有已存在的文件
        
        Returns:
            bool: 是否有已存在的文件
        """
        return self._get_state('dfm_training_data_file') is not None
    
    def _get_existing_file(self):
        """
        获取已存在的文件
        
        Returns:
            已存在的文件对象或None
        """
        return self._get_state('dfm_training_data_file')
    
    def _render_existing_file_info(self, st_obj) -> None:
        """
        渲染已存在文件的信息
        
        Args:
            st_obj: Streamlit对象
        """
        try:
            current_file = self._get_existing_file()
            if current_file:
                st_obj.info(f"当前已加载训练数据: {current_file.name}. 您可以上传新文件替换它。")
                
                # 渲染文件结构诊断工具
                self._render_file_diagnosis(st_obj)
                
        except Exception as e:
            logger.error(f"渲染已存在文件信息失败: {e}")
    
    def _render_file_diagnosis(self, st_obj) -> None:
        """
        渲染文件结构诊断工具
        
        Args:
            st_obj: Streamlit对象
        """
        with st_obj.expander("[VIEW] 文件结构诊断工具 (可选)", expanded=False):
            if st_obj.button("检查文件结构", 
                           key=f"{self.get_state_key_prefix()}_diagnosis",
                           help="查看已上传文件的内部结构，帮助诊断格式问题"):
                try:
                    self._perform_file_diagnosis(st_obj)
                except Exception as e:
                    st_obj.error(f"文件结构检查出错: {e}")
                    logger.error(f"文件诊断失败: {e}")
    
    def _perform_file_diagnosis(self, st_obj) -> None:
        """
        执行文件结构诊断
        
        Args:
            st_obj: Streamlit对象
        """
        current_file = self._get_existing_file()
        if not current_file:
            st_obj.warning("没有找到已上传的文件")
            return

        # 检查文件对象是否有效
        if not hasattr(current_file, 'getvalue'):
            st_obj.error("文件对象无效，无法读取文件内容")
            return

        try:
            # 读取文件内容
            file_bytes = current_file.getvalue()
            if not file_bytes:
                st_obj.warning("文件内容为空")
                return
            excel_file = io.BytesIO(file_bytes)
        except Exception as e:
            st_obj.error(f"读取文件内容失败: {e}")
            return
        
        if current_file.name.endswith('.xlsx'):
            # Excel文件诊断
            self._diagnose_excel_file(st_obj, excel_file)
        else:
            st_obj.warning("当前只支持Excel文件的结构诊断")
    
    def _diagnose_excel_file(self, st_obj, excel_file) -> None:
        """
        诊断Excel文件结构
        
        Args:
            st_obj: Streamlit对象
            excel_file: Excel文件对象
        """
        try:
            xl_file = pd.ExcelFile(excel_file)
            sheet_names = xl_file.sheet_names
            
            st_obj.write("**[INFO] 工作表信息:**")
            st_obj.write(f"- 工作表数量: {len(sheet_names)}")
            st_obj.write(f"- 工作表名称: {', '.join(sheet_names)}")
            
            # 显示每个工作表的基本信息
            for sheet_name in sheet_names[:3]:  # 最多显示前3个工作表
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
                    st_obj.write(f"**工作表 '{sheet_name}' 预览:**")
                    st_obj.write(f"- 列数: {len(df.columns)}")
                    st_obj.write(f"- 列名: {', '.join(df.columns.astype(str)[:10])}")
                    if len(df.columns) > 10:
                        st_obj.write("  (显示前10列)")
                    
                    # 显示数据预览
                    st_obj.dataframe(df.head(3))
                    
                except Exception as e:
                    st_obj.warning(f"无法读取工作表 '{sheet_name}': {e}")
            
            if len(sheet_names) > 3:
                st_obj.info(f"还有 {len(sheet_names) - 3} 个工作表未显示")
                
        except Exception as e:
            st_obj.error(f"Excel文件诊断失败: {e}")
            logger.error(f"Excel诊断失败: {e}")
    
    def _detect_date_range(self, uploaded_file) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        检测文件中的日期范围
        
        Args:
            uploaded_file: 上传的文件对象
            
        Returns:
            Tuple[开始日期, 结束日期]
        """
        try:
            if uploaded_file is None:
                return None, None

            # 检查文件对象是否有效
            if not hasattr(uploaded_file, 'getvalue'):
                logger.warning("文件对象无效，缺少getvalue方法")
                return None, None

            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                logger.warning("文件内容为空")
                return None, None

            excel_file = io.BytesIO(file_bytes)
            
            all_dates_found = []
            
            # 获取所有工作表
            xl_file = pd.ExcelFile(excel_file)
            sheet_names = xl_file.sheet_names
            
            # 在每个工作表中查找日期
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # 查找可能的日期列
                    for col in df.columns:
                        if any(keyword in str(col).lower() for keyword in ['date', '日期', 'time', '时间']):
                            try:
                                dates = pd.to_datetime(df[col], errors='coerce')
                                valid_dates = dates.dropna()
                                if len(valid_dates) > 0:
                                    all_dates_found.extend(valid_dates.tolist())
                            except:
                                continue
                                
                except Exception as e:
                    logger.warning(f"处理工作表 {sheet_name} 时出错: {e}")
                    continue
            
            if all_dates_found:
                return min(all_dates_found), max(all_dates_found)
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"日期范围检测失败: {e}")
            return None, None
    
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
