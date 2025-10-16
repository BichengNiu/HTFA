# -*- coding: utf-8 -*-
"""
侧边栏组件
提供侧边栏相关的UI组件
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from dashboard.ui.components.module_selector import render_main_module_selector, render_sub_module_selector, reset_button_key_tracking
from dashboard.ui.utils.state_helpers import get_staged_data, get_tools_manager_instance
from dashboard.ui.utils.button_state_manager import clear_button_state_cache
from dashboard.core import get_unified_manager, get_global_dfm_manager

logger = logging.getLogger(__name__)

class SidebarComponent:
    """侧边栏组件基类"""

    def __init__(self):
        pass

    def render(self, st_obj, **kwargs) -> None:
        """渲染侧边栏组件"""
        pass

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return []

# 注意: DataUploadSidebar 已移至 ui/components/data_input/upload.py
# 使用统一的数据上传组件: from dashboard.ui.components.data_input import DataUploadSidebar

class ParameterSidebar(SidebarComponent):
    """参数配置侧边栏组件"""
    
    def __init__(self, title: str = "参数配置", 
                 parameters: List[Dict[str, Any]] = None):
        super().__init__()
        self.title = title
        self.parameters = parameters or []
    
    def render(self, st_obj, **kwargs) -> Dict[str, Any]:
        """渲染参数配置侧边栏"""
        with st_obj.sidebar:
            st_obj.markdown(f"### {self.title}")
            
            param_values = {}
            
            for param in self.parameters:
                param_type = param.get('type', 'text')
                param_key = param['key']
                param_label = param.get('label', param_key)
                param_default = param.get('default')
                param_help = param.get('help')
                
                if param_type == 'text':
                    value = st_obj.text_input(
                        param_label,
                        value=param_default or "",
                        help=param_help,
                        key=f"param_{param_key}"
                    )
                elif param_type == 'number':
                    value = st_obj.number_input(
                        param_label,
                        value=param_default or 0,
                        help=param_help,
                        key=f"param_{param_key}"
                    )
                elif param_type == 'select':
                    options = param.get('options', [])
                    value = st_obj.selectbox(
                        param_label,
                        options=options,
                        index=options.index(param_default) if param_default in options else 0,
                        help=param_help,
                        key=f"param_{param_key}"
                    )
                elif param_type == 'checkbox':
                    value = st_obj.checkbox(
                        param_label,
                        value=param_default or False,
                        help=param_help,
                        key=f"param_{param_key}"
                    )
                elif param_type == 'slider':
                    min_val = param.get('min', 0)
                    max_val = param.get('max', 100)
                    value = st_obj.slider(
                        param_label,
                        min_value=min_val,
                        max_value=max_val,
                        value=param_default or min_val,
                        help=param_help,
                        key=f"param_{param_key}"
                    )
                else:
                    value = param_default
                
                param_values[param_key] = value
            
            return param_values
    
    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return [f"param_{param['key']}" for param in self.parameters]


class DataExplorationSidebar(SidebarComponent):
    """
    数据探索专用侧边栏组件

    完全基于UnifiedDataUploadComponent实现。
    """

    def __init__(self,
                 title: str = "数据探索 - 数据上传",
                 accepted_types: List[str] = None,
                 help_text: str = None):
        super().__init__()
        self.title = title
        self.accepted_types = accepted_types or ['csv', 'xlsx', 'xls']
        self.help_text = help_text or "上传CSV或Excel文件进行时间序列数据探索分析"

        # 使用UnifiedDataUploadComponent
        from dashboard.ui.components.data_input import UnifiedDataUploadComponent
        self.upload_component = UnifiedDataUploadComponent(
            accepted_types=self.accepted_types,
            help_text=self.help_text,
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="data_exploration_upload"
        )

    def render(self, st_obj, **kwargs) -> Optional[pd.DataFrame]:
        """渲染数据探索专用侧边栏"""
        with st_obj.sidebar:
            st_obj.markdown(f"### {self.title}")

            # 使用UnifiedDataUploadComponent渲染
            data = self.upload_component.render_file_upload_section(
                st_obj,
                upload_key="data_exploration_sidebar_upload",
                show_overview=False,
                show_preview=False
            )

            if data is not None:
                # 获取文件名
                file_name = self.upload_component.get_state('file_name')

                # 手动调用数据存储逻辑
                if file_name:
                    self.store_exploration_data(data, file_name)

                st_obj.info(f"数据形状: {data.shape[0]} 行 x {data.shape[1]} 列")

            return data

    def store_exploration_data(self, data: pd.DataFrame, file_name: str):
        """存储数据到统一状态管理器 - 确保所有分析模块都能访问"""
        state_manager = get_unified_manager()

        current_data_hash = hash(str(data.shape) + file_name)
        last_stored_hash = getattr(self, '_last_stored_data_hash', None)

        if current_data_hash != last_stored_hash:
            success_count = 0
            modules = ['stationarity', 'time_lag_corr', 'lead_lag']
            for module in modules:
                # 存储数据到各个模块
                success = state_manager.set_state(f"exploration.{module}.data", data)
                if success:
                    state_manager.set_state(f"exploration.{module}.file_name", file_name)
                    state_manager.set_state(f"exploration.{module}.data_source", 'upload')
                    success_count += 1

            if success_count == len(modules):
                logger.info(f"数据已存储到所有分析模块: {data.shape}")
            else:
                logger.warning(f"数据存储部分成功: {success_count}/{len(modules)} 个模块")

            self._last_stored_data_hash = current_data_hash

    def render_data_overview(self, st_obj, data: pd.DataFrame, file_name: str):
        """渲染数据概览 - 增强版本"""
        # 基本统计信息
        col1, col2, col3 = st_obj.columns(3)

        with col1:
            st_obj.metric("行数", f"{data.shape[0]:,}")

        with col2:
            st_obj.metric("列数", data.shape[1])

        with col3:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            st_obj.metric("内存", f"{memory_mb:.1f}MB")

        # 数据类型信息
        st_obj.markdown("**数据类型分布：**")
        type_counts = data.dtypes.value_counts()
        type_info = " | ".join([f"{dtype}: {count}" for dtype, count in type_counts.items()])
        st_obj.caption(type_info)

        # 显示数据预览
        with st_obj.expander("数据预览 (前5行)", expanded=False):
            st_obj.dataframe(data.head(5), use_container_width=True)

        # 显示列名信息
        with st_obj.expander("列名列表", expanded=False):
            cols_text = ", ".join(data.columns.tolist())
            st_obj.text(cols_text)

    def render_current_data_status(self, st_obj):
        """显示当前数据状态 - 增强版本"""
        try:
            tools_manager = get_tools_manager_instance()
            if tools_manager:
                current_data = tools_manager.get_tools_state('data_exploration', 'current_data')
                current_data_name = tools_manager.get_tools_state('data_exploration', 'current_data_name')
                current_data_info = tools_manager.get_tools_state('data_exploration', 'current_data_info')

                st_obj.markdown("---")
                st_obj.markdown("#### 当前数据状态")

                if current_data is not None and current_data_info:
                    # 显示数据信息
                    st_obj.success(f"已加载：{current_data_name}")

                    # 显示详细信息
                    col1, col2 = st_obj.columns(2)
                    with col1:
                        st_obj.caption(f"{current_data_info['shape'][0]}行 x {current_data_info['shape'][1]}列")
                    with col2:
                        upload_time = current_data_info.get('upload_time', '')
                        if upload_time:
                            time_str = upload_time.split('T')[1][:8] if 'T' in upload_time else upload_time
                            st_obj.caption(f"{time_str}")

                    # 数据可用性提示
                    st_obj.info("数据已在三个分析模块间共享，可直接进行分析")
                else:
                    st_obj.warning("暂无数据，请上传数据文件")
                    st_obj.markdown("""
                    <div style="font-size: 0.8em; color: #666;">
                    <strong>支持格式：</strong><br>
                    • CSV文件 (UTF-8, GBK, GB2312编码)<br>
                    • Excel文件 (.xlsx, .xls)
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st_obj.error(f"显示数据状态失败: {e}")
            logger.error(f"显示数据状态失败: {e}")

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return ['data_exploration_sidebar_upload']


class DFMDataUploadSidebar(SidebarComponent):
    """
    DFM数据上传侧边栏组件

    完全基于UnifiedDataUploadComponent实现。
    """

    def __init__(self):
        super().__init__()
        self._supported_formats = ['xlsx', 'xls']
        self._max_file_size = 100  # MB

        # 使用UnifiedDataUploadComponent
        from dashboard.ui.components.data_input import UnifiedDataUploadComponent
        self.upload_component = UnifiedDataUploadComponent(
            accepted_types=self._supported_formats,
            max_file_size=self._max_file_size,
            help_text="请上传包含时间序列数据的Excel文件（支持.xlsx, .xls格式）",
            show_data_source_selector=False,
            show_staging_data_option=False,
            component_id="dfm_data_upload",
            return_file_object=True  # DFM需要文件对象
        )

    def render(self, st_obj, **kwargs) -> Dict[str, Any]:
        """渲染DFM数据上传侧边栏"""
        st_obj.subheader("DFM 数据上传")

        # 检查是否已有上传的文件
        existing_file = self._get_existing_file()

        if existing_file:
            # 显示已上传文件的信息
            st_obj.success(f"已上传文件: {existing_file['name']}")

            # 显示文件信息
            col1, col2 = st_obj.columns(2)
            with col1:
                st_obj.metric("文件大小", f"{existing_file.get('size', 0):.2f} MB")
            with col2:
                st_obj.metric("上传时间", existing_file.get('upload_time', '未知'))

            # 提供重新上传选项
            if st_obj.button("重新上传文件", key="dfm_reupload_btn"):
                self._clear_existing_file()
                st_obj.rerun()

            return {
                'has_file': True,
                'file_info': existing_file,
                'uploaded_file': None
            }

        # 使用UnifiedDataUploadComponent上传
        uploaded_file = self.upload_component.render_file_upload_section(
            st_obj,
            upload_key=kwargs.get('upload_key', 'dfm_data_upload_unified'),
            show_overview=False,
            show_preview=False
        )

        if uploaded_file is not None:
            file_name = self.upload_component.get_state('file_name')
            logger.debug(f"检测到文件上传，文件名: {file_name}")

            if file_name:
                self._save_to_dfm_state(uploaded_file, file_name)
            else:
                logger.warning("file_name为空，尝试使用uploaded_file.name")
                if hasattr(uploaded_file, 'name'):
                    file_name = uploaded_file.name
                    self._save_to_dfm_state(uploaded_file, file_name)
                else:
                    logger.error("无法获取文件名，跳过保存")

            st_obj.info("文件已加载，可用于DFM模型的各个功能模块。")
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024

            return {
                'has_file': True,
                'uploaded_file': uploaded_file,
                'file_size': file_size,
                'save_result': {'success': True}
            }

        return {
            'has_file': False,
            'uploaded_file': None
        }

    def _get_existing_file(self) -> Optional[Dict[str, Any]]:
        """获取已存在的文件信息"""
        # 使用DFM统一状态管理系统
        dfm_manager = get_global_dfm_manager()
        if dfm_manager:
            # 获取文件对象和路径
            file_obj = dfm_manager.get_dfm_state('data_prep', 'dfm_training_data_file', None)
            file_path = dfm_manager.get_dfm_state('data_prep', 'dfm_uploaded_excel_file_path', None)

            if file_obj and file_path:
                # 计算文件大小
                file_size = len(file_obj.getvalue()) / 1024 / 1024 if hasattr(file_obj, 'getvalue') else 0

                return {
                    'name': file_path,
                    'size': file_size,
                    'upload_time': datetime.now().strftime('%H:%M:%S'),
                    'file_obj': file_obj
                }
        return None

    def _clear_existing_file(self) -> None:
        """清除已存在的文件"""
        dfm_manager = get_global_dfm_manager()
        if dfm_manager:
            # 清除相关状态
            dfm_manager.set_dfm_state('data_prep', 'dfm_training_data_file', None)
            dfm_manager.set_dfm_state('data_prep', 'dfm_uploaded_excel_file_path', None)
            dfm_manager.set_dfm_state('data_prep', 'dfm_file_processed', False)
            dfm_manager.set_dfm_state('data_prep', 'dfm_date_detection_needed', True)

    def _validate_file(self, uploaded_file) -> Dict[str, Any]:
        """验证上传的文件"""
        try:
            # 检查文件格式
            if not self._is_valid_format(uploaded_file.name):
                return {
                    'valid': False,
                    'error': f"不支持的文件格式。请上传 {', '.join(self._supported_formats)} 格式的文件。"
                }

            # 检查文件大小
            file_size = len(uploaded_file.getvalue())
            if file_size > self._max_file_size:
                return {
                    'valid': False,
                    'error': f"文件大小超过限制（{self._max_file_size / 1024 / 1024:.0f}MB）。"
                }

            # 检查文件是否为空
            if file_size == 0:
                return {
                    'valid': False,
                    'error': "文件为空，请选择有效的数据文件。"
                }

            return {'valid': True}

        except Exception as e:
            return {
                'valid': False,
                'error': f"文件验证失败: {str(e)}"
            }

    def _is_valid_format(self, filename: str) -> bool:
        """检查文件格式是否有效"""
        if not filename:
            return False

        file_extension = filename.lower().split('.')[-1]
        return file_extension in self._supported_formats

    def _save_to_dfm_state(self, data_or_file, filename: str) -> None:
        """保存文件到DFM状态管理（回调函数）"""
        uploaded_file = data_or_file

        logger.info(f"开始保存文件到DFM状态: {filename}")

        dfm_manager = get_global_dfm_manager()
        if dfm_manager:
            file_bytes = uploaded_file.getvalue()

            dfm_manager.set_dfm_state('data_prep', 'dfm_training_data_bytes', file_bytes)
            dfm_manager.set_dfm_state('data_prep', 'dfm_training_data_file', uploaded_file)
            dfm_manager.set_dfm_state('data_prep', 'dfm_uploaded_excel_file_path', filename)
            dfm_manager.set_dfm_state('data_prep', 'dfm_use_full_data_preparation', True)
            dfm_manager.set_dfm_state('data_prep', 'dfm_file_processed', False)
            dfm_manager.set_dfm_state('data_prep', 'dfm_date_detection_needed', True)
            logger.info(f"文件保存成功: {filename}, 字节大小: {len(file_bytes)}")

            saved_bytes = dfm_manager.get_dfm_state('data_prep', 'dfm_training_data_bytes', None)
            saved_path = dfm_manager.get_dfm_state('data_prep', 'dfm_uploaded_excel_file_path', None)
            logger.debug(f"验证保存 - 字节内容: {saved_bytes is not None and len(saved_bytes) > 0}, 路径: {saved_path}")
        else:
            logger.error("DFM状态管理器不可用")

    def get_state_keys(self) -> List[str]:
        """获取组件相关的状态键"""
        return [
            'dfm_training_data_file',
            'dfm_uploaded_excel_file_path',
            'dfm_use_full_data_preparation',
            'dfm_file_processed',
            'dfm_date_detection_needed'
        ]


def render_complete_sidebar(
    module_config: Dict[str, Any],
    nav_manager: Any,
    key_prefix: str = "sidebar"
) -> Dict[str, Any]:
    """
    渲染完整的侧边栏，包括标题、模块选择器、暂存数据等

    Args:
        module_config: 模块配置字典
        nav_manager: 导航管理器
        key_prefix: 组件key前缀

    Returns:
        Dict[str, Any]: 侧边栏渲染结果
    """
    reset_button_key_tracking()

    if not validate_sidebar_config(module_config):
        return {'error': 'Invalid module configuration'}

    with create_sidebar_container():
        # 移除侧边栏标题，保持简洁

        # === 第一层：主模块选择器 ===
        st.markdown("### 主模块")

        # 暂时显示所有模块，包括用户管理模块
        main_module_options = list(module_config.keys())

        # 获取当前状态，初次进入时可能为None
        current_main_module = nav_manager.get_current_main_module() if nav_manager else None

        try:
            clear_button_state_cache()
        except Exception as e:
            logger.error(f"清除按钮状态缓存失败: {e}")

        # 只有在current_main_module不为None且不在选项中时才设置默认值
        if current_main_module is not None and current_main_module not in main_module_options:
            current_main_module = main_module_options[0]
            if nav_manager:
                nav_manager.set_current_main_module(current_main_module)

        # 渲染主模块选择器
        main_module_result = render_main_module_selector(
            main_module_options, current_main_module, nav_manager, f"{key_prefix}_main"
        )

        # 获取更新后的主模块状态用于子模块选择器
        updated_main_module = main_module_result.get('selected_module', current_main_module)

        # === 视觉分割线 ===
        st.markdown("---")

        # === 第二层：子模块选择器 ===
        current_sub_module = nav_manager.get_current_sub_module() if nav_manager else None
        sub_module_result = None

        # 只有当选择了主模块时才显示子模块选择器
        if updated_main_module and updated_main_module in module_config:
            sub_config = module_config[updated_main_module]
            if isinstance(sub_config, dict):  # 有子模块
                st.markdown("### 子模块")
                st.caption(f"当前主模块：{updated_main_module}")

                # 添加一些间距来实现视觉分割
                st.markdown("")  # 空行

                sub_module_options = list(sub_config.keys())

                # 渲染子模块选择器
                sub_module_result = render_sub_module_selector(
                    sub_module_options, current_sub_module, updated_main_module,
                    nav_manager, f"{key_prefix}_sub"
                )

                st.markdown("")  # 空行
            else:
                # 没有子模块的情况，显示提示
                st.markdown("### 子模块")
        else:
            # 没有选择主模块时，显示提示
            st.markdown("### 子模块")
            st.info("请先选择一个主模块")

        # 渲染分隔线
        st.markdown("---")

        # 渲染暂存数据部分 - 仅在数据预处理模块中显示
        staged_data_info = None
        current_sub_module = sub_module_result.get('selected_sub_module') if sub_module_result else None

        # 不显示暂存数据，返回默认值
        staged_data_info = {
            'has_data': False,
            'data_count': 0,
            'datasets': {}
        }

        # 渲染数据上传部分（如果适用）
        upload_info = None
        if sub_module_result and sub_module_result.get('selected_sub_module'):
            upload_config = get_upload_section_config(
                updated_main_module, sub_module_result['selected_sub_module']
            )
            if upload_config['show_upload']:
                upload_info = render_data_upload_section(
                    updated_main_module, sub_module_result['selected_sub_module']
                )

    result = {
        'main_module_result': main_module_result,
        'sub_module_result': sub_module_result,
        'staged_data_info': staged_data_info,
        'upload_info': upload_info
    }

    return result


def render_staged_data_section(staged_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    渲染暂存数据部分

    Args:
        staged_data: 暂存数据字典

    Returns:
        Dict[str, Any]: 暂存数据信息
    """
    st.subheader("暂存数据")

    if not staged_data:
        st.info("暂无暂存数据")
        return {
            'has_data': False,
            'data_count': 0,
            'datasets': {}
        }

    # 显示暂存数据信息
    st.write(f"共有 {len(staged_data)} 个数据集：")
    for name, data_info in staged_data.items():
        if isinstance(data_info, dict) and 'name' in data_info:
            st.write(f"• {data_info['name']}")
        else:
            st.write(f"• {name}")

    return {
        'has_data': True,
        'data_count': len(staged_data),
        'datasets': staged_data
    }


def render_data_upload_section(main_module: str, sub_module: str) -> Dict[str, Any]:
    """
    渲染数据上传部分 - 支持DFM模块和其他模块

    Args:
        main_module: 主模块名称
        sub_module: 子模块名称

    Returns:
        Dict[str, Any]: 上传部分信息
    """
    upload_config = get_upload_section_config(main_module, sub_module)

    if not upload_config['show_upload']:
        return {'show_upload': False}

    if main_module == "模型分析" and sub_module == "DFM 模型":
        logger.debug(f"侧边栏选择 - 主模块: '{main_module}', 子模块: '{sub_module}'")
        dfm_upload_sidebar = DFMDataUploadSidebar()
        stable_upload_key = 'dfm_sidebar_upload_stable'
        upload_result = dfm_upload_sidebar.render(st, upload_key=stable_upload_key)
        logger.debug(f"DFM上传组件渲染结果: {upload_result.get('has_file', False)}")

        return {
            'show_upload': True,
            'main_module': main_module,
            'sub_module': sub_module,
            'upload_type': 'dfm_data',
            'upload_result': upload_result,
            'upload_title': upload_config['title'],
            'upload_description': upload_config['description']
        }

    elif main_module != "数据探索":
        st.subheader(upload_config['title'])
        st.write(upload_config['description'])

    return {
        'show_upload': True,
        'main_module': main_module,
        'sub_module': sub_module,
        'upload_title': upload_config['title'],
        'upload_description': upload_config['description']
    }


@contextmanager
def create_sidebar_container():
    """
    创建侧边栏容器的上下文管理器

    Yields:
        侧边栏上下文
    """
    with st.sidebar:
        yield st.sidebar


def get_sidebar_title(custom_title: str = None) -> str:
    """
    获取侧边栏标题

    Args:
        custom_title: 自定义标题

    Returns:
        str: 侧边栏标题
    """
    return custom_title or "经济运行分析平台"


def get_module_selector_title(selector_type: str, main_module: str = None) -> str:
    """
    获取模块选择器标题

    Args:
        selector_type: 选择器类型 ('main' 或 'sub')
        main_module: 主模块名称（用于子模块标题）

    Returns:
        str: 选择器标题
    """
    if selector_type == 'main':
        return "选择子模块"
    elif selector_type == 'sub':
        if main_module:
            return f"{main_module} 子选项"
        else:
            return "子选项"
    else:
        return "模块选择"


def validate_sidebar_config(config: Any) -> bool:
    """
    验证侧边栏配置的有效性

    Args:
        config: 配置对象

    Returns:
        bool: 配置是否有效
    """
    if not config or not isinstance(config, dict):
        return False

    if len(config) == 0:
        return False

    return True


def get_upload_section_config(main_module: str, sub_module: str) -> Dict[str, Any]:
    """
    获取上传部分的配置

    Args:
        main_module: 主模块名称
        sub_module: 子模块名称

    Returns:
        Dict[str, Any]: 上传配置
    """
    # 定义支持上传的模块组合
    upload_configs = {
        ('监测分析', '工业'): {
            'show_upload': True,
            'title': '工业监测分析数据上传',
            'description': '上传一个Excel文件，同时支持宏观运行和企业经营分析'
        },
        ('数据探索', None): {
            'show_upload': True,
            'title': '数据探索 - 数据上传',
            'description': '上传数据文件进行探索性分析'
        },
        ('模型分析', 'DFM 模型'): {
            'show_upload': True,
            'title': 'DFM 模型数据上传',
            'description': '上传Excel数据文件，用于DFM模型的数据准备、训练和分析',
            'upload_type': 'dfm_data'
        }
    }

    key = (main_module, sub_module)
    return upload_configs.get(key, {'show_upload': False})


def handle_sidebar_state_changes(
    main_result: Dict[str, Any],
    sub_result: Optional[Dict[str, Any]],
    nav_manager: Any
) -> Dict[str, Any]:
    """
    处理侧边栏状态变化

    Args:
        main_result: 主模块选择结果
        sub_result: 子模块选择结果
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 状态变化处理结果
    """
    main_changed = main_result.get('has_change', False)
    sub_changed = sub_result.get('has_change', False) if sub_result else False

    current_main = main_result.get('selected_module')
    current_sub = sub_result.get('selected_sub_module') if sub_result else None

    if main_changed or sub_changed:
        logger.debug(
            f"侧边栏状态变化 - 主模块: {current_main}, 子模块: {current_sub}, "
            f"主模块变化: {main_changed}, 子模块变化: {sub_changed}"
        )

    return {
        'main_changed': main_changed,
        'sub_changed': sub_changed,
        'current_main': current_main,
        'current_sub': current_sub,
        'nav_manager_available': nav_manager is not None
    }


def filter_modules_by_permission(all_modules: List[str]) -> List[str]:
    """
    根据用户权限过滤模块列表

    Args:
        all_modules: 所有可用模块列表

    Returns:
        List[str]: 用户有权限的模块列表
    """
    # 从统一状态管理器获取用户权限信息
    state_manager = get_unified_manager()
    if not state_manager:
        # 如果状态管理器不可用，返回除用户管理外的所有模块
        return [module for module in all_modules if module != '用户管理']

    user_accessible_modules = state_manager.get_state('auth.user_accessible_modules', set())
    current_user = state_manager.get_state('auth.current_user', None)

    if not current_user:
        # 如果用户未登录，返回除用户管理外的所有模块
        return [module for module in all_modules if module != '用户管理']

    # 过滤模块列表
    filtered_modules = []
    for module in all_modules:
        if module == '用户管理':
            # 用户管理模块只有有相应权限的用户才能看到
            if '用户管理' in user_accessible_modules:
                filtered_modules.append(module)
        else:
            # 其他模块按现有逻辑处理
            if module in user_accessible_modules:
                filtered_modules.append(module)

    return filtered_modules


__all__ = [
    'SidebarComponent', 'DataUploadSidebar', 'NavigationSidebar', 'FilterSidebar',
    'DataExplorationSidebar', 'DFMDataUploadSidebar',
    'render_complete_sidebar', 'render_staged_data_section', 'render_data_upload_section',
    'create_sidebar_container', 'get_sidebar_title', 'get_module_selector_title',
    'validate_sidebar_config', 'get_upload_section_config', 'handle_sidebar_state_changes',
    'filter_modules_by_permission'
]
