# -*- coding: utf-8 -*-
"""
UI组件基类
提供所有UI组件的基础接口和通用功能
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging
import time

from dashboard.core.ui.utils.error_handler import get_ui_error_handler

logger = logging.getLogger(__name__)


class UnifiedBaseComponent:
    """使用命名空间管理状态的UI组件基类"""

    def __init__(self, component_name: str = None):
        self.component_name = component_name or self.__class__.__name__
        self.logger = logging.getLogger(f"UI.{self.component_name}")

    def get_component_state(self, key: str, default=None):
        """获取组件状态"""
        full_key = f"ui.{self.component_name}.{key}"
        return st.session_state.get(full_key, default)

    def set_component_state(self, key: str, value) -> bool:
        """设置组件状态"""
        try:
            full_key = f"ui.{self.component_name}.{key}"
            st.session_state[full_key] = value
            self.logger.debug(f"状态设置成功: {key}")
            return True
        except Exception as e:
            self.logger.error(f"状态设置失败: {key}, 错误: {e}")
            return False

    def clear_component_state(self, key: str = None) -> bool:
        """清理组件状态"""
        try:
            if key:
                # 清理单个状态
                full_key = f"ui.{self.component_name}.{key}"
                if full_key in st.session_state:
                    del st.session_state[full_key]
            else:
                # 清理所有组件状态
                prefix = f"ui.{self.component_name}."
                keys_to_delete = [k for k in st.session_state.keys() if str(k).startswith(prefix)]
                for k in keys_to_delete:
                    del st.session_state[k]
            return True
        except Exception as e:
            self.logger.error(f"清理组件状态失败: {key}, 错误: {e}")
            return False

    def get_component_logs(self) -> list:
        """获取组件日志"""
        return self.get_component_state('logs', [])

    def add_component_log(self, action: str, details: dict = None, level: str = "INFO"):
        """添加组件日志"""
        logs = self.get_component_logs()
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'action': action,
            'component': self.component_name
        }

        if details:
            log_entry.update(details)

        logs.append(log_entry)

        # 保持最近100条日志
        if len(logs) > 100:
            logs = logs[-100:]

        return self.set_component_state('logs', logs)

    def get_current_time(self) -> float:
        """获取当前时间"""
        return self.get_component_state('current_time', time.time())

    def update_current_time(self) -> bool:
        """更新当前时间"""
        return self.set_component_state('current_time', time.time())

    def is_healthy(self) -> bool:
        """检查组件是否健康"""
        return True  # 不再依赖state_manager，始终返回True

    def get_component_summary(self) -> dict:
        """获取组件状态摘要"""
        try:
            all_keys = list(st.session_state.keys())
            component_prefix = f"ui.{self.component_name}."
            component_keys = [key for key in all_keys if str(key).startswith(component_prefix)]

            return {
                'component_name': self.component_name,
                'total_states': len(component_keys),
                'current_time': self.get_current_time(),
                'logs_count': len(self.get_component_logs()),
                'is_healthy': self.is_healthy()
            }
        except Exception as e:
            self.logger.error(f"获取组件摘要失败: {e}")
            return {'error': str(e)}


class UIComponent(UnifiedBaseComponent, ABC):
    """
    UI组件基类 - 使用命名空间管理状态

    所有UI组件都应该继承此基类，自动获得命名空间状态管理功能
    """

    def __init__(self, component_name: str = None):
        """初始化UI组件"""
        # 初始化统一基类
        super().__init__(component_name)

        # 基本属性初始化
        self.component_id = component_name or self.__class__.__name__

    @abstractmethod
    def render(self, st_obj, **kwargs) -> None:
        """
        渲染组件

        Args:
            st_obj: Streamlit对象
            **kwargs: 其他参数
        """
        pass

    def render_with_monitoring(self, st_obj, **kwargs) -> None:
        """
        渲染方法（已移除性能监控）

        Args:
            st_obj: Streamlit对象
            **kwargs: 其他参数
        """
        return self.render(st_obj, **kwargs)

    def process_async(self, st_obj, processor_func, data, description: str = "处理中..."):
        """
        异步处理数据

        Args:
            st_obj: Streamlit对象
            processor_func: 处理函数
            data: 要处理的数据
            description: 处理描述

        Returns:
            处理结果
        """
        if self.async_helper:
            return self.async_helper.process_data_async(st_obj, processor_func, data, description)
        else:
            # 同步处理作为备选
            return processor_func(data)

    @abstractmethod
    def get_state_keys(self) -> List[str]:
        """
        获取组件相关的状态键

        Returns:
            List[str]: 状态键列表
        """
        pass

    def get_component_id(self) -> str:
        """
        获取组件ID

        Returns:
            str: 组件ID
        """
        # 移除Component后缀，转换为小写
        class_name = self.__class__.__name__
        if class_name.endswith('Component'):
            class_name = class_name[:-9]  # 移除'Component'
        return class_name.lower()

    def get_state(self, key: str, default=None):
        """
        获取组件状态

        Args:
            key: 状态键
            default: 默认值

        Returns:
            状态值
        """
        try:
            full_key = f"ui.{self.component_id}.{key}"
            return st.session_state.get(full_key, default)
        except Exception as e:
            logger.error(f"获取组件状态失败: {self.component_id}.{key}, 错误: {e}")
            return default

    def set_state(self, key: str, value):
        """
        设置组件状态

        Args:
            key: 状态键
            value: 状态值

        Returns:
            bool: 是否设置成功
        """
        try:
            full_key = f"ui.{self.component_id}.{key}"
            st.session_state[full_key] = value
            return True
        except Exception as e:
            logger.error(f"设置组件状态失败: {self.component_id}.{key}, 错误: {e}")
            return False

    def get_all_states(self) -> Dict[str, Any]:
        """
        获取组件的所有状态

        Returns:
            Dict[str, Any]: 组件的所有状态
        """
        try:
            prefix = f"ui.{self.component_id}."
            return {
                key[len(prefix):]: value
                for key, value in st.session_state.items()
                if key.startswith(prefix)
            }
        except Exception as e:
            logger.error(f"获取组件所有状态失败: {self.component_id}, 错误: {e}")
            return {}

    def cleanup(self):
        """
        组件清理

        在组件销毁时调用，清理相关状态和资源
        """
        try:
            # 清理组件状态
            prefix = f"ui.{self.component_id}."
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                del st.session_state[k]

            logger.debug(f"组件清理完成: {self.component_id}")

        except Exception as e:
            logger.error(f"组件清理失败: {self.component_id}, 错误: {e}")

    def handle_error(self, st_obj, error: Exception, context: str = "", **kwargs):
        """
        统一错误处理 - 使用标准化的UI错误处理器

        Args:
            st_obj: Streamlit对象
            error: 异常对象
            context: 错误上下文
            **kwargs: 额外参数
        """
        try:
            # 使用标准化的UI错误处理器
            error_handler = get_ui_error_handler()
            result = error_handler.handle_ui_error(
                error=error,
                component_id=self.component_id,
                context=context,
                st_obj=st_obj,
                **kwargs
            )

            # 记录错误到组件状态
            if result.get('success'):
                self.set_state('last_error', result['error_info'])
                self.set_state('error_count', self.get_state('error_count', 0) + 1)

            return result

        except Exception as e:
            # 错误处理本身出错时直接抛出
            logger.error(f"标准化错误处理失败: {self.component_id}, 原始错误: {error}, 处理错误: {e}")
            raise RuntimeError(f"组件 {self.component_id} 错误处理失败: {e}") from error

    def validate_props(self, props: Dict[str, Any]) -> bool:
        """
        验证组件属性
        
        Args:
            props: 组件属性字典
            
        Returns:
            bool: 验证是否通过
        """
        return True


    def log_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """
        记录组件操作

        Args:
            action: 操作名称
            details: 操作详情
        """
        # 使用统一基类的日志记录方法
        self.add_component_log(action, details, "INFO")

        # 同时记录到传统日志系统
        if details:
            self.logger.info(f"组件操作: {action}, 详情: {details}")
        else:
            self.logger.info(f"组件操作: {action}")


class BaseWelcomePage(UIComponent):
    """欢迎页面基类"""

    def __init__(self):
        # 调用父类初始化
        super().__init__()

        self.constants = None
        self.module_config = None
    
    def render(self, st_obj, **kwargs) -> None:
        """渲染欢迎页面"""
        try:
            # 显示标题和介绍
            self._render_header(st_obj)
            
            # 显示子模块选择卡片
            self._render_sub_modules(st_obj)
            
            # 处理导航事件
            self._handle_navigation(st_obj)
            
        except Exception as e:
            self.handle_error(st_obj, e, "渲染欢迎页面")
    
    def _render_header(self, st_obj) -> None:
        """渲染页面头部"""
        if self.module_config and 'title' in self.module_config:
            st_obj.title(self.module_config['title'])
        
        if self.module_config and 'description' in self.module_config:
            st_obj.markdown(self.module_config['description'])
    
    def _render_sub_modules(self, st_obj) -> None:
        """渲染子模块选择卡片"""
        if not self.module_config or 'sub_modules' not in self.module_config:
            return
        
        sub_modules = self.module_config['sub_modules']
        if not sub_modules:
            return
        
        # 创建列布局
        cols = st_obj.columns(len(sub_modules))
        
        for i, (sub_module_name, sub_module_config) in enumerate(sub_modules.items()):
            with cols[i]:
                if st_obj.button(
                    sub_module_name,
                    key=f"welcome_sub_module_{sub_module_name}",
                    use_container_width=True
                ):
                    self._handle_sub_module_click(st_obj, sub_module_name)
    
    def _handle_sub_module_click(self, st_obj, sub_module_name: str) -> None:
        """处理子模块点击事件"""
        # 记录操作
        self.log_action('sub_module_click', {'sub_module': sub_module_name})
        
        # 这里可以添加导航逻辑
        # 目前只是显示信息
        st_obj.info(f"点击了子模块: {sub_module_name}")
    
    def _handle_navigation(self, st_obj) -> None:
        """处理导航事件"""
        # 子类可以重写此方法来实现具体的导航逻辑
        pass
    
    def get_state_keys(self) -> List[str]:
        """获取状态键"""
        return [
            'component_logs',
            'current_time'
        ]


__all__ = [
    'UnifiedBaseComponent',
    'UIComponent',
    'BaseWelcomePage'
]
