# -*- coding: utf-8 -*-
"""
UI模块状态管理适配器
正确集成统一状态管理系统，替代重复的状态管理实现
"""

from typing import Any, Dict, List, Optional, Callable
import logging
from datetime import datetime

# 导入统一状态管理系统
from dashboard.core import get_unified_manager

logger = logging.getLogger(__name__)


class UIStateAdapter:
    """
    UI模块状态管理适配器
    直接使用统一状态管理系统，不依赖抽象基类
    """

    def __init__(self):
        """初始化UI状态适配器"""
        # 获取统一状态管理器
        try:
            self.unified_manager = get_unified_manager()
        except Exception as e:
            logger.warning(f"获取统一状态管理器失败: {e}")
            self.unified_manager = None

        self.module_name = "ui"
        self.logger = logger

        # UI模块特定的配置
        self._component_registry = {}
        self._ui_config = {
            'cache_enabled': True,
            'validation_enabled': True,
            'auto_cleanup': True
        }

        # 初始化UI模块
        self._initialize_module()
        logger.info("UI状态适配器初始化完成")
    
    def _initialize_module(self):
        """初始化UI模块"""
        try:
            # 注册UI模块的默认状态键
            default_keys = [
                'ui.current_page',
                'ui.navigation_state',
                'ui.theme_config',
                'ui.user_preferences'
            ]

            # 初始化默认状态
            for key in default_keys:
                if self.get_state(key) is None:
                    self.set_state(key, {})

            self.logger.info("UI模块初始化完成")

        except Exception as e:
            self.logger.error(f"UI模块初始化失败: {e}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        try:
            if not self.unified_manager:
                return default

            if not isinstance(key, str):
                self.logger.warning(f"键名不是字符串类型: {type(key)} = {key}")
                key = str(key)

            return self.unified_manager.get_state(key, default)

        except Exception as e:
            self.logger.error(f"从统一管理器获取状态失败: {key}, 错误: {e}")
            return default

    def set_state(self, key: str, value: Any) -> bool:
        """设置状态"""
        try:
            if not self.unified_manager:
                return False

            if not isinstance(key, str):
                self.logger.warning(f"键名不是字符串类型: {type(key)} = {key}")
                key = str(key)

            return self.unified_manager.set_state(key, value)

        except Exception as e:
            self.logger.error(f"设置到统一管理器失败: {key}, 错误: {e}")
            return False

    def delete_state(self, key: str) -> bool:
        """删除状态"""
        try:
            if not self.unified_manager:
                return False

            if not isinstance(key, str):
                self.logger.warning(f"键名不是字符串类型: {type(key)} = {key}")
                key = str(key)

            return self.unified_manager.set_state(key, None)

        except Exception as e:
            self.logger.error(f"从统一管理器删除状态失败: {key}, 错误: {e}")
            return False
    
    def register_component(self, component_id: str, component_instance):
        """
        注册UI组件

        Args:
            component_id: 组件ID
            component_instance: 组件实例
        """
        try:
            # 延迟获取状态键，避免在组件初始化过程中访问未设置的属性
            def get_state_keys_safe():
                try:
                    return getattr(component_instance, 'get_state_keys', lambda: [])()
                except Exception:
                    return []

            self._component_registry[component_id] = {
                'instance': component_instance,
                'registered_at': datetime.now(),
                'state_keys': get_state_keys_safe()
            }

            self.logger.info(f"UI组件已注册: {component_id}")

        except Exception as e:
            self.logger.error(f"注册UI组件失败: {component_id}, 错误: {e}")
    
    def unregister_component(self, component_id: str):
        """
        注销UI组件
        
        Args:
            component_id: 组件ID
        """
        try:
            if component_id in self._component_registry:
                # 清理组件相关的状态
                self.cleanup_component_state(component_id)
                
                # 从注册表中移除
                del self._component_registry[component_id]


                self.logger.info(f"UI组件已注销: {component_id}")

        except Exception as e:
            self.logger.error(f"注销UI组件失败: {component_id}, 错误: {e}")
    
    def get_component_state(self, component_id: str, key: str, default=None):
        """
        获取组件状态
        
        Args:
            component_id: 组件ID
            key: 状态键
            default: 默认值
            
        Returns:
            状态值
        """
        try:
            # 构造完整的状态键：ui.{component_id}.{key}
            full_key = f"ui.{component_id}.{key}"
            return self.get_state(full_key, default)

        except Exception as e:
            self.logger.error(f"获取组件状态失败: {component_id}.{key}, 错误: {e}")
            return default
    
    def set_component_state(self, component_id: str, key: str, value):
        """
        设置组件状态
        
        Args:
            component_id: 组件ID
            key: 状态键
            value: 状态值
            
        Returns:
            bool: 是否设置成功
        """
        try:
            # 构造完整的状态键：ui.{component_id}.{key}
            full_key = f"ui.{component_id}.{key}"
            success = self.set_state(full_key, value)

            if success:
                self.logger.debug(f"组件状态设置成功: {component_id}.{key}")
            else:
                self.logger.warning(f"组件状态设置失败: {component_id}.{key}")

            return success

        except Exception as e:
            self.logger.error(f"设置组件状态失败: {component_id}.{key}, 错误: {e}")
            return False
    
    def cleanup_component_state(self, component_id: str):
        """
        清理组件状态

        Args:
            component_id: 组件ID
        """
        try:
            # 清理所有以ui.{component_id}.开头的状态键
            prefix = f"ui.{component_id}."

            # 获取统一状态管理器中的所有状态键
            if self.unified_manager and hasattr(self.unified_manager, 'get_all_keys'):
                all_keys = self.unified_manager.get_all_keys()
                keys_to_clean = [key for key in all_keys if key.startswith(prefix)]

                for key in keys_to_clean:
                    self.delete_state(key)

                self.logger.info(f"组件状态已清理: {component_id}, 清理了 {len(keys_to_clean)} 个状态键")
            else:
                # 统一状态管理器不可用时，记录错误
                self.logger.error(f"统一状态管理器不可用，无法清理组件状态: {component_id}")

        except Exception as e:
            self.logger.error(f"清理组件状态失败: {component_id}, 错误: {e}")
    
    def get_all_component_states(self, component_id: str) -> Dict[str, Any]:
        """
        获取组件的所有状态
        
        Args:
            component_id: 组件ID
            
        Returns:
            Dict[str, Any]: 组件的所有状态
        """
        try:
            component_info = self._component_registry.get(component_id)
            if not component_info:
                return {}
            
            state_keys = component_info.get('state_keys', [])
            states = {}
            
            for key in state_keys:
                value = self.get_component_state(component_id, key)
                if value is not None:
                    states[key] = value
            
            return states

        except Exception as e:
            self.logger.error(f"获取组件所有状态失败: {component_id}, 错误: {e}")
            return {}
    
    def get_registered_components(self) -> List[str]:
        """
        获取已注册的组件列表
        
        Returns:
            List[str]: 组件ID列表
        """
        return list(self._component_registry.keys())
    
    def get_component_info(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        获取组件信息
        
        Args:
            component_id: 组件ID
            
        Returns:
            Optional[Dict[str, Any]]: 组件信息
        """
        return self._component_registry.get(component_id)
    
    def validate_component_state(self, component_id: str, key: str, value: Any) -> bool:
        """
        验证组件状态值
        
        Args:
            component_id: 组件ID
            key: 状态键
            value: 状态值
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 基本验证逻辑
            if value is None:
                return True  # None值总是有效的
            
            # 可以根据需要添加更多验证逻辑
            return True

        except Exception as e:
            self.logger.error(f"验证组件状态失败: {component_id}.{key}, 错误: {e}")
            return False
    
    def get_ui_config(self, key: str, default=None):
        """
        获取UI配置
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._ui_config.get(key, default)
    
    def set_ui_config(self, key: str, value: Any):
        """
        设置UI配置
        
        Args:
            key: 配置键
            value: 配置值
        """
        self._ui_config[key] = value
        self.logger.debug(f"UI配置已更新: {key} = {value}")


# 全局UI状态适配器实例
_ui_state_adapter = None


def get_ui_state_adapter() -> UIStateAdapter:
    """
    获取UI状态适配器实例（单例模式）
    
    Returns:
        UIStateAdapter: UI状态适配器实例
    """
    global _ui_state_adapter
    if _ui_state_adapter is None:
        _ui_state_adapter = UIStateAdapter()
    return _ui_state_adapter


def reset_ui_state_adapter():
    """重置UI状态适配器实例"""
    global _ui_state_adapter
    _ui_state_adapter = None
