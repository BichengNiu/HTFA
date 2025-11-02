# -*- coding: utf-8 -*-
"""
UI错误处理器
集成统一状态管理的错误处理机制，提供用户友好的错误处理
"""

from typing import Dict, Any, Optional, Callable, List
import logging
import traceback
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    USER_INPUT = "user_input"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"
    UI_COMPONENT = "ui_component"


class UIErrorHandler:
    """
    UI错误处理器
    集成统一状态管理的错误处理机制
    """
    
    def __init__(self):
        """初始化UI错误处理器"""
        # 用户友好的错误消息映射
        self.error_messages = {
            # 文件操作错误
            'FileNotFoundError': {
                'message': '文件未找到，请检查文件路径是否正确',
                'suggestion': '请重新选择文件或检查文件是否存在',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.FILE_OPERATION
            },
            'PermissionError': {
                'message': '权限不足，无法访问该文件',
                'suggestion': '请检查文件权限或以管理员身份运行',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.FILE_OPERATION
            },
            'IsADirectoryError': {
                'message': '选择的是文件夹，请选择文件',
                'suggestion': '请选择具体的文件而不是文件夹',
                'severity': ErrorSeverity.LOW,
                'category': ErrorCategory.FILE_OPERATION
            },
            
            # 数据验证错误
            'ValueError': {
                'message': '数据格式错误，请检查输入数据',
                'suggestion': '请确保输入的数据格式正确',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.VALIDATION
            },
            'TypeError': {
                'message': '数据类型错误',
                'suggestion': '请检查输入数据的类型是否正确',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.VALIDATION
            },
            'KeyError': {
                'message': '缺少必要的数据字段',
                'suggestion': '请确保数据包含所有必要的字段',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.VALIDATION
            },
            
            # 网络错误
            'ConnectionError': {
                'message': '网络连接错误，请检查网络连接',
                'suggestion': '请检查网络连接或稍后重试',
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.NETWORK
            },
            'TimeoutError': {
                'message': '操作超时，请稍后重试',
                'suggestion': '请检查网络连接或稍后重试',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.NETWORK
            },
            'HTTPError': {
                'message': '服务器响应错误',
                'suggestion': '请稍后重试或联系技术支持',
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.NETWORK
            },
            
            # 系统错误
            'MemoryError': {
                'message': '内存不足，无法完成操作',
                'suggestion': '请关闭其他程序或处理较小的数据集',
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.SYSTEM
            },
            'OSError': {
                'message': '系统操作错误',
                'suggestion': '请检查系统状态或联系技术支持',
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.SYSTEM
            }
        }
        
        # 错误恢复策略
        self.recovery_strategies = {
            ErrorCategory.FILE_OPERATION: self._file_operation_recovery,
            ErrorCategory.NETWORK: self._network_recovery,
            ErrorCategory.VALIDATION: self._validation_recovery,
            ErrorCategory.SYSTEM: self._system_recovery,
            ErrorCategory.UI_COMPONENT: self._ui_component_recovery
        }
        
        # 错误统计
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_category': {},
            'last_error_time': None
        }
        
        logger.info("UI错误处理器初始化完成")
    
    def handle_ui_error(self, error: Exception, component_id: str, 
                       context: str, st_obj, **kwargs) -> Dict[str, Any]:
        """
        处理UI错误
        
        Args:
            error: 异常对象
            component_id: 组件ID
            context: 错误上下文
            st_obj: Streamlit对象
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 错误处理结果
        """
        try:
            # 获取错误类型
            error_type = type(error).__name__
            error_message = str(error)
            
            # 更新错误统计
            self._update_error_stats(error_type)
            
            # 获取错误信息
            error_info = self.error_messages.get(error_type, {
                'message': '发生未知错误，请联系技术支持',
                'suggestion': '请尝试刷新页面或联系技术支持',
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.SYSTEM
            })
            
            # 构造详细错误信息
            detailed_error_info = {
                'component_id': component_id,
                'context': context,
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat(),
                'severity': error_info['severity'].value,
                'category': error_info['category'].value,
                'user_message': error_info['message'],
                'suggestion': error_info['suggestion'],
                'traceback': traceback.format_exc() if kwargs.get('include_traceback', False) else None
            }
            
            # 显示用户友好的错误消息
            self._display_error_message(st_obj, error_info, detailed_error_info)
            
            # 尝试错误恢复
            recovery_result = self._attempt_recovery(error_info['category'], error, st_obj, **kwargs)
            
            # 记录错误日志
            logger.error(
                f"UI错误处理: 组件={component_id}, 上下文={context}, "
                f"错误类型={error_type}, 错误消息={error_message}"
            )
            
            return {
                'success': True,
                'error_info': detailed_error_info,
                'recovery_attempted': recovery_result['attempted'],
                'recovery_success': recovery_result['success'],
                'user_notified': True
            }
            
        except Exception as handling_error:
            # 错误处理本身出错时的兜底处理
            logger.error(f"错误处理失败: {handling_error}")
            self._emergency_error_handling(st_obj, component_id, error)
            
            return {
                'success': False,
                'error_info': {'error': str(handling_error)},
                'recovery_attempted': False,
                'recovery_success': False,
                'user_notified': True
            }
    
    def _display_error_message(self, st_obj, error_info: Dict, detailed_info: Dict):
        """显示错误消息"""
        try:
            # 根据严重程度选择显示方式
            severity = error_info['severity']
            
            if severity == ErrorSeverity.CRITICAL:
                st_obj.error(f"严重错误: {error_info['message']}")
            elif severity == ErrorSeverity.HIGH:
                st_obj.error(f"{error_info['message']}")
            elif severity == ErrorSeverity.MEDIUM:
                st_obj.warning(f"{error_info['message']}")
            else:
                st_obj.info(f"{error_info['message']}")
            
            # 显示建议
            if error_info['suggestion']:
                st_obj.info(f"建议: {error_info['suggestion']}")
            
            # 在开发模式下显示详细信息
            if detailed_info.get('traceback') and st_obj.session_state.get('debug_mode', False):
                with st_obj.expander("详细错误信息 (开发模式)"):
                    st_obj.code(detailed_info['traceback'])
                    
        except Exception as e:
            logger.error(f"显示错误消息失败: {e}")
    
    def _attempt_recovery(self, category: ErrorCategory, error: Exception, 
                         st_obj, **kwargs) -> Dict[str, Any]:
        """尝试错误恢复"""
        try:
            recovery_func = self.recovery_strategies.get(category)
            if recovery_func:
                return recovery_func(error, st_obj, **kwargs)
            else:
                return {'attempted': False, 'success': False, 'message': '无可用恢复策略'}
                
        except Exception as e:
            logger.error(f"错误恢复失败: {e}")
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _file_operation_recovery(self, error: Exception, st_obj, **kwargs) -> Dict[str, Any]:
        """文件操作错误恢复"""
        try:
            # 提供文件重新选择选项
            if hasattr(st_obj, 'button'):
                if st_obj.button("重新选择文件", key=f"retry_file_{id(error)}"):
                    # 触发文件重新选择
                    return {'attempted': True, 'success': True, 'message': '已触发文件重新选择'}
            
            return {'attempted': True, 'success': False, 'message': '请手动重新选择文件'}
            
        except Exception as e:
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _network_recovery(self, error: Exception, st_obj, **kwargs) -> Dict[str, Any]:
        """网络错误恢复"""
        try:
            # 提供重试选项
            if hasattr(st_obj, 'button'):
                if st_obj.button("重试", key=f"retry_network_{id(error)}"):
                    return {'attempted': True, 'success': True, 'message': '已触发重试'}
            
            return {'attempted': True, 'success': False, 'message': '请手动重试'}
            
        except Exception as e:
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _validation_recovery(self, error: Exception, st_obj, **kwargs) -> Dict[str, Any]:
        """数据验证错误恢复"""
        try:
            # 提供数据重新输入提示
            st_obj.info("请检查并重新输入数据")
            return {'attempted': True, 'success': True, 'message': '已提供数据重新输入提示'}
            
        except Exception as e:
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _system_recovery(self, error: Exception, st_obj, **kwargs) -> Dict[str, Any]:
        """系统错误恢复"""
        try:
            # 提供页面刷新建议
            st_obj.info("建议刷新页面或重启应用")
            return {'attempted': True, 'success': True, 'message': '已提供系统恢复建议'}
            
        except Exception as e:
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _ui_component_recovery(self, error: Exception, st_obj, **kwargs) -> Dict[str, Any]:
        """UI组件错误恢复"""
        try:
            # 提供组件重新加载选项
            if hasattr(st_obj, 'button'):
                if st_obj.button("重新加载组件", key=f"reload_component_{id(error)}"):
                    return {'attempted': True, 'success': True, 'message': '已触发组件重新加载'}
            
            return {'attempted': True, 'success': False, 'message': '请手动刷新页面'}
            
        except Exception as e:
            return {'attempted': True, 'success': False, 'message': f'恢复失败: {e}'}
    
    def _emergency_error_handling(self, st_obj, component_id: str, error: Exception):
        """兜底错误处理"""
        st_obj.error(f"组件 {component_id} 发生错误")
        logger.error(f"错误处理失败: 组件={component_id}, 错误={error}")
        raise
    
    def _update_error_stats(self, error_type: str):
        """更新错误统计"""
        try:
            self.error_stats['total_errors'] += 1
            self.error_stats['errors_by_type'][error_type] = \
                self.error_stats['errors_by_type'].get(error_type, 0) + 1
            self.error_stats['last_error_time'] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"更新错误统计失败: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return self.error_stats.copy()
    
    def clear_error_stats(self):
        """清除错误统计"""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_category': {},
            'last_error_time': None
        }
        logger.info("错误统计已清除")


# 全局UI错误处理器实例
_ui_error_handler = None


def get_ui_error_handler() -> UIErrorHandler:
    """
    获取UI错误处理器实例（单例模式）
    
    Returns:
        UIErrorHandler: UI错误处理器实例
    """
    global _ui_error_handler
    if _ui_error_handler is None:
        _ui_error_handler = UIErrorHandler()
    return _ui_error_handler


def reset_ui_error_handler():
    """重置UI错误处理器实例"""
    global _ui_error_handler
    _ui_error_handler = None
