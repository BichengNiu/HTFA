# -*- coding: utf-8 -*-
"""
持久化存储工具
提供本地文件持久化存储功能，解决页面刷新后状态丢失问题
支持多用户session隔离
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import logging
import tempfile
from typing import Any, Optional, Dict
import time
import hashlib
from pathlib import Path

def _get_session_id() -> str:
    """获取当前Streamlit会话ID"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx:
            return ctx.session_id
        else:
            return "default"
    except Exception:
        return "default"


class FilePersistentStorage:
    """基于文件的持久化存储管理器"""
    
    def __init__(self, storage_dir: str = None):
        """
        初始化持久化存储管理器
        
        Args:
            storage_dir: 存储目录，默认为临时目录
        """
        self.logger = logging.getLogger(__name__)
        
        if storage_dir is None:
            # 使用临时目录
            self.storage_dir = Path(tempfile.gettempdir()) / "htfa_persistent_storage"
        else:
            self.storage_dir = Path(storage_dir)
        
        # 确保存储目录存在
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"持久化存储目录: {self.storage_dir}")
        
    def _get_file_path(self, key: str, storage_type: str = "local") -> Path:
        """
        获取存储文件路径（包含session隔离）

        Args:
            key: 存储键
            storage_type: 存储类型

        Returns:
            文件路径
        """
        session_id = _get_session_id()

        # 为当前session创建独立目录
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 创建安全的文件名
        safe_key = hashlib.md5(key.encode()).hexdigest()
        filename = f"{storage_type}_{safe_key}.json"
        return session_dir / filename
    
    def set_item(self, key: str, value: Any, storage_type: str = "local") -> bool:
        """
        设置持久化存储项
        
        Args:
            key: 存储键
            value: 存储值  
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            是否设置成功
        """
        try:
            file_path = self._get_file_path(key, storage_type)
            
            # 准备存储数据
            storage_data = {
                'key': key,
                'value': value,
                'storage_type': storage_type,
                'timestamp': int(time.time() * 1000),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, default=str, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"持久化存储设置成功: {key} -> {type(value).__name__}")
            return True
            
        except Exception as e:
            self.logger.error(f"设置持久化存储失败: {key}, 错误: {e}")
            return False
    
    def get_item(self, key: str, default: Any = None, storage_type: str = "local") -> Any:
        """
        获取持久化存储项
        
        Args:
            key: 存储键
            default: 默认值
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            存储值或默认值
        """
        try:
            file_path = self._get_file_path(key, storage_type)
            
            if not file_path.exists():
                self.logger.debug(f"持久化存储文件不存在: {key}")
                return default
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                storage_data = json.load(f)
            
            # 检查数据完整性
            if 'value' not in storage_data:
                self.logger.warning(f"持久化存储数据格式错误: {key}")
                return default
            
            # 检查过期时间（对于session类型，1天过期；local类型，7天过期）
            max_age = 7 * 24 * 60 * 60 * 1000 if storage_type == "local" else 24 * 60 * 60 * 1000
            current_time = int(time.time() * 1000)
            
            if current_time - storage_data.get('timestamp', 0) > max_age:
                self.logger.debug(f"持久化存储已过期: {key}")
                # 删除过期文件
                file_path.unlink(missing_ok=True)
                return default
            
            self.logger.debug(f"持久化存储获取成功: {key}")
            return storage_data['value']
            
        except Exception as e:
            self.logger.error(f"获取持久化存储失败: {key}, 错误: {e}")
            return default
    
    def remove_item(self, key: str, storage_type: str = "local") -> bool:
        """
        移除持久化存储项
        
        Args:
            key: 存储键
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            是否移除成功
        """
        try:
            file_path = self._get_file_path(key, storage_type)
            
            if file_path.exists():
                file_path.unlink()
                self.logger.debug(f"持久化存储移除成功: {key}")
                return True
            else:
                self.logger.debug(f"持久化存储文件不存在，无需移除: {key}")
                return True
            
        except Exception as e:
            self.logger.error(f"移除持久化存储失败: {key}, 错误: {e}")
            return False
    
    def clear_all(self, storage_type: str = None) -> bool:
        """
        清空当前session的持久化存储

        Args:
            storage_type: 存储类型，None表示清空所有类型

        Returns:
            是否清空成功
        """
        try:
            session_id = _get_session_id()
            session_dir = self.storage_dir / session_id

            if not session_dir.exists():
                self.logger.debug(f"Session目录不存在: {session_id}")
                return True

            cleared_count = 0

            for file_path in session_dir.glob("*.json"):
                if storage_type is None or file_path.name.startswith(f"{storage_type}_"):
                    file_path.unlink()
                    cleared_count += 1

            self.logger.info(f"持久化存储已清空: session={session_id}, 类型={storage_type or '所有'}, 清理文件数: {cleared_count}")
            return True

        except Exception as e:
            self.logger.error(f"清空持久化存储失败: {storage_type}, 错误: {e}")
            return False

    def clear_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        清理过期的session数据

        Args:
            max_age_hours: 最大保留时间（小时）

        Returns:
            清理的session数量
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleared_sessions = 0

            for session_dir in self.storage_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                # 检查目录的最后修改时间
                dir_mtime = session_dir.stat().st_mtime
                if current_time - dir_mtime > max_age_seconds:
                    try:
                        import shutil
                        shutil.rmtree(session_dir)
                        cleared_sessions += 1
                        self.logger.info(f"清理过期session: {session_dir.name}")
                    except Exception as e:
                        self.logger.warning(f"清理session目录失败: {session_dir.name}, 错误: {e}")

            if cleared_sessions > 0:
                self.logger.info(f"清理完成: 共清理 {cleared_sessions} 个过期session")

            return cleared_sessions

        except Exception as e:
            self.logger.error(f"清理过期session失败: {e}")
            return 0


class BrowserPersistentStorage:
    """基于浏览器的持久化存储管理器（使用JavaScript + Streamlit通信）"""
    
    def __init__(self):
        """初始化浏览器持久化存储管理器"""
        self.logger = logging.getLogger(__name__)
    
    def set_item_async(self, key: str, value: Any, storage_type: str = "local") -> str:
        """
        异步设置持久化存储项
        
        Args:
            key: 存储键
            value: 存储值
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            JavaScript代码字符串
        """
        try:
            # 序列化值
            serialized_value = json.dumps(value, default=str, ensure_ascii=False).replace("'", "\\'")
            
            # 生成存储操作的JavaScript代码
            storage_method = "localStorage" if storage_type == "local" else "sessionStorage"
            
            js_code = f"""
            <div style="display: none;">
            <script>
            (function() {{
                try {{
                    {storage_method}.setItem('{key}', '{serialized_value}');
                    console.log('持久化存储设置成功: {key}');
                    
                    // 触发自定义事件通知存储成功
                    window.dispatchEvent(new CustomEvent('htfa_storage_set', {{
                        detail: {{
                            key: '{key}',
                            success: true,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }} catch (error) {{
                    console.error('持久化存储设置失败:', error);
                    window.dispatchEvent(new CustomEvent('htfa_storage_set', {{
                        detail: {{
                            key: '{key}',
                            success: false,
                            error: error.message,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }}
            }})();
            </script>
            </div>
            """
            
            return js_code
            
        except Exception as e:
            self.logger.error(f"生成存储JavaScript失败: {key}, 错误: {e}")
            return f"<div><!-- 存储JavaScript生成失败: {e} --></div>"
    
    def get_item_async(self, key: str, storage_type: str = "local") -> str:
        """
        异步获取持久化存储项
        
        Args:
            key: 存储键
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            JavaScript代码字符串
        """
        try:
            storage_method = "localStorage" if storage_type == "local" else "sessionStorage"
            
            js_code = f"""
            <div style="display: none;">
            <script>
            (function() {{
                try {{
                    var storedValue = {storage_method}.getItem('{key}');
                    var result = storedValue ? JSON.parse(storedValue) : null;
                    
                    console.log('持久化存储获取:', '{key}', result ? '成功' : '无数据');
                    
                    // 将结果存储到window对象
                    if (!window.htfa_storage_results) {{
                        window.htfa_storage_results = {{}};
                    }}
                    window.htfa_storage_results['{key}'] = result;
                    
                    // 触发自定义事件
                    window.dispatchEvent(new CustomEvent('htfa_storage_get', {{
                        detail: {{
                            key: '{key}',
                            value: result,
                            success: true,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }} catch (error) {{
                    console.error('持久化存储获取失败:', error);
                    window.dispatchEvent(new CustomEvent('htfa_storage_get', {{
                        detail: {{
                            key: '{key}',
                            success: false,
                            error: error.message,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }}
            }})();
            </script>
            </div>
            """
            
            return js_code
            
        except Exception as e:
            self.logger.error(f"生成获取JavaScript失败: {key}, 错误: {e}")
            return f"<div><!-- 获取JavaScript生成失败: {e} --></div>"
    
    def remove_item_async(self, key: str, storage_type: str = "local") -> str:
        """
        异步移除持久化存储项
        
        Args:
            key: 存储键
            storage_type: 存储类型 ("local" 或 "session")
            
        Returns:
            JavaScript代码字符串
        """
        try:
            storage_method = "localStorage" if storage_type == "local" else "sessionStorage"
            
            js_code = f"""
            <div style="display: none;">
            <script>
            (function() {{
                try {{
                    {storage_method}.removeItem('{key}');
                    console.log('持久化存储移除成功: {key}');
                    
                    // 清除window对象中的结果
                    if (window.htfa_storage_results && window.htfa_storage_results['{key}']) {{
                        delete window.htfa_storage_results['{key}'];
                    }}
                    
                    // 触发自定义事件
                    window.dispatchEvent(new CustomEvent('htfa_storage_remove', {{
                        detail: {{
                            key: '{key}',
                            success: true,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }} catch (error) {{
                    console.error('持久化存储移除失败:', error);
                    window.dispatchEvent(new CustomEvent('htfa_storage_remove', {{
                        detail: {{
                            key: '{key}',
                            success: false,
                            error: error.message,
                            storage_type: '{storage_type}',
                            timestamp: Date.now()
                        }}
                    }}));
                }}
            }})();
            </script>
            </div>
            """
            
            return js_code
            
        except Exception as e:
            self.logger.error(f"生成移除JavaScript失败: {key}, 错误: {e}")
            return f"<div><!-- 移除JavaScript生成失败: {e} --></div>"


# Session级别的存储管理器获取函数

def get_file_persistent_storage() -> FilePersistentStorage:
    """获取当前session的文件持久化存储实例"""
    if '_file_persistent_storage' not in st.session_state:
        st.session_state['_file_persistent_storage'] = FilePersistentStorage()
    return st.session_state['_file_persistent_storage']


def get_browser_persistent_storage() -> BrowserPersistentStorage:
    """获取当前session的浏览器持久化存储实例"""
    if '_browser_persistent_storage' not in st.session_state:
        st.session_state['_browser_persistent_storage'] = BrowserPersistentStorage()
    return st.session_state['_browser_persistent_storage']

