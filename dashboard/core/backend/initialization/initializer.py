# -*- coding: utf-8 -*-
"""
应用初始化器 - 重构版
按职责拆分为多个专门初始化器
"""

import os
import sys
import warnings
import logging
import shutil
import glob
import time
from pathlib import Path
from typing import Dict, Any
import streamlit as st
from dashboard.core.backend.config import get_core_config
# TODO: 更新为 dashboard.core.ui.utils.style_initializer
from dashboard.core.ui.utils.style_initializer import get_ui_initializer
from dashboard.core.backend.resource import get_resource_loader

logger = logging.getLogger(__name__)


# ==================== 环境初始化器 ====================

class EnvironmentInitializer:
    """环境初始化器 - 负责环境变量和路径设置"""

    @staticmethod
    def setup_environment():
        """设置环境变量"""
        start_time = time.time()

        # 从配置系统加载环境变量
        config = get_core_config()
        env_vars = config.get_env_vars()

        for key, value in env_vars.items():
            os.environ[key] = value

        warnings.filterwarnings("ignore")

        streamlit_logger = logging.getLogger("streamlit")
        streamlit_logger.setLevel(logging.CRITICAL)
        streamlit_logger.propagate = False

        setup_time = time.time() - start_time
        logger.debug(f"Environment setup completed in {setup_time:.3f}s")

    @staticmethod
    def setup_paths():
        """设置项目路径"""
        start_time = time.time()

        current_dir = Path(__file__).parent.parent.parent
        project_root = current_dir.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        setup_time = time.time() - start_time
        logger.debug(f"Path setup completed in {setup_time:.3f}s")


# ==================== Streamlit初始化器 ====================

class StreamlitInitializer:
    """Streamlit初始化器 - 负责Streamlit配置"""

    @staticmethod
    def configure_streamlit():
        """配置Streamlit"""
        start_time = time.time()

        already_configured = st.session_state.get('core.streamlit_configured', False)

        if not already_configured:
            st.session_state['core.streamlit_configured'] = True
            logger.info("Streamlit页面配置状态记录完成")

        setup_time = time.time() - start_time
        logger.debug(f"Streamlit configuration completed in {setup_time:.3f}s")

    @staticmethod
    def load_styles():
        """加载应用样式"""
        start_time = time.time()

        ui_initializer = get_ui_initializer()
        ui_initializer.load_styles()
        logger.info("使用UI初始化器加载样式成功")

        setup_time = time.time() - start_time
        logger.debug(f"Styles loaded in {setup_time:.3f}s")


# ==================== 缓存清理管理器 ====================

class CacheCleanupManager:
    """缓存清理管理器 - 负责清理缓存和日志"""

    @staticmethod
    def _get_cleanup_targets(current_dir: Path, project_root: Path):
        """
        获取清理目标列表

        Args:
            current_dir: 当前目录
            project_root: 项目根目录

        Returns:
            清理目标路径列表
        """
        return [
            project_root / "cache",
            current_dir / "cache",
            project_root / "logs",
            current_dir / "logs",
            project_root / "*.tmp",
            project_root / "*.log",
            current_dir / "*.tmp",
            current_dir / "*.log"
        ]

    @staticmethod
    def _cleanup_single_target(target: Path, cleaned_items: list):
        """
        清理单个目标

        Args:
            target: 目标路径
            cleaned_items: 已清理项目列表（会被修改）
        """
        try:
            if target.exists():
                if target.is_dir():
                    for item in target.iterdir():
                        if item.is_file():
                            item.unlink()
                            cleaned_items.append(f"文件: {item}")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            cleaned_items.append(f"目录: {item}")
                elif target.is_file():
                    target.unlink()
                    cleaned_items.append(f"文件: {target}")
            else:
                if "*" in str(target):
                    pattern_files = glob.glob(str(target))
                    for file_path in pattern_files:
                        os.remove(file_path)
                        cleaned_items.append(f"模式文件: {file_path}")
        except Exception as e:
            logger.warning(f"清理 {target} 时出错: {e}")

    @staticmethod
    def _cleanup_paths(targets: list, cleaned_items: list):
        """
        批量清理路径

        Args:
            targets: 目标路径列表
            cleaned_items: 已清理项目列表（会被修改）
        """
        for target in targets:
            CacheCleanupManager._cleanup_single_target(target, cleaned_items)

    @staticmethod
    def clear_cache_and_logs(skip_pycache=True):
        """
        清理所有缓存、日志和旧结果文件

        Args:
            skip_pycache: 是否跳过__pycache__清理（默认True，避免耗时过长）
        """
        start_time = time.time()

        try:
            current_dir = Path(__file__).parent.parent.parent
            project_root = current_dir.parent

            cleanup_targets = CacheCleanupManager._get_cleanup_targets(current_dir, project_root)
            cleaned_items = []

            CacheCleanupManager._cleanup_paths(cleanup_targets, cleaned_items)

            if not skip_pycache:
                CacheCleanupManager._clean_pycache_directories(project_root, cleaned_items)

            setup_time = time.time() - start_time

            if cleaned_items:
                logger.info(f"缓存、日志清理完成，共清理 {len(cleaned_items)} 项，耗时 {setup_time:.3f}s")
            else:
                logger.info(f"没有发现需要清理的缓存或日志文件，耗时 {setup_time:.3f}s")

        except Exception as e:
            logger.error(f"缓存和日志清理失败: {e}")

    @staticmethod
    def _clean_pycache_directories(root_path, cleaned_items):
        """递归清理所有__pycache__目录"""
        try:
            for item in root_path.rglob("__pycache__"):
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                        cleaned_items.append(f"__pycache__目录: {item}")
                    except Exception as e:
                        logger.warning(f"清理__pycache__目录 {item} 时出错: {e}")
        except Exception as e:
            logger.warning(f"搜索__pycache__目录时出错: {e}")


# ==================== 应用初始化器 ====================

class AppInitializer:
    """应用初始化器 - 总协调器"""

    def __init__(self):
        self.initialization_time = 0
        self.initialized_components = set()

    def _initialize_component(self, component_name: str, init_func, post_init_func=None):
        """
        通用组件初始化方法

        Args:
            component_name: 组件名称
            init_func: 初始化函数，返回组件实例
            post_init_func: 初始化后的回调函数，接收组件实例作为参数
        """
        start_time = time.time()

        try:
            component_instance = init_func()

            if post_init_func:
                post_init_func(component_instance)

            logger.info(f"{component_name}初始化完成")

        except Exception as e:
            logger.error(f"Failed to initialize {component_name}: {e}")

        setup_time = time.time() - start_time
        logger.debug(f"{component_name} setup completed in {setup_time:.3f}s")
        self.initialized_components.add(component_name)


    def initialize_resource_loader(self):
        """初始化资源加载器"""
        def init_func():
            return get_resource_loader()

        def post_init_func(resource_loader):
            resource_loader.preload_critical_resources()

        self._initialize_component('resource_loader', init_func, post_init_func)

    def full_initialize(self, enable_cleanup=False) -> Dict[str, Any]:
        """
        完整初始化应用

        Args:
            enable_cleanup: 是否启用启动时清理缓存（默认False，提升性能）
        """
        total_start_time = time.time()

        initialization_steps = [
            ('environment', EnvironmentInitializer.setup_environment),
            ('paths', EnvironmentInitializer.setup_paths),
            ('streamlit', StreamlitInitializer.configure_streamlit),
            ('styles', StreamlitInitializer.load_styles),
            ('resource_loader', self.initialize_resource_loader)
        ]

        if enable_cleanup:
            initialization_steps.insert(0, ('cleanup', CacheCleanupManager.clear_cache_and_logs))

        step_times = {}

        for step_name, step_func in initialization_steps:
            if step_name not in self.initialized_components:
                step_start = time.time()
                try:
                    step_func()
                    step_times[step_name] = time.time() - step_start
                    self.initialized_components.add(step_name)
                except Exception as e:
                    logger.error(f"Failed to initialize {step_name}: {e}")
                    step_times[step_name] = -1

        self.initialization_time = time.time() - total_start_time

        return {
            'total_time': self.initialization_time,
            'step_times': step_times,
            'initialized_components': list(self.initialized_components)
        }

    def get_initialization_stats(self) -> Dict[str, Any]:
        """获取初始化统计信息"""
        return {
            'initialization_time': self.initialization_time,
            'initialized_components': list(self.initialized_components),
            'component_count': len(self.initialized_components)
        }


# ==================== 全局函数 ====================

_app_initializer = None


def get_app_initializer() -> AppInitializer:
    """获取全局应用初始化器实例"""
    global _app_initializer
    if _app_initializer is None:
        _app_initializer = AppInitializer()
    return _app_initializer


def initialize_app(enable_cleanup=False) -> Dict[str, Any]:
    """
    应用初始化函数

    Args:
        enable_cleanup: 是否启用启动时清理缓存
    """
    initializer = get_app_initializer()
    return initializer.full_initialize(enable_cleanup=enable_cleanup)
