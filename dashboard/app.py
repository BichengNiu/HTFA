# -*- coding: utf-8 -*-
"""
经济运行分析平台 - 主dashboard
优化版本：使用统一初始化器和懒加载机制
"""

import sys
import os
import time
import warnings
import logging
import multiprocessing

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if multiprocessing.current_process().name != 'MainProcess':
    # 如果这是子进程，立即退出，不执行任何UI相关代码
    sys.exit(0)

# Dashboard状态管理使用st.session_state

# 在任何其他导入之前立即抑制 Streamlit 警告
def _suppress_streamlit_warnings():
    """在模块导入前抑制 Streamlit 警告"""
    # 设置环境变量 - 包括日志级别优化
    os.environ.update({
        'STREAMLIT_LOGGER_LEVEL': 'CRITICAL',
        'STREAMLIT_CLIENT_TOOLBAR_MODE': 'minimal',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_CLIENT_SHOW_ERROR_DETAILS': 'false',
        'PYTHONWARNINGS': 'ignore',
        'STREAMLIT_SILENT_IMPORTS': 'true',
        'STREAMLIT_SUPPRESS_WARNINGS': 'true',
        # 优化日志输出级别 - 减少冗余日志
        'LOG_LEVEL': 'WARNING',  # 设置为WARNING级别减少DEBUG/INFO日志
        'ENVIRONMENT': 'production'  # 设置为生产环境模式
    })

    # 抑制所有警告
    warnings.filterwarnings("ignore")

    # 预先配置 Streamlit 日志器
    streamlit_loggers = [
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner_utils",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.caching",
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.state",
        "streamlit.runtime.state.session_state_proxy",
        "streamlit.web",
        "streamlit.web.server",
        "streamlit.web.bootstrap"
    ]

    for logger_name in streamlit_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True
        logger.propagate = False

# 立即执行警告抑制
_suppress_streamlit_warnings()

# 配置全局日志级别，抑制INFO级别日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
    force=True
)
# 禁用dashboard模块的INFO日志
logging.getLogger('dashboard').setLevel(logging.WARNING)

# 添加项目根目录到Python路径，确保能正确导入dashboard包
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上一级目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 立即导入streamlit并设置页面配置 - 必须在任何其他操作之前
import streamlit as st

# 直接设置页面配置，避免导入问题
if 'dashboard_page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="经济运行分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state['dashboard_page_config_set'] = True

# 强制直接注入CSS
import os
from pathlib import Path

# 获取CSS文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, "ui", "static", "styles.css")

if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        css_content = f.read()

    # 直接注入CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# 全局CSS注入标志 - 使用进程级别缓存
_css_injection_cache = {}

# CSS注入函数 - 修改为每次都注入以确保样式一致性
def inject_styles_always():
    """每次页面渲染都注入CSS - 修复样式变化问题"""
    inject_cached_styles()

def inject_styles_once():
    """只执行一次CSS注入"""
    process_id = os.getpid()
    css_cache_key = f"css_injected_{process_id}"

    # 检查进程级别缓存
    if _css_injection_cache.get(css_cache_key, False):
        return

    if not st.session_state.get(f'dashboard.css.{css_cache_key}', False):
        inject_cached_styles()
        st.session_state[f'dashboard.css.{css_cache_key}'] = True
        _css_injection_cache[css_cache_key] = True
    else:
        _css_injection_cache[css_cache_key] = True

# 基础库导入
import pandas as pd
import re
import altair as alt
from datetime import datetime
import inspect
import traceback as tb

# 应用初始化将在页面配置后进行
# 延迟导入避免触发 @st.cache_resource

# === 导入UI调试工具 ===
from dashboard.core.ui.utils.debug_helpers import (
    debug_log, debug_state_change, debug_navigation, debug_button_click
)

# === 导入核心模块 ===
from dashboard.core import get_resource_loader
from dashboard.core import get_config_cache
from dashboard.core import get_navigation_manager
from dashboard.core.ui.utils.style_loader import inject_cached_styles

# 简化初始化，避免循环导入
lazy_loader = None
nav_manager = None

# 第二阶段优化：导入资源加载器
def initialize_resource_loading():
    """初始化资源加载系统"""
    # 获取资源加载器
    resource_loader = get_resource_loader()
    config_cache = get_config_cache()

    # 预加载关键资源
    resource_loader.preload_critical_resources()

    return resource_loader, config_cache

# 注释：已使用统一的 navigation_manager，不再需要额外的状态同步函数

# 延迟初始化函数
def get_managers():
    """延迟获取管理器实例 - 优化版本，使用session_state缓存"""
    # streamlit, time, os模块已在顶部导入

    global lazy_loader, nav_manager

    # 获取当前进程ID用于缓存键
    process_id = os.getpid()
    init_lock_key = f"managers_init_lock_{process_id}"

    # 检查初始化锁
    if st.session_state.get(f'dashboard.init_lock.{init_lock_key}', False):
        # 正在初始化，返回缓存的实例（如果有）
        cached_managers = st.session_state.get('dashboard.managers_cache', None)
        if cached_managers:
            return cached_managers
        return None, None

    # 检查缓存
    current_time = time.time()
    cached_managers = st.session_state.get('dashboard.managers_cache', None)
    cached_time = st.session_state.get('dashboard.managers_cache_time', 0)
    cached_health = st.session_state.get('dashboard.managers_health', False)

    # 如果缓存在30分钟内且健康检查通过，使用缓存
    if (cached_managers and cached_time and
        current_time - cached_time < 1800 and cached_health):
        lazy_loader, nav_manager = cached_managers
        return lazy_loader, nav_manager

    # 记录初始化开始时间
    start_time = time.time()

    # 设置初始化锁
    st.session_state[f'dashboard.init_lock.{init_lock_key}'] = True
    debug_log(f"初始化锁设置成功: {init_lock_key}", "DEBUG")

    # 重新初始化管理器
    if lazy_loader is None:
        lazy_loader = get_resource_loader()

    if nav_manager is None:
        nav_manager = get_navigation_manager()
        debug_log("导航管理器初始化成功", "INFO")

    # 执行健康检查
    health_check = _perform_managers_health_check(nav_manager)

    # 第二阶段优化：初始化资源加载系统
    resource_loader, config_cache = initialize_resource_loading()

    # 计算初始化耗时
    end_time = time.time()
    init_time = (end_time - start_time) * 1000  # 转换为毫秒

    # 缓存到session_state
    managers_tuple = (lazy_loader, nav_manager)
    st.session_state['dashboard.managers_cache'] = managers_tuple
    st.session_state['dashboard.managers_cache_time'] = current_time
    st.session_state['dashboard.managers_health'] = health_check

    # 缓存资源加载器
    if resource_loader:
        st.session_state[f'dashboard.resource_loader_{process_id}'] = resource_loader
    if config_cache:
        st.session_state[f'dashboard.config_cache_{process_id}'] = config_cache

    # 释放初始化锁
    st.session_state[f'dashboard.init_lock.{init_lock_key}'] = False

    return lazy_loader, nav_manager

def _perform_managers_health_check(nav_manager):
    """执行管理器健康检查"""
    try:
        # 检查导航管理器
        if nav_manager is not None:
            # 尝试调用一个简单的方法来验证导航管理器是否正常工作
            if hasattr(nav_manager, 'get_current_main_module'):
                nav_manager.get_current_main_module()

        return True
    except Exception as e:
        print(f"[DEBUG Init] 管理器健康检查失败: {e}")
        return False

# 注释：已使用统一的 navigation_manager，不再需要 check_navigation_change 函数

# 使用UI模块的按钮状态管理
from dashboard.core.ui.utils.button_state_manager import optimize_button_state_management


# 配置Altair数据转换器
alt.data_transformers.enable("vegafusion")

# st.sidebar.title("[CHART] 经济运行分析平台")


# extract_industry_name函数已移至utils模块，避免重复定义

from dashboard.core.ui.utils.state_helpers import (
    get_dashboard_state, set_dashboard_state, get_staged_data,
    get_staged_data_options, clear_analysis_states as ui_clear_analysis_states,
    set_analysis_data, clear_analysis_data
)

def clear_analysis_states(analysis_type: str, selected_name: str = None):
    """清理分析相关状态 - 使用UI模块的状态管理器"""
    return ui_clear_analysis_states(analysis_type)


# extract_industry_name函数已移至dashboard.utils.industry_utils模块，避免重复定义


MODULE_CONFIG = {
    "数据预览": None,  # 直接显示工业数据预览功能，不区分子模块
    "监测分析": {
        "工业": ["工业增加值", "工业企业利润拆解"]
    },
    "模型分析": {
        "DFM 模型": ["数据准备", "模型训练", "模型分析", "新闻分析"]
    },
    "数据探索": None,  # 直接显示数据探索功能，包含平稳性分析和相关性分析
    "用户管理": None  # 直接显示用户管理功能
}


def _perform_intelligent_state_cleanup():
    """执行智能状态清理，避免循环渲染"""
    try:
        # 获取所有状态键
        all_keys = list(st.session_state.keys())

        # 清理导航相关的临时状态
        navigation_patterns = ['navigate_to', 'temp_selected', 'rerun', '_transition', '_loading']
        navigation_keys = [k for k in all_keys if any(pattern in str(k) for pattern in navigation_patterns)]

        for key in navigation_keys:
            if key in st.session_state:
                del st.session_state[key]

        # 清理可能导致循环的组件状态
        component_patterns = ['_preview_data', '_processed_data', '_analysis_result', '_cached_']
        component_keys = [k for k in all_keys if any(pattern in str(k) for pattern in component_patterns)]

        for key in component_keys:
            if key in st.session_state:
                del st.session_state[key]

        print(f"[DEBUG Recovery] 清理了 {len(navigation_keys + component_keys)} 个状态键")

    except Exception as e:
        print(f"[DEBUG Recovery] 状态清理失败: {e}")


# 每次都注入CSS样式以确保样式一致性（修复模块切换时样式变化问题）
inject_styles_always()

# 获取管理器实例
lazy_loader, nav_manager = get_managers()

# 集成认证中间件
from dashboard.auth.ui.middleware import get_auth_middleware
from dashboard.auth.config import AuthConfig

auth_middleware = get_auth_middleware()

# 根据调试模式决定是否需要认证
if AuthConfig.is_debug_mode():
    # 调试模式：跳过认证
    current_user = None
else:
    # 正常模式：强制要求认证
    current_user = auth_middleware.require_authentication(show_login=True)

# 如果用户已认证，渲染用户信息到侧边栏
if current_user:
    # 在侧边栏显示用户信息和登出按钮
    auth_middleware.render_user_info()

# time模块已在顶部导入
current_time = time.time()

# 检查是否有模块正在加载
all_keys = list(st.session_state.keys())
loading_modules = [key for key in all_keys if key.startswith('_loading_')]
if loading_modules:
    st.stop()

# 检测快速连续重新渲染（回弹检测）- 改进版本
last_render_time = st.session_state.get('dashboard.last_render_time', 0)
render_interval = current_time - last_render_time

# 检查是否是用户主动导航操作导致的重新渲染
is_navigation_triggered = False
if nav_manager:
    # 检查导航状态是否在变化中
    is_navigation_triggered = nav_manager.is_transitioning()

    # 检查是否在导航操作的时间窗口内（2秒内的导航操作都认为是用户主动的）
    if not is_navigation_triggered:
        last_nav_time = st.session_state.get('dashboard.last_navigation_time', 0)
        if current_time - last_nav_time < 2.0:  # 2秒的导航操作窗口
            is_navigation_triggered = True

# 只有在非导航触发且间隔很短的情况下才视为回弹
if render_interval < 0.05 and last_render_time > 0 and not is_navigation_triggered:  # 50ms阈值，排除导航触发
    st.stop()

st.session_state['dashboard.last_render_time'] = current_time

# 如果管理器正在初始化，跳过本次渲染
if lazy_loader is None or nav_manager is None:
    st.info("系统正在初始化，请稍候...")
    st.stop()

try:
    st.session_state['dashboard.initialized'] = True
    st.session_state['dashboard.start_time'] = datetime.now()
except Exception as e:
    st.error(f"状态初始化失败: {e}")
    st.stop()

# 管理器初始化结果已在get_managers()函数中打印，避免重复

# 确保键的唯一性和防止重复渲染
if not st.session_state.get('dashboard.sidebar.rendered', False):
    st.session_state['dashboard.sidebar.rendered'] = True
    st.session_state['dashboard.sidebar.key_counter'] = 0
    st.session_state['dashboard.main_content.rendered'] = False

# 改进的循环渲染检测机制
current_time = time.time()

# 获取渲染跟踪数据
default_tracking = {'count': 0, 'last_reset': time.time(), 'last_render': 0}
tracking = st.session_state.get('dashboard.render_tracking', default_tracking)

# 每30秒重置计数器，避免正常使用被误判
if current_time - tracking['last_reset'] > 30:
    current_ts = time.time()
    tracking = {'count': 0, 'last_reset': current_ts, 'last_render': current_ts}
    st.session_state['dashboard.render_tracking'] = tracking

# 增加渲染计数
tracking['count'] += 1
render_interval = current_time - tracking['last_render']
tracking['last_render'] = time.time()
st.session_state['dashboard.render_tracking'] = tracking

# 改进循环渲染检测：更严格的条件，避免误判用户正常操作
# 只有在极短时间内（<0.05秒）连续渲染超过10次且非导航触发时才认为是循环渲染
is_user_navigation = False
if nav_manager:
    # 检查是否是用户导航操作
    is_user_navigation = (nav_manager.is_transitioning() or
                         (current_time - st.session_state.get('dashboard.last_navigation_time', 0) < 3.0))

if (render_interval and render_interval < 0.05 and tracking and tracking['count'] > 10 and
    not is_user_navigation):
    # 只在真正的循环渲染时显示警告，避免对用户造成困扰
    st.warning("检测到异常的页面渲染循环，正在自动修复...")

    # 智能状态清理
    _perform_intelligent_state_cleanup()

    # 重置渲染计数
    current_ts = time.time()
    tracking = {'count': 0, 'last_reset': current_ts, 'last_render': current_ts}
    st.session_state['dashboard.render_tracking'] = tracking

    # 简化恢复信息，减少对用户的干扰
    with st.expander("系统状态", expanded=False):
        st.info("系统已自动修复渲染问题")
        st.info("如果页面仍有异常，请刷新页面")
        if st.button("刷新页面", key="manual_refresh_button"):
            st.rerun()

# 确保key稳定性
stable_key_prefix = st.session_state.get('dashboard.sidebar.stable_key_prefix')
if not stable_key_prefix:
    # 使用更长的时间戳和随机数确保唯一性
    import time
    import random
    import uuid
    timestamp = str(int(time.time() * 1000))
    random_suffix = str(random.randint(1000, 9999))
    session_id = str(uuid.uuid4())[:8]  # 使用UUID的前8个字符
    stable_key_prefix = f"sidebar_{timestamp}_{random_suffix}_{session_id}"
    st.session_state['dashboard.sidebar.stable_key_prefix'] = stable_key_prefix
key_prefix = stable_key_prefix

# 使用UI模块的完整侧边栏组件
from dashboard.core.ui.components.sidebar import render_complete_sidebar
from dashboard.core.ui.components.content_router import force_navigation_state_sync

# 在侧边栏渲染前强制同步导航状态，确保按钮状态正确
if nav_manager:
    current_main = nav_manager.get_current_main_module()
    current_sub = nav_manager.get_current_sub_module()

    if current_main or current_sub:
        force_navigation_state_sync(current_main, current_sub)

    # 获取用户可访问的模块信息
    user_accessible_modules = set()
    if current_user and auth_middleware:
        try:
            # 正常模式：使用权限管理器获取用户可访问的模块列表
            accessible_modules_list = auth_middleware.permission_manager.get_accessible_modules(current_user)
            user_accessible_modules = set(accessible_modules_list)
            debug_navigation("权限检查", f"用户 {current_user.username} 可访问 {len(user_accessible_modules)} 个模块: {user_accessible_modules}")
        except Exception as e:
            print(f"[ERROR] 模块权限检查失败: {e}")
            # 如果检查失败，不给任何默认权限，让用户联系管理员
            user_accessible_modules = set()
    else:
        # 调试模式：没有用户时给所有权限
        if AuthConfig.is_debug_mode():
            user_accessible_modules = set(AuthConfig.DEBUG_ACCESSIBLE_MODULES)
            debug_navigation("权限检查", f"调试模式：允许访问所有模块 ({len(user_accessible_modules)} 个)")
        else:
            # 正常模式且未登录：无权限
            user_accessible_modules = set()

    # 将权限信息和调试模式状态存储到session_state
    st.session_state['auth.debug_mode'] = AuthConfig.is_debug_mode()
    st.session_state['auth.user_accessible_modules'] = user_accessible_modules
    st.session_state['auth.current_user'] = current_user
    
    # 渲染完整侧边栏（显示所有模块）
    sidebar_result = render_complete_sidebar(MODULE_CONFIG, nav_manager, key_prefix)

    # 提取结果
    main_selector_result = sidebar_result.get('main_module_result', {})
    sub_selector_result = sidebar_result.get('sub_module_result')

    selected_main_module_val = main_selector_result.get('selected_module', None)
    change_result = main_selector_result.get('success', True)

    if sub_selector_result:
        selected_sub_module_val = sub_selector_result.get('selected_sub_module')
    else:
        selected_sub_module_val = None

    # 处理临时状态恢复
    temp_main = st.session_state.get('navigation.temp_selected_main_module')
    if temp_main:
        selected_main_module_val = temp_main
        if 'navigation.temp_selected_main_module' in st.session_state:
            del st.session_state['navigation.temp_selected_main_module']
        debug_navigation("临时状态恢复", f"恢复主模块选择: {selected_main_module_val}")

    temp_sub = st.session_state.get('navigation.temp_selected_sub_module')
    if temp_sub:
        selected_sub_module_val = temp_sub
        st.session_state['navigation.last_clicked_sub_module'] = selected_sub_module_val
        if 'navigation.temp_selected_sub_module' in st.session_state:
            del st.session_state['navigation.temp_selected_sub_module']

    # 获取当前状态用于后续逻辑
    current_main_module = nav_manager.get_current_main_module() if nav_manager else None
    current_sub_module = nav_manager.get_current_sub_module() if nav_manager else None

    # 立即更新当前模块变量以确保按钮状态同步
    if selected_main_module_val != current_main_module:
        debug_state_change("主模块切换", current_main_module, selected_main_module_val, "用户点击按钮")
else:
    selected_main_module_val = None
    selected_sub_module_val = None
    change_result = True
    current_main_module = None
    current_sub_module = None

# 处理主模块切换逻辑
if selected_main_module_val != current_main_module:
    # 设置导航状态为转换中
    if nav_manager:
        nav_manager.set_transitioning(True)
        debug_navigation("转换状态设置", "设置transitioning=True")

    current_main_module = selected_main_module_val

    # 主模块切换时，清除子模块状态以避免状态污染
    debug_navigation("状态清除", "主模块切换，开始清除子模块状态")
    if nav_manager:
        nav_manager.set_current_sub_module(None)

    # 避免重复渲染，只在必要时重新运行
    # 检查是否真的需要重新渲染
    last_change = st.session_state.get('dashboard.last_main_module_change')
    if last_change != selected_main_module_val:
        st.session_state['dashboard.last_main_module_change'] = selected_main_module_val
        debug_navigation("重新渲染", f"主模块切换到 {selected_main_module_val}，触发重新渲染")
else:
    # 确保非切换时清除transitioning状态
    if nav_manager:
        nav_manager.set_transitioning(False)


# 使用UI模块的主内容路由组件
from dashboard.core.ui.components.content_router import render_main_content

# 渲染主内容
content_result = render_main_content(nav_manager)

debug_navigation(
    "主内容渲染",
    f"内容渲染完成 - 模块: {content_result.get('main_module')}, "
    f"子模块: {content_result.get('sub_module')}, "
    f"状态: {content_result.get('status')}"
)

# 在内容渲染完成后清除transitioning状态
if nav_manager:
    nav_manager.set_transitioning(False)

# (End of script)