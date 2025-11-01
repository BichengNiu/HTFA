# -*- coding: utf-8 -*-
"""
主内容路由组件
提供主内容区域的路由和渲染功能
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dashboard.ui.utils.tab_detector import TabStateDetector
from dashboard.ui.components.analysis.timeseries import (
    StationarityAnalysisComponent,
    UnifiedCorrelationAnalysisComponent,
    LeadLagAnalysisComponent
)
from dashboard.ui.components.sidebar import DataExplorationSidebar
from dashboard.ui.pages.main_modules.user_management import UserManagementWelcomePage, render_user_management_sub_module

logger = logging.getLogger(__name__)


def force_navigation_state_sync(main_module: str, sub_module: str = None):
    """
    强制同步导航状态并刷新相关缓存。

    Args:
        main_module: 主模块名称
        sub_module: 子模块名称
    """
    st.session_state['navigation.main_module'] = main_module
    if sub_module:
        st.session_state['navigation.sub_module'] = sub_module

    cache_keys_to_clear = [
        'ui.button_state_cache',
        'ui.button_state_time',
        'ui.navigation_cache',
        'ui.module_selector_cache'
    ]

    existing_keys = set(st.session_state.keys())
    for key in cache_keys_to_clear:
        if key in existing_keys:
            del st.session_state[key]

    from dashboard.ui.utils.button_state_manager import clear_button_state_cache, update_button_state_cache
    clear_button_state_cache()

    from dashboard.ui.components.sidebar import filter_modules_by_permission
    all_module_options = ['数据预览', '监测分析', '模型分析', '数据探索', '用户管理']
    main_module_options = filter_modules_by_permission(all_module_options)
    update_button_state_cache(main_module_options, main_module)


def check_user_permission(module_name: str) -> tuple[bool, Optional[str]]:
    """
    检查用户是否有访问指定模块的权限

    Args:
        module_name: 模块名称

    Returns:
        tuple[bool, Optional[str]]: (是否有权限, 错误信息)
    """
    try:
        # 读取调试模式状态
        debug_mode = st.session_state.get('auth.debug_mode', True)

        # 调试模式：直接放行
        if debug_mode:
            logger.debug(f"调试模式：允许访问模块 {module_name}")
            return True, None

        # 正常模式：检查用户权限
        user_accessible_modules = st.session_state.get('auth.user_accessible_modules', set())
        current_user = st.session_state.get('auth.current_user', None)

        if not current_user:
            error_msg = f"请先登录后访问「{module_name}」模块"
            logger.warning(f"权限检查失败：{error_msg}")
            return False, error_msg

        # 获取管理员权限检查器
        from dashboard.ui.components.auth.auth_middleware import get_auth_middleware
        auth_middleware = get_auth_middleware()
        is_admin = auth_middleware.permission_manager.is_admin(current_user)

        # 用户管理模块：仅管理员可访问
        if module_name == '用户管理':
            if is_admin:
                logger.debug(f"权限检查通过：管理员可以访问用户管理模块")
                return True, None
            else:
                error_msg = f"只有管理员才能访问「{module_name}」模块"
                logger.warning(f"权限检查失败：{error_msg}")
                return False, error_msg

        # 其他模块：管理员不可访问，普通用户按权限访问
        if is_admin:
            error_msg = f"管理员账户无法访问「{module_name}」模块，仅可访问用户管理"
            logger.warning(f"权限检查失败：{error_msg}")
            return False, error_msg

        # 普通用户：检查模块是否在可访问列表中
        if module_name in user_accessible_modules:
            logger.debug(f"权限检查通过：用户可以访问模块 {module_name}")
            return True, None
        else:
            error_msg = f"您没有访问「{module_name}」模块的权限，请联系管理员"
            logger.warning(f"权限检查失败：{error_msg}")
            return False, error_msg

    except Exception as e:
        logger.error(f"权限检查失败: {e}")
        return False, f"权限检查失败: {e}"


def render_permission_denied(module_name: str, error_message: str = None) -> Dict[str, Any]:
    """
    渲染权限拒绝页面
    
    Args:
        module_name: 模块名称
        error_message: 错误信息
        
    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 只显示一个简洁的红色提示信息
    st.error("无访问权限")
    
    return {
        'status': 'permission_denied',
        'content_type': 'permission_denied',
        'main_module': module_name,
        'error_message': error_message
    }


def render_main_content(nav_manager: Any) -> Dict[str, Any]:
    """
    渲染主内容区域

    Args:
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 获取当前导航状态
    current_main_module = nav_manager.get_current_main_module() if nav_manager else None
    current_sub_module = nav_manager.get_current_sub_module() if nav_manager else None

    # 如果没有选择主模块，显示欢迎页面
    if not current_main_module:
        with create_content_container():
            render_welcome_page(None, None)
        return {
            'main_module': None,
            'sub_module': None,
            'content_type': 'welcome',
            'status': 'success'
        }

    # 检查用户权限
    has_permission, permission_error = check_user_permission(current_main_module)
    if not has_permission:
        with create_content_container():
            return render_permission_denied(current_main_module, permission_error)

    _clear_previous_module_state(current_main_module)

    logger.debug(f"渲染主内容 - 主模块: {current_main_module}, 子模块: {current_sub_module}")

    # 获取内容配置
    content_config = get_content_config(current_main_module, current_sub_module)
    
    if not validate_content_config(content_config):
        st.error("内容配置无效")
        return {
            'main_module': current_main_module,
            'sub_module': current_sub_module,
            'content_type': 'error',
            'status': 'error'
        }
    
    # 创建内容容器
    with create_content_container():
        # 路由到具体内容（平台标题只在欢迎页面显示）
        content_result = route_to_content(content_config, nav_manager)
    
    return {
        'main_module': current_main_module,
        'sub_module': current_sub_module,
        'content_type': content_config['content_type'],
        'status': 'success',
        'content_result': content_result
    }


def get_content_config(main_module: str, sub_module: Optional[str] = None) -> Dict[str, Any]:
    """
    获取内容配置
    
    Args:
        main_module: 主模块名称
        sub_module: 子模块名称
        
    Returns:
        Dict[str, Any]: 内容配置
    """
    # 基础配置
    config = {
        'title': main_module,
        'icon': get_module_icon(main_module),
        'description': get_module_description(main_module, sub_module),
        'main_module': main_module,
        'sub_module': sub_module
    }
    
    # 根据主模块设置内容类型
    if main_module == '数据预览':
        config['content_type'] = 'data_preview'
    elif main_module == '监测分析':
        config['content_type'] = 'monitoring_analysis'
    elif main_module == '模型分析':
        config['content_type'] = 'model_analysis'
    elif main_module == '数据探索':
        config['content_type'] = 'data_exploration'
    elif main_module == '用户管理':
        config['content_type'] = 'user_management'
    else:
        config['content_type'] = 'unknown'

    return config


def render_content_header(config: Dict[str, Any]) -> None:
    """
    渲染内容头部
    
    Args:
        config: 内容配置
    """
    # 渲染标题
    st.markdown(f"{config['icon']}")
    st.title(config['title'])
    
    # 渲染描述
    if config.get('description'):
        st.markdown(config['description'])


def route_to_content(config: Dict[str, Any], nav_manager: Any) -> Dict[str, Any]:
    """
    路由到具体内容

    Args:
        config: 内容配置
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 内容渲染结果
    """
    content_type = config['content_type']
    main_module = config.get('main_module')
    sub_module = config.get('sub_module')

    # 检测导航层次
    navigation_level = detect_navigation_level(main_module, sub_module, nav_manager)

    try:
        # 根据导航层次决定渲染内容
        if navigation_level == 'MAIN_MODULE_ONLY':
            # 特殊处理：用户管理模块没有子模块时直接显示内容
            if main_module == '用户管理':
                return render_user_management_content(sub_module, nav_manager)
            # 第一层：只选择了主模块，显示子模块选择界面
            return render_module_selection_guide(main_module, 'sub_module')
        elif navigation_level == 'SUB_MODULE_ONLY':
            # 第二层：选择了子模块，但没有活跃的第三层tab
            # 对于数据探索模块，直接显示tab界面
            if main_module == '数据探索':
                return render_data_exploration_content(sub_module, nav_manager)
            # 对于模型分析模块，直接显示tab界面而不是功能选择指导
            elif main_module == '模型分析' and sub_module:
                return render_model_analysis_content(sub_module, nav_manager)
            else:
                # 其他模块显示功能选择界面
                return render_module_selection_guide(main_module, 'function', sub_module)
        elif navigation_level == 'FUNCTION_ACTIVE':
            # 第三层：有活跃的功能tab，渲染具体内容
            if content_type == 'data_preview':
                return render_data_preview_content(sub_module, nav_manager)
            elif content_type == 'monitoring_analysis':
                return render_monitoring_analysis_content(sub_module, nav_manager)
            elif content_type == 'model_analysis':
                return render_model_analysis_content(sub_module, nav_manager)
            elif content_type == 'data_exploration':
                return render_data_exploration_content(sub_module, nav_manager)
            elif content_type == 'user_management':
                return render_user_management_content(sub_module, nav_manager)
            else:
                st.warning(f"未知的内容类型: {content_type}")
                return {'status': 'warning', 'message': f'未知的内容类型: {content_type}'}
        else:
            # 默认情况，显示欢迎页面
            return render_welcome_page(main_module, sub_module)

    except Exception as e:
        st.error(f"内容渲染失败: {e}")
        logger.error(f"渲染{content_type}失败: {e}")
        return {'status': 'error', 'message': str(e)}


def render_data_preview_content(sub_module: Optional[str], nav_manager: Any) -> Dict[str, Any]:
    """
    渲染数据预览内容 - 直接显示工业数据预览

    Args:
        sub_module: 子模块名称 (现在被忽略，直接显示工业数据预览)
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 直接渲染工业数据预览内容，不区分子模块
    from datetime import datetime
    print(f"\n[DEBUG-ROUTER] render_data_preview_content 被调用 - 时间: {datetime.now().strftime('%H:%M:%S.%f')}")

    from dashboard.preview.main import display_industrial_tabs
    from dashboard.preview.data_loader import extract_industry_name

    print(f"[DEBUG-ROUTER] 准备调用 display_industrial_tabs")
    # 调用工业数据预览的主要功能
    display_industrial_tabs(extract_industry_name)
    print(f"[DEBUG-ROUTER] display_industrial_tabs 调用完成\n")

    return {'status': 'success', 'content_type': 'data_preview', 'sub_module': None}


def render_monitoring_analysis_content(sub_module: Optional[str], nav_manager: Any) -> Dict[str, Any]:
    """
    渲染监测分析内容

    Args:
        sub_module: 子模块名称
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 渲染结果
    """
    if sub_module == '工业':
        # 调用实际的工业分析模块
        from dashboard.analysis.industrial import render_industrial_analysis
        render_industrial_analysis(st)
    else:
        st.info("请选择一个子模块以开始监测分析")

    return {'status': 'success', 'content_type': 'monitoring_analysis', 'sub_module': sub_module}


def render_model_analysis_content(sub_module: Optional[str], nav_manager: Any) -> Dict[str, Any]:
    """
    渲染模型分析内容

    Args:
        sub_module: 子模块名称
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 渲染结果
    """
    try:
        # 如果选择了DFM模型，显示DFM功能的tab界面
        if sub_module == "DFM 模型":
            # 状态同步已在dashboard.py主流程中完成，这里不再重复设置

            # 创建DFM功能标签页（移除Tab内容中的状态设置）
            tab1, tab2, tab3, tab4 = st.tabs(["数据准备", "模型训练", "模型分析", "新闻分析"])

            with tab1:
                from dashboard.ui.pages.dfm import render_dfm_data_prep_page
                render_dfm_data_prep_page(st)

            with tab2:
                from dashboard.ui.pages.dfm import render_dfm_model_training_page
                render_dfm_model_training_page(st)

            with tab3:
                from dashboard.ui.pages.dfm import render_dfm_model_analysis_page
                render_dfm_model_analysis_page(st)

            with tab4:
                from dashboard.ui.pages.dfm import render_dfm_news_analysis_page
                render_dfm_news_analysis_page(st)
        else:
            st.info("请选择一个模型分析子模块以开始分析")

        return {'status': 'success', 'content_type': 'model_analysis', 'sub_module': sub_module}
    except Exception as e:
        st.error(f"加载DFM模块时出错: {str(e)}")
        return {'status': 'error', 'content_type': 'model_analysis', 'sub_module': sub_module, 'error': str(e)}


def render_data_exploration_content(sub_module: Optional[str], nav_manager: Any) -> Dict[str, Any]:
    """
    渲染数据探索内容

    Args:
        sub_module: 子模块名称（现在数据探索是主模块，此参数未使用）
        nav_manager: 导航管理器

    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 渲染完整的数据探索界面
    # 状态同步已在dashboard.py主流程中完成，这里不再重复设置

    # 渲染侧边栏数据上传
    sidebar = DataExplorationSidebar()
    uploaded_data = sidebar.render(st)

    # 创建分析标签页（相关分析标签页包含DTW分析）
    tab1, tab2, tab3 = st.tabs(["平稳性分析", "相关分析", "领先滞后分析"])

    with tab1:
        stationarity_component = StationarityAnalysisComponent()
        stationarity_component.render(st, tab_index=0)

    with tab2:
        try:
            unified_correlation_component = UnifiedCorrelationAnalysisComponent()
            unified_correlation_component.render(st, tab_index=1)
        except Exception as e:
            st.error(f"统一相关分析组件加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab3:
        try:
            lead_lag_component = LeadLagAnalysisComponent()
            lead_lag_component.render(st, tab_index=2)
        except Exception as e:
            st.error(f"领先滞后分析组件加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())

    return {'status': 'success', 'content_type': 'data_exploration', 'sub_module': None}


@contextmanager
def create_content_container():
    """
    创建内容容器的上下文管理器
    
    Yields:
        内容容器上下文
    """
    with st.container():
        yield st.container()


def get_module_icon(main_module: str) -> str:
    """
    获取模块图标
    
    Args:
        main_module: 主模块名称
        
    Returns:
        str: 模块图标
    """
    icons = {
        '数据预览': '📊',
        '监测分析': '📈',
        '模型分析': '🤖',
        '数据探索': '🔍'
    }
    return icons.get(main_module, 'ℹ️')


def get_module_description(main_module: str, sub_module: Optional[str] = None) -> str:
    """
    获取模块描述
    
    Args:
        main_module: 主模块名称
        sub_module: 子模块名称
        
    Returns:
        str: 模块描述
    """
    descriptions = {
        '数据预览': '查看和预览工业领域的经济数据',
        '监测分析': '对经济运行数据进行深度监测和分析，提供专业的分析报告',
        '模型分析': '使用先进的数学模型对经济数据进行建模和预测分析',
        '数据探索': '深入探索时间序列数据的统计特性和内在规律，包括平稳性分析和相关性分析'
    }
    
    base_desc = descriptions.get(main_module, '经济数据分析功能')
    
    if sub_module:
        return f"{base_desc} - {sub_module}"
    else:
        return base_desc


def validate_content_config(config: Optional[Dict[str, Any]]) -> bool:
    """
    验证内容配置的有效性
    
    Args:
        config: 内容配置
        
    Returns:
        bool: 配置是否有效
    """
    if not config or not isinstance(config, dict):
        return False
    
    required_fields = ['title', 'icon', 'description', 'content_type']
    for field in required_fields:
        if field not in config:
            return False
    
    return True


def _clear_previous_module_state(current_main_module: str) -> None:
    """
    清理之前模块的状态残留

    Args:
        current_main_module: 当前主模块名称
    """
    try:
        # 使用统一状态管理器进行清理
        import sys
        import os
        
        # 添加项目根目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 定义需要清理的状态键模式
        state_patterns_to_clear = [
            'temp_selected_',
            'navigate_to_',
            '_preview_data',
            '_processed_data',
            '_analysis_result'
        ]

        # 模块特定的状态清理
        module_specific_states = {
            '数据预览': ['monitoring_', 'model_', 'tools_'],
            '监测分析': ['preview_', 'model_', 'tools_'],
            '模型分析': ['preview_', 'monitoring_', 'tools_'],
            '数据探索': ['preview_', 'monitoring_', 'model_']
        }

        # 获取需要清理的模块前缀
        prefixes_to_clear = module_specific_states.get(current_main_module, [])

        # 获取所有状态键
        all_keys = list(st.session_state.keys())

        # 清理状态
        keys_to_remove = []
        for key in all_keys:
            key_str = str(key)

            # 清理通用状态模式
            for pattern in state_patterns_to_clear:
                if pattern in key_str:
                    keys_to_remove.append(key)
                    break

            # 清理模块特定状态
            for prefix in prefixes_to_clear:
                if key_str.startswith(prefix):
                    keys_to_remove.append(key)
                    break

        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]

        logger.debug(f"清理了 {len(keys_to_remove)} 个状态键")

    except Exception as e:
        logger.error(f"状态清理失败: {e}")


def detect_navigation_level(main_module: str, sub_module: Optional[str], nav_manager: Any) -> str:
    """
    检测当前导航层次

    Args:
        main_module: 主模块名称
        sub_module: 子模块名称
        nav_manager: 导航管理器

    Returns:
        str: 导航层次 ('MAIN_MODULE_ONLY', 'SUB_MODULE_ONLY', 'FUNCTION_ACTIVE')
    """
    try:
        # 对于数据预览模块，直接显示内容（无子模块）
        if main_module == '数据预览':
            return 'FUNCTION_ACTIVE'

        # 对于数据探索模块，直接显示内容（无子模块）
        if main_module == '数据探索':
            return 'FUNCTION_ACTIVE'

        # 对于用户管理模块，直接显示内容（管理员专属，无子模块）
        if main_module == '用户管理':
            return 'FUNCTION_ACTIVE'

        # 如果没有子模块，说明只选择了主模块
        if not sub_module:
            return 'MAIN_MODULE_ONLY'

        # 对于监测分析模块，如果已选择子模块，直接进入功能层
        if main_module == '监测分析' and sub_module == '工业':
            return 'FUNCTION_ACTIVE'

        return 'SUB_MODULE_ONLY'

    except Exception as e:
        logger.error(f"导航层次检测失败: {e}")
        return 'SUB_MODULE_ONLY'


def render_module_selection_guide(main_module: str, guide_type: str, sub_module: Optional[str] = None) -> Dict[str, Any]:
    """
    渲染模块选择指导界面

    Args:
        main_module: 主模块名称
        guide_type: 指导类型 ('sub_module' 或 'function')
        sub_module: 子模块名称（当guide_type为'function'时需要）

    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 所有主模块都显示统一的欢迎页面样式
    if guide_type == 'sub_module':
        # 对于数据预览模块，使用专门的欢迎页面
        if main_module == '数据预览':
            from dashboard.ui.pages.main_modules.data_preview import DataPreviewWelcomePage
            welcome_page = DataPreviewWelcomePage()
        else:
            # 对于其他模块，使用通用欢迎页面
            from dashboard.ui.pages.main_modules.data_preview import UniversalWelcomePage
            welcome_page = UniversalWelcomePage(main_module)

        welcome_page.render(st)
        return {
            'status': 'success',
            'content_type': f'{main_module}_welcome',
            'guide_type': guide_type,
            'main_module': main_module,
            'sub_module': sub_module
        }
    else:
        st.markdown("")

    return {
        'status': 'success',
        'content_type': 'selection_guide',
        'guide_type': guide_type,
        'main_module': main_module,
        'sub_module': sub_module
    }


def render_platform_header() -> None:
    """
    渲染平台标题头部 - 经济运行分析平台标题、分割线、机构信息
    """
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">经济运行分析平台</h1>
        <hr class="platform-divider">
        <p class="platform-subtitle">国家信息中心</p>
    </div>
    """, unsafe_allow_html=True)


def render_welcome_page(main_module: str, sub_module: Optional[str] = None) -> Dict[str, Any]:
    """
    渲染欢迎页面 - 只在首页显示平台标题和机构信息

    Args:
        main_module: 主模块名称
        sub_module: 子模块名称

    Returns:
        Dict[str, Any]: 渲染结果
    """
    # 只在欢迎页面显示平台标题和机构信息
    render_platform_header()

    return {
        'status': 'success',
        'content_type': 'welcome',
        'main_module': main_module,
        'sub_module': sub_module
    }


def render_user_management_content(sub_module: Optional[str], nav_manager: Any) -> Dict[str, Any]:
    """
    渲染用户管理内容
    
    Args:
        sub_module: 子模块名称
        nav_manager: 导航管理器
        
    Returns:
        Dict[str, Any]: 渲染结果
    """
    try:
        # 如果有子模块，渲染对应的子模块内容
        if sub_module:
            result = render_user_management_sub_module(sub_module)
            if result == "success":
                return {
                    'status': 'success',
                    'content_type': 'user_management',
                    'main_module': '用户管理',
                    'sub_module': sub_module
                }
            else:
                return {
                    'status': 'error',
                    'content_type': 'user_management',
                    'main_module': '用户管理',
                    'sub_module': sub_module,
                    'message': result
                }
        else:
            # 没有子模块，显示用户管理主页
            UserManagementWelcomePage.render()
            return {
                'status': 'success',
                'content_type': 'user_management',
                'main_module': '用户管理',
                'sub_module': None
            }
            
    except Exception as e:
        st.error(f"用户管理模块渲染失败: {e}")
        logger.error(f"用户管理模块渲染失败: {e}")
        return {
            'status': 'error',
            'content_type': 'user_management',
            'main_module': '用户管理',
            'sub_module': sub_module,
            'message': str(e)
        }


__all__ = [
    'render_main_content', 'get_content_config', 'render_content_header',
    'route_to_content', 'render_data_preview_content', 'render_monitoring_analysis_content',
    'render_model_analysis_content', 'render_data_exploration_content', 'render_user_management_content',
    'create_content_container', 'get_module_icon', 'get_module_description',
    'validate_content_config', 'detect_navigation_level',
    'render_module_selection_guide', 'render_welcome_page', 'render_platform_header',
    'check_user_permission', 'render_permission_denied'
]
