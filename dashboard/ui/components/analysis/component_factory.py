# -*- coding: utf-8 -*-
"""
分析组件工厂
解决导入冲突问题，提供安全的组件获取机制
"""

from typing import Optional, Type, Dict, Any
import logging
import importlib
import importlib.util

logger = logging.getLogger(__name__)


class AnalysisComponentFactory:
    """分析组件工厂类"""
    
    def __init__(self):
        self._component_registry = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件注册表"""
        try:
            # 注册图表组件
            from dashboard.ui.components.analysis.charts import TimeSeriesChartComponent, EnterpriseIndicatorsChartComponent
            self._component_registry['TimeSeriesChartComponent'] = TimeSeriesChartComponent
            self._component_registry['EnterpriseIndicatorsChartComponent'] = EnterpriseIndicatorsChartComponent

            # 注册可视化组件
            from dashboard.ui.components.analysis.visualization import VisualizationComponent
            self._component_registry['VisualizationComponent'] = VisualizationComponent

            # 注册基础分析组件
            from dashboard.ui.components.analysis.industrial.base_analysis_component import BaseAnalysisComponent
            self._component_registry['BaseAnalysisComponent'] = BaseAnalysisComponent
            
            logger.info("分析组件工厂初始化成功")
            
        except Exception as e:
            logger.error(f"分析组件工厂初始化失败: {e}")
            raise
    
    def _load_industrial_components(self):
        """动态加载工业分析组件"""
        try:
            # 使用绝对导入路径
            import sys
            import os

            # 添加项目根目录到sys.path
            current_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # 获取industrial.py的完整路径
            industrial_file = os.path.join(current_dir, 'industrial.py')

            if os.path.exists(industrial_file):
                # 使用importlib动态导入，使用相对模块名
                module_name = "ui.components.analysis.industrial_components"
                spec = importlib.util.spec_from_file_location(module_name, industrial_file)
                industrial_module = importlib.util.module_from_spec(spec)

                # 将模块添加到sys.modules中
                sys.modules[module_name] = industrial_module
                spec.loader.exec_module(industrial_module)

                # 注册工业分析组件
                components_to_register = [
                    'IndustrialFileUploadComponent',
                    'IndustrialTimeRangeSelectorComponent',
                    'IndustrialGroupDetailsComponent',
                    'IndustrialWelcomeComponent'
                ]

                registered_count = 0
                for component_name in components_to_register:
                    if hasattr(industrial_module, component_name):
                        self._component_registry[component_name] = getattr(industrial_module, component_name)
                        registered_count += 1
                        logger.debug(f"注册组件: {component_name}")

                logger.info(f"工业分析组件加载成功，注册了 {registered_count} 个组件")
                return registered_count > 0
            else:
                logger.warning("industrial.py文件不存在")
                return False

        except Exception as e:
            logger.error(f"加载工业分析组件失败: {e}")
            return False
    
    def get_component(self, component_name: str) -> Optional[Type]:
        """
        获取组件类
        
        Args:
            component_name: 组件名称
            
        Returns:
            组件类或None
        """
        # 首先尝试从已注册的组件获取
        if component_name in self._component_registry:
            return self._component_registry[component_name]
        
        # 如果是工业分析组件，尝试动态加载
        if component_name.startswith('Industrial'):
            if self._load_industrial_components():
                return self._component_registry.get(component_name)

        logger.warning(f"组件未找到: {component_name}")
        return None
    
    def get_available_components(self) -> Dict[str, Type]:
        """获取所有可用组件"""
        # 确保工业组件已加载
        self._load_industrial_components()
        return self._component_registry.copy()
    
    def register_component(self, component_name: str, component_class: Type):
        """
        注册新组件
        
        Args:
            component_name: 组件名称
            component_class: 组件类
        """
        self._component_registry[component_name] = component_class
        logger.info(f"组件已注册: {component_name}")


# 全局组件工厂实例
_component_factory = None


def get_component_factory() -> AnalysisComponentFactory:
    """获取组件工厂实例"""
    global _component_factory
    if _component_factory is None:
        _component_factory = AnalysisComponentFactory()
    return _component_factory


def get_analysis_component(component_name: str) -> Optional[Type]:
    """
    获取分析组件的便捷函数
    
    Args:
        component_name: 组件名称
        
    Returns:
        组件类或None
    """
    factory = get_component_factory()
    return factory.get_component(component_name)


# 为了向后兼容，提供直接导入接口
def get_industrial_file_upload_component():
    """获取工业文件上传组件"""
    return get_analysis_component('IndustrialFileUploadComponent')


def get_industrial_time_range_selector_component():
    """获取工业时间范围选择器组件"""
    return get_analysis_component('IndustrialTimeRangeSelectorComponent')


def get_industrial_group_details_component():
    """获取工业分组详情组件"""
    return get_analysis_component('IndustrialGroupDetailsComponent')


def get_industrial_welcome_component():
    """获取工业欢迎页面组件"""
    return get_analysis_component('IndustrialWelcomeComponent')


def get_time_series_chart_component():
    """获取时间序列图表组件"""
    return get_analysis_component('TimeSeriesChartComponent')


def get_enterprise_indicators_chart_component():
    """获取企业指标图表组件"""
    return get_analysis_component('EnterpriseIndicatorsChartComponent')
