# -*- coding: utf-8 -*-
"""
资源加载模块
"""

from dashboard.core.backend.resource.loader import (
    ResourceLoader,
    get_resource_loader
)
from dashboard.core.backend.resource.lazy_loader import (
    LazyModuleLoader,
    get_lazy_loader,
    get_cached_lazy_loader
)
from dashboard.core.backend.resource.component_loader import (
    ComponentLoader,
    get_component_loader,
    lazy_load_component,
    preload_components_async
)

__all__ = [
    'ResourceLoader',
    'get_resource_loader',
    'LazyModuleLoader',
    'get_lazy_loader',
    'get_cached_lazy_loader',
    'ComponentLoader',
    'get_component_loader',
    'lazy_load_component',
    'preload_components_async'
]
