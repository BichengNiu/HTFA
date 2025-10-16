"""
主题样式集成组件
提供工业数据预览模块的主题样式集成功能
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import logging
import colorsys
import re

logger = logging.getLogger(__name__)


class IndustrialPreviewThemeManager:
    """
    工业数据预览主题管理器
    负责管理工业数据预览模块的主题样式
    """
    
    def __init__(self):
        """初始化主题管理器"""
        self.theme_name = "industrial_preview"
        self.style_constants = StyleConstants()

        # 主题配置
        self.theme_config = self._create_theme_config()
        self.color_scheme = self._create_color_scheme()
        self.typography = self._create_typography_config()

        logger.info(f"初始化工业数据预览主题管理器")
    
    def _create_theme_config(self) -> Dict[str, Any]:
        """创建主题配置"""
        return {
            'name': self.theme_name,
            'version': '1.0.0',
            'description': '工业数据预览模块主题',
            'colors': self.style_constants.get_color_palette(),
            'typography': self.style_constants.get_typography_scale(),
            'spacing': self.style_constants.get_spacing_scale(),
            'components': {
                'chart': {
                    'background': '#ffffff',
                    'grid_color': '#e5e7eb',
                    'text_color': '#374151'
                },
                'table': {
                    'header_bg': '#f9fafb',
                    'border_color': '#d1d5db',
                    'hover_bg': '#f3f4f6'
                },
                'button': {
                    'primary_bg': '#3b82f6',
                    'primary_text': '#ffffff',
                    'secondary_bg': '#e5e7eb',
                    'secondary_text': '#374151'
                }
            }
        }
    
    def _create_color_scheme(self) -> Dict[str, str]:
        """创建颜色方案"""
        return {
            'primary': '#3b82f6',      # 蓝色
            'secondary': '#6b7280',    # 灰色
            'success': '#10b981',      # 绿色
            'warning': '#f59e0b',      # 橙色
            'error': '#ef4444',        # 红色
            'background': '#ffffff',   # 白色背景
            'surface': '#f9fafb',      # 浅灰表面
            'text': '#111827',         # 深色文本
            'text_secondary': '#6b7280', # 次要文本
            'border': '#d1d5db'        # 边框色
        }
    
    def _create_typography_config(self) -> Dict[str, Any]:
        """创建字体配置"""
        return {
            'font_family': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'font_sizes': {
                'xs': '12px',
                'sm': '14px',
                'base': '16px',
                'lg': '18px',
                'xl': '20px',
                '2xl': '24px',
                '3xl': '30px'
            },
            'font_weights': {
                'normal': '400',
                'medium': '500',
                'semibold': '600',
                'bold': '700'
            },
            'line_heights': {
                'tight': '1.25',
                'normal': '1.5',
                'relaxed': '1.75'
            }
        }
    
    def get_theme_config(self) -> Dict[str, Any]:
        """
        获取主题配置
        
        Returns:
            主题配置字典
        """
        return self.theme_config.copy()
    
    def get_color_scheme(self) -> Dict[str, str]:
        """
        获取颜色方案
        
        Returns:
            颜色方案字典
        """
        return self.color_scheme.copy()
    
    def get_typography_config(self) -> Dict[str, Any]:
        """
        获取字体配置
        
        Returns:
            字体配置字典
        """
        return self.typography.copy()
    
    def apply_theme_to_component(self, component_type: str, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用主题到组件
        
        Args:
            component_type: 组件类型
            component_config: 组件配置
            
        Returns:
            应用主题后的组件配置
        """
        try:
            # 获取组件样式配置
            component_styles = self.theme_config['components'].get(component_type, {})
            
            # 应用样式
            styled_config = component_config.copy()
            styled_config['style'] = {
                **component_styles,
                **styled_config.get('style', {})
            }
            
            return styled_config
            
        except Exception as e:
            logger.error(f"应用主题到组件失败: {e}")
            return component_config
    
    def generate_css_styles(self) -> str:
        """
        生成CSS样式
        
        Returns:
            CSS样式字符串
        """
        try:
            css_parts = []
            
            # 根样式变量
            css_parts.append(":root {")
            for key, value in self.color_scheme.items():
                css_parts.append(f"  --color-{key.replace('_', '-')}: {value};")
            css_parts.append("}")
            
            # 组件样式
            for component_type, styles in self.theme_config['components'].items():
                css_parts.append(f".industrial-preview-{component_type} {{")
                for prop, value in styles.items():
                    css_prop = prop.replace('_', '-')
                    css_parts.append(f"  {css_prop}: {value};")
                css_parts.append("}")
            
            return "\n".join(css_parts)
            
        except Exception as e:
            logger.error(f"生成CSS样式失败: {e}")
            return ""
    
    def update_theme_config(self, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新主题配置
        
        Args:
            custom_config: 自定义配置
            
        Returns:
            更新结果
        """
        try:
            # 深度合并配置
            self._deep_merge(self.theme_config, custom_config)
            
            # 更新颜色方案
            if 'colors' in custom_config:
                self.color_scheme.update(custom_config['colors'])
            
            # 更新字体配置
            if 'typography' in custom_config:
                self._deep_merge(self.typography, custom_config['typography'])
            
            return {
                'success': True,
                'updated_keys': list(custom_config.keys())
            }
            
        except Exception as e:
            logger.error(f"更新主题配置失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


class StyleConstants:
    """
    样式常量类
    定义所有样式相关的常量
    """
    
    def __init__(self):
        """初始化样式常量"""
        self.COLORS = self._define_colors()
        self.TYPOGRAPHY = self._define_typography()
        self.SPACING = self._define_spacing()
        self.BREAKPOINTS = self._define_breakpoints()
    
    def _define_colors(self) -> Dict[str, str]:
        """定义颜色常量"""
        return {
            'primary': '#3b82f6',
            'primary_light': '#60a5fa',
            'primary_dark': '#1d4ed8',
            'secondary': '#6b7280',
            'secondary_light': '#9ca3af',
            'secondary_dark': '#374151',
            'success': '#10b981',
            'success_light': '#34d399',
            'success_dark': '#047857',
            'warning': '#f59e0b',
            'warning_light': '#fbbf24',
            'warning_dark': '#d97706',
            'error': '#ef4444',
            'error_light': '#f87171',
            'error_dark': '#dc2626',
            'background': '#ffffff',
            'surface': '#f9fafb',
            'surface_dark': '#f3f4f6',
            'text': '#111827',
            'text_secondary': '#6b7280',
            'text_light': '#9ca3af',
            'border': '#d1d5db',
            'border_light': '#e5e7eb'
        }
    
    def _define_typography(self) -> Dict[str, Any]:
        """定义字体常量"""
        return {
            'font_family': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'font_sizes': {
                'xs': '12px',
                'sm': '14px',
                'base': '16px',
                'lg': '18px',
                'xl': '20px',
                '2xl': '24px',
                '3xl': '30px',
                '4xl': '36px'
            },
            'font_weights': {
                'light': '300',
                'normal': '400',
                'medium': '500',
                'semibold': '600',
                'bold': '700',
                'extrabold': '800'
            }
        }
    
    def _define_spacing(self) -> Dict[str, str]:
        """定义间距常量"""
        return {
            'xs': '4px',
            'sm': '8px',
            'md': '16px',
            'lg': '24px',
            'xl': '32px',
            '2xl': '48px',
            '3xl': '64px'
        }
    
    def _define_breakpoints(self) -> Dict[str, str]:
        """定义断点常量"""
        return {
            'mobile': '640px',
            'tablet': '768px',
            'desktop': '1024px',
            'wide': '1280px'
        }
    
    def get_color_palette(self) -> Dict[str, str]:
        """获取颜色调色板"""
        return self.COLORS.copy()
    
    def get_typography_scale(self) -> Dict[str, Any]:
        """获取字体比例"""
        return self.TYPOGRAPHY.copy()
    
    def get_spacing_scale(self) -> Dict[str, str]:
        """获取间距比例"""
        return self.SPACING.copy()
    
    def get_breakpoints(self) -> Dict[str, str]:
        """获取断点"""
        return self.BREAKPOINTS.copy()


class ComponentStyler:
    """
    组件样式器
    负责为不同组件应用样式
    """
    
    def __init__(self, theme_manager=None):
        """初始化组件样式器"""
        self.theme_manager = theme_manager
        self.style_registry = self._create_style_registry()
    
    def _create_style_registry(self) -> Dict[str, callable]:
        """创建样式注册表"""
        return {
            'chart': self._style_chart,
            'table': self._style_table,
            'button': self._style_button,
            'card': self._style_card,
            'input': self._style_input
        }
    
    def style_chart_component(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        样式化图表组件
        
        Args:
            chart_config: 图表配置
            
        Returns:
            样式化后的图表配置
        """
        return self._style_chart(chart_config)
    
    def style_table_component(self, table_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        样式化表格组件
        
        Args:
            table_config: 表格配置
            
        Returns:
            样式化后的表格配置
        """
        return self._style_table(table_config)
    
    def style_button_component(self, button_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        样式化按钮组件
        
        Args:
            button_config: 按钮配置
            
        Returns:
            样式化后的按钮配置
        """
        return self._style_button(button_config)
    
    def register_custom_style(self, component_type: str, styler_func: callable) -> bool:
        """
        注册自定义样式
        
        Args:
            component_type: 组件类型
            styler_func: 样式函数
            
        Returns:
            是否注册成功
        """
        try:
            self.style_registry[component_type] = styler_func
            return True
        except Exception as e:
            logger.error(f"注册自定义样式失败: {e}")
            return False
    
    def apply_global_styles(self) -> str:
        """
        应用全局样式

        Returns:
            全局CSS样式字符串
        """
        if self.theme_manager:
            return self.theme_manager.generate_css_styles()
        else:
            # 创建临时主题管理器
            temp_manager = IndustrialPreviewThemeManager()
            return temp_manager.generate_css_styles()
    
    def _style_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """样式化图表"""
        if self.theme_manager:
            colors = self.theme_manager.get_color_scheme()
        else:
            # 使用默认颜色
            colors = StyleConstants().get_color_palette()
        
        styled_config = config.copy()
        styled_config['layout'] = {
            'paper_bgcolor': colors['background'],
            'plot_bgcolor': colors['background'],
            'font': {'color': colors['text']},
            **styled_config.get('layout', {})
        }
        
        return styled_config
    
    def _style_table(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """样式化表格"""
        if self.theme_manager:
            colors = self.theme_manager.get_color_scheme()
        else:
            colors = StyleConstants().get_color_palette()
        
        styled_config = config.copy()
        styled_config['style'] = {
            'backgroundColor': colors['background'],
            'color': colors['text'],
            'borderColor': colors['border'],
            **styled_config.get('style', {})
        }
        
        return styled_config
    
    def _style_button(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """样式化按钮"""
        if self.theme_manager:
            colors = self.theme_manager.get_color_scheme()
        else:
            colors = StyleConstants().get_color_palette()
        button_type = config.get('type', 'primary')
        
        styled_config = config.copy()
        
        if button_type == 'primary':
            styled_config['style'] = {
                'backgroundColor': colors['primary'],
                'color': colors['background'],
                'border': 'none',
                **styled_config.get('style', {})
            }
        else:
            styled_config['style'] = {
                'backgroundColor': colors['surface'],
                'color': colors['text'],
                'border': f"1px solid {colors['border']}",
                **styled_config.get('style', {})
            }
        
        return styled_config
    
    def _style_card(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """样式化卡片"""
        if self.theme_manager:
            colors = self.theme_manager.get_color_scheme()
        else:
            colors = StyleConstants().get_color_palette()
        
        styled_config = config.copy()
        styled_config['style'] = {
            'backgroundColor': colors['background'],
            'border': f"1px solid {colors['border']}",
            'borderRadius': '8px',
            'padding': '16px',
            **styled_config.get('style', {})
        }
        
        return styled_config
    
    def _style_input(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """样式化输入框"""
        if self.theme_manager:
            colors = self.theme_manager.get_color_scheme()
        else:
            colors = StyleConstants().get_color_palette()
        
        styled_config = config.copy()
        styled_config['style'] = {
            'backgroundColor': colors['background'],
            'border': f"1px solid {colors['border']}",
            'borderRadius': '4px',
            'color': colors['text'],
            **styled_config.get('style', {})
        }
        
        return styled_config


class ResponsiveDesignManager:
    """
    响应式设计管理器
    负责管理响应式设计和布局适配
    """

    def __init__(self):
        """初始化响应式设计管理器"""
        self.style_constants = StyleConstants()
        self.breakpoints = self.style_constants.get_breakpoints()
        self.grid_system = self._create_grid_system()
        self.responsive_rules = self._create_responsive_rules()

    def _create_grid_system(self) -> Dict[str, Any]:
        """创建网格系统"""
        return {
            'columns': 12,
            'gutter': '16px',
            'container_max_width': {
                'mobile': '100%',
                'tablet': '768px',
                'desktop': '1024px',
                'wide': '1280px'
            }
        }

    def _create_responsive_rules(self) -> Dict[str, Dict[str, Any]]:
        """创建响应式规则"""
        return {
            'chart': {
                'mobile': {
                    'height': '300px',
                    'margin': '8px 0'
                },
                'tablet': {
                    'height': '400px',
                    'margin': '16px 0'
                },
                'desktop': {
                    'height': '500px',
                    'margin': '24px 0'
                }
            },
            'table': {
                'mobile': {
                    'font_size': '12px',
                    'scroll': 'horizontal'
                },
                'tablet': {
                    'font_size': '14px',
                    'scroll': 'auto'
                },
                'desktop': {
                    'font_size': '16px',
                    'scroll': 'auto'
                }
            }
        }

    def get_responsive_config(self, component_type: str) -> Dict[str, Any]:
        """
        获取响应式配置

        Args:
            component_type: 组件类型

        Returns:
            响应式配置
        """
        return self.responsive_rules.get(component_type, {})

    def generate_responsive_css(self) -> str:
        """
        生成响应式CSS

        Returns:
            响应式CSS字符串
        """
        css_parts = []

        # 基础样式
        css_parts.append(".industrial-preview-container { width: 100%; }")

        # 响应式媒体查询
        for breakpoint, width in self.breakpoints.items():
            css_parts.append(f"@media (min-width: {width}) {{")
            css_parts.append(f"  .industrial-preview-{breakpoint} {{ display: block; }}")
            css_parts.append("}")

        return "\n".join(css_parts)

    def get_grid_config(self) -> Dict[str, Any]:
        """
        获取网格配置

        Returns:
            网格配置
        """
        return self.grid_system.copy()

    def adapt_component_for_screen_size(
        self,
        component_config: Dict[str, Any],
        screen_size: str
    ) -> Dict[str, Any]:
        """
        为屏幕尺寸适配组件

        Args:
            component_config: 组件配置
            screen_size: 屏幕尺寸

        Returns:
            适配后的组件配置
        """
        adapted_config = component_config.copy()

        # 根据屏幕尺寸调整
        if screen_size == 'mobile':
            adapted_config['width'] = min(adapted_config.get('width', 800), 400)
            adapted_config['height'] = min(adapted_config.get('height', 600), 300)
        elif screen_size == 'tablet':
            adapted_config['width'] = min(adapted_config.get('width', 800), 600)
            adapted_config['height'] = min(adapted_config.get('height', 600), 400)

        return adapted_config


class AccessibilityManager:
    """
    可访问性管理器
    负责管理可访问性功能和合规性
    """

    def __init__(self):
        """初始化可访问性管理器"""
        self.a11y_rules = self._create_accessibility_rules()
        self.color_contrast_checker = ColorContrastChecker()
        self.keyboard_navigation = self._create_keyboard_navigation_rules()

    def _create_accessibility_rules(self) -> Dict[str, Any]:
        """创建可访问性规则"""
        return {
            'color_contrast': {
                'aa_normal': 4.5,
                'aa_large': 3.0,
                'aaa_normal': 7.0,
                'aaa_large': 4.5
            },
            'keyboard_navigation': {
                'tab_order': True,
                'focus_indicators': True,
                'skip_links': True
            },
            'screen_reader': {
                'alt_text': True,
                'aria_labels': True,
                'semantic_markup': True
            }
        }

    def _create_keyboard_navigation_rules(self) -> Dict[str, Any]:
        """创建键盘导航规则"""
        return {
            'tab_index': {
                'interactive': 0,
                'non_interactive': -1
            },
            'key_bindings': {
                'Enter': 'activate',
                'Space': 'activate',
                'Escape': 'close',
                'ArrowUp': 'previous',
                'ArrowDown': 'next'
            }
        }

    def check_color_contrast(self, foreground: str, background: str) -> Dict[str, Any]:
        """
        检查颜色对比度

        Args:
            foreground: 前景色
            background: 背景色

        Returns:
            对比度检查结果
        """
        return self.color_contrast_checker.check_contrast(foreground, background)

    def generate_aria_attributes(self, component_config: Dict[str, Any]) -> Dict[str, str]:
        """
        生成ARIA属性

        Args:
            component_config: 组件配置

        Returns:
            ARIA属性字典
        """
        aria_attrs = {}

        # 基于组件类型生成ARIA属性
        component_type = component_config.get('type', '')

        if component_type == 'chart':
            aria_attrs['aria-label'] = component_config.get('title', 'Chart')
            if 'description' in component_config:
                aria_attrs['aria-describedby'] = 'chart-description'

        elif component_type == 'table':
            aria_attrs['role'] = 'table'
            aria_attrs['aria-label'] = component_config.get('caption', 'Data table')

        elif component_type == 'button':
            aria_attrs['role'] = 'button'
            aria_attrs['aria-label'] = component_config.get('text', 'Button')

        return aria_attrs

    def add_keyboard_navigation(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加键盘导航

        Args:
            component_config: 组件配置

        Returns:
            增强后的组件配置
        """
        enhanced_config = component_config.copy()

        # 添加tabindex
        if component_config.get('interactive', True):
            enhanced_config['tabindex'] = 0

        # 添加键盘事件处理
        enhanced_config['onKeyDown'] = self._create_keyboard_handler(component_config)

        return enhanced_config

    def validate_accessibility_compliance(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证可访问性合规性

        Args:
            component_config: 组件配置

        Returns:
            合规性验证结果
        """
        issues = []
        recommendations = []

        # 检查颜色对比度
        if 'colors' in component_config and 'background' in component_config:
            for color in component_config['colors']:
                contrast_result = self.check_color_contrast(
                    color,
                    component_config['background']
                )
                if not contrast_result['passes_aa']:
                    issues.append(f"颜色 {color} 与背景对比度不足")
                    recommendations.append("调整颜色以提高对比度")

        # 检查ARIA属性
        if not any(key.startswith('aria-') for key in component_config.keys()):
            issues.append("缺少ARIA属性")
            recommendations.append("添加适当的ARIA标签")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'score': max(0, 100 - len(issues) * 20)
        }

    def get_accessibility_guidelines(self) -> Dict[str, Any]:
        """
        获取可访问性指南

        Returns:
            可访问性指南
        """
        return {
            'color_contrast': {
                'description': '确保足够的颜色对比度',
                'requirements': self.a11y_rules['color_contrast']
            },
            'keyboard_navigation': {
                'description': '支持键盘导航',
                'requirements': self.a11y_rules['keyboard_navigation']
            },
            'screen_reader': {
                'description': '屏幕阅读器兼容',
                'requirements': self.a11y_rules['screen_reader']
            }
        }

    def _create_keyboard_handler(self, component_config: Dict[str, Any]) -> str:
        """创建键盘事件处理器"""
        return """
        function(event) {
            switch(event.key) {
                case 'Enter':
                case ' ':
                    event.preventDefault();
                    // 激活组件
                    break;
                case 'Escape':
                    // 关闭或取消
                    break;
            }
        }
        """


class ColorContrastChecker:
    """
    颜色对比度检查器
    """

    def check_contrast(self, foreground: str, background: str) -> Dict[str, Any]:
        """
        检查颜色对比度

        Args:
            foreground: 前景色
            background: 背景色

        Returns:
            对比度检查结果
        """
        try:
            # 转换颜色为RGB
            fg_rgb = self._hex_to_rgb(foreground)
            bg_rgb = self._hex_to_rgb(background)

            # 计算相对亮度
            fg_luminance = self._calculate_luminance(fg_rgb)
            bg_luminance = self._calculate_luminance(bg_rgb)

            # 计算对比度
            ratio = self._calculate_contrast_ratio(fg_luminance, bg_luminance)

            return {
                'ratio': round(ratio, 2),
                'passes_aa': ratio >= 4.5,
                'passes_aaa': ratio >= 7.0,
                'passes_aa_large': ratio >= 3.0,
                'passes_aaa_large': ratio >= 4.5
            }

        except Exception as e:
            logger.error(f"颜色对比度检查失败: {e}")
            return {
                'ratio': 0,
                'passes_aa': False,
                'passes_aaa': False,
                'passes_aa_large': False,
                'passes_aaa_large': False,
                'error': str(e)
            }

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _calculate_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """计算相对亮度"""
        def normalize_channel(channel):
            channel = channel / 255.0
            if channel <= 0.03928:
                return channel / 12.92
            else:
                return pow((channel + 0.055) / 1.055, 2.4)

        r, g, b = rgb
        r_norm = normalize_channel(r)
        g_norm = normalize_channel(g)
        b_norm = normalize_channel(b)

        return 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm

    def _calculate_contrast_ratio(self, luminance1: float, luminance2: float) -> float:
        """计算对比度比率"""
        lighter = max(luminance1, luminance2)
        darker = min(luminance1, luminance2)
        return (lighter + 0.05) / (darker + 0.05)
