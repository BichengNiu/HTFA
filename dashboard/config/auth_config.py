"""
认证系统配置模块

通过环境变量控制调试模式和认证行为
"""
import os


class AuthConfig:
    """认证系统配置类"""

    # 调试模式开关 - 从环境变量读取，默认为true
    # 设置环境变量 HTFA_DEBUG_MODE=false 启用完整认证
    DEBUG_MODE = os.getenv('HTFA_DEBUG_MODE', 'true').lower() == 'true'

    # 调试模式下的默认权限（所有模块）
    DEBUG_DEFAULT_PERMISSIONS = [
        'data_preview',
        'monitoring_analysis',
        'model_analysis',
        'data_exploration',
        'user_management'
    ]

    # 调试模式下可访问的所有模块
    DEBUG_ACCESSIBLE_MODULES = [
        '数据预览',
        '监测分析',
        '模型分析',
        '数据探索',
        '用户管理'
    ]

    # 会话配置
    SESSION_DURATION_HOURS = 8  # 默认会话时长
    REMEMBER_ME_DURATION_HOURS = 24  # 记住登录时长

    # 账户安全配置
    MAX_FAILED_LOGIN_ATTEMPTS = 5  # 最大登录失败次数
    ACCOUNT_LOCK_DURATION_MINUTES = 30  # 账户锁定时长

    @classmethod
    def is_debug_mode(cls) -> bool:
        """检查是否处于调试模式"""
        return cls.DEBUG_MODE

    @classmethod
    def get_mode_name(cls) -> str:
        """获取当前模式名称"""
        return "调试模式" if cls.DEBUG_MODE else "正常模式"

    @classmethod
    def reload(cls):
        """重新加载配置（从环境变量读取）"""
        cls.DEBUG_MODE = os.getenv('HTFA_DEBUG_MODE', 'true').lower() == 'true'


# 全局配置实例
auth_config = AuthConfig()
