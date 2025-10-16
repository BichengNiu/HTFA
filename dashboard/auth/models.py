# -*- coding: utf-8 -*-
"""
定义用户、会话等数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid
import json

@dataclass
class User:
    """用户数据模型"""
    id: int
    username: str
    password_hash: str
    email: Optional[str] = None
    wechat: Optional[str] = None  # 微信号
    phone: Optional[str] = None  # 手机号
    organization: Optional[str] = None  # 单位名称
    permissions: List[str] = field(default_factory=list)  # 用户直接拥有的权限列表
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    def is_locked(self) -> bool:
        """检查用户是否被锁定"""
        return self.locked_until and datetime.now() < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30):
        """锁定账户"""
        self.locked_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.failed_login_attempts = 0
    
    def unlock_account(self):
        """解锁账户"""
        self.locked_until = None
        self.failed_login_attempts = 0


@dataclass
class UserSession:
    """用户会话数据模型"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=8))
    last_accessed: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at
    
    def extend_session(self, hours: int = 8):
        """延长会话时间"""
        now = datetime.now()
        self.expires_at = now + timedelta(hours=hours)
        self.last_accessed = now



# 权限模块映射
PERMISSION_MODULE_MAP = {
    "数据预览": ["data_preview"],
    "监测分析": ["monitoring_analysis"], 
    "模型分析": ["model_analysis"],
    "数据探索": ["data_exploration"],
    "用户管理": ["user_management"]
}
