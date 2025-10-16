# CLAUDE.md

这个文件为 Claude Code (claude.ai/code) 在本代码库中工作时提供指导。

## 项目概述

经济运行分析平台 - 基于Streamlit的数据分析Dashboard系统，主要用于工业经济数据的监测、分析和建模。

## 核心架构

### 1. 统一状态管理系统

项目采用统一状态管理器作为核心架构，**严禁使用其他状态管理方式**。

- **核心组件**: `dashboard/core/unified_state.py` - UnifiedStateManager类
- **获取实例**: 使用 `from dashboard.core import get_unified_manager`
- **导航管理**: `dashboard/core/navigation_manager.py` - NavigationManager类，必须通过统一状态管理器初始化
- **状态管理器**: 管理器遵循单例模式，使用线程锁确保线程安全

**关键规则**:
- 所有涉及状态管理的功能必须使用统一状态管理模块
- 不要使用 `st.session_state` 直接操作状态
- 通过 `state_manager.get_state()` 和 `state_manager.set_state()` 访问状态
- 导航状态必须通过 NavigationManager 管理

### 2. 模块化UI架构

- **主入口**: `dashboard/app.py`
- **UI组件**: `dashboard/ui/components/`
  - `sidebar.py` - 侧边栏导航组件
  - `content_router.py` - 主内容路由器
  - `auth/` - 认证相关组件
  - `analysis/` - 分析功能组件
  - `dfm/` - DFM模型相关UI组件
- **页面路由**: `dashboard/ui/pages/`

### 3. 功能模块

系统包含以下主要模块：

1. **数据预览** (`dashboard/preview/`)
   - 工业数据预览和浏览
   - 数据加载和可视化

2. **监测分析** (`dashboard/analysis/industrial/`)
   - 工业增加值分析
   - 工业企业利润拆解

3. **模型分析** (`dashboard/DFM/`)
   - DFM模型数据准备 (`data_prep/`)
   - 模型训练 (`train_model/`)
   - 模型分析 (`model_analysis/`)
   - 新闻分析 (`news_analysis/`)

4. **数据探索** (`dashboard/explore/`)
   - 平稳性分析
   - 相关性分析
   - DTW动态时间规整

5. **用户管理** (`dashboard/auth/`)
   - 用户认证和授权
   - 权限管理

### 4. DFM模型架构

DFM (动态因子模型) 是项目的核心分析工具：

- **数据准备**: 模块化数据处理管道 (`DFM/data_prep/modules/`)
  - `data_loader.py` - 数据加载
  - `data_cleaner.py` - 数据清洗
  - `data_aligner.py` - 数据对齐
  - `stationarity_processor.py` - 平稳性处理
  - `main_data_processor.py` - 主处理流程

- **模型训练**: 基于卡尔曼滤波的动态因子模型
  - `DynamicFactorModel.py` - 核心模型实现
  - `DiscreteKalmanFilter.py` - 卡尔曼滤波器
  - `dfm_core.py` - 模型核心逻辑

## 开发命令

### 启动应用

```bash
# Windows (PowerShell)
cd "C:\Users\niu\Desktop\工作同步\HTFA"
python -m streamlit run dashboard/app.py --server.headless true

# 或使用环境变量
$env:PYTHONPATH="$PWD"
streamlit run dashboard/app.py --server.headless true
```

### 依赖管理

```bash
# 安装依赖
pip install -r requirements.txt

# 关键依赖包
# - streamlit>=1.50.0 (核心框架)
# - pandas>=2.3.0 (数据处理)
# - numpy>=2.3.0 (科学计算)
# - statsmodels>=0.14.0 (统计分析)
# - scikit-learn>=1.7.0 (机器学习)
```

## 开发规范

### 1. 状态管理规范

```python
# 正确方式
from dashboard.core import get_unified_manager

state_manager = get_unified_manager()
value = state_manager.get_state('key', default_value)
state_manager.set_state('key', value)

# 错误方式 - 严禁使用
st.session_state['key'] = value  # 禁止直接使用
```

### 2. 导航状态管理

```python
from core.navigation_manager import get_navigation_manager

nav_manager = get_navigation_manager(state_manager)
nav_manager.set_current_main_module('模块名称')
nav_manager.set_current_sub_module('子模块名称')
```

### 3. 调试信息

涉及修改状态管理的功能，必须在关键节点添加调试信息：

```python
from dashboard.ui.utils.debug_helpers import (
    debug_log, debug_state_change, debug_navigation, debug_button_click
)

debug_log("操作描述", "INFO")
debug_state_change("描述", old_value, new_value, "原因")
debug_navigation("操作", "详细信息")
```

### 4. 代码清理

功能测试成功后删除：
- 所有生成的测试脚本
- 测试产生的临时文件
- 调试用的print语句（保留使用debug_helpers的调试信息）

### 5. 组件开发

- 新UI组件应放在 `dashboard/ui/components/` 对应目录
- 后端逻辑应放在 `dashboard/` 对应功能模块
- 组件必须通过统一状态管理器管理状态

## 重要约束

1. **不写兼容代码**: 不要写任何兼容、回退的功能或代码
2. **无emoji**: 代码和回复中不要加入任何emoji图标
3. **必须使用统一状态管理**: 所有状态操作必须通过 UnifiedStateManager
4. **添加调试信息**: 涉及BUG修复，需要在关键节点添加调试信息
5. **中文文档**: 所有对话和文档都使用中文，文档使用markdown格式

## 项目结构

```
HTFA/
├── dashboard/
│   ├── app.py                    # 主应用入口
│   ├── core/                     # 核心基础设施
│   │   ├── unified_state.py      # 统一状态管理器
│   │   ├── navigation_manager.py # 导航管理器
│   │   ├── component_loader.py   # 组件加载器
│   │   ├── config_cache.py       # 配置缓存
│   │   ├── app_initializer.py    # 应用初始化器
│   │   └── lazy_loader.py        # 懒加载器
│   ├── ui/                       # UI组件
│   │   ├── components/           # 可复用组件
│   │   ├── pages/                # 页面组件
│   │   └── utils/                # UI工具
│   ├── DFM/                      # DFM模型模块
│   │   ├── data_prep/            # 数据准备
│   │   ├── train_model/          # 模型训练
│   │   ├── model_analysis/       # 模型分析
│   │   └── news_analysis/        # 新闻分析
│   ├── analysis/                 # 分析功能
│   │   └── industrial/           # 工业分析
│   ├── explore/                  # 数据探索
│   ├── preview/                  # 数据预览
│   └── auth/                     # 认证授权
└── requirements.txt              # 依赖清单
```

## 常见问题

### 状态管理器初始化失败

如果遇到 "统一状态管理器不可用" 错误：
1. 检查是否正确导入: `from dashboard.core import get_unified_manager`
2. 确保在正确的目录下运行应用
3. 检查PYTHONPATH是否包含项目根目录

### 导航状态不同步

如果按钮状态与实际模块不一致：
1. 使用 `force_navigation_state_sync()` 强制同步
2. 检查是否直接操作了 session_state (不应该这样做)
3. 确保通过 NavigationManager 管理导航状态

### 模块加载失败

1. 检查相对导入路径是否正确
2. 确保 `__init__.py` 文件存在
3. 验证PYTHONPATH配置
