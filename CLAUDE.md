<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HTFA (High-frequency Time-series Factor Analysis) 是一个基于Streamlit的经济运行分析平台,支持多源数据接入、时间序列分析、动态因子模型(DFM)预测和数据探索等功能。

- **技术栈**: Python 3.11.5 + Streamlit >= 1.50.0
- **代码规模**: 254个Python文件,约75,000行代码
- **主入口**: `dashboard/app.py`
- **运行端口**: 8501

## 常用命令

### 启动应用

```bash
# Windows推荐方式(自动清理缓存和端口)
./start.bat

# 直接启动
python -m streamlit run dashboard/app.py --server.port=8501

# Docker方式
docker-compose up -d
docker-compose logs -f htfa-app
```

### 清理Python缓存

```bash
# Windows
for /r %i in (*.pyc *.pyo) do del /f /q "%i"
for /d /r %i in (__pycache__) do rd /s /q "%i"

# Linux/Mac
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### 端口管理

```bash
# Windows - 查看8501端口占用
netstat -ano | findstr ":8501"

# Windows - 关闭占用进程
taskkill /F /PID <进程ID>

# Linux - 释放端口
fuser -k 8501/tcp
```

## 核心架构

### 分层设计

```
dashboard/
├── app.py              # 主入口(713行)
├── core/               # 核心框架层
│   ├── unified_state.py        # 统一状态管理
│   ├── navigation_manager.py   # 导航管理
│   ├── resource_loader.py      # 资源懒加载
│   └── app_initializer.py      # 应用初始化
├── auth/               # 认证与权限
│   ├── database.py             # SQLite数据库
│   ├── models.py               # 数据模型
│   ├── authentication.py       # 认证逻辑
│   └── permissions.py          # 权限管理
├── ui/                 # UI组件层
│   ├── components/             # 可复用组件
│   ├── pages/                  # 页面模块
│   └── utils/                  # UI工具库
├── DFM/                # 动态因子模型
│   ├── data_prep/              # 数据准备
│   ├── train_ref/              # 重构版训练模块(10,800行)
│   ├── model_analysis/         # 模型分析
│   └── news_analysis/          # 新闻分析
├── preview/            # 数据预览
├── explore/            # 数据探索
└── analysis/           # 监测分析
```

### 统一状态管理系统(CRITICAL)

**所有状态操作必须通过UnifiedStateManager进行,严禁直接使用st.session_state!**

```python
from dashboard.core.unified_state import get_unified_manager

# 获取管理器实例
state_mgr = get_unified_manager()

# 基本操作
state_mgr.set_state("key", value)          # 设置状态
value = state_mgr.get_state("key", default) # 获取状态
state_mgr.delete_state("key")              # 删除状态
state_mgr.has_state("key")                 # 检查存在

# 命名空间操作
state_mgr.set_namespaced("preview", "data", df)
data = state_mgr.get_namespaced("preview", "data")
state_mgr.clear_namespace("preview")

# DFM模块专用API
state_mgr.set_dfm_state("data_prep", "indicators", indicators)
state_mgr.get_dfm_state("data_prep", "indicators", [])
state_mgr.clear_dfm_state("train_model", "results")
```

**特性**:
- 线程安全(RLock保护)
- 单例模式(ThreadSafeSingleton)
- 支持命名空间(避免键冲突)
- 基于Streamlit session_state持久化

### 导航系统

导航由NavigationManager统一管理,状态同步机制确保主模块与子模块一致性。

**主模块结构** (dashboard/app.py定义):
```python
MODULE_CONFIG = {
    "数据预览": None,
    "监测分析": {
        "工业": ["工业增加值", "工业企业利润拆解"]
    },
    "模型分析": {
        "DFM模型": ["数据准备", "模型训练", "模型分析", "新闻分析"]
    },
    "数据探索": None,
    "用户管理": {
        "用户管理": ["用户列表", "权限配置", "系统设置"]
    }
}
```

**导航状态键**:
- `selected_main_module` - 主模块
- `selected_sub_module` - 子模块
- `selected_detail_module` - 详细模块

### 数据库架构 (SQLite)

**位置**: `data/users.db`

**users表**:
```sql
id INTEGER PRIMARY KEY
username TEXT UNIQUE NOT NULL
password_hash TEXT NOT NULL        -- bcrypt加密
email, wechat, phone, organization TEXT
permissions TEXT                   -- JSON数组: ["data_preview", "model_analysis"]
created_at, last_login DATETIME
is_active INTEGER                  -- 布尔值
failed_login_attempts INTEGER      -- 登录失败次数
locked_until TEXT                  -- 账户锁定时间
```

**user_sessions表**:
```sql
session_id TEXT PRIMARY KEY        -- UUID
user_id INTEGER                    -- 外键
created_at, expires_at DATETIME    -- 默认8小时过期
last_accessed DATETIME
is_active INTEGER
```

## DFM模块重构架构

DFM模块经过重构,从15,049行优化到10,800行(减少28%),采用分层设计:

```
dashboard/DFM/train_ref/
├── core/                  # 核心算法层
│   ├── kalman.py          # 卡尔曼滤波(预测/平滑)
│   ├── factor_model.py    # DFM模型实现
│   └── estimator.py       # EM参数估计
├── selection/             # 变量选择层
│   └── backward_selector.py  # 后向逐步变量选择
├── training/              # 训练协调层
│   ├── config.py          # 配置管理(TrainingConfig)
│   ├── trainer.py         # DFMTrainer统一接口(包含ModelEvaluator)
│   └── pipeline.py        # 训练流程管道
├── analysis/              # 分析输出层
│   ├── reporter.py        # 分析报告生成(PCA/贡献度/R²)
│   ├── analysis_utils.py  # 分析工具函数
│   └── visualizer.py      # 结果可视化
├── evaluation/            # 评估层
│   ├── evaluator.py       # 模型评估器
│   ├── metrics.py         # 评估指标
│   └── validator.py       # 数据验证器
├── utils/                 # 工具层
│   ├── data_utils.py      # 数据加载与预处理
│   ├── precompute.py      # 预计算引擎
│   └── logger.py          # 日志工具
├── optimization/          # 优化层
│   └── cache.py           # LRU缓存
└── facade.py              # 统一API入口
```

**关键配置类**:
```python
@dataclass
class TrainingConfig:
    # 数据配置
    data_path: str                  # 数据文件路径
    target_variable: str            # 目标变量
    selected_indicators: List[str]  # 选中指标
    train_end_date: str             # 训练集结束日期
    validation_end_date: str        # 验证集结束日期

    # 模型配置
    k_factors: int = 2              # 因子个数
    max_iterations: int = 30        # 最大迭代次数
    tolerance: float = 1e-6         # 收敛容差

    # 变量选择配置
    enable_variable_selection: bool = False  # 是否启用变量选择
    selection_criterion: str = 'rmse'        # 选择准则(rmse/hit_rate)

    # 因子数选择配置
    factor_selection_method: str = 'fixed'   # 因子数选择方法(fixed/cumulative/elbow)
    pca_threshold: float = 0.85             # PCA累积方差阈值
```

**训练流程**（两阶段）:
1. **数据加载与验证** - DataValidator, data_utils
2. **阶段1：变量选择**（可选） - BackwardSelector
   - 固定k=块数，后向逐步剔除变量
   - 优化目标：RMSE或Hit Rate
3. **阶段2：因子数选择** - PCA分析
   - fixed: 固定因子数
   - cumulative: PCA累积方差阈值
   - elbow: 边际方差肘部法则
4. **最终模型训练** - DFMModel (core/)
   - EM算法参数估计 (estimator.py)
   - 卡尔曼滤波状态估计 (kalman.py)
5. **模型评估** - ModelEvaluator
   - 样本内/样本外RMSE, Hit Rate, 相关系数
6. **结果分析与可视化** - AnalysisReporter, ResultVisualizer

**快速开始示例**:

```python
from dashboard.DFM.train_ref import DFMTrainer, TrainingConfig

# 1. 构建配置
config = TrainingConfig(
    # 数据配置
    data_path="data/经济数据库.xlsx",
    target_variable="规模以上工业增加值:当月同比",
    selected_indicators=["钢铁产量", "发电量", "货运量", ...],
    train_end_date="2023-12-31",
    validation_end_date="2024-06-30",

    # 模型配置
    k_factors=3,
    max_iterations=50,
    tolerance=1e-6,

    # 变量选择（可选）
    enable_variable_selection=True,
    selection_criterion='rmse',

    # 因子数选择
    factor_selection_method='cumulative',
    pca_threshold=0.85
)

# 2. 训练模型
trainer = DFMTrainer(config)

def progress_callback(msg):
    print(f"[训练进度] {msg}")

results = trainer.train(progress_callback=progress_callback)

# 3. 查看结果
print(f"样本外RMSE: {results.metrics.rmse_oos:.4f}")
print(f"样本外Hit Rate: {results.metrics.hit_rate_oos:.2%}")
print(f"选定变量数: {len(results.selected_variables)}")
print(f"因子个数: {results.k_factors}")

# 4. 生成分析报告（可选）
from dashboard.DFM.train_ref.analysis import AnalysisReporter
reporter = AnalysisReporter()
reporter.generate_report_with_params(results, output_dir="results/")
```

详细使用说明请参考: dashboard/DFM/train_ref/README.md

## 数据预览模块

**频率支持**: 周度、月度、日度、旬度、年度

**配置文件**: `dashboard/preview/config.py`
```python
UNIFIED_FREQUENCY_CONFIGS = {
    "周度": {
        "sort_column": "周次",
        "percentage_columns": ["同比增速(%)", "环比增速(%)"],
        "color": "#1f77b4"
    },
    "月度": {...},
    "日度": {...}
}
```

**关键功能**:
- Excel文件自动解析 (data_loader.py)
- 频率转换与对齐 (frequency_utils.py)
- 基于文件名的缓存机制
- 多频率数据合并展示

## 调试与性能监控

### 添加调试日志

```python
from dashboard.ui.utils.debug_helpers import debug_log, debug_navigation

debug_log("关键节点信息", "DEBUG")
debug_navigation("导航标签", "导航详情")
```

### 性能监控

```python
from dashboard.ui.utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()
# ... 执行操作
monitor.report()
```

## 开发注意事项

### 关键原则

1. **状态管理**: 必须使用UnifiedStateManager,严禁直接操作st.session_state
2. **调试信息**: BUG修复时需在关键节点添加调试日志
3. **测试清理**: 功能测试成功后删除所有测试脚本和临时文件
4. **代码质量**: 遵循KISS、DRY、YAGNI、SOC、SRP原则
5. **无emoji**: 代码和文档中不使用emoji图标
6. **无兼容代码**: 不写兼容或回退逻辑

### 添加新模块步骤

1. 在`dashboard/`下创建模块目录
2. 创建`__init__.py`导出公共接口
3. 在`dashboard/ui/pages/`创建对应UI页面
4. 在`app.py`的`MODULE_CONFIG`中注册
5. 在`permissions.py`的`PERMISSION_MODULE_MAP`中添加权限映射

### UI样式注意

每次页面渲染都需要重新注入CSS(Streamlit特性),使用:
```python
from dashboard.ui.utils.style_loader import inject_custom_css
inject_custom_css()
```

### 数据文件路径

- 用户数据: `data/users.db`
- 经济数据: `data/*.xlsx`
- WIND数据: `data/wind数据/*.xlsx`
- 缓存文件: `data/DTW分析结果.csv`

## 权限系统

**权限模块映射** (dashboard/auth/permissions.py):
```python
PERMISSION_MODULE_MAP = {
    "数据预览": ["data_preview"],
    "监测分析": ["monitoring_analysis"],
    "模型分析": ["model_analysis"],
    "数据探索": ["data_exploration"],
    "用户管理": ["user_management"]
}
```

**检查用户权限**:
```python
from dashboard.auth.permissions import has_permission

if has_permission(user, "model_analysis"):
    # 显示模型分析模块
```

## Docker部署

**Dockerfile关键配置**:
- 基础镜像: python:3.11.5-slim
- 工作目录: /app
- 环境变量: PYTHONPATH=/app
- 健康检查: 30秒间隔HTTP检查

**持久化卷**:
- `./data:/app/data` - 数据文件
- `./logs:/app/logs` - 日志文件
- `./config:/app/config` - 配置文件

## 依赖关系

**核心依赖**:
- streamlit >= 1.50.0 (Web框架)
- pandas >= 2.3.0, numpy >= 2.3.0 (数据处理)
- scipy >= 1.16.0, statsmodels >= 0.14.0 (科学计算)
- scikit-learn >= 1.7.0 (机器学习)
- dtaidistance >= 2.3.0 (DTW分析)
- bcrypt >= 5.0.0 (密码加密)
- plotly >= 6.3.0, matplotlib >= 3.10.0 (可视化)

**安装**:
```bash
pip install -r requirements.txt
```

## 关键文件路径参考

| 功能 | 文件路径 | 说明 |
|------|---------|------|
| 主入口 | dashboard/app.py:713 | 应用启动入口 |
| 状态管理 | dashboard/core/unified_state.py:14 | UnifiedStateManager类 |
| 导航管理 | dashboard/core/navigation_manager.py | NavigationManager类 |
| 用户数据库 | dashboard/auth/database.py | AuthDatabase类 |
| **DFM训练器** | dashboard/DFM/train_ref/training/trainer.py | DFMTrainer类(845行) |
| **DFM配置** | dashboard/DFM/train_ref/training/config.py | TrainingConfig类 |
| **DFM核心算法** | dashboard/DFM/train_ref/core/factor_model.py | DFMModel类(514行) |
| **卡尔曼滤波** | dashboard/DFM/train_ref/core/kalman.py | KalmanFilter类(341行) |
| **EM参数估计** | dashboard/DFM/train_ref/core/estimator.py | estimate_parameters函数(298行) |
| **变量选择** | dashboard/DFM/train_ref/selection/backward_selector.py | BackwardSelector类(339行) |
| **结果分析** | dashboard/DFM/train_ref/analysis/reporter.py | AnalysisReporter类(289行) |
| **结果可视化** | dashboard/DFM/train_ref/analysis/visualizer.py | ResultVisualizer类(634行) |
| **模型评估** | dashboard/DFM/train_ref/evaluation/evaluator.py | ModelEvaluator类 |
| **统一API** | dashboard/DFM/train_ref/facade.py | 完整使用示例 |
| 数据预览 | dashboard/preview/main.py | 预览模块主逻辑 |
| 频率配置 | dashboard/preview/config.py | UNIFIED_FREQUENCY_CONFIGS |
| 平稳性分析 | dashboard/explore/analysis/stationarity.py | ADF/KPSS检验 |
