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
├── models/DFM/         # 动态因子模型
│   ├── prep/                   # 数据准备
│   ├── train/                  # 模型训练
│   ├── decomp/                 # 新闻分析与分解
│   └── results/                # 结果处理
├── preview/            # 数据预览
├── explore/            # 数据探索
└── analysis/           # 监测分析
```

### 状态管理

**项目使用Streamlit官方推荐的st.session_state进行状态管理**

```python
import streamlit as st

# 初始化状态
if 'my_state' not in st.session_state:
    st.session_state.my_state = initial_value

# 读取状态
value = st.session_state.my_state

# 修改状态
st.session_state.my_state = new_value

# 删除状态
if 'temp_data' in st.session_state:
    del st.session_state.temp_data
```

**命名约定**：使用点分命名空间避免键冲突

```python
# 模块命名空间
st.session_state['navigation.main_module'] = 'DFM'
st.session_state['navigation.sub_module'] = '数据准备'

# DFM模块命名空间
st.session_state['train_model.dfm_training_status'] = '训练完成'
st.session_state['train_model.dfm_prepared_data_df'] = dataframe

# 组件命名空间
st.session_state['preview.current_file'] = file_path
```

**特性**:
- Streamlit原生支持，无需额外封装
- 会话隔离，每个用户独立状态
- 自动持久化，页面刷新时保留
- 符合框架设计理念，学习成本低

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
        "DFM模型": ["数据准备", "模型训练", "模型分析", "影响分析"]
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

## DFM Decomp模块重构架构

DFM Decomp模块（新闻分析/Nowcast演变）经过重构，采用分层架构设计，共2,000行代码（从原3,200行减少37%）：

### 架构概述

```
dashboard/models/DFM/decomp/
├── domain/                # 领域模型层
│   ├── models.py          # 数据类定义(8个dataclass)
│   └── exceptions.py      # 自定义异常(7个)
├── utils/                 # 工具层
│   ├── validators.py      # 验证装饰器与工具
│   ├── matrix_ops.py      # 矩阵运算优化(LRU缓存)
│   └── environment.py     # 环境配置(编码/路径/模块别名)
├── config.py             # 配置管理(无副作用)
├── core/                 # 核心业务层
│   ├── nowcast_model.py   # DFM Nowcast模型
│   ├── evolution_calculator.py  # Nowcast演变计算
│   ├── news_calculator.py       # 新闻分解计算
│   └── news_aggregator.py       # 新闻聚合
├── infrastructure/       # 基础设施层
│   ├── model_loader.py    # 模型加载器
│   ├── kalman_runner.py   # 卡尔曼滤波执行器
│   └── workspace_manager.py  # 工作空间管理(临时文件)
├── plotting/             # 可视化层
│   ├── plot_config.py     # 图表配置
│   ├── base_plotter.py    # 抽象基类
│   ├── evolution_plotter.py    # 演变图绘制
│   └── decomposition_plotter.py # 分解图绘制
└── analysis/             # 分析协调层
    ├── pipeline.py        # 流水线编排器
    ├── orchestrator.py    # 图表编排器
    └── backend.py         # UI后端接口
```

### 关键改进

- **代码行数**: 3,200 → 2,000行（减少37%）
- **最大文件**: 1,106 → 300行（减少73%）
- **最大方法**: 266 → 50行（减少81%）
- **单一职责**: 每个类/函数职责明确
- **可测试性**: 组件隔离，支持单元测试
- **可维护性**: 清晰分层，低耦合

### 核心流程

分析流水线执行6个步骤：

1. **加载模型** (ModelLoader) - 从joblib/pkl文件加载DFM模型和元数据
2. **运行卡尔曼滤波** (KalmanRunner) - 重新执行Kalman Filter & Smoother
3. **计算Nowcast演变** (EvolutionCalculator) - 生成时间序列预测演变
4. **计算新闻分解** (NewsCalculator) - 归因预测变化到数据更新
5. **生成图表** (PlotOrchestrator) - 创建演变图和分解图(Plotly HTML)
6. **保存结果** (Pipeline) - 导出CSV和图表文件

### 使用示例

#### 方式1: UI后端接口

```python
from dashboard.models.DFM.decomp import execute_news_analysis

# 执行新闻分析
result = execute_news_analysis(
    dfm_model_file_content=model_bytes,
    dfm_metadata_file_content=metadata_bytes,
    target_month="2024-06",
    plot_start_date="2023-01-01",
    plot_end_date="2024-06-30",
    base_workspace_dir=None  # 使用系统临时目录
)

# 访问PipelineResult对象
if result.returncode == 0:
    print(f"演变图: {result.plot_paths['evolution']}")
    print(f"分解图: {result.plot_paths['decomposition']}")
    print(f"演变CSV: {result.csv_paths['evolution']}")
    print(f"新闻CSV: {result.csv_paths['news']}")
    print(f"演变数据: {result.evolution_df.shape}")
else:
    print(f"错误: {result.error_message}")
```

#### 方式2: 编程接口

```python
from dashboard.models.DFM.decomp import NowcastPipeline, create_default_config

# 1. 创建配置
config = create_default_config()

# 2. 创建流水线
pipeline = NowcastPipeline(config)

# 3. 执行分析
result = pipeline.execute(
    model_files_dir="/path/to/model",
    output_dir="/path/to/output",
    target_month="2024-06",
    plot_start_date="2023-01-01",
    plot_end_date="2024-06-30"
)

# 4. 访问结果
print(f"Nowcast演变数据形状: {result.evolution_df.shape}")
print(f"新闻分解项数: {len(result.news_decomposition.items)}")
print(f"图表路径: {result.plot_paths['evolution']}")
```

#### 方式3: 低级API

```python
from dashboard.models.DFM.decomp.infrastructure import ModelLoader, KalmanRunner
from dashboard.models.DFM.decomp.core import EvolutionCalculator, NewsCalculator
from dashboard.models.DFM.decomp import create_default_config

config = create_default_config()

# 步骤1: 加载模型
loader = ModelLoader(config)
context = loader.load("/path/to/model", target_month="2024-06")

# 步骤2: 运行卡尔曼滤波
kalman = KalmanRunner(config)
context = kalman.run(context)

# 步骤3: 计算演变
evo_calc = EvolutionCalculator(config)
evolution_df = evo_calc.calculate(context)

# 步骤4: 计算新闻
news_calc = NewsCalculator(config)
news_decomposition = news_calc.calculate(context, evolution_df)
```

### 关键数据类

```python
# domain/models.py
@dataclass
class AnalysisContext:
    """分析上下文（不可变）"""
    model: DFMNowcastModel
    metadata: Dict[str, Any]
    prepared_data: pd.DataFrame
    target_month: pd.Timestamp
    analysis_start_date: pd.Timestamp
    analysis_end_date: pd.Timestamp

@dataclass
class NewsItem:
    """单个新闻项"""
    update_date: pd.Timestamp
    updated_variable: str
    observed: float          # 观测值
    forecast_prev: float     # 先验预测
    news: float              # 新息(observed - forecast_prev)
    weight: float            # 卡尔曼增益
    impact: float            # 对目标变量的影响

@dataclass
class PipelineResult:
    """流水线执行结果"""
    evolution_df: pd.DataFrame
    news_decomposition: NewsDecomposition
    plot_paths: Dict[str, str]
    csv_paths: Dict[str, str]
    returncode: int
    error_message: Optional[str]
```

### 技术特性

- **矩阵运算优化**: MatrixPowerCache使用LRU缓存加速A^k计算
- **验证装饰器**: @require_valid_context确保数据完整性
- **工作空间管理**: WorkspaceManager自动管理临时文件生命周期
- **策略模式**: BasePlotter抽象基类支持多种图表类型
- **依赖注入**: 组件通过构造函数接收config和logger
- **上下文管理器**: 支持with语句自动清理资源
- **显式配置**: 移除全局单例，使用工厂函数create_default_config()

### 影响分解公式（CRITICAL）

**核心公式**（已于2025-10-27修复）：

```
Δy_t = λ_y' × K_t[:, i] × v_i,t
```

其中：
- `λ_y` - 目标变量的因子载荷向量 (n_factors,)，从观测矩阵H中提取
- `K_t` - 第t期卡尔曼增益矩阵 (n_factors, n_variables)
- `K_t[:, i]` - 变量i对应的卡尔曼增益列向量 (n_factors,)
- `v_i,t` - 变量i在第t期的新息（观测值 - 先验预测）
- `Δy_t` - 变量i的数据更新对目标变量y的影响（标量）

**数学推导**：

根据卡尔曼滤波更新公式：

```
f_t|t = f_t|t-1 + K_t × v_t              [因子状态更新]
y_t = H × f_t|t                          [观测方程]
```

当变量i更新时：

```
Δf_t = K_t[:, i] × v_i,t                 [因子状态增量]
Δy_t = λ_y' × Δf_t                       [传递到目标变量]
```

**数据要求（重要）**：

1. **训练模块必须保存K_t历史**: 从2025-10-27起，训练模块会自动保存完整的卡尔曼增益历史到元数据中
2. **旧模型不兼容**: 2025-10-27之前训练的模型无法进行影响分解分析
3. **重新训练**: 使用旧模型时会收到警告提示，需要重新训练模型

**实现位置**：

- 核心算法: `dashboard/models/DFM/decomp/core/impact_analyzer.py:85-176`
- K_t保存: `dashboard/models/DFM/train/core/kalman.py:98-188`
- K_t导出: `dashboard/models/DFM/train/export/exporter.py:189-200`
- K_t加载: `dashboard/models/DFM/decomp/core/model_loader.py:230-253`

**数据流**：

```
训练阶段 (train):
  1. kalman.py: 卡尔曼滤波时保存每个时刻K_t → kalman_gains_history
  2. factor_model.py: DFMModelResult携带kalman_gains_history
  3. exporter.py: 将kalman_gains_history写入元数据.pkl文件

分析阶段 (decomp):
  1. model_loader.py: 从元数据.pkl加载kalman_gains_history
  2. impact_analyzer.py: 使用K_t计算影响 Δy = λ_y' × K_t[:, i] × v_i
```

**向后兼容说明**：

- 旧模型：元数据中没有`kalman_gains_history`字段
- 加载器行为：返回None并打印警告信息
- 影响分析器：检测到None时抛出DataFormatError，提示用户重新训练模型
- 不提供fallback逻辑：避免使用错误的H矩阵导致计算结果错误

**验证方法**：

```python
# 检查模型是否包含K_t历史
import joblib
metadata = joblib.load("dfm_metadata.pkl")
has_kt = 'kalman_gains_history' in metadata
print(f"模型支持影响分解: {has_kt}")

if has_kt:
    kt_history = metadata['kalman_gains_history']
    print(f"K_t历史长度: {len(kt_history)}")
    print(f"K_t矩阵形状: {kt_history[-1].shape}")  # (n_factors, n_variables)
```

### 与UI层集成

新闻分析页面通过统一接口调用：

```python
# dashboard/ui/pages/dfm/news_analysis_page.py
from dashboard.models.DFM.decomp import execute_news_analysis

result = execute_news_analysis(
    dfm_model_file_content=uploaded_model.getvalue(),
    dfm_metadata_file_content=uploaded_metadata.getvalue(),
    target_month=target_month,
    plot_start_date=start_date,
    plot_end_date=end_date
)

if result.returncode == 0:
    # 访问PipelineResult对象属性
    evolution_plot = result.plot_paths['evolution']
    evo_csv = result.csv_paths['evolution']

    # 显示图表和数据
    with open(evolution_plot, 'r', encoding='utf-8') as f:
        st.components.v1.html(f.read(), height=500)
    st.dataframe(pd.read_csv(evo_csv))
else:
    st.error(f"分析失败: {result.error_message}")
```

### 测试策略

关键路径测试覆盖：
- ✅ 模型加载与验证
- ✅ 卡尔曼滤波执行
- ✅ Nowcast演变计算
- ✅ 新闻分解计算
- ✅ 图表生成
- ✅ 端到端流水线

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

1. **状态管理**: 使用Streamlit官方的st.session_state进行状态管理，遵循点分命名空间约定（如'data_prep.key'）
2. **调试信息**: BUG修复时需在关键节点添加调试日志
3. **测试清理**: 功能测试成功后删除所有测试脚本和临时文件
4. **代码质量**: 遵循KISS、DRY、YAGNI、SOC、SRP原则
5. **无emoji**: 代码和文档中不使用emoji图标
6. **无兼容代码**: 不写兼容或回退逻辑

### 添加新模块步骤

1. 在 `dashboard/`下创建模块目录
2. 创建 `__init__.py`导出公共接口
3. 在 `dashboard/ui/pages/`创建对应UI页面
4. 在 `app.py`的 `MODULE_CONFIG`中注册
5. 在 `permissions.py`的 `PERMISSION_MODULE_MAP`中添加权限映射

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

| 功能                  | 文件路径                                               | 说明                            |
| --------------------- | ------------------------------------------------------ | ------------------------------- |
| 主入口                | dashboard/app.py:713                                   | 应用启动入口                    |
| 导航管理              | dashboard/core/navigation_manager.py                   | NavigationManager类             |
| 用户数据库            | dashboard/auth/database.py                             | AuthDatabase类                  |
| **DFM训练**     | dashboard/models/DFM/train/                            | 模型训练模块                    |
| **DFM分解**     | dashboard/models/DFM/decomp/                           | 新闻分析与影响分解模块          |
| 数据预览              | dashboard/preview/main.py                              | 预览模块主逻辑                  |
| 频率配置              | dashboard/preview/config.py                            | UNIFIED_FREQUENCY_CONFIGS       |
| 平稳性分析            | dashboard/explore/analysis/stationarity.py             | ADF/KPSS检验                    |
