# Project Context

## Purpose
HTFA (High-frequency Time-series Factor Analysis) 是一个基于Streamlit的经济运行分析平台，用于：
- 多源经济数据接入与预览（支持周度、月度、日度、旬度、年度）
- 时间序列分析（DTW相似度分析、铅滞后关系、平稳性检验）
- 动态因子模型（DFM）预测与评估
- 经济指标监测分析（工业、投资、消费等）
- 数据探索与可视化

目标用户：经济分析师、研究人员、决策支持人员

## Tech Stack

### 核心框架
- Python 3.11.5
- Streamlit >= 1.50.0（Web应用框架）
- SQLite 3（用户认证数据库）

### 数据处理
- Pandas >= 2.3.0（数据框操作）
- Numpy >= 2.3.0（数值计算）
- Openpyxl（Excel文件读写）

### 统计分析
- Statsmodels >= 0.14.0（时间序列分析、状态空间模型）
- Scipy >= 1.16.0（科学计算、卡尔曼滤波）
- Scikit-learn >= 1.7.0（机器学习、数据预处理）
- dtaidistance >= 2.3.0（DTW动态时间规整）

### 可视化
- Plotly >= 6.3.0（交互式图表）
- Matplotlib >= 3.10.0（静态图表）

### 安全
- Bcrypt >= 5.0.0（密码哈希）

### 部署
- Docker + Docker Compose
- Nginx（生产环境反向代理）

## Project Conventions

### Code Style
- **语言**: 所有文档和注释使用中文
- **格式**: Markdown文档，代码遵循PEP 8
- **命名**:
  - 文件/模块: snake_case (unified_state.py)
  - 类名: PascalCase (UnifiedStateManager)
  - 函数/变量: snake_case (get_unified_manager)
  - 常量: UPPER_SNAKE_CASE (MODULE_CONFIG)
- **禁止**: 代码和文档中不使用emoji图标
- **原则**: 遵循KISS、DRY、YAGNI、SOC、SRP编码实践

### Architecture Patterns

#### 分层架构
```
dashboard/
├── core/               # 核心框架层（状态管理、导航、资源加载）
├── auth/               # 认证与权限层
├── ui/                 # UI组件层（可复用组件、页面、工具）
├── [功能模块]/         # 业务逻辑层（DFM、preview、explore、analysis）
└── app.py              # 应用入口
```

#### 关键模式

**1. 统一状态管理（CRITICAL）**
- 所有状态操作必须通过 `UnifiedStateManager` 进行
- 严禁直接使用 `st.session_state`
- 支持命名空间和DFM专用API
- 线程安全（RLock保护）+ 单例模式

**2. 导航管理**
- `NavigationManager` 统一管理模块导航
- 三级导航：主模块 → 子模块 → 详细模块
- 状态同步机制确保一致性

**3. 资源懒加载**
- `ResourceLoader` 按需加载模块
- 减少启动时间和内存占用

**4. 权限控制**
- 基于用户权限的模块访问控制
- 权限与模块映射在 `permissions.py` 中定义

**5. DFM重构架构**
- 分层设计：核心算法层 → 评估层 → 训练协调层 → 优化层
- 从15,343行优化到6,000行（减少60%）
- LRU缓存机制

### Testing Strategy
- **调试**: 在关键节点添加调试日志（使用 `debug_log`, `debug_navigation`）
- **性能监控**: 使用 `PerformanceMonitor` 跟踪性能
- **测试清理**: 功能测试成功后删除所有测试脚本和临时文件
- **数据验证**: DFM模块使用 `DataValidator` 验证数据质量

### Git Workflow
- **主分支**: main
- **提交**: 功能完成后统一提交，附带清晰的中文说明
- **最近提交示例**:
  - 34fb8c4 修正数据准备\模型训练模块UI显示问题
  - cf8e12a 修复优化DTW,lead_lag分析
  - 9e206bc 修复DFM数据准备模块bug

## Domain Context

### 经济数据特点
- **多频率**: 周度、月度、日度、旬度、年度数据并存
- **数据源**: Wind数据库、统计局、行业协会等
- **指标类型**: 工业增加值、PMI、CPI、固定资产投资、消费品零售等
- **时间范围**: 通常为2000年至今的月度/季度数据

### DFM模型核心概念
- **动态因子模型**: 从多个观测指标中提取少数几个共同因子
- **卡尔曼滤波**: 用于状态空间模型的最优估计（预测步骤 + 平滑步骤）
- **EM算法**: 参数估计的迭代算法
- **评估指标**: RMSE、Hit Rate、相关系数

### 时间序列分析术语
- **DTW**: 动态时间规整，用于序列相似度计算
- **铅滞后关系**: 不同指标之间的先行/滞后相关性
- **平稳性检验**: ADF检验、KPSS检验

## Important Constraints

### 技术约束
- **不写兼容代码**: 不编写回退或兼容逻辑
- **状态管理强制**: 必须使用UnifiedStateManager，违反会导致状态不一致
- **Streamlit特性**: 每次页面渲染需重新注入CSS
- **Python版本**: 固定使用Python 3.11.5

### 业务约束
- **数据安全**: 用户数据存储在 `data/users.db`，密码必须bcrypt加密
- **会话管理**: 会话默认8小时过期
- **登录保护**: 5次失败后锁定账户30分钟

### 性能约束
- **启动端口**: 固定8501
- **缓存机制**: 基于文件名的数据预览缓存、DFM结果LRU缓存
- **Docker资源**: 生产环境使用Docker部署，需配置持久化卷

## External Dependencies

### 数据源
- **Wind数据**: `data/wind数据/*.xlsx`
- **本地数据**: `data/*.xlsx`（经济指标Excel文件）

### 配置文件
- **频率配置**: `dashboard/preview/config.py` (UNIFIED_FREQUENCY_CONFIGS)
- **模块配置**: `dashboard/app.py` (MODULE_CONFIG)
- **权限映射**: `dashboard/auth/permissions.py` (PERMISSION_MODULE_MAP)

### 持久化存储
- **用户数据库**: `data/users.db` (SQLite)
- **缓存文件**: `data/DTW分析结果.csv`
- **日志文件**: `logs/` (Docker部署时)

### 外部服务
- 无外部API依赖（完全离线可运行）

## Key File Paths Reference

| 功能分类 | 文件路径 | 说明 |
|---------|---------|------|
| **核心框架** |
| 主入口 | dashboard/app.py:713 | 应用启动入口 |
| 状态管理 | dashboard/core/unified_state.py:14 | UnifiedStateManager类 |
| 导航管理 | dashboard/core/navigation_manager.py | NavigationManager类 |
| 资源加载 | dashboard/core/resource_loader.py | ResourceLoader类 |
| **认证系统** |
| 数据库 | dashboard/auth/database.py | AuthDatabase类 |
| 数据模型 | dashboard/auth/models.py | User、Session模型 |
| 认证逻辑 | dashboard/auth/authentication.py | 登录/登出/会话管理 |
| 权限管理 | dashboard/auth/permissions.py | 权限检查和映射 |
| **DFM模块** |
| 训练器 | dashboard/DFM/train_ref/training/trainer.py | DFMTrainer类 |
| 卡尔曼滤波 | dashboard/DFM/train_ref/core/kalman.py | 预测/平滑算法 |
| 因子模型 | dashboard/DFM/train_ref/core/factor_model.py | DFM核心实现 |
| 评估器 | dashboard/DFM/train_ref/evaluation/evaluator.py | 模型评估 |
| **数据预览** |
| 主逻辑 | dashboard/preview/main.py | 预览模块入口 |
| 频率配置 | dashboard/preview/config.py | UNIFIED_FREQUENCY_CONFIGS |
| 数据加载 | dashboard/preview/data_loader.py | Excel解析 |
| **数据探索** |
| 平稳性分析 | dashboard/explore/analysis/stationarity.py | ADF/KPSS检验 |
| DTW分析 | dashboard/explore/analysis/dtw_analysis.py | 动态时间规整 |

## Development Workflow

### 启动应用
```bash
# Windows推荐（自动清理缓存和端口）
./start.bat

# 直接启动
python -m streamlit run dashboard/app.py --server.port=8501
```

### 添加新模块步骤
1. 在 `dashboard/` 下创建模块目录
2. 创建 `__init__.py` 导出公共接口
3. 在 `dashboard/ui/pages/` 创建对应UI页面
4. 在 `app.py` 的 `MODULE_CONFIG` 中注册
5. 在 `permissions.py` 的 `PERMISSION_MODULE_MAP` 中添加权限映射

### 清理缓存
```bash
# Windows
for /r %i in (*.pyc *.pyo) do del /f /q "%i"
for /d /r %i in (__pycache__) do rd /s /q "%i"

# Linux/Mac
find . -type d -name __pycache__ -exec rm -rf {} +
```
