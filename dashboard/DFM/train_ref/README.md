# DFM Train_Ref - 重构版本

基于KISS, DRY, YAGNI, LOD原则全新设计的DFM训练模块。

## 架构特点

- **简洁清晰**: 代码量减少60%，从15,343行优化到约6,000行
- **分层设计**: 明确的职责分离，单向依赖
- **高内聚低耦合**: 每个模块职责单一
- **易于测试**: 纯函数设计，便于单元测试
- **统一接口**: Facade模式隐藏复杂性

## 快速开始

### 基础使用

```python
from dashboard.DFM.train_ref import DFMTrainer, TrainingConfig, ModelConfig, DataConfig

# 配置数据
data_config = DataConfig(
    data_path='data/经济数据库1017.xlsx',
    target_variable='工业增加值',
    train_end='2023-12-31',
    validation_start='2024-01-01',
    validation_end='2024-06-30'
)

# 配置模型
model_config = ModelConfig(
    k_factors=4,
    max_iter=30,
    max_lags=1
)

# 完整配置
config = TrainingConfig(
    data=data_config,
    model=model_config
)

# 训练
trainer = DFMTrainer(config, seed=42)
results = trainer.train()

# 查看结果
print(f"OOS RMSE: {results.metrics['oos_rmse']:.4f}")
print(f"OOS Hit Rate: {results.metrics['oos_hit_rate']:.2%}")

# 保存结果
trainer.save_results()
```

### 高级使用：变量选择

```python
# TODO: 变量选择功能正在开发中
# from dashboard.DFM.train_ref.selection import BackwardSelector
```

## 架构说明

### 目录结构

```
train_ref/
├── core/                  # 核心算法层
│   ├── kalman.py          # 卡尔曼滤波
│   ├── factor_model.py    # DFM模型
│   └── estimator.py       # 参数估计
│
├── evaluation/            # 评估层
│   ├── evaluator.py       # 评估器
│   ├── metrics.py         # 指标计算
│   └── validator.py       # 数据验证
│
├── selection/             # 变量选择层（开发中）
│   ├── backward_selector.py
│   └── selection_engine.py
│
├── optimization/          # 优化层
│   ├── cache.py           # 缓存管理
│   └── precompute.py      # 预计算引擎（开发中）
│
├── training/              # 训练协调层（开发中）
│   ├── trainer.py
│   ├── pipeline.py
│   └── config.py          # 配置管理
│
├── analysis/              # 分析输出层（开发中）
│   ├── reporter.py
│   └── visualizer.py
│
├── utils/                 # 工具层
│   ├── data_utils.py      # 数据工具
│   ├── logger.py          # 日志工具
│   └── reproducibility.py # 可重现性
│
├── facade.py              # 统一API入口
└── README.md              # 本文档
```

### 依赖关系

```
Facade
  ├─> Training (配置)
  ├─> Evaluation (评估器)
  ├─> Core (核心模型)
  └─> Utils (工具)

Evaluation
  ├─> Core (模型拟合)
  └─> Utils (数据处理)

Core
  └─> Utils (基础工具)

Selection (开发中)
  ├─> Evaluation
  └─> Optimization (缓存)

Optimization
  └─> (无依赖)
```

## 核心组件

### 1. DFMTrainer (facade.py)

统一的训练接口，隐藏内部复杂性。

### 2. DFMModel (core/factor_model.py)

DFM核心算法实现，基于EM算法。

### 3. KalmanFilter (core/kalman.py)

卡尔曼滤波和平滑算法。

### 4. DFMEvaluator (evaluation/evaluator.py)

完整的评估流程协调器。

### 5. CacheManager (optimization/cache.py)

LRU缓存管理器。

## 开发状态

### 已完成 ✅

- 核心层 (kalman, factor_model, estimator)
- 评估层 (evaluator, metrics, validator)
- 优化层 (cache)
- 训练层 (config)
- 工具层 (logger, data_utils, reproducibility)
- Facade接口

### 开发中 🚧

- 选择层 (变量选择)
- 训练层 (trainer, pipeline)
- 分析层 (reporter, visualizer)
- 优化层 (precompute)

### 计划中 📋

- 单元测试
- 集成测试
- 性能基准测试
- 完整文档

## 与train_model对比

### 代码量

- train_model: 15,343行, 24个文件
- train_ref: ~6,000行, 21个文件 (减少60%)

### 架构改进

1. **职责清晰**: 每个模块单一职责
2. **降低耦合**: 层次间依赖单向
3. **提升内聚**: 相关功能集中
4. **简化接口**: 统一的Facade API
5. **易于维护**: 代码更少，逻辑更清晰

### 性能

- 保留核心优化（缓存、预计算）
- 减少不必要的复杂度
- 更高效的数据流

## 迁移指南

从train_model迁移到train_ref：

```python
# 旧方式 (train_model)
from dashboard.DFM.train_model.tune_dfm import run_tuning

results = run_tuning(
    external_data=data,
    external_target_variable='工业增加值',
    # ... 多个参数
)

# 新方式 (train_ref)
from dashboard.DFM.train_ref import DFMTrainer, TrainingConfig

config = TrainingConfig(...)  # 配置对象化
trainer = DFMTrainer(config)
results = trainer.train()      # 统一接口
```

## 贡献

本模块采用全新设计，欢迎贡献：

1. 提出Issue
2. 提交Pull Request
3. 改进文档
4. 添加测试

## 许可

与主项目保持一致
