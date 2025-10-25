# 规范：DFM训练日期参数传递

## Purpose

确保UI层设置的训练期和验证期日期参数能够正确传递到后端，并在数据切分时正确应用。

## ADDED Requirements

### Requirement: UI到后端的日期参数传递必须完整且格式统一

**优先级**: P0 (关键)

**要求**:
- UI层使用 `date.strftime('%Y-%m-%d')` 转换为字符串
- Config层存储为字符串类型
- Trainer层使用 `pd.to_datetime()` 转换为Timestamp
- Metrics层接收Timestamp或字符串，统一转换为Timestamp后使用

#### Scenario: UI设置日期参数后成功传递到Trainer

**Given**:
- 用户在UI中选择：
  - 训练期结束日期：2023年12月31日
  - 验证期开始日期：2024年1月1日
  - 验证期结束日期：2024年6月30日

**When**:
- UI调用 `trainer.train()` 并传递参数：
  ```python
  train_end=date(2023, 12, 31).strftime('%Y-%m-%d'),
  validation_start=date(2024, 1, 1).strftime('%Y-%m-%d'),
  validation_end=date(2024, 6, 30).strftime('%Y-%m-%d')
  ```

**Then**:
- `TrainingConfig` 接收到：
  ```python
  config.train_end = "2023-12-31"
  config.validation_start = "2024-01-01"
  config.validation_end = "2024-06-30"
  ```
- `DFMTrainer._train_final_model()` 中日志输出：
  ```
  [INFO] 训练配置 - 训练结束: 2023-12-31, 验证期: 2024-01-01 至 2024-06-30
  ```

#### Scenario: 日期参数在metrics层正确应用于数据切分

**Given**:
- `config.train_end = "2023-12-31"`
- `config.validation_start = "2024-01-01"`
- `config.validation_end = "2024-06-30"`
- `aligned_df.index` 为 DatetimeIndex，范围 2020-01 至 2024-12

**When**: 调用 `calculate_metrics(..., train_end="2023-12-31", validation_start="2024-01-01", validation_end="2024-06-30")`

**Then**:
- 内部转换：
  ```python
  train_end_ts = pd.to_datetime("2023-12-31")  # Timestamp('2023-12-31')
  val_start_ts = pd.to_datetime("2024-01-01")  # Timestamp('2024-01-01')
  val_end_ts = pd.to_datetime("2024-06-30")    # Timestamp('2024-06-30')
  ```
- 数据切分：
  ```python
  train_data = aligned_df.loc[:train_end_ts]  # 包含2023-12-31及之前
  oos_data = aligned_df.loc[val_start_ts:val_end_ts]  # 包含2024-01 至 2024-06
  ```
- 日志输出：
  ```
  [INFO] 训练期样本数: 48 (截止 2023-12-31)
  [INFO] 验证期样本数: 6 (2024-01-01 至 2024-06-30)
  ```

### Requirement: 必须验证日期参数的合理性

**优先级**: P1 (重要)

**要求**:
- 训练结束日期必须早于验证开始日期
- 验证开始日期必须早于验证结束日期
- 日期范围必须在数据可用范围内

#### Scenario: 检测不合理的日期设置

**Given**:
- 用户设置训练结束：2024年6月30日
- 用户设置验证开始：2024年1月1日
- （训练结束晚于验证开始，不合理）

**When**: 点击"开始训练"按钮

**Then**:
- UI层在 `model_training_page.py:1370-1376` 检测到错误
- 显示错误提示："训练期结束日期必须早于验证期开始日期"
- 禁用"开始训练"按钮
- `date_validation_passed = False`

#### Scenario: 自动调整超出范围的日期

**Given**:
- 数据实际范围：2020-01-01 至 2024-06-30
- 用户设置验证结束：2024年12月31日（超出范围）

**When**: 调用 `calculate_metrics(..., validation_end="2024-12-31")`

**Then**:
- 记录警告日志："验证结束日期 2024-12-31 超出数据范围，自动调整为 2024-06-30"
- 自动修正：`validation_end = aligned_df.index.max()`
- 继续执行，不中断训练

### Requirement: 必须添加日期传递的诊断日志

**优先级**: P2 (有用)

**要求**:
- 在每个传递环节记录日期参数
- 便于排查日期参数丢失或格式错误问题

#### Scenario: 完整的日期参数传递日志链

**Given**: 正常的训练流程

**When**: 从UI发起训练到指标计算完成

**Then**: 日志输出包含完整链路（按时间顺序）：
```
[UI层] model_training_page.py:1478
  传递日期参数: train_end=2023-12-31, validation_start=2024-01-01, validation_end=2024-06-30

[Config层] training/config.py
  接收日期参数: TrainingConfig(train_end='2023-12-31', validation_start='2024-01-01', validation_end='2024-06-30')

[Trainer层] trainer.py:800
  转换日期参数: train_end_date=Timestamp('2023-12-31'), val_start=Timestamp('2024-01-01'), val_end=Timestamp('2024-06-30')

[Metrics层] metrics.py:168
  应用日期切分: train_end='2023-12-31', validation_start='2024-01-01', validation_end='2024-06-30'
  训练期样本数: 48, 验证期样本数: 6
```

## References

- UI层：`dashboard/ui/pages/dfm/model_training_page.py:1478-1480`
- 配置层：`dashboard/DFM/train_ref/training/config.py`
- 训练层：`dashboard/DFM/train_ref/training/trainer.py:800-810`
- 指标层：`dashboard/DFM/train_ref/evaluation/metrics.py:168-180`

## Rollout

### 阶段1：添加日志
- 在4个关键环节添加日志输出
- 运行一次测试验证日志完整性

### 阶段2：修正日期处理
- 在 `metrics.py` 入口统一转换日期格式
- 确保使用 `pd.Timestamp` 对象索引

### 阶段3：增强验证
- UI层增强日期合理性检查
- 后端添加日期范围自动调整逻辑
