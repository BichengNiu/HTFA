# 规范：DFM模型评估指标计算

## Purpose

定义DFM模型训练过程中评估指标（RMSE、Hit Rate、MAE）的计算方法，确保：
1. 计算结果准确可靠
2. 新老代码一致性
3. 异常情况有合理处理
4. 日期参数传递正确

## MODIFIED Requirements

### Requirement: Hit Rate计算必须匹配老代码逻辑

**优先级**: P0 (关键)

**当前问题**:
- 新代码 (`train_ref/evaluation/metrics.py:32-58`) 使用 `valid_mask` 过滤方式
- 老代码 (`train_model/analysis_utils.py:650-670`) 使用 `dropna()` + `intersection` 方式
- 可能导致索引不对齐和计算结果差异

**修正要求**:
- 必须采用老代码的 `diff().dropna()` + `intersection()` 逻辑
- 确保在相同输入下结果完全一致（差异 < 0.1%）

#### Scenario: 计算正常时间序列的Hit Rate

**Given**:
- 完整的目标变量时间序列 `y_true`（无缺失值）
- 完整的预测值时间序列 `y_pred`（无缺失值）
- 两者索引对齐，长度相同

**When**: 调用 `calculate_hit_rate(y_true, y_pred)`

**Then**:
- 返回值为 0 到 1 之间的浮点数
- 计算逻辑：
  1. 分别计算 `y_true.diff().dropna()` 和 `y_pred.diff().dropna()`
  2. 对齐索引：`common_idx = true_diff.index.intersection(pred_diff.index)`
  3. 计算方向：`np.sign()` 应用于对齐后的差分序列
  4. 统计命中：`(true_direction == pred_direction).sum() / len(common_idx)`

#### Scenario: 处理包含NaN的时间序列

**Given**:
- 目标变量 `y_true = [1, 2, NaN, 4, 5]`
- 预测值 `y_pred = [1, 2, 3, NaN, 5]`

**When**: 调用 `calculate_hit_rate(y_true, y_pred)`

**Then**:
- `diff()` 后自动跳过NaN对应的差分值
- `dropna()` 移除包含NaN的项
- `intersection()` 确保只比较两者都有效的时间点
- 返回基于有效数据的命中率

#### Scenario: 时间序列过短无法计算

**Given**:
- `y_true` 长度 < 2 或 `y_pred` 长度 < 2

**When**: 调用 `calculate_hit_rate(y_true, y_pred)`

**Then**:
- 返回 `np.nan`
- 记录警告日志："时间序列长度不足，无法计算Hit Rate"

### Requirement: RMSE计算必须使用sklearn标准方法

**优先级**: P0 (关键)

**要求**:
- 使用 `sklearn.metrics.mean_squared_error` 计算MSE
- 手动应用 `np.sqrt()` 得到RMSE
- 确保与老代码计算方法一致

#### Scenario: 计算样本内RMSE

**Given**:
- 对齐后的训练期数据 `train_data`（DataFrame包含 'Target' 和 'Nowcast' 列）
- `len(train_data) >= 2`

**When**: 计算样本内RMSE

**Then**:
```python
is_rmse = np.sqrt(mean_squared_error(
    train_data['Target'], train_data['Nowcast']
))
```
- 返回值为正浮点数
- 如果 `len(train_data) < 2`，返回 `np.nan` 而非 `np.inf`

#### Scenario: 计算样本外RMSE

**Given**:
- 对齐后的验证期数据 `oos_data`
- `len(oos_data) >= 2`

**When**: 计算样本外RMSE

**Then**:
- 使用相同的 `mean_squared_error` 方法
- 如果验证期数据不足，返回 `np.nan` 并记录警告

### Requirement: 日期参数必须正确传递和使用

**优先级**: P0 (关键)

**当前问题**:
- UI传递字符串格式日期
- 后端在多处进行 `pd.to_datetime()` 转换
- `metrics.py` 使用字符串索引 `DatetimeIndex`，可能有边界问题

**修正要求**:
- 在 `calculate_metrics` 入口处统一转换日期格式
- 使用 `pd.Timestamp` 对象进行索引切片
- 添加日志记录日期切分结果

#### Scenario: UI传递的日期参数正确应用于数据切分

**Given**:
- UI设置：`train_end = "2023-12-31"`
- UI设置：`validation_start = "2024-01-01"`
- UI设置：`validation_end = "2024-06-30"`
- 数据索引为 `DatetimeIndex`，范围 2020-01-01 至 2024-12-31

**When**: 调用 `calculate_metrics(nowcast, target, train_end, validation_start, validation_end)`

**Then**:
- 训练期数据：`aligned_df.loc[:pd.Timestamp(train_end)]`
  - 包含 2020-01-01 至 2023-12-31 的所有数据
- 验证期数据：`aligned_df.loc[pd.Timestamp(validation_start):pd.Timestamp(validation_end)]`
  - 包含 2024-01-01 至 2024-06-30 的所有数据
- 日志输出：
  ```
  训练期样本数: XXX (截止 2023-12-31)
  验证期样本数: YYY (2024-01-01 至 2024-06-30)
  ```

#### Scenario: 日期参数超出数据范围

**Given**:
- 数据范围：2020-01-01 至 2024-06-30
- UI设置：`validation_end = "2024-12-31"`（超出数据范围）

**When**: 调用 `calculate_metrics(..., validation_end="2024-12-31")`

**Then**:
- 记录警告日志："验证结束日期 2024-12-31 超出数据范围 2024-06-30"
- 自动调整为数据最后日期：`validation_end = aligned_df.index.max()`
- 继续计算，不抛出异常

### Requirement: 异常情况必须返回NaN而非负无穷

**优先级**: P1 (重要)

**当前问题**:
- `metrics.py:150-152` 在数据不足时返回 `(np.inf, np.inf, -np.inf, ...)`
- 导致UI显示 "Hit Rate: -inf%" 的异常值

**修正要求**:
- 所有异常情况返回 `np.nan`
- 添加详细的警告日志说明原因
- UI层检测 `np.nan` 并显示友好提示

#### Scenario: nowcast和target无共同索引

**Given**:
- `nowcast.index = DatetimeIndex(['2020-01', '2020-02', ...])`
- `target.index = DatetimeIndex(['2021-01', '2021-02', ...])`
- 两者无交集

**When**: 调用 `calculate_metrics(nowcast, target, ...)`

**Then**:
- 返回 `MetricsResult` 所有字段为 `np.nan`：
  ```python
  MetricsResult(
      is_rmse=np.nan,
      is_mae=np.nan,
      is_hit_rate=np.nan,
      oos_rmse=np.nan,
      oos_mae=np.nan,
      oos_hit_rate=np.nan,
      aligned_data=None
  )
  ```
- 记录错误日志："nowcast和target没有共同的时间索引，无法计算指标"

#### Scenario: 训练期数据不足

**Given**:
- 对齐后的训练期数据仅有1条记录
- `len(train_data) < 2`

**When**: 计算样本内指标

**Then**:
- `is_rmse = np.nan`
- `is_mae = np.nan`
- `is_hit_rate = np.nan`
- 记录警告日志："训练期数据不足（1条），无法计算样本内指标"

## ADDED Requirements

### Requirement: 必须提供新老代码一致性验证

**优先级**: P0 (关键)

**要求**:
- 创建自动化测试验证新老代码在相同输入下输出一致
- 测试覆盖：RMSE、Hit Rate、MAE
- 允许误差：RMSE < 1e-6，Hit Rate < 0.1%

#### Scenario: 验证完整训练流程的指标一致性

**Given**:
- 测试数据：`data/经济数据库.xlsx`
- 目标变量："规模以上工业增加值:当月同比"
- 指标列表：固定5个指标
- 训练期：2020-01-01 至 2023-12-31
- 验证期：2024-01-01 至 2024-06-30
- 因子数：k=2

**When**:
- 运行老代码 `train_model` 模块
- 运行新代码 `train_ref` 模块

**Then**:
- 样本内RMSE误差：`|new_is_rmse - old_is_rmse| < 1e-6`
- 样本外RMSE误差：`|new_oos_rmse - old_oos_rmse| < 1e-6`
- 样本内Hit Rate差异：`|new_is_hr - old_is_hr| < 0.001` (0.1%)
- 样本外Hit Rate差异：`|new_oos_hr - old_oos_hr| < 0.001` (0.1%)
- 测试结果记录在 `tests/consistency/CONSISTENCY_REPORT.md`

### Requirement: 必须添加详细的诊断日志

**优先级**: P1 (重要)

**要求**:
- 在关键路径添加日志记录
- 日志包含：日期参数、样本数、索引范围
- 使用 `logger.info` 记录正常流程，`logger.warning` 记录异常情况

#### Scenario: 记录完整的指标计算过程

**Given**: 正常的训练流程

**When**: 执行 `calculate_metrics()`

**Then**: 日志输出包含以下信息（按顺序）：
```
[INFO] 开始计算评估指标
[DEBUG] 应用滞后对齐: True
[DEBUG] 最优滞后: 0个月, 相关系数: 0.8234
[INFO] nowcast和target对齐后样本数: 48
[INFO] 训练期样本数: 36 (截止 2023-12-31)
[INFO] 验证期样本数: 6 (2024-01-01 至 2024-06-30)
[INFO] 指标计算完成 - IS: RMSE=4.5678, MAE=3.2345, HR=65.43% | OOS: RMSE=5.1234, MAE=4.0123, HR=58.33%
```

## References

- 老代码参考：`train_model/analysis_utils.py:650-690` (Hit Rate和指标计算)
- 新代码位置：`train_ref/evaluation/metrics.py`
- 相关模块：`train_ref/evaluation/evaluator.py`（调用指标计算）

## Rollout

### 阶段1：修正核心计算逻辑
- 修改 `metrics.py::calculate_hit_rate` 匹配老代码
- 修改 `metrics.py::calculate_metrics` 日期参数处理
- 修改异常返回值从 `-np.inf` 改为 `np.nan`

### 阶段2：添加验证和日志
- 添加一致性测试脚本
- 添加详细诊断日志
- 验证所有场景通过

### 阶段3：UI集成和文档
- UI层处理 `np.nan` 显示
- 更新 README 文档
- 记录一致性测试结果
