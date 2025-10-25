# 设计文档：DFM指标计算修正

## 问题根因分析

### 1. Hit Rate为-inf的触发条件

通过代码分析，发现以下触发路径：

```python
# metrics.py:148-152
if len(common_idx) == 0:
    logger.error("nowcast和target没有共同的时间索引")
    return MetricsResult(
        np.inf, np.inf, -np.inf, np.inf, np.inf, -np.inf, None
    )
```

**可能原因**：
1. 日期格式不匹配导致索引对齐失败
2. `nowcast` 和 `target` 的时间索引类型不一致（如 `DatetimeIndex` vs `Index`）
3. 日期切分后数据为空

### 2. 新老代码Hit Rate计算差异

**老代码逻辑** (`train_model/analysis_utils.py:650-670`)：
```python
def calculate_hit_rate(data_df):
    # 1. 先diff再dropna
    target_diff = data_df['target'].diff().dropna()
    nowcast_diff = data_df['nowcast'].diff().dropna()

    # 2. 对齐索引
    common_index = target_diff.index.intersection(nowcast_diff.index)
    target_diff_aligned = target_diff.loc[common_index]
    nowcast_diff_aligned = nowcast_diff.loc[common_index]

    # 3. 计算方向
    target_direction = np.sign(target_diff_aligned)
    nowcast_direction = np.sign(nowcast_diff_aligned)

    # 4. 统计命中
    hits = (target_direction == nowcast_direction).sum()
    return hits / len(common_index)
```

**新代码逻辑** (`train_ref/evaluation/metrics.py:32-58`)：
```python
def calculate_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    # 1. 先diff（不dropna）
    true_direction = np.sign(y_true.diff())
    pred_direction = np.sign(y_pred.diff())

    # 2. 创建有效性掩码
    valid_mask = ~(true_direction.isna() | pred_direction.isna())

    # 3. 过滤有效数据
    hits = (true_direction[valid_mask] == pred_direction[valid_mask]).sum()
    total = valid_mask.sum()

    return hits / total if total > 0 else np.nan
```

**关键差异**：
- 老代码先 `dropna()` 再 `intersection`，确保数据对齐
- 新代码使用 `valid_mask` 过滤，但可能在 `diff()` 后第一个元素为NaN时产生索引不对齐

### 3. 日期参数传递链路

```
UI层 (model_training_page.py:1478-1480)
  ↓ 字符串格式 'YYYY-MM-DD'
  train_end=train_end_date.strftime('%Y-%m-%d')
  validation_start=validation_start_value.strftime('%Y-%m-%d')
  validation_end=validation_end_value.strftime('%Y-%m-%d')

配置层 (training/config.py)
  ↓ 存储为字符串
  TrainingConfig.train_end: str
  TrainingConfig.validation_start: Optional[str]
  TrainingConfig.validation_end: Optional[str]

训练器层 (trainer.py:800-810)
  ↓ 转换为datetime
  train_end_date = pd.to_datetime(self.config.train_end)
  val_start_date = pd.to_datetime(self.config.validation_start)
  val_end_date = pd.to_datetime(self.config.validation_end)

指标计算层 (metrics.py:168-180)
  ↓ 使用字符串索引（可能有问题！）
  train_data = aligned_df.loc[:train_end]
  oos_data = aligned_df.loc[validation_start:validation_end]
```

**潜在问题**：
- `aligned_df` 的索引是 `DatetimeIndex`
- `train_end`, `validation_start`, `validation_end` 是字符串
- Pandas允许用字符串索引 `DatetimeIndex`，但可能在边界处理上有差异

## 解决方案设计

### 方案A：统一使用Datetime对象

**优点**：
- 类型安全，避免字符串匹配问题
- 与Pandas索引类型一致

**缺点**：
- 需要修改多处代码
- 配置序列化需要特殊处理

**实施**：
1. 修改 `TrainingConfig` 使用 `datetime.date` 类型
2. 修改 `metrics.py` 接收 `datetime` 参数并转换为字符串索引

### 方案B：保持字符串但确保格式一致（推荐）

**优点**：
- 改动最小
- 配置序列化简单

**缺点**：
- 需要严格验证日期字符串格式

**实施**：
1. 在 `metrics.py` 中添加日期格式验证
2. 确保使用 `pd.to_datetime()` 转换后再索引
3. 添加详细日志记录日期切分结果

### 方案C：修正Hit Rate计算逻辑以匹配老代码

**实施**：
```python
def calculate_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """计算方向命中率（匹配老代码逻辑）"""
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan

    # 1. 先diff再dropna（匹配老代码）
    true_diff = y_true.diff().dropna()
    pred_diff = y_pred.diff().dropna()

    # 2. 对齐索引（匹配老代码）
    common_idx = true_diff.index.intersection(pred_diff.index)
    if len(common_idx) == 0:
        return np.nan

    # 3. 计算方向（匹配老代码）
    true_direction = np.sign(true_diff.loc[common_idx])
    pred_direction = np.sign(pred_diff.loc[common_idx])

    # 4. 统计命中（匹配老代码）
    hits = (true_direction == pred_direction).sum()
    return hits / len(common_idx)
```

## 实施计划

### 优先级1：修正Hit Rate计算（方案C）
- **理由**：老代码经过验证，逻辑可靠
- **风险**：低，仅替换计算逻辑
- **工作量**：2小时

### 优先级2：修正日期参数处理（方案B）
- **理由**：改动最小，风险可控
- **实施细节**：
  ```python
  # metrics.py:168行前添加
  train_end = pd.to_datetime(train_end) if isinstance(train_end, str) else train_end
  validation_start = pd.to_datetime(validation_start) if isinstance(validation_start, str) else validation_start
  validation_end = pd.to_datetime(validation_end) if isinstance(validation_end, str) else validation_end

  # 使用datetime对象索引
  train_data = aligned_df.loc[:train_end]
  oos_data = aligned_df.loc[validation_start:validation_end]
  ```
- **工作量**：1小时

### 优先级3：改进异常值处理
- **实施**：
  ```python
  # metrics.py:150-152
  if len(common_idx) == 0:
      logger.error("nowcast和target没有共同的时间索引，无法计算指标")
      # 使用NaN而非-inf，更符合统计惯例
      return MetricsResult(
          np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None
      )
  ```
- **工作量**：0.5小时

### 优先级4：添加日志和验证
- **日志点**：
  1. UI层：传递的日期参数
  2. Config层：接收的日期参数
  3. Trainer层：转换后的datetime对象
  4. Metrics层：切分后的样本数
- **验证逻辑**：
  ```python
  # trainer.py:800行后添加
  logger.info(f"日期切分 - 训练期结束: {train_end_date}, "
              f"验证期: {val_start_date} 至 {val_end_date}")
  logger.info(f"训练样本数: {len(train_data_filtered)}, "
              f"验证样本数: {len(val_data_filtered)}")
  ```
- **工作量**：1小时

## 测试策略

### 单元测试

创建 `tests/consistency/test_hit_rate.py`：
```python
import pandas as pd
import numpy as np
from dashboard.DFM.train_ref.evaluation.metrics import calculate_hit_rate

def test_hit_rate_basic():
    """基础命中率测试"""
    y_true = pd.Series([1, 2, 3, 2, 1], index=pd.date_range('2020-01-01', periods=5, freq='M'))
    y_pred = pd.Series([1, 2, 3, 2, 1], index=pd.date_range('2020-01-01', periods=5, freq='M'))
    hr = calculate_hit_rate(y_true, y_pred)
    assert hr == 1.0, f"完全命中应为1.0，实际为{hr}"

def test_hit_rate_with_nan():
    """包含NaN的命中率测试"""
    y_true = pd.Series([1, np.nan, 3, 2, 1])
    y_pred = pd.Series([1, 2, 3, np.nan, 1])
    hr = calculate_hit_rate(y_true, y_pred)
    assert not np.isnan(hr), "应能处理NaN值"
```

### 一致性测试

创建 `tests/consistency/test_metrics_consistency.py`：
```python
def test_rmse_consistency():
    """对比新老代码RMSE计算"""
    # 加载测试数据
    # 调用老代码
    # 调用新代码
    # 断言误差 < 1e-6

def test_hit_rate_consistency():
    """对比新老代码Hit Rate计算"""
    # 同上
    # 断言差异 < 0.1%
```

### 端到端测试

使用真实数据测试完整流程：
- 数据：`data/经济数据库.xlsx`
- 目标：规模以上工业增加值:当月同比
- 训练期：2020-01-01 至 2023-12-31
- 验证期：2024-01-01 至 2024-12-31
- 因子数：k=2
- 预期结果：Hit Rate > 0.5, RMSE < 10

## 回滚计划

如果修复导致回归问题：

1. **立即回滚**：恢复 `metrics.py::calculate_hit_rate` 到原版本
2. **保留日志**：保留诊断阶段添加的日志代码
3. **重新诊断**：使用日志分析新的问题
4. **逐步修复**：每次只修改一个函数，测试通过后再修改下一个

## 验收标准

### 必须满足
1. Hit Rate不再出现 `-inf`
2. 新老代码RMSE误差 < 1e-6
3. 新老代码Hit Rate差异 < 0.1%
4. 所有单元测试通过
5. 端到端测试通过

### 建议满足
1. 日志完整记录日期传递链路
2. 异常情况有友好提示
3. 文档更新完整

## 后续优化

1. **性能优化**：缓存指标计算结果
2. **可视化增强**：在UI中展示样本内外分布
3. **指标扩展**：添加MAE、相关系数等指标的详细验证
