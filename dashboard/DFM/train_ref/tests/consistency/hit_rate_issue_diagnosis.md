# Hit Rate=-inf%问题诊断报告

## 问题现象

所有5个已完成测试用例（T1, T2, T3, T4, T7）的Hit Rate均显示为-inf%，而其他指标正常：

| 测试 | RMSE_OOS | Hit Rate_OOS | 状态 |
|------|----------|--------------|------|
| T1   | 5.0278   | -inf%        | ✅ 完成 |
| T2   | 5.2186   | -inf%        | ✅ 完成 |
| T3   | 5.2118   | -inf%        | ✅ 完成 |
| T4   | 5.3159   | -inf%        | ✅ 完成 |
| T7   | 4.9508   | -inf%        | ✅ 完成 |

## 代码分析

### 1. Hit Rate格式化输出位置

`dashboard/DFM/train_ref/training/trainer.py:926-927`

```python
样本内命中率: {result.metrics.is_hit_rate:.2f}%
样本外命中率: {result.metrics.oos_hit_rate:.2f}%
```

当`oos_hit_rate = -np.inf`时，格式化为`-inf%`。

### 2. Hit Rate默认值

`dashboard/DFM/train_ref/training/trainer.py:36`

```python
@dataclass
class EvaluationMetrics:
    oos_hit_rate: float = -np.inf  # 默认值
```

### 3. Hit Rate计算触发条件

`dashboard/DFM/train_ref/training/trainer.py:322`

```python
# 命中率(需要前一期值)
if len(actual_oos) > 1:
    metrics.oos_hit_rate = self.calculate_hit_rate(...)
```

**关键**：只有当`len(actual_oos) > 1`时才计算Hit Rate。

### 4. Hit Rate计算方法

`dashboard/DFM/train_ref/training/trainer.py:164-212`

```python
def calculate_hit_rate(...) -> float:
    # 返回-inf的情况：
    if len(predictions) == 0 or len(actuals) == 0 or len(previous_values) == 0:
        return -np.inf  # 情况1：输入数据为空

    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isnan(previous_values))
    if not valid_mask.any():
        return -np.inf  # 情况2：所有数据都是NaN

    if total == 0:
        return -np.inf  # 情况3：有效数据点为0
```

## 问题假设

基于RMSE能正常计算（说明forecast_oos和actual_oos有有效数据），Hit Rate=-inf的可能原因：

### 假设1：验证期数据点<=1

- **现象**：`len(actual_oos) <= 1`，导致不执行hit_rate计算
- **可能原因**：
  - validation_start/validation_end配置错误
  - 数据库实际数据不足（文件名：经济数据库1017.xlsx，可能只到2024-10-17）
  - forecast_oos生成时长度被截断

### 假设2：所有预测值都是NaN

- **现象**：`forecast_oos`中所有值都是NaN
- **可能原因**：
  - 因子矩阵X在验证期包含NaN
  - 回归系数beta计算有问题
  - _generate_target_forecast方法的索引计算错误

### 假设3：验证期没有设置

- **现象**：UI未传递validation_start/validation_end参数
- **可能原因**：
  - 测试时忘记设置验证期日期
  - UI组件默认值为None
  - TrainingConfig未正确接收参数

## 关键发现

从测试报告可知：
- **训练集结束**: 2024-06-30
- **验证集**: 2024-07-01 至 2024-12-31

如果是月度数据且数据库只到2024-10-17，验证期应该有：
- 2024-07-31
- 2024-08-31
- 2024-09-30

共**3个月**数据，应该足够计算Hit Rate（需要>1个数据点）。

## 下一步诊断方案

### 方案A：添加调试日志（推荐）

在`trainer.py`的evaluate方法中添加日志：

```python
# Line 308之后
logger.info(f"[DEBUG Hit Rate] val_data长度: {len(val_data)}")
logger.info(f"[DEBUG Hit Rate] forecast_oos: {forecast_oos[:5] if forecast_oos is not None else None}")
logger.info(f"[DEBUG Hit Rate] actual_oos长度: {len(actual_oos) if 'actual_oos' in locals() else 'N/A'}")

# Line 322之后
if len(actual_oos) > 1:
    logger.info(f"[DEBUG Hit Rate] 开始计算hit_rate...")
    logger.info(f"[DEBUG Hit Rate] previous_oos: {previous_oos}")
else:
    logger.info(f"[DEBUG Hit Rate] actual_oos数据不足: {len(actual_oos)}")
```

### 方案B：检查UI配置

检查UI测试时validation_start/validation_end是否正确传递：
1. 检查date_range组件的状态
2. 确认TrainingConfig接收的参数
3. 验证validation_end是否超出数据范围

### 方案C：单元测试

创建简单的单元测试验证calculate_hit_rate逻辑：

```python
evaluator = ModelEvaluator()
predictions = np.array([1.0, 2.0, 3.0])
actuals = np.array([1.1, 1.9, 3.2])
previous = np.array([0.9, 1.1, 2.0])
result = evaluator.calculate_hit_rate(predictions, actuals, previous)
print(f"Hit Rate: {result}")  # 应该得到正常值，不是-inf
```

## 建议修复顺序

1. ✅ **首先**：添加调试日志（方案A），重新执行一个测试用例（如T1）
2. 🔍 根据日志输出确定根本原因
3. 🛠️ 实施针对性修复：
   - 如果是数据不足：调整validation_end日期
   - 如果是NaN问题：修复forecast生成逻辑
   - 如果是计算逻辑bug：修复calculate_hit_rate方法

## 预计影响

- **严重程度**：P1 - 功能性问题
- **影响范围**：所有使用样本外hit_rate评估的场景
- **用户体验**：Hit Rate=-inf%显示不友好，但不影响RMSE等核心指标

---

**生成时间**: 2025-10-24
**状态**: 待添加调试日志确认根因
