# Proposal: 修正DFM模型训练中的指标计算问题

## 问题描述

当前DFM模型训练(`train_ref`模块)中存在以下问题：

1. **Hit Rate计算异常**：训练过程中Hit Rate显示为负无穷(`-inf`)
2. **日期参数传递疑问**：怀疑UI层的训练开始日、验证开始日和验证结束日未正确传递到后端
3. **新老代码一致性未验证**：`train_ref`(新代码)与`train_model`(老代码)在相同数据下的RMSE和Hit Rate计算结果未经对比验证

## 现状分析

### 代码结构对比

**老代码 (`train_model`)：**
- Hit Rate计算逻辑位于 `analysis_utils.py:650-670`
- 使用内嵌函数 `calculate_hit_rate(data_df)`
- 基于 `data_df['target'].diff()` 和 `data_df['nowcast'].diff()` 计算方向一致性

**新代码 (`train_ref`)：**
- Hit Rate计算逻辑位于 `evaluation/metrics.py:32-58`
- 使用独立函数 `calculate_hit_rate(y_true, y_pred)`
- 基于 `np.sign(y_true.diff())` 和 `np.sign(y_pred.diff())` 计算方向命中率

### 潜在问题点

1. **日期切分逻辑**：
   - UI层：`model_training_page.py:1478-1480` 传递 `train_end`, `validation_start`, `validation_end`
   - 后端接收：`trainer.py:261-263` 接收这三个参数
   - 切分实现：`metrics.py:168-180` 使用 `loc[:train_end]` 和 `loc[validation_start:validation_end]`
   - 可能问题：日期格式转换、索引对齐、边界处理

2. **Hit Rate计算差异**：
   - 老代码：先 `diff().dropna()` 再对齐索引
   - 新代码：先计算 `diff()` 再用 `valid_mask` 过滤
   - 可能导致计算基数不同

3. **返回值异常处理**：
   - `metrics.py:150-165` 在数据不足时返回 `(np.inf, np.inf, -np.inf)`
   - 这会导致Hit Rate显示为`-inf`

## 变更目标

1. **根本原因诊断**：精确定位Hit Rate为`-inf`的触发条件
2. **日期传递验证**：确认UI到后端的日期参数传递链路完整性
3. **新老代码一致性**：在相同数据、相同参数下，新老代码的RMSE和Hit Rate必须完全一致
4. **修复计算错误**：修正所有导致指标计算异常的逻辑问题

## 变更范围

### 核心模块

- `dashboard/DFM/train_ref/evaluation/metrics.py` - 指标计算逻辑
- `dashboard/DFM/train_ref/training/trainer.py` - 日期参数处理
- `dashboard/DFM/train_ref/evaluation/evaluator.py` - 评估器调用
- `dashboard/ui/pages/dfm/model_training_page.py` - UI日期参数传递

### 测试模块

- 新增：`dashboard/DFM/train_ref/tests/metrics/` - 指标计算一致性测试
- 新增：对比测试脚本验证新老代码输出一致性

## 不涉及范围

- DFM核心算法实现（EM估计、卡尔曼滤波）
- UI界面布局和交互逻辑
- 数据加载和预处理流程

## 成功标准

1. **诊断完整**：明确Hit Rate为`-inf`的根本原因并提供修复方案
2. **日期验证**：通过日志验证UI传递的日期参数在后端正确接收和使用
3. **一致性验证**：新老代码在以下场景完全一致：
   - 相同数据文件（`data/经济数据库.xlsx`）
   - 相同目标变量（如"规模以上工业增加值:当月同比"）
   - 相同指标集合
   - 相同训练/验证日期切分
   - 相同因子数k
   - 验证指标：RMSE误差 < 1e-6，Hit Rate差异 < 0.1%
4. **回归测试**：所有现有测试通过

## 依赖关系

### 前置依赖
- 无（独立变更）

### 后置影响
- 可能影响依赖指标计算的其他模块（如`analysis/reporter.py`）

## 风险评估

### 中等风险
- **计算逻辑调整**：修改指标计算可能影响历史结果的可比性
- **缓解措施**：保留老代码作为基准，仅在验证一致后替换

### 低风险
- **日期参数传递**：仅涉及参数传递链路，不影响核心算法
- **测试增强**：新增测试用例，不影响现有功能

## 实施计划

### 阶段1：诊断分析（1天）
1. 添加详细日志追踪日期参数传递链路
2. 复现Hit Rate为`-inf`的场景
3. 对比新老代码在相同输入下的中间结果

### 阶段2：修复验证（2天）
1. 修正指标计算逻辑
2. 编写一致性测试脚本
3. 验证新老代码输出一致性

### 阶段3：测试清理（1天）
1. 运行完整回归测试
2. 清理测试脚本和临时文件
3. 更新相关文档

## 相关规范

此变更将创建或修改以下能力规范：

- `metrics-calculation` - 评估指标计算规范（新增）
- `date-parameter-handling` - 日期参数传递规范（新增）
- `legacy-code-consistency` - 新老代码一致性验证规范（新增）
