# 规范：新老代码一致性验证

## Purpose

确保重构后的 `train_ref` 模块与原始 `train_model` 模块在评估指标计算上完全一致，避免因重构引入计算误差。

## ADDED Requirements

### Requirement: 必须提供自动化一致性测试

**优先级**: P0 (关键)

**要求**:
- 创建测试脚本比对新老代码输出
- 测试覆盖：RMSE、Hit Rate、MAE、相关系数
- 允许误差范围：RMSE < 1e-6，Hit Rate < 0.1%
- 测试结果记录在文档中

#### Scenario: 验证相同参数下RMSE计算一致

**Given**:
- 测试数据：`data/经济数据库.xlsx`
- 目标变量："规模以上工业增加值:当月同比"
- 预测变量：["发电量:当月同比", "钢铁产量:当月同比", "水泥产量:当月同比"]
- 训练期：2020-01-01 至 2023-12-31
- 验证期：2024-01-01 至 2024-06-30
- 因子数：k=2
- EM迭代：max_iter=30, tol=1e-6

**When**:
- 运行老代码：
  ```python
  from train_model import evaluate_dfm_params
  old_results = evaluate_dfm_params(...)
  ```
- 运行新代码：
  ```python
  from dashboard.DFM.train_ref import DFMTrainer, TrainingConfig
  config = TrainingConfig(...)
  trainer = DFMTrainer(config)
  new_results = trainer.train()
  ```

**Then**:
- 样本内RMSE：`abs(new_results.metrics.rmse_is - old_results['is_rmse']) < 1e-6`
- 样本外RMSE：`abs(new_results.metrics.rmse_oos - old_results['oos_rmse']) < 1e-6`
- 测试通过标准：所有指标误差在允许范围内

#### Scenario: 验证相同参数下Hit Rate计算一致

**Given**: 与上述RMSE测试相同的配置

**When**: 对比新老代码的Hit Rate输出

**Then**:
- 样本内Hit Rate：`abs(new_results.metrics.hit_rate_is - old_results['is_hit_rate']) < 0.001` (0.1%)
- 样本外Hit Rate：`abs(new_results.metrics.hit_rate_oos - old_results['oos_hit_rate']) < 0.001`
- 如果差异超过0.1%，测试失败并输出详细对比

### Requirement: 必须验证多种参数组合

**优先级**: P1 (重要)

**要求**:
- 测试不同因子数：k=1, 2, 3
- 测试不同训练期长度：12个月、24个月、36个月
- 测试启用/禁用变量选择
- 每种组合都必须通过一致性检查

#### Scenario: 验证不同因子数下的一致性

**Given**:
- 基础配置与前述相同
- 测试因子数：k ∈ {1, 2, 3}

**When**: 分别用k=1, 2, 3运行新老代码

**Then**:
- 每个k值下，新老代码的RMSE误差 < 1e-6
- 每个k值下，新老代码的Hit Rate差异 < 0.1%
- 测试结果记录在表格：

| 因子数 | 老代码RMSE(IS) | 新代码RMSE(IS) | 误差 | 老代码HR(OOS) | 新代码HR(OOS) | 差异 |
|--------|----------------|----------------|------|---------------|---------------|------|
| k=1    | 5.234          | 5.234          | 0.000| 0.583         | 0.583         | 0.000|
| k=2    | 4.567          | 4.567          | 0.000| 0.667         | 0.667         | 0.000|
| k=3    | 4.123          | 4.123          | 0.000| 0.708         | 0.708         | 0.000|

#### Scenario: 验证变量选择模式下的一致性

**Given**:
- 启用后向逐步变量选择
- 选择准则：RMSE
- 固定因子数：k=块数

**When**:
- 老代码运行变量选择流程
- 新代码运行变量选择流程

**Then**:
- 最终选定的变量列表完全相同
- 最终模型的RMSE误差 < 1e-6
- 最终模型的Hit Rate差异 < 0.1%

### Requirement: 必须记录一致性测试结果

**优先级**: P1 (重要)

**要求**:
- 创建 `CONSISTENCY_REPORT.md` 文档
- 记录每次测试的详细结果
- 包含时间戳、数据版本、代码版本

#### Scenario: 生成完整的一致性测试报告

**Given**: 完成所有一致性测试

**When**: 运行 `generate_consistency_report.py`

**Then**: 生成的报告包含以下章节：

```markdown
# DFM Train_ref 新老代码一致性测试报告

## 测试环境
- 测试时间：2025-10-25 15:30:00
- 数据文件：data/经济数据库.xlsx (MD5: abc123...)
- 老代码版本：train_model (commit: 4d6982c)
- 新代码版本：train_ref (commit: 当前)

## 测试场景1：基础指标计算
- 配置：k=2, 训练期36个月
- RMSE(IS): 老代码=4.5678, 新代码=4.5678, 误差=0.000000
- RMSE(OOS): 老代码=5.1234, 新代码=5.1234, 误差=0.000000
- Hit Rate(IS): 老代码=0.6543, 新代码=0.6543, 差异=0.00%
- Hit Rate(OOS): 老代码=0.5833, 新代码=0.5833, 差异=0.00%
- 结论：✓ 通过

## 测试场景2：不同因子数
[表格数据...]

## 测试场景3：变量选择
[详细结果...]

## 总结
- 总测试场景：5
- 通过场景：5
- 失败场景：0
- 一致性验证：✓ 完全一致
```

### Requirement: 必须在CI/CD中集成一致性测试

**优先级**: P2 (有用)

**要求**:
- 将一致性测试加入回归测试套件
- 每次修改评估相关代码时自动运行
- 测试失败时阻止合并

#### Scenario: 自动触发一致性测试

**Given**: 开发者修改了 `metrics.py` 文件

**When**: 提交代码并创建Pull Request

**Then**:
- CI系统自动运行 `pytest tests/consistency/`
- 执行所有一致性测试用例
- 如果任一测试失败，PR显示"检查失败"状态
- 必须修复一致性问题后才能合并

## References

- 老代码入口：`train_model/tune_dfm.py`
- 新代码入口：`dashboard/DFM/train_ref/training/trainer.py`
- 测试脚本位置：`dashboard/DFM/train_ref/tests/consistency/`
- 报告模板：`dashboard/DFM/train_ref/tests/consistency/CONSISTENCY_REPORT.md`

## Rollout

### 阶段1：创建基础测试框架
- 编写 `test_metrics_consistency.py` 脚本
- 实现新老代码调用接口
- 验证基本的RMSE和Hit Rate对比

### 阶段2：扩展测试覆盖
- 添加多参数组合测试
- 添加变量选择模式测试
- 添加边界情况测试

### 阶段3：自动化和文档
- 生成一致性报告
- 集成到回归测试套件
- 更新开发文档
