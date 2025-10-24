# train_ref模块自动化UI测试

## Why

### 问题背景

train_ref模块已完成完全重构并合并到main分支(commit 6ec36c0),代码从15,049行精简到10,800行。虽然已完成:
- ✅ 单元测试(核心层89%、分析层91%、工具层82%覆盖率)
- ✅ 数值一致性验证(13/13核心算法测试通过)
- ✅ 基础UI集成测试(Phase 6.3单次验证)

但**缺少系统化的参数组合测试**,存在以下风险:

### 测试覆盖缺口

1. **参数组合不全**: Phase 6.3仅测试了单一参数配置(k=10, 变量选择=global_backward, BIC)
2. **边界情况未覆盖**:
   - 极小因子数(k=1,2)
   - 极大因子数(k=15,20)
   - 不同变量选择方法(forward/backward/none)
   - 不同因子选择方法(fixed/cumulative/elbow)
3. **缺少回归验证**: 未建立自动化回归测试机制,无法快速验证未来修改
4. **后台输出未监控**: 训练过程的后台打印输出未被检查,可能隐藏警告或错误

### 为什么需要Playwright自动化测试

1. **端到端验证**: 从UI上传数据→data_prep预处理→模型训练→结果展示的完整流程
2. **真实环境**: 测试实际的Streamlit应用,不是模拟环境
3. **后台监控**: Playwright可捕获控制台输出,发现隐藏问题
4. **可重复性**: 自动化测试可快速回归,保障代码质量

### 测试价值

通过系统化参数组合测试,可以:
- 🔍 **发现潜在问题**: 特定参数组合下的崩溃、异常、数值错误
- 📊 **验证健壮性**: 确保各种合理参数配置都能正常工作
- 🚀 **建立质量基线**: 为未来修改提供回归测试保障
- 📝 **生成测试报告**: 记录所有参数组合的训练结果和性能指标

## What Changes

### 核心变更

创建**系统化的Playwright自动化UI测试套件**,使用真实数据(`经济数据库1017.xlsx`)测试train_ref模块在不同参数配置下的完整训练流程。

### 测试范围

#### 1. 数据准备(data_prep)自动化

- 上传Excel文件(`经济数据库1017.xlsx`)
- 设置开始日期为2020-01-01
- 其他参数使用默认值
- 验证数据加载成功

#### 2. 参数组合测试矩阵(10个左右预测变量)

测试以下参数组合(共12-15个测试用例):

| 测试ID | 变量数 | 变量选择 | 因子选择方法 | 因子数(k) | EM迭代 | 说明 |
|--------|-------|---------|-------------|----------|--------|------|
| T1 | 10 | 禁用 | fixed | 2 | 30 | 基线:小因子数 |
| T2 | 10 | 禁用 | fixed | 3 | 30 | 基线:中因子数 |
| T3 | 10 | 禁用 | fixed | 5 | 30 | 基线:大因子数 |
| T4 | 10 | 禁用 | cumulative | auto(0.85阈值) | 30 | PCA自动选择 |
| T5 | 10 | 禁用 | cumulative | auto(0.90阈值) | 30 | PCA高阈值 |
| T6 | 10 | 禁用 | elbow | auto | 30 | Elbow方法 |
| T7 | 12 | backward | fixed | 3 | 30 | 变量选择+固定因子 |
| T8 | 12 | backward | cumulative | auto(0.85) | 30 | 变量选择+PCA |
| T9 | 10 | 禁用 | fixed | 1 | 30 | 边界:单因子 |
| T10 | 10 | 禁用 | fixed | 10 | 30 | 边界:高维因子 |
| T11 | 10 | 禁用 | fixed | 3 | 10 | EM迭代少 |
| T12 | 10 | 禁用 | fixed | 3 | 50 | EM迭代多 |

**预期扩展**: 根据执行结果,可能增加forward变量选择、更多PCA阈值等

#### 3. 验证内容

对每个测试用例,验证:

1. **训练成功完成**
   - 无Python异常或崩溃
   - 训练状态显示"completed"
   - 结果摘要正确展示

2. **结果合理性**
   - RMSE > 0
   - Hit Rate在[0,1]区间
   - 相关系数在[-1,1]区间
   - 选定变量数 ≤ 初始变量数
   - 选定因子数符合配置

3. **后台输出监控**
   - 捕获所有console.log/print输出
   - 检查ERROR关键字
   - 检查WARNING关键字
   - 记录EM收敛信息

4. **性能指标**
   - 记录训练用时
   - 记录数据预处理用时
   - 生成性能对比报告

### 新增文件

```
dashboard/DFM/train_ref/tests/ui/
├── __init__.py
├── test_training_param_combinations.py  # 主测试套件(Playwright)
├── test_data_prep_automation.py         # 数据准备自动化测试
├── conftest.py                          # Pytest配置和fixtures
├── utils/
│   ├── __init__.py
│   ├── playwright_helpers.py            # Playwright工具函数
│   ├── console_monitor.py               # 控制台输出监控
│   └── test_report_generator.py         # 测试报告生成器
└── test_results/                        # 测试结果输出目录
    ├── test_report.html                 # HTML测试报告
    ├── test_summary.json                # JSON汇总数据
    └── screenshots/                     # 失败截图
```

## Impact

### 影响的规范

- **ui-test-automation** (新增): DFM模块UI自动化测试能力规范
- **dfm-model-training**: 补充自动化测试验收标准

### 影响的代码

**新增代码** (~1,200行):
- `tests/ui/test_training_param_combinations.py`: 主测试套件(~400行)
- `tests/ui/test_data_prep_automation.py`: 数据准备测试(~200行)
- `tests/ui/conftest.py`: Pytest配置(~150行)
- `tests/ui/utils/playwright_helpers.py`: 工具函数(~200行)
- `tests/ui/utils/console_monitor.py`: 控制台监控(~150行)
- `tests/ui/utils/test_report_generator.py`: 报告生成(~100行)

**修改代码** (可选,视测试发现的问题而定):
- `dashboard/ui/pages/dfm/model_training_page.py`: 如发现UI问题需修复
- `dashboard/DFM/train_ref/training/trainer.py`: 如发现训练逻辑问题需修复

**无删除代码**

### 依赖项

新增测试依赖(已在项目中可用):
- `playwright` (通过MCP已集成)
- `pytest-playwright` (如未安装需添加)
- `pytest-html` (测试报告生成)

### 风险与缓解

**风险**:
1. **测试执行时间长**: 12-15个用例,每个~3分钟,总计~45分钟
   - **缓解**: 提供快速模式(仅测试3个核心用例)
2. **Streamlit应用不稳定**: 测试可能因UI加载问题失败
   - **缓解**: 添加智能等待和重试机制
3. **测试发现大量问题**: 可能需要额外修复时间
   - **缓解**: 分阶段修复,优先修复阻塞性问题

### 预期收益

1. **质量保障**: 建立自动化回归测试机制,防止未来修改引入bug
2. **参数验证**: 确认所有合理参数组合都能正常工作
3. **问题发现**: 通过后台输出监控发现隐藏问题
4. **文档价值**: 测试用例即文档,展示各参数配置的使用方法
5. **CI/CD集成**: 为未来的持续集成提供基础

### 交付物

1. 完整的Playwright自动化测试套件
2. HTML测试报告(包含所有参数组合的结果)
3. 问题清单(如发现任何问题)
4. 测试执行指南(README.md)
