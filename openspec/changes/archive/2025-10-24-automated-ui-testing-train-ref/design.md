# train_ref模块自动化UI测试设计文档

## Context

train_ref模块已完成完全重构并合并到main分支,但仅进行了单次UI集成测试(Phase 6.3)。为确保模块在各种参数配置下都能稳定工作,需要建立系统化的自动化UI测试套件。

测试使用Playwright MCP(已集成到项目中),对真实Streamlit应用进行端到端测试。

## Goals / Non-Goals

### Goals

1. **全面参数覆盖**: 测试12-15种参数组合,覆盖因子数、变量选择、因子选择等维度
2. **后台监控**: 捕获并分析训练过程的控制台输出,发现潜在问题
3. **回归保障**: 建立可重复的自动化测试,防止未来修改引入bug
4. **结果验证**: 验证训练结果的合理性(RMSE、Hit Rate等指标)
5. **性能基线**: 记录各参数组合的训练用时,建立性能基准

### Non-Goals

1. **不测试数值精度**: 不进行与train_model的数值对比(已在Phase 5完成)
2. **不测试UI美观性**: 仅测试功能,不关注UI布局和样式
3. **不测试所有可能配置**: 不穷举所有参数组合,仅覆盖代表性用例
4. **不进行压力测试**: 不测试大规模数据或并发训练

## Decisions

### 1. 测试框架选择: Playwright + Pytest

**决策**: 使用Playwright MCP + Pytest作为测试框架

**理由**:
- ✅ Playwright MCP已集成到项目中,无需额外安装
- ✅ 支持真实浏览器自动化,可测试实际Streamlit应用
- ✅ 可捕获控制台输出(console.log/print)
- ✅ Pytest提供丰富的fixtures和插件(如pytest-html报告)
- ✅ 支持参数化测试(pytest.mark.parametrize)

**替代方案**:
- ❌ Selenium: 功能类似但需额外配置
- ❌ 手动测试: 不可重复,无法回归

### 2. 测试参数矩阵设计

**决策**: 采用分层抽样策略,而非全组合测试

**参数维度**:
1. **变量数**: 固定10个(用户需求)
2. **变量选择**: backward(1个用例), 禁用(其余用例)
3. **因子选择方法**: fixed(8个), cumulative(3个), elbow(1个)
4. **因子数(k)**: 1, 2, 3, 5, 10(边界和代表性值)
5. **EM迭代次数**: 10, 30, 50(少/中/多)

**12个测试用例**:

| ID | 目的 | 变量选择 | 因子方法 | k | EM | 预期 |
|----|------|---------|---------|---|----|----|
| T1 | 基线:小k | 禁用 | fixed | 2 | 30 | 成功 |
| T2 | 基线:中k | 禁用 | fixed | 3 | 30 | 成功 |
| T3 | 基线:大k | 禁用 | fixed | 5 | 30 | 成功 |
| T4 | PCA自动(85%) | 禁用 | cumulative | auto | 30 | 成功,k≈2-4 |
| T5 | PCA自动(90%) | 禁用 | cumulative | auto | 30 | 成功,k≈3-5 |
| T6 | Elbow方法 | 禁用 | elbow | auto | 30 | 成功 |
| T7 | 变量选择+固定k | backward | fixed | 3 | 30 | 成功,变量减少 |
| T8 | 变量选择+PCA | backward | cumulative | auto | 30 | 成功 |
| T9 | 边界:单因子 | 禁用 | fixed | 1 | 30 | 成功 |
| T10 | 边界:高维 | 禁用 | fixed | 10 | 30 | 成功或警告 |
| T11 | EM迭代少 | 禁用 | fixed | 3 | 10 | 成功或未收敛警告 |
| T12 | EM迭代多 | 禁用 | fixed | 3 | 50 | 成功 |

**理由**:
- ✅ 覆盖关键参数组合
- ✅ 包含边界情况(k=1, k=10, EM=10)
- ✅ 测试自动选择方法(cumulative, elbow)
- ✅ 测试变量选择流程(T7, T8)
- ✅ 执行时间可控(12用例 × 3分钟 ≈ 36分钟)

### 3. 测试架构设计

```
tests/ui/
├── conftest.py                          # Pytest fixtures
│   ├── playwright_browser fixture       # 浏览器启动/关闭
│   ├── streamlit_app fixture            # Streamlit应用URL
│   └── test_data_file fixture           # 测试数据文件路径
│
├── test_data_prep_automation.py         # 数据准备测试
│   └── test_data_prep_flow()            # 完整data_prep流程
│
├── test_training_param_combinations.py  # 主测试套件
│   ├── test_training_with_params()      # 参数化测试(12个用例)
│   └── collect_test_results()           # 收集所有结果
│
└── utils/
    ├── playwright_helpers.py            # Playwright工具函数
    │   ├── navigate_to_dfm_module()     # 导航到DFM模块
    │   ├── upload_data_file()           # 上传Excel文件
    │   ├── set_data_prep_params()       # 设置数据准备参数
    │   ├── select_training_variables()  # 选择训练变量
    │   ├── set_training_params()        # 设置训练参数
    │   ├── start_training()             # 启动训练
    │   └── wait_for_training_complete() # 等待训练完成
    │
    ├── console_monitor.py               # 控制台监控
    │   ├── ConsoleMonitor类
    │   ├── capture_console_output()     # 捕获输出
    │   ├── check_errors()               # 检查ERROR
    │   ├── check_warnings()             # 检查WARNING
    │   └── extract_metrics()            # 提取指标
    │
    └── test_report_generator.py         # 测试报告
        ├── TestResultCollector类
        ├── collect_result()             # 收集单个结果
        ├── generate_html_report()       # HTML报告
        └── generate_json_summary()      # JSON汇总
```

### 4. 控制台输出监控策略

**决策**: 使用Playwright的`page.on('console', handler)`捕获所有控制台输出

```python
class ConsoleMonitor:
    def __init__(self, page):
        self.page = page
        self.console_logs = []
        self.errors = []
        self.warnings = []

        # 监听控制台事件
        page.on('console', self._handle_console_message)

    def _handle_console_message(self, msg):
        # 记录所有消息
        self.console_logs.append({
            'type': msg.type,
            'text': msg.text,
            'timestamp': datetime.now()
        })

        # 分类ERROR和WARNING
        text = msg.text.lower()
        if 'error' in text or msg.type == 'error':
            self.errors.append(msg.text)
        elif 'warning' in text or msg.type == 'warning':
            self.warnings.append(msg.text)

    def get_summary(self):
        return {
            'total_logs': len(self.console_logs),
            'errors': self.errors,
            'warnings': self.warnings
        }
```

**监控内容**:
- ❌ Python异常堆栈
- ⚠️  训练警告(如EM未收敛)
- 📊 训练进度信息(EM迭代、RMSE等)
- ⏱️  性能日志(加载时间、训练时间)

### 5. 结果验证策略

**分层验证**:

**Level 1: 训练成功性**
```python
def verify_training_success(page):
    # 检查训练状态
    assert page.locator("text=训练完成").is_visible(timeout=300000)  # 5分钟超时

    # 检查无异常
    assert not page.locator("text=训练失败").is_visible()
    assert not page.locator("text=Error").is_visible()
```

**Level 2: 结果合理性**
```python
def verify_training_results(page, config):
    # 提取指标
    rmse_oos = float(page.locator("text=样本外RMSE").text_content().split(':')[1])
    hit_rate_oos = float(page.locator("text=样本外Hit Rate").text_content().split(':')[1])

    # 合理性检查
    assert rmse_oos > 0, "RMSE应大于0"
    assert 0 <= hit_rate_oos <= 1, "Hit Rate应在[0,1]区间"

    # 因子数验证
    if config['factor_method'] == 'fixed':
        assert selected_k == config['k_factors'], f"因子数不符: 期望{config['k_factors']}, 实际{selected_k}"
```

**Level 3: 后台输出检查**
```python
def verify_console_output(console_monitor):
    # 不应有ERROR(除非预期)
    if console_monitor.errors:
        pytest.fail(f"发现{len(console_monitor.errors)}个ERROR: {console_monitor.errors}")

    # 记录WARNING但不失败
    if console_monitor.warnings:
        pytest.warns(f"发现{len(console_monitor.warnings)}个WARNING: {console_monitor.warnings}")
```

### 6. 测试报告设计

**HTML报告**(pytest-html):
- 测试摘要(总数/通过/失败)
- 每个用例的详细信息
- 失败截图
- 控制台输出摘要

**JSON汇总**(自定义):
```json
{
  "test_suite": "train_ref_param_combinations",
  "total_tests": 12,
  "passed": 10,
  "failed": 2,
  "execution_time_seconds": 2160,
  "test_results": [
    {
      "test_id": "T1",
      "params": {"k": 2, "method": "fixed", ...},
      "status": "PASSED",
      "metrics": {"rmse_oos": 1.82, "hit_rate_oos": 0.63},
      "training_time_seconds": 125,
      "console_errors": 0,
      "console_warnings": 0
    },
    ...
  ]
}
```

## Risks / Trade-offs

### 风险1: 测试执行时间长

**描述**: 12个用例 × ~3分钟/用例 ≈ 36-45分钟

**缓解**:
- 提供快速模式(仅测试T1,T2,T7三个核心用例,~10分钟)
- 支持并行执行(pytest-xdist,可减少50%时间)
- CI/CD中仅运行快速模式,完整测试按需执行

### 风险2: Streamlit应用不稳定

**描述**: UI元素加载顺序不确定,可能导致测试flaky

**缓解**:
- 使用Playwright的智能等待(`wait_for_selector`)
- 添加重试机制(`pytest-rerunfailures`)
- 增加超时容错(5分钟训练超时)

### 风险3: 测试发现大量问题

**描述**: 如发现多个参数组合失败,修复工作量大

**缓解**:
- 分阶段修复: P0阻塞性问题→P1功能问题→P2性能优化
- 允许部分用例失败(如k=10可能合理失败)
- 记录已知问题到test_report.md

### 关键权衡

1. **全面性 vs 速度**: 选择12个用例(不是50+),平衡覆盖和速度
2. **真实环境 vs 稳定性**: 使用真实Streamlit应用(而非模拟),接受一定flaky
3. **自动化 vs 灵活性**: 固定参数矩阵,但提供参数化扩展能力

## Migration Plan

### Phase 1: 基础设施(Week 1, Days 1-2)

1. 创建测试目录结构
2. 编写conftest.py(fixtures)
3. 编写playwright_helpers.py工具函数
4. 验证Playwright MCP可用性

### Phase 2: 数据准备测试(Week 1, Days 3-4)

1. 实现test_data_prep_automation.py
2. 测试上传Excel文件
3. 测试设置开始日期(2020-01-01)
4. 验证data_prep流程成功

### Phase 3: 核心参数测试(Week 1-2, Days 5-10)

1. 实现test_training_param_combinations.py
2. 编写12个参数化测试用例
3. 实现console_monitor.py
4. 验证3个核心用例通过(快速模式)
5. 运行完整12个用例

### Phase 4: 报告与修复(Week 2, Days 11-14)

1. 实现test_report_generator.py
2. 生成HTML和JSON报告
3. 分析测试结果
4. 修复发现的问题(如有)
5. 文档化已知问题

### 总计: 2周

## Open Questions

1. **是否需要测试forward变量选择方法**?
   - **建议**: 暂不测试,backward已足够验证变量选择功能

2. **k=10是否过大,是否应该失败**?
   - **建议**: 允许成功或合理警告,记录实际行为

3. **是否需要测试不同的数据集**?
   - **建议**: 暂时仅测试`经济数据库1017.xlsx`,足够验证参数组合

4. **是否需要CI/CD集成**?
   - **建议**: 先手动执行,如测试稳定再考虑CI集成
