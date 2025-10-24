# train_ref模块自动化UI测试任务清单

## 重要原则

- 所有测试代码遵循KISS、DRY、YAGNI原则
- 测试失败时必须记录详细日志和截图
- 测试成功后删除临时文件
- 不添加emoji到代码或文档中

## 前置准备

### 0.1 环境验证

- [ ] 0.1.1 验证Playwright MCP可用性
  - 检查`mcp__playwright__*`工具可用
  - 测试基本浏览器操作
  - 确认可访问本地Streamlit应用(http://localhost:8501)

- [ ] 0.1.2 验证测试数据文件
  - 确认`data/经济数据库1017.xlsx`存在
  - 验证文件可正常读取
  - 检查数据维度和格式

- [ ] 0.1.3 安装测试依赖
  - pytest-html (HTML报告生成)
  - pytest-rerunfailures (失败重试)
  - pytest-xdist (可选,并行执行)

## 1. 测试基础设施(Week 1, Days 1-2)

### 1.1 目录结构创建

- [ ] 1.1.1 创建测试目录
  - `dashboard/DFM/train_ref/tests/ui/__init__.py`
  - `dashboard/DFM/train_ref/tests/ui/utils/__init__.py`
  - `dashboard/DFM/train_ref/tests/ui/test_results/`

- [ ] 1.1.2 创建配置文件
  - `tests/ui/pytest.ini`: Pytest配置
  - `tests/ui/.env.test`: 测试环境变量(如Streamlit URL)

### 1.2 Pytest Fixtures

- [ ] 1.2.1 实现conftest.py
  - `streamlit_url` fixture: 返回Streamlit应用URL
  - `test_data_file` fixture: 返回测试数据文件路径
  - `browser_context` fixture: Playwright浏览器上下文(会话级)
  - `page` fixture: Playwright页面对象(函数级)
  - `console_monitor` fixture: 控制台监控器

- [ ] 1.2.2 编写fixture测试
  - 验证所有fixtures可正常工作
  - 测试浏览器启动/关闭
  - 测试页面导航

### 1.3 Playwright工具函数

- [ ] 1.3.1 实现playwright_helpers.py
  - `navigate_to_dfm_module(page)`: 导航到DFM模块
  - `wait_for_element(page, selector, timeout)`: 智能等待
  - `click_and_wait(page, selector)`: 点击并等待加载
  - `get_text_content(page, selector)`: 获取文本内容
  - `take_screenshot(page, filename)`: 截图保存

- [ ] 1.3.2 实现data_prep_helpers.py
  - `upload_excel_file(page, file_path)`: 上传Excel文件
  - `set_start_date(page, date_str)`: 设置开始日期
  - `click_process_button(page)`: 点击处理按钮
  - `wait_for_data_prep_complete(page)`: 等待处理完成
  - `verify_data_prep_success(page)`: 验证处理成功

- [ ] 1.3.3 实现training_helpers.py
  - `navigate_to_training_tab(page)`: 切换到模型训练tab
  - `select_training_variables(page, variable_names)`: 选择训练变量
  - `set_variable_selection_method(page, method)`: 设置变量选择方法
  - `set_factor_selection_method(page, method)`: 设置因子选择方法
  - `set_k_factors(page, k)`: 设置因子数
  - `set_em_iterations(page, max_iter)`: 设置EM迭代次数
  - `click_train_button(page)`: 点击训练按钮
  - `wait_for_training_complete(page, timeout=300000)`: 等待训练完成(5分钟超时)
  - `extract_training_results(page)`: 提取训练结果

## 2. 控制台监控(Week 1, Days 3-4)

### 2.1 控制台监控器实现

- [ ] 2.1.1 实现ConsoleMonitor类(utils/console_monitor.py)
  - `__init__(page)`: 初始化并注册控制台事件监听器
  - `_handle_console_message(msg)`: 处理控制台消息
  - `get_all_logs()`: 获取所有日志
  - `get_errors()`: 获取ERROR日志
  - `get_warnings()`: 获取WARNING日志
  - `check_for_keywords(keywords)`: 检查关键字
  - `extract_em_convergence_info()`: 提取EM收敛信息
  - `get_summary()`: 生成日志摘要

- [ ] 2.1.2 编写控制台监控测试
  - 测试日志捕获
  - 测试ERROR/WARNING分类
  - 测试关键字匹配
  - 覆盖率 > 85%

## 3. 数据准备自动化测试(Week 1, Days 3-4)

### 3.1 数据准备测试实现

- [ ] 3.1.1 实现test_data_prep_automation.py
  - `test_data_prep_upload_file()`: 测试上传Excel文件
  - `test_data_prep_set_start_date()`: 测试设置开始日期(2020-01-01)
  - `test_data_prep_default_params()`: 测试默认参数
  - `test_data_prep_complete_flow()`: 完整data_prep流程测试
  - `test_data_prep_console_output()`: 验证控制台无ERROR

- [ ] 3.1.2 验证data_prep测试通过
  - 运行`pytest tests/ui/test_data_prep_automation.py`
  - 所有测试通过
  - 生成HTML报告

## 4. 参数组合测试(Week 1-2, Days 5-10)

### 4.1 主测试套件实现

- [ ] 4.1.1 实现test_training_param_combinations.py
  - 定义参数配置列表(12个测试用例)
  - 实现`test_training_with_params()`参数化测试
  - 使用`@pytest.mark.parametrize`装饰器
  - 预计~400行

- [ ] 4.1.2 实现12个测试用例配置
  - T1: k=2, fixed, 禁用变量选择
  - T2: k=3, fixed, 禁用变量选择
  - T3: k=5, fixed, 禁用变量选择
  - T4: k=auto, cumulative(0.85), 禁用变量选择
  - T5: k=auto, cumulative(0.90), 禁用变量选择
  - T6: k=auto, elbow, 禁用变量选择
  - T7: k=3, fixed, backward变量选择
  - T8: k=auto, cumulative(0.85), backward变量选择
  - T9: k=1, fixed, 禁用变量选择(边界)
  - T10: k=10, fixed, 禁用变量选择(边界)
  - T11: k=3, fixed, EM=10(少迭代)
  - T12: k=3, fixed, EM=50(多迭代)

### 4.2 结果验证逻辑

- [ ] 4.2.1 实现训练成功性验证
  - `verify_training_success(page)`: 检查训练状态为"completed"
  - `verify_no_errors(page)`: 检查无异常消息
  - `verify_results_displayed(page)`: 检查结果正确展示

- [ ] 4.2.2 实现结果合理性验证
  - `verify_metrics_valid(results)`: 验证RMSE/Hit Rate/相关系数范围
  - `verify_factor_number(results, config)`: 验证因子数符合配置
  - `verify_variable_selection(results, config)`: 验证变量选择结果

- [ ] 4.2.3 实现控制台输出验证
  - `verify_no_console_errors(console_monitor)`: 检查无ERROR
  - `verify_em_convergence(console_monitor)`: 检查EM收敛信息
  - `verify_training_progress(console_monitor)`: 检查训练进度日志

### 4.3 快速模式实现

- [ ] 4.3.1 添加pytest marker
  - `@pytest.mark.quick`: 标记快速测试(T1, T2, T7)
  - `@pytest.mark.full`: 标记完整测试(所有12个)

- [ ] 4.3.2 编写快速模式配置
  - `pytest.ini`: 配置quick marker
  - 运行命令: `pytest -m quick`(~10分钟)

## 5. 测试报告生成(Week 2, Days 11-12)

### 5.1 报告生成器实现

- [ ] 5.1.1 实现TestResultCollector类(utils/test_report_generator.py)
  - `collect_result(test_id, config, status, metrics, console_summary)`: 收集单个结果
  - `get_all_results()`: 获取所有结果
  - `calculate_statistics()`: 计算统计信息

- [ ] 5.1.2 实现HTML报告生成
  - `generate_html_report(results, output_path)`: 生成HTML报告
  - 包含测试摘要表格
  - 包含每个用例的详细信息
  - 包含失败截图(如有)
  - 包含控制台输出摘要

- [ ] 5.1.3 实现JSON汇总生成
  - `generate_json_summary(results, output_path)`: 生成JSON文件
  - 结构化测试结果数据
  - 便于后续分析和CI/CD集成

### 5.2 pytest-html集成

- [ ] 5.2.1 配置pytest-html插件
  - `pytest.ini`: 配置报告输出路径
  - 添加自定义CSS样式
  - 添加失败截图附件

- [ ] 5.2.2 编写pytest hook
  - `pytest_runtest_makereport`: 测试结果收集
  - `pytest_html_results_table_header`: 自定义表头
  - `pytest_html_results_table_row`: 自定义表行

## 6. 完整测试执行(Week 2, Days 13-14)

### 6.1 快速模式测试

- [ ] 6.1.1 运行快速测试(3个用例)
  - 命令: `pytest -m quick tests/ui/test_training_param_combinations.py`
  - 验证所有测试通过
  - 用时 < 15分钟

- [ ] 6.1.2 分析快速测试结果
  - 检查HTML报告
  - 检查JSON汇总
  - 检查控制台输出
  - 记录任何问题

### 6.2 完整测试执行

- [ ] 6.2.1 运行完整测试(12个用例)
  - 命令: `pytest tests/ui/test_training_param_combinations.py --html=test_results/report.html`
  - 用时预计 36-45分钟
  - 记录每个用例的执行时间

- [ ] 6.2.2 分析完整测试结果
  - 统计通过/失败数量
  - 分析失败原因(如有)
  - 检查控制台ERROR/WARNING
  - 生成问题清单

### 6.3 问题修复(如需要)

- [ ] 6.3.1 修复阻塞性问题(P0)
  - 导致训练失败的bug
  - Python异常或崩溃
  - 数据处理错误

- [ ] 6.3.2 修复功能性问题(P1)
  - 结果不合理(如RMSE异常)
  - 参数配置不生效
  - UI交互问题

- [ ] 6.3.3 记录已知问题(P2)
  - 性能问题
  - 边界情况警告
  - 可选优化项

## 7. 文档与交付(Week 2, Day 14)

### 7.1 测试文档

- [ ] 7.1.1 编写测试README.md
  - 测试目的和范围
  - 环境准备步骤
  - 运行测试命令
  - 报告解读指南

- [ ] 7.1.2 编写测试用例文档
  - 12个测试用例详细说明
  - 参数配置说明
  - 预期结果说明

- [ ] 7.1.3 编写问题报告(如有)
  - 问题清单(标题、描述、优先级)
  - 复现步骤
  - 截图和日志
  - 建议修复方案

### 7.2 代码清理

- [ ] 7.2.1 删除测试临时文件
  - 删除测试过程产生的临时数据
  - 保留测试报告和截图

- [ ] 7.2.2 代码审查
  - 检查代码符合项目规范
  - 添加必要的docstring
  - 删除调试代码

- [ ] 7.2.3 提交代码
  - 提交所有测试代码
  - 提交测试文档
  - 提交测试报告

## 验收标准

完成后必须满足:

1. **基础设施**:
   - ✅ 所有fixtures正常工作
   - ✅ Playwright工具函数覆盖率 > 80%
   - ✅ 控制台监控器功能完整

2. **数据准备测试**:
   - ✅ data_prep流程测试通过
   - ✅ 上传、设置参数、处理全流程成功

3. **参数组合测试**:
   - ✅ 12个测试用例全部实现
   - ✅ 快速模式(3个用例)测试通过
   - ✅ 完整测试(12个用例)通过率 >= 80% (至少10个通过)

4. **控制台监控**:
   - ✅ 捕获所有控制台输出
   - ✅ ERROR/WARNING正确分类
   - ✅ EM收敛信息正确提取

5. **测试报告**:
   - ✅ HTML报告生成成功
   - ✅ JSON汇总数据完整
   - ✅ 失败用例包含截图和日志

6. **文档**:
   - ✅ README.md完整
   - ✅ 测试用例文档清晰
   - ✅ 问题报告(如有)详细

## 时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| 0 | 前置准备 | 0.5天 |
| 1 | 测试基础设施 | 2天 |
| 2 | 控制台监控 | 1天 |
| 3 | 数据准备测试 | 1天 |
| 4 | 参数组合测试 | 4天 |
| 5 | 测试报告 | 1.5天 |
| 6 | 完整测试执行 | 2天 |
| 7 | 文档与交付 | 1天 |
| **总计** | | **13天 ≈ 2周** |

## 关键成功因素

1. **Playwright稳定性**: 确保页面元素定位器准确,等待策略合理
2. **参数配置正确**: 12个测试用例配置准确反映设计意图
3. **控制台监控有效**: 能捕获并分析关键日志信息
4. **问题及时修复**: 发现问题后快速分析和修复
5. **文档清晰完整**: 便于未来维护和扩展
