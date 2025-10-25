# ui-test-automation Specification

## Purpose

定义DFM模块train_ref的UI自动化测试能力规范,确保通过Playwright端到端测试验证各种参数配置下的训练流程稳定性和结果合理性。

## ADDED Requirements

### Requirement: Playwright测试基础设施

系统SHALL提供基于Playwright的自动化测试基础设施,支持端到端UI测试。

#### Scenario: Playwright fixtures配置

**GIVEN** pytest测试环境已配置
**WHEN** 测试用例需要浏览器和页面对象
**THEN** 系统提供以下fixtures:
- `streamlit_url`: 返回Streamlit应用URL(默认http://localhost:8501)
- `test_data_file`: 返回测试数据文件路径(data/经济数据库1017.xlsx)
- `browser_context`: Playwright浏览器上下文(会话级,测试间共享)
- `page`: Playwright页面对象(函数级,每个测试独立)
- `console_monitor`: 控制台输出监控器

#### Scenario: Playwright工具函数库

**GIVEN** 需要执行常见UI操作
**WHEN** 测试用例调用工具函数
**THEN** 系统提供以下工具函数:
- `navigate_to_dfm_module(page)`: 导航到DFM模块页面
- `wait_for_element(page, selector, timeout)`: 智能等待元素出现
- `click_and_wait(page, selector)`: 点击元素并等待页面加载完成
- `get_text_content(page, selector)`: 安全获取元素文本内容
- `take_screenshot(page, filename)`: 保存页面截图到test_results/screenshots/

#### Scenario: 测试环境隔离

**GIVEN** 多个测试用例并发或顺序执行
**WHEN** 每个测试启动时
**THEN** 系统确保:
- 浏览器上下文在测试间共享但页面独立
- 每个测试使用新的页面对象,避免状态污染
- 测试失败时自动截图保存到test_results/screenshots/
- 测试完成后自动清理临时文件

---

### Requirement: 数据准备(data_prep)自动化测试

系统SHALL自动化测试DFM模块的数据准备流程,包括上传数据、设置参数和处理验证。

#### Scenario: 上传Excel数据文件

**GIVEN** 测试数据文件`data/经济数据库1017.xlsx`存在
**WHEN** 测试用例调用`upload_excel_file(page, file_path)`
**THEN** 系统应:
- 定位文件上传组件(file_uploader)
- 上传指定的Excel文件
- 等待文件上传完成(上传状态消失)
- 验证文件名显示正确

#### Scenario: 设置数据准备参数

**GIVEN** 数据文件已上传
**WHEN** 测试用例调用`set_start_date(page, "2020-01-01")`
**THEN** 系统应:
- 定位开始日期输入框
- 设置日期为2020-01-01
- 其他参数使用默认值(不修改)
- 验证参数设置成功

#### Scenario: 执行数据处理并验证成功

**GIVEN** 数据文件和参数已设置
**WHEN** 测试用例调用`click_process_button(page)`并等待完成
**THEN** 系统应:
- 点击"处理数据"按钮
- 等待处理完成(最多60秒)
- 验证处理状态显示为"完成"或"成功"
- 验证数据维度信息正确展示(如"数据维度: 1342×88")
- 验证控制台无ERROR消息

---

### Requirement: 控制台输出监控

系统SHALL实时监控浏览器控制台输出,捕获Python后台打印的日志、警告和错误信息。

#### Scenario: 捕获所有控制台消息

**GIVEN** ConsoleMonitor已初始化并绑定到page对象
**WHEN** Streamlit应用执行训练流程
**THEN** ConsoleMonitor应:
- 捕获所有console.log、console.warn、console.error消息
- 捕获Python后台print输出(通过Streamlit重定向)
- 记录每条消息的类型(log/warn/error)、文本内容和时间戳
- 提供`get_all_logs()`方法返回完整日志列表

#### Scenario: 分类ERROR和WARNING消息

**GIVEN** 控制台输出包含多种类型消息
**WHEN** 测试用例调用`console_monitor.get_errors()`和`get_warnings()`
**THEN** 系统应:
- 自动识别包含"error"关键字的消息为ERROR
- 自动识别包含"warning"关键字的消息为WARNING
- 自动识别msg.type=='error'的消息为ERROR
- 返回按类型分类的消息列表

#### Scenario: 提取EM收敛信息

**GIVEN** 训练过程输出包含EM迭代日志
**WHEN** 测试用例调用`console_monitor.extract_em_convergence_info()`
**THEN** 系统应:
- 解析形如"EM迭代 X/30: log-likelihood = -123.45"的消息
- 提取最终迭代次数
- 提取最终对数似然值
- 提取收敛状态(是否达到tolerance或最大迭代次数)
- 返回结构化的收敛信息字典

#### Scenario: 验证无关键错误

**GIVEN** 训练流程执行完成
**WHEN** 测试用例调用`verify_no_console_errors(console_monitor)`
**THEN** 验证逻辑应:
- 检查`console_monitor.get_errors()`为空列表
- 如有ERROR,测试失败并附上ERROR消息内容
- WARNING不导致测试失败,但记录到报告中

---

### Requirement: 训练参数配置自动化

系统SHALL自动化设置模型训练的各种参数组合,包括变量选择、因子数、因子选择方法等。

#### Scenario: 选择训练变量

**GIVEN** 已切换到"模型训练"tab
**WHEN** 测试用例调用`select_training_variables(page, variable_names)`
**THEN** 系统应:
- 定位变量选择组件(multiselect或checkbox列表)
- 勾选指定的变量名称列表(如10个预测变量)
- 验证选中变量数量正确
- 验证选中变量名称匹配

#### Scenario: 设置变量选择方法

**GIVEN** 需要测试变量选择功能
**WHEN** 测试用例调用`set_variable_selection_method(page, "backward")`
**THEN** 系统应:
- 定位变量选择方法选择框(selectbox或radio)
- 选择"后向逐步剔除"或"backward"选项
- 如method="none"或False,则禁用变量选择
- 验证选择生效

#### Scenario: 设置因子选择方法和因子数

**GIVEN** 需要测试不同因子配置
**WHEN** 测试用例调用`set_factor_selection_method(page, "fixed")`和`set_k_factors(page, 3)`
**THEN** 系统应:
- 定位因子选择方法选择框
- 选择"固定因子数"/"PCA累积方差"/"Elbow方法"
- 如method="fixed",设置因子数输入框为指定值(如3)
- 如method="cumulative",设置PCA阈值(如0.85)
- 如method="elbow",不需要设置因子数(自动)
- 验证所有参数设置成功

#### Scenario: 设置EM算法参数

**GIVEN** 需要测试不同EM配置
**WHEN** 测试用例调用`set_em_iterations(page, 30)`
**THEN** 系统应:
- 定位EM最大迭代次数输入框
- 设置为指定值(如30)
- 收敛容差使用默认值(1e-6)
- 验证参数设置成功

---

### Requirement: 训练执行与结果验证

系统SHALL自动化启动训练、等待完成并验证结果的合理性。

#### Scenario: 启动训练并等待完成

**GIVEN** 所有训练参数已设置
**WHEN** 测试用例调用`click_train_button(page)`和`wait_for_training_complete(page, timeout=300000)`
**THEN** 系统应:
- 点击"开始训练"按钮
- 等待训练状态变为"completed"或"训练完成"(最多5分钟)
- 如超时,测试失败并保存超时时刻的截图
- 如训练失败,测试失败并附上失败原因

#### Scenario: 提取训练结果指标

**GIVEN** 训练已成功完成
**WHEN** 测试用例调用`extract_training_results(page)`
**THEN** 系统应提取以下结果:
- 样本外RMSE (rmse_oos)
- 样本外Hit Rate (hit_rate_oos)
- 样本外相关系数 (corr_oos)
- 选定变量数量 (num_selected_variables)
- 选定因子数 (k_factors)
- 训练用时 (training_time_seconds, 可选)
- 返回结构化的结果字典

#### Scenario: 验证结果合理性

**GIVEN** 训练结果已提取
**WHEN** 测试用例调用`verify_training_results(results, config)`
**THEN** 验证逻辑应:
- RMSE > 0 (必须为正数)
- Hit Rate在[0, 1]区间
- 相关系数在[-1, 1]区间
- 如配置为fixed,验证k_factors == config.k_factors
- 如配置为cumulative/elbow,验证k_factors > 0且合理
- 如启用变量选择,验证num_selected_variables <= 初始变量数
- 所有验证失败时,测试失败并附上详细错误信息

#### Scenario: 验证训练成功性

**GIVEN** 训练流程执行完毕
**WHEN** 测试用例调用`verify_training_success(page)`
**THEN** 验证逻辑应:
- 页面显示"训练完成"或类似成功消息
- 页面不显示"训练失败"或"Error"消息
- 结果摘要区域可见且包含关键指标
- 控制台无ERROR消息(通过console_monitor验证)

---

### Requirement: 参数组合测试矩阵

系统SHALL测试12个预定义的参数组合,覆盖不同因子数、因子选择方法、变量选择和EM配置。

#### Scenario: 基线参数测试(T1-T3)

**GIVEN** 禁用变量选择,使用固定因子数
**WHEN** 分别测试k=2, k=3, k=5三种配置
**THEN** 每个配置应:
- 10个预测变量
- 因子选择方法=fixed
- EM最大迭代=30
- 训练成功完成
- 结果合理(RMSE>0, Hit Rate∈[0,1])

#### Scenario: PCA自动选择测试(T4-T5)

**GIVEN** 禁用变量选择,使用PCA累积方差方法
**WHEN** 分别测试PCA阈值=0.85和0.90
**THEN** 每个配置应:
- 10个预测变量
- 因子选择方法=cumulative
- 自动选定因子数k(根据累积方差)
- 训练成功完成
- 选定的k在合理范围(如2-6)

#### Scenario: Elbow方法测试(T6)

**GIVEN** 禁用变量选择,使用Elbow方法
**WHEN** 测试Elbow自动选择配置
**THEN** 配置应:
- 10个预测变量
- 因子选择方法=elbow
- 自动选定因子数k(根据边际方差下降)
- 训练成功完成
- 选定的k在合理范围

#### Scenario: 变量选择+固定因子测试(T7)

**GIVEN** 启用backward变量选择,固定因子数
**WHEN** 测试backward + k=3配置
**THEN** 配置应:
- 初始12个预测变量
- 变量选择方法=backward
- 因子选择方法=fixed, k=3
- EM最大迭代=30
- 训练成功完成
- 选定变量数 <= 12(通常8-11个)
- 结果合理

#### Scenario: 变量选择+PCA测试(T8)

**GIVEN** 启用backward变量选择,使用PCA
**WHEN** 测试backward + cumulative(0.85)配置
**THEN** 配置应:
- 初始12个预测变量
- 变量选择方法=backward
- 因子选择方法=cumulative, 阈值=0.85
- 训练成功完成
- 选定变量数 <= 12
- 选定因子数k自动确定

#### Scenario: 边界条件测试(T9-T10)

**GIVEN** 测试极端因子数配置
**WHEN** 分别测试k=1(单因子)和k=10(高维)
**THEN** 每个配置应:
- 10个预测变量
- 因子选择方法=fixed
- k=1: 训练成功,结果可能精度较低但合理
- k=10: 训练成功或合理警告(如"因子数过多")

#### Scenario: EM迭代次数测试(T11-T12)

**GIVEN** 测试不同EM迭代配置
**WHEN** 分别测试max_iter=10(少)和max_iter=50(多)
**THEN** 每个配置应:
- 10个预测变量
- 因子选择方法=fixed, k=3
- max_iter=10: 可能未收敛但训练成功,记录WARNING
- max_iter=50: 充分收敛,训练成功

---

### Requirement: 测试报告生成

系统SHALL生成详细的HTML和JSON测试报告,包含所有测试用例的执行结果、性能指标和控制台输出摘要。

#### Scenario: 收集测试结果

**GIVEN** 每个测试用例执行完毕
**WHEN** 测试框架调用`TestResultCollector.collect_result()`
**THEN** 系统应记录:
- test_id (如"T1", "T2")
- 参数配置 (k, method, 变量选择等)
- 测试状态 (PASSED/FAILED)
- 训练结果指标 (rmse_oos, hit_rate_oos, corr_oos)
- 训练用时 (training_time_seconds)
- 控制台摘要 (错误数、警告数)
- 失败截图路径 (如失败)

#### Scenario: 生成HTML测试报告

**GIVEN** 所有测试执行完毕
**WHEN** 测试框架调用`generate_html_report(results, "test_results/report.html")`
**THEN** HTML报告应包含:
- 测试摘要表格(总数、通过数、失败数、执行时间)
- 参数组合对比表格(12行,包含配置和结果)
- 每个失败用例的详细信息(失败原因、截图、日志)
- 控制台输出统计(ERROR数、WARNING数)
- 性能对比图表(可选,各用例训练时间柱状图)

#### Scenario: 生成JSON汇总数据

**GIVEN** 所有测试执行完毕
**WHEN** 测试框架调用`generate_json_summary(results, "test_results/summary.json")`
**THEN** JSON文件应包含:
- test_suite元数据(名称、执行时间、总数)
- 每个测试用例的完整结果(test_id, params, status, metrics, console_summary)
- 统计信息(通过率、平均训练时间、最快/最慢用例)
- 便于后续分析和CI/CD集成

---

### Requirement: 快速测试模式

系统SHALL提供快速测试模式,仅执行3个核心用例,用于快速验证和回归测试。

#### Scenario: 标记快速测试用例

**GIVEN** 使用pytest marker机制
**WHEN** 测试用例定义时标记`@pytest.mark.quick`
**THEN** 系统应:
- T1(k=2, fixed)标记为quick
- T2(k=3, fixed)标记为quick
- T7(backward + k=3)标记为quick
- 其他用例不标记quick

#### Scenario: 运行快速测试

**GIVEN** 需要快速验证train_ref功能
**WHEN** 执行命令`pytest -m quick tests/ui/test_training_param_combinations.py`
**THEN** 系统应:
- 仅执行3个标记为quick的用例
- 执行时间 < 15分钟
- 生成快速测试报告(test_results/quick_report.html)
- 如所有用例通过,确认train_ref核心功能正常

#### Scenario: 运行完整测试

**GIVEN** 需要全面验证所有参数组合
**WHEN** 执行命令`pytest tests/ui/test_training_param_combinations.py`
**THEN** 系统应:
- 执行所有12个测试用例
- 执行时间预计36-45分钟
- 生成完整测试报告
- 统计通过率(目标 >= 80%,即至少10个通过)

---

### Requirement: 测试可靠性与重试机制

系统SHALL提供测试重试机制和智能等待策略,确保测试稳定性和可重复性。

#### Scenario: 失败自动重试

**GIVEN** 配置pytest-rerunfailures插件
**WHEN** 某个测试用例首次执行失败(如UI加载超时)
**THEN** 系统应:
- 自动重试失败的用例(最多1次)
- 如重试成功,标记为PASSED但记录"通过重试"
- 如重试仍失败,标记为FAILED
- 减少因临时网络或UI加载问题导致的flaky测试

#### Scenario: 智能等待元素

**GIVEN** UI元素可能异步加载
**WHEN** 测试用例调用`wait_for_element(page, selector, timeout)`
**THEN** 系统应:
- 使用Playwright的`wait_for_selector()`
- 默认超时30秒(可配置)
- 轮询检查元素是否出现
- 如超时,抛出清晰的错误消息(包含selector和当前页面状态)

#### Scenario: 训练完成智能等待

**GIVEN** 训练时间不确定(可能1-5分钟)
**WHEN** 测试用例调用`wait_for_training_complete(page, timeout=300000)`
**THEN** 系统应:
- 轮询检查训练状态元素(如"训练完成"文本)
- 超时5分钟
- 每10秒检查一次(避免CPU占用)
- 如检测到"训练失败",立即失败并返回失败原因
- 如超时,截图并失败

---

### Requirement: 测试环境配置

系统SHALL提供灵活的测试环境配置,支持不同的Streamlit URL和数据文件路径。

#### Scenario: 环境变量配置

**GIVEN** 测试环境可能不同(本地/Docker/CI)
**WHEN** 测试启动时读取配置
**THEN** 系统应支持:
- 环境变量`STREAMLIT_URL`(默认http://localhost:8501)
- 环境变量`TEST_DATA_FILE`(默认data/经济数据库1017.xlsx)
- 环境变量`TEST_TIMEOUT`(默认300秒)
- 通过`.env.test`文件配置

#### Scenario: pytest配置文件

**GIVEN** pytest.ini配置文件存在
**WHEN** pytest读取配置
**THEN** 配置应包含:
- HTML报告输出路径(--html=test_results/report.html)
- 失败重试次数(--reruns=1)
- 并行执行workers(可选,--numprocesses=2)
- marker定义(quick, full)

---
