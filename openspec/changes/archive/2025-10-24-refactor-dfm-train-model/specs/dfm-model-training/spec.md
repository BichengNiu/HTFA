# DFM模型训练能力规范

## ADDED Requirements

### Requirement: 统一训练接口

系统SHALL提供统一的DFM模型训练接口，支持配置化训练流程。

#### Scenario: 基本训练流程

- **WHEN** 用户提供数据配置、模型配置和训练配置
- **THEN** 系统执行完整的训练流程并返回训练结果
- **AND** 结果包含模型参数、评估指标、预测值、实际值

#### Scenario: 配置对象验证

- **WHEN** 用户提供的配置缺少必填字段（如data_path、target_variable）
- **THEN** 系统抛出配置验证错误，明确指出缺失字段
- **AND** 不执行任何训练操作

#### Scenario: 可重现性保证

- **WHEN** 用户提供相同的数据、配置和随机种子
- **THEN** 系统返回完全一致的训练结果
- **AND** 数值误差小于1e-6

### Requirement: 两阶段训练流程

系统SHALL支持两阶段训练流程：阶段1为变量选择，阶段2为因子数选择。

#### Scenario: 阶段1变量选择

- **WHEN** 用户启用变量选择（enable_variable_selection=True）
- **THEN** 系统固定因子数为数据块数，执行后向逐步变量选择
- **AND** 优化目标依次为Hit Rate和RMSE
- **AND** 返回最优变量子集

#### Scenario: 阶段1跳过变量选择

- **WHEN** 用户禁用变量选择（enable_variable_selection=False）
- **THEN** 系统使用全部选中变量
- **AND** 直接进入阶段2

#### Scenario: 阶段2因子数选择（PCA累积方差）

- **WHEN** 用户选择因子选择方法为'cumulative'且设置PCA阈值为0.9
- **THEN** 系统基于阶段1的变量，使用PCA计算累积方差贡献
- **AND** 选择使累积方差 >= 90%的最小因子数
- **AND** 使用该因子数训练最终模型

#### Scenario: 阶段2因子数选择（Elbow方法）

- **WHEN** 用户选择因子选择方法为'elbow'且设置Elbow阈值为0.1
- **THEN** 系统计算边际方差贡献下降率
- **AND** 选择第一个下降率 < 10%的因子数
- **AND** 使用该因子数训练最终模型

#### Scenario: 阶段2固定因子数

- **WHEN** 用户选择因子选择方法为'fixed'且设置k_factors=4
- **THEN** 系统直接使用k=4训练最终模型
- **AND** 不执行PCA或Elbow分析

### Requirement: 数据处理流程

系统SHALL支持多频率数据加载、预处理和对齐。

#### Scenario: 加载Excel数据

- **WHEN** 用户提供Excel文件路径和目标变量名称
- **THEN** 系统加载指定Sheet中的目标变量
- **AND** 解析时间索引并转换为周度频率（W-FRI）
- **AND** 验证数据格式有效性

#### Scenario: 缺失值处理

- **WHEN** 数据中存在缺失值
- **THEN** 系统使用线性插值方法填充缺失值
- **AND** 记录缺失值数量和位置到日志

#### Scenario: 数据集划分

- **WHEN** 用户指定训练集结束日期和验证集起止日期
- **THEN** 系统按日期划分训练集和验证集
- **AND** 确保训练集和验证集时间上不重叠
- **AND** 验证集至少包含4个时间点

### Requirement: 模型评估指标

系统SHALL计算样本内（IS）和样本外（OOS）评估指标。

#### Scenario: RMSE计算

- **WHEN** 模型预测完成
- **THEN** 系统计算训练集RMSE（样本内）
- **AND** 计算验证集RMSE（样本外）
- **AND** RMSE = sqrt(mean((预测值 - 实际值)^2))

#### Scenario: Hit Rate计算

- **WHEN** 模型预测完成
- **THEN** 系统计算训练集Hit Rate（样本内）
- **AND** 计算验证集Hit Rate（样本外）
- **AND** Hit Rate = (预测方向正确的次数) / 总次数
- **AND** 预测方向正确定义为：(预测值 - 前值) * (实际值 - 前值) > 0

#### Scenario: 相关系数计算

- **WHEN** 模型预测完成
- **THEN** 系统计算预测值与实际值的Pearson相关系数
- **AND** 分别计算样本内和样本外相关系数

### Requirement: 训练进度反馈

系统SHALL在训练过程中提供实时进度反馈。

#### Scenario: 进度回调

- **WHEN** 用户提供progress_callback函数
- **THEN** 系统在关键节点调用回调函数
- **AND** 回调消息包含阶段标识（[TRAIN_PROGRESS]）
- **AND** 关键节点包括：数据加载完成、EM迭代进度、阶段1完成、阶段2完成

#### Scenario: EM迭代进度

- **WHEN** EM算法执行迭代
- **THEN** 系统每次迭代后报告当前迭代次数和对数似然值
- **AND** 收敛时报告最终迭代次数和收敛状态

#### Scenario: 无回调函数

- **WHEN** 用户未提供progress_callback
- **THEN** 系统静默训练，不输出进度信息
- **AND** 训练逻辑不受影响

### Requirement: 结果输出

系统SHALL输出完整的训练结果，包括模型参数、预测值、评估指标和分析报告。

#### Scenario: 基本结果输出

- **WHEN** 训练完成
- **THEN** 结果对象包含以下字段：
  - `params`: 模型参数（A, Q, H, R）
  - `forecast_is`: 样本内预测值
  - `forecast_oos`: 样本外预测值
  - `actual_is`: 样本内实际值
  - `actual_oos`: 样本外实际值
  - `metrics`: 评估指标字典（is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, is_corr, oos_corr）

#### Scenario: PCA分析结果

- **WHEN** 使用PCA或Elbow方法选择因子数
- **THEN** 结果对象包含PCA分析DataFrame
- **AND** DataFrame包含列：因子索引、特征值、方差贡献率、累积方差贡献率

#### Scenario: 贡献度分解结果

- **WHEN** 训练完成
- **THEN** 结果对象包含贡献度分解DataFrame
- **AND** DataFrame包含各指标对目标变量的贡献度
- **AND** 贡献度 = 因子载荷 * 因子值 * 标准差

#### Scenario: 结果保存

- **WHEN** 用户调用save_results()方法
- **THEN** 系统将结果保存到指定目录
- **AND** 保存文件包括：模型参数（.pkl）、预测结果（.csv）、评估指标（.json）、分析报告（.csv）

### Requirement: 数值一致性验证

系统SHALL通过对比测试验证与原train_model模块的数值一致性。

#### Scenario: 参数估计一致性

- **WHEN** 使用相同数据、配置和随机种子
- **THEN** EM算法估计的模型参数（A, Q, H, R）与原模块相比
- **AND** 每个参数矩阵的L2范数差异 < 1e-6

#### Scenario: 预测结果一致性

- **WHEN** 使用相同数据、配置和随机种子
- **THEN** 卡尔曼滤波预测值与原模块相比
- **AND** 每个时间点的预测值差异 < 1e-6

#### Scenario: 评估指标一致性

- **WHEN** 使用相同数据、配置和随机种子
- **THEN** RMSE、Hit Rate、相关系数与原模块相比
- **AND** RMSE差异 < 1e-4
- **AND** Hit Rate差异 < 1%
- **AND** 相关系数差异 < 1e-4

### Requirement: 错误处理

系统SHALL对常见错误情况提供清晰的错误提示。

#### Scenario: 数据文件不存在

- **WHEN** 用户提供的data_path文件不存在
- **THEN** 系统抛出FileNotFoundError
- **AND** 错误消息明确指出文件路径

#### Scenario: 目标变量不存在

- **WHEN** 用户指定的target_variable在数据中不存在
- **THEN** 系统抛出ValueError
- **AND** 错误消息列出可用变量列表

#### Scenario: 验证集过小

- **WHEN** 验证集时间点数量 < 4
- **THEN** 系统抛出ValueError
- **AND** 错误消息说明最小验证集要求

#### Scenario: EM算法不收敛

- **WHEN** EM算法达到最大迭代次数仍未收敛
- **THEN** 系统记录警告日志
- **AND** 返回当前最优参数
- **AND** 结果对象标记收敛状态为False

### Requirement: UI集成接口

系统SHALL提供与Streamlit UI无缝集成的接口。

#### Scenario: 状态管理兼容

- **WHEN** UI通过UnifiedStateManager设置训练状态
- **THEN** 训练器读取状态并更新训练进度
- **AND** 训练完成后更新状态为'completed'
- **AND** 错误时更新状态为'failed'

#### Scenario: 异步训练支持

- **WHEN** UI在后台线程启动训练
- **THEN** 训练器通过progress_callback同步进度
- **AND** 训练结果存储到UnifiedStateManager
- **AND** UI可通过轮询检查训练完成状态

#### Scenario: 配置UI映射

- **WHEN** UI收集用户输入的训练参数
- **THEN** 系统提供工具函数将UI参数转换为配置对象
- **AND** 转换函数验证参数有效性
- **AND** 提供默认值填充机制
