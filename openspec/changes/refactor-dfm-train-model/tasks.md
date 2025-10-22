# DFM模型训练模块完全重构实施任务清单

## 前置准备

### 0.1 Baseline生成

- [ ] 0.1.1 运行train_model生成baseline结果
  - 准备5个典型测试案例（不同配置组合）
  - 运行train_model并保存所有中间结果
  - 保存到tests/consistency/baseline/

- [ ] 0.1.2 创建feature分支
  - 从main分支创建feature/refactor-train-model
  - 在该分支进行所有开发工作

## 1. 变量选择层实现（Week 1-3）

### 1.1 后向选择器

- [ ] 1.1.1 实现BackwardSelector类（selection/backward_selector.py）
  - 后向逐步变量剔除算法
  - 支持RMSE和Hit Rate作为优化目标
  - 使用PrecomputeEngine优化性能
  - 返回SelectionResult对象

- [ ] 1.1.2 实现SelectionEngine类（selection/selection_engine.py）
  - 统一的变量选择入口
  - 支持'none', 'backward'两种方法
  - 返回标准化的SelectionResult对象

- [ ] 1.1.3 编写变量选择单元测试（tests/selection/）
  - 测试后向选择逻辑正确性
  - 测试边界情况（单变量、无改进停止）
  - 对比baseline验证数值一致性
  - 覆盖率 > 85%

## 2. 训练协调层实现（Week 4-7）

### 2.1 流程管道

- [ ] 2.1.1 实现TrainingPipeline类（training/pipeline.py）
  - Pipeline设计模式实现两阶段流程
  - 阶段1：变量选择（固定k=块数）
  - 阶段2：因子数选择（PCA/Elbow/Fixed）
  - 支持progress_callback进度反馈

- [ ] 2.1.2 实现因子数选择方法
  - PCA累积方差法（cumulative）
  - Elbow方法（elbow）
  - 固定值方法（fixed）

### 2.2 训练器

- [ ] 2.2.1 完善DFMTrainer类（training/trainer.py）
  - 集成TrainingPipeline
  - 实现完整的train()方法
  - 支持progress_callback进度反馈
  - 实现save_results()结果保存

- [ ] 2.2.2 补充配置管理（training/config.py）
  - 添加SelectionConfig
  - 完善ModelConfig（factor_selection_method等字段）
  - 添加配置验证方法

- [ ] 2.2.3 编写训练流程单元测试（tests/training/）
  - 测试两阶段流程正确性
  - 测试各种因子选择方法
  - 验证配置验证逻辑
  - 覆盖率 > 80%

## 3. 分析输出层实现（Week 8-11）

### 3.1 分析报告器

- [ ] 3.1.1 实现AnalysisReporter类（analysis/reporter.py）
  - generate_pca_report(): PCA方差贡献分析
  - generate_contribution_report(): 贡献度分解
  - generate_r2_report(): 个体R²和行业R²
  - Excel多Sheet报告生成

- [ ] 3.1.2 实现分析工具函数（analysis/analysis_utils.py）
  - calculate_metrics_with_lagged_target()
  - calculate_factor_contributions()
  - calculate_individual_variable_r2()
  - calculate_industry_r2()

- [ ] 3.1.3 实现报告生成器（analysis/generate_report.py）
  - generate_report_with_params()
  - 文件路径管理
  - 参数化报告生成

### 3.2 可视化器

- [ ] 3.2.1 实现ResultVisualizer类（analysis/visualizer.py）
  - plot_forecast_vs_actual(): 预测vs实际对比图
  - plot_residuals(): 残差分析图
  - plot_pca_variance(): PCA方差贡献图
  - plot_factor_loadings(): 因子载荷热力图
  - 支持Plotly和Matplotlib两种后端

- [ ] 3.2.2 编写分析输出单元测试（tests/analysis/）
  - 测试报告生成逻辑
  - 验证数值计算正确性
  - 测试可视化图表生成
  - 覆盖率 > 80%

## 4. 优化层实现（Week 12-13）

### 4.1 预计算引擎

- [ ] 4.1.1 实现PrecomputeEngine类（optimization/precompute.py）
  - 预计算协方差矩阵
  - 预计算数据标准化参数
  - 返回PrecomputedContext对象

- [ ] 4.1.2 实现优化评估器（optimization/optimized_evaluator.py）
  - 使用预计算上下文的评估器
  - 批量优化评估
  - 内存优化版本

- [ ] 4.1.3 编写优化层单元测试（tests/optimization/）
  - 验证预计算结果正确性
  - 测试缓存命中逻辑
  - 性能基准测试

## 5. 数值一致性验证（Week 14-16）

### 5.1 对比测试框架

- [ ] 5.1.1 创建对比测试基类（tests/consistency/base.py）
  - 加载baseline结果工具函数
  - 设置相同随机种子
  - 提供结果对比工具函数

### 5.2 核心算法对比

- [ ] 5.2.1 参数估计对比测试（tests/consistency/test_parameter_estimation.py）
  - 对比EM算法估计的A, Q, H, R矩阵
  - 验证L2范数差异 < 1e-6
  - 对比收敛迭代次数

- [ ] 5.2.2 状态估计对比测试（tests/consistency/test_state_estimation.py）
  - 对比卡尔曼滤波预测步骤结果
  - 对比平滑步骤结果
  - 验证每个时间点差异 < 1e-6

- [ ] 5.2.3 评估指标对比测试（tests/consistency/test_metrics.py）
  - 对比RMSE（容差 < 1e-4）
  - 对比Hit Rate（容差 < 1%）
  - 对比相关系数（容差 < 1e-4）

### 5.3 端到端对比

- [ ] 5.3.1 完整训练流程对比（tests/consistency/test_end_to_end.py）
  - 使用5个测试案例
  - 对比最终预测结果
  - 对比所有输出文件（PCA、贡献度等）

- [ ] 5.3.2 不同配置下的对比
  - 测试不同因子数（k=2,3,4,5）
  - 测试不同选择方法（cumulative, elbow, fixed）
  - 测试不同训练集划分

- [ ] 5.3.3 性能基准测试（tests/consistency/test_performance.py）
  - 对比执行时间（目标：差异 < 10%）
  - 对比内存占用
  - 生成性能报告

## 6. UI层迁移（Week 17-18）

### 6.1 更新模型训练页面

- [ ] 6.1.1 修改model_training_page.py
  - 删除train_model导入
  - 使用train_ref导入
  - 构建TrainingConfig对象
  - 调用DFMTrainer.train()

- [ ] 6.1.2 适配progress_callback
  - 确保回调消息格式一致
  - 保持训练日志显示逻辑不变

- [ ] 6.1.3 适配结果显示
  - 确保结果字典结构兼容
  - 更新结果路径存储逻辑
  - 测试所有结果展示组件

### 6.2 更新训练状态组件

- [ ] 6.2.1 更新TrainingStatusComponent（ui/components/dfm/train_model/training_status.py）
  - 支持train_ref的状态更新
  - 保持状态键名一致

- [ ] 6.2.2 更新变量选择组件（ui/components/dfm/train_model/variable_selection.py）
  - 适配train_ref接口
  - 保持UI交互不变

### 6.3 UI集成测试

- [ ] 6.3.1 手动测试所有UI路径
  - 测试无变量选择 + 固定因子数
  - 测试后向变量选择 + PCA选择
  - 测试后向变量选择 + Elbow选择
  - 测试不同迭代次数设置

- [ ] 6.3.2 测试边界情况
  - 测试配置错误提示
  - 测试训练中断处理
  - 测试结果保存和加载

## 7. 文档更新（Week 19）

### 7.1 项目文档

- [ ] 7.1.1 更新CLAUDE.md
  - 更新DFM模块架构说明（指向train_ref）
  - 删除train_model相关说明
  - 更新关键文件路径表格

- [ ] 7.1.2 更新train_ref/README.md
  - 移除"开发中"标记
  - 添加完整使用示例
  - 添加API文档链接

### 7.2 代码文档

- [ ] 7.2.1 补充Docstring
  - 所有公共类和方法添加docstring
  - 使用Google风格
  - 添加参数类型和返回值说明

- [ ] 7.2.2 添加代码示例
  - 在facade.py中添加使用示例
  - 在README.md中添加快速开始指南

## 8. 部署上线（Week 20-21）

### 8.1 部署准备

- [ ] 8.1.1 代码审查
  - 审查所有新增代码
  - 检查代码风格一致性
  - 运行静态代码分析

- [ ] 8.1.2 测试覆盖率检查
  - 运行pytest --cov
  - 确保核心算法层覆盖率 > 90%
  - 确保其他层覆盖率 > 80%

- [ ] 8.1.3 性能验证
  - 运行性能基准测试
  - 确保执行时间增加 < 10%
  - 生成性能对比报告

### 8.2 用户测试

- [ ] 8.2.1 准备测试计划
  - 列出测试场景清单
  - 准备测试数据集
  - 设计验证表格

- [ ] 8.2.2 用户验收测试
  - 提供train_ref使用指南
  - 收集用户反馈
  - 记录发现的问题

- [ ] 8.2.3 修复反馈问题
  - 优先级排序
  - 修复高优先级问题
  - 回归测试

### 8.3 正式部署

- [ ] 8.3.1 部署到测试环境
  - 验证部署成功
  - 运行冒烟测试
  - 检查日志无异常

- [ ] 8.3.2 部署到生产环境
  - 执行部署脚本
  - 验证功能正常
  - 通知所有用户

## 9. 清理工作（Week 22）

### 9.1 代码清理

- [ ] 9.1.1 删除train_model模块
  - 删除`dashboard/DFM/train_model/`目录（15,049行）
  - 删除相关导入
  - 清理未使用的依赖

- [ ] 9.1.2 移除冗余代码
  - 删除废弃的工具函数
  - 删除未使用的导入
  - 清理注释掉的代码

- [ ] 9.1.3 更新依赖
  - 检查并移除train_model特有的依赖
  - 更新requirements.txt

### 9.2 归档与发布

- [ ] 9.2.1 创建归档标签
  - 创建`v2.0.0-train-ref`标签
  - 添加Release说明文档

- [ ] 9.2.2 更新CHANGELOG
  - 记录重大变更
  - 记录性能改进
  - 记录API变化

### 9.3 文档最终更新

- [ ] 9.3.1 更新所有文档中的引用
  - 删除train_model相关说明
  - 统一使用train_ref术语
  - 更新架构图

- [ ] 9.3.2 发布完成公告
  - 编写重构完成总结
  - 发布到团队文档
  - 更新项目README

## 验收标准（完全重构）

所有任务完成后，必须满足以下标准：

1. 所有单元测试通过（总覆盖率 > 80%，核心层 > 90%）
2. 对比测试数值差异在容差内（RMSE < 1e-4, Hit Rate < 1%）
3. 性能测试执行时间增加 < 10%
4. UI功能完全一致，无回归问题
5. train_model代码完全删除，无遗留引用
6. 文档完整更新，无train_model相关内容
7. 代码审查通过，符合项目规范
8. 用户测试反馈良好，无阻塞性问题

## 时间估算（完全重构）

| 阶段 | 任务 | 时间 |
|------|------|------|
| 0 | 前置准备（Baseline生成） | 1周 |
| 1 | 变量选择层实现 | 3周 |
| 2 | 训练协调层实现 | 4周 |
| 3 | 分析输出层实现 | 4周 |
| 4 | 优化层实现 | 2周 |
| 5 | 数值一致性验证 | 3周 |
| 6 | UI层迁移 | 2周 |
| 7 | 文档更新 | 1周 |
| 8 | 部署上线 | 2周 |
| 9 | 清理工作 | 1周 |
| **总计** | | **22周** |

## 关键成功因素

1. **数值一致性**：必须与baseline结果完全一致（误差 < 1e-6）
2. **高测试覆盖**：核心算法层 > 90%，保证质量
3. **分阶段验证**：每完成一个模块都要进行对比测试
4. **彻底删除**：train_model代码必须完全删除，不保留任何遗留
