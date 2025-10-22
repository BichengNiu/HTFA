# DFM模型训练模块精简重构实施任务清单（方案B）

## 架构精简说明

**采用方案B：精简5层架构**
- 代码量：10,800行（vs train_model减少28%）
- 目录数：5个（vs 方案A的7个）
- 文件数：12个（vs 方案A的21个）
- 工作时间：19周（vs 方案A的22周，节省3周）

**关键精简**：
- ❌ 删除evaluation/目录 → 合并到training/trainer.py
- ❌ 删除selection_engine.py → 直接使用BackwardSelector
- ❌ 删除pipeline.py → 合并到training/trainer.py
- ❌ 删除optimized_evaluator.py → 冗余
- ❌ 删除interfaces.py和wrapper.py → 过度抽象
- ❌ 删除generate_report.py → 合并到analysis/reporter.py

## 重要原则

**train_model模块处理策略**：
- ⚠️ 在所有重构工作、一致性验证、用户测试完成前，**不得删除**train_model模块
- 📖 train_model代码仅作为**参考**，不得被调用或修改
- 🔒 所有新功能必须在train_ref中实现，不得依赖train_model
- ✅ 只有在Phase 8（部署上线）完全完成后，才在Phase 9删除train_model

**分支管理策略**：
- 🌿 所有重构工作必须在feature/refactor-train-model分支进行
- 🚫 重构期间不得合并到main分支
- ✅ 只有在Phase 9（清理工作）完全完成后，才合并到main分支
- 📌 合并前需要完整的代码审查和最终验证

## 前置准备

### 0.1 Baseline生成

- [ ] 0.1.1 运行train_model生成baseline结果
  - 准备5个典型测试案例（不同配置组合）
  - 运行train_model并保存所有中间结果
  - 保存到tests/consistency/baseline/

- [ ] 0.1.2 创建并切换到feature分支
  - 从main分支创建feature/refactor-train-model
  - 立即切换到该分支
  - **所有后续开发工作都在此分支进行**
  - 保持main分支的train_model作为稳定参考
  - 只有Phase 9完成后才合并回main

## 1. 变量选择层实现（Week 1-2.5）⏱️ 节省0.5周

### 1.1 后向选择器

- [x] 1.1.1 实现BackwardSelector类（selection/backward_selector.py）
  - 后向逐步变量剔除算法
  - 支持RMSE和Hit Rate作为优化目标
  - 直接使用PrecomputeEngine（来自utils/precompute.py）
  - 返回SelectionResult对象（简单dataclass）
  - **注意**: 不需要SelectionEngine抽象层，直接实例化BackwardSelector
  - **已完成**: 实现了339行的BackwardSelector类，包含完整的后向选择逻辑
  - **已完成**: 删除了selection_engine.py，更新了__init__.py导出

- [ ] 1.1.2 编写变量选择单元测试（tests/selection/）
  - 测试后向选择逻辑正确性
  - 测试边界情况（单变量、无改进停止）
  - 对比baseline验证数值一致性
  - 覆盖率 > 85%

## 2. 训练协调层实现（Week 3-5.5）⏱️ 节省1周

### 2.1 模型评估器（合并到trainer.py）

- [x] 2.1.1 实现ModelEvaluator类（training/trainer.py内部类）
  - calculate_rmse(): RMSE计算
  - calculate_hit_rate(): Hit Rate计算
  - calculate_correlation(): 相关系数计算
  - evaluate(): 样本内和样本外评估
  - **注意**: 不单独建evaluation/目录，直接作为trainer.py的内部类
  - **已完成**: 实现了ModelEvaluator类(约230行)，包含完整的指标计算逻辑

### 2.2 主训练器（合并pipeline逻辑）

- [x] 2.2.1 实现DFMTrainer类（training/trainer.py）基础框架
  - __init__(): 初始化，包含环境设置和evaluator实例化
  - train(): 完整两阶段训练流程框架
  - 数据类定义: EvaluationMetrics, DFMModelResult, TrainingResult
  - **已完成**: 实现了463行的trainer.py基础框架
  - **待完成**: 补充完整的训练流程逻辑(变量选择、因子数选择、模型训练)

- [x] 2.2.2 实现环境初始化（training/trainer.py::_init_environment）
  - 多线程BLAS配置（OMP_NUM_THREADS, MKL_NUM_THREADS等）
  - 全局静默模式控制（环境变量DFM_SILENT_WARNINGS）
  - 随机种子设置（numpy, random, SEED=42）
  - **已完成**: _init_environment方法已实现

- [ ] 2.2.3 补充配置管理（training/config.py）
  - TrainingConfig: 完整训练配置（包含selection和model参数）
  - 配置验证方法（validate()）
  - 参数默认值设置

- [ ] 2.2.4 编写训练层单元测试（tests/training/）
  - 测试两阶段流程正确性
  - 测试各种因子选择方法（fixed/cumulative/elbow）
  - 测试ModelEvaluator的指标计算
  - 验证配置验证逻辑
  - 测试可重现性（相同种子相同结果）
  - 覆盖率 > 80%

## 3. 分析输出层实现（Week 6-9）⏱️ 节省0.5周

### 3.1 分析报告器（合并generate_report逻辑）

- [ ] 3.1.1 实现AnalysisReporter类（analysis/reporter.py）
  - generate_report_with_params(): 参数化报告生成（合并原generate_report.py）
  - generate_pca_report(): PCA方差贡献分析
  - generate_contribution_report(): 贡献度分解
  - generate_r2_report(): 个体R²和行业R²
  - _format_excel_sheet(): Excel格式化工具函数
  - Excel多Sheet报告生成
  - **注意**: 不单独创建generate_report.py，合并到reporter.py

- [ ] 3.1.2 实现分析工具函数（analysis/analysis_utils.py）
  - calculate_metrics_with_lagged_target(): 带滞后目标的指标计算
  - calculate_factor_contributions(): 因子贡献度分解
  - calculate_individual_variable_r2(): 个体变量R²
  - calculate_industry_r2(): 行业聚合R²
  - calculate_monthly_friday_metrics(): 月度周五指标计算
  - calculate_factor_industry_r2(): 因子-行业交叉R²
  - calculate_factor_type_r2(): 因子-类型R²
  - calculate_pca_variance(): PCA方差贡献计算

### 3.2 可视化器

- [ ] 3.2.1 实现ResultVisualizer类（analysis/visualizer.py）
  - plot_forecast_vs_actual(): 预测vs实际对比图
  - plot_residuals(): 残差分析图
  - plot_pca_variance(): PCA方差贡献图
  - plot_factor_loadings(): 因子载荷热力图
  - plot_industry_vs_driving_factor(): 行业vs驱动因子对比图
  - plot_aligned_loading_comparison(): 因子载荷对比图
  - plot_factor_loading_clustermap(): 因子载荷聚类图
  - 支持Plotly和Matplotlib两种后端

- [ ] 3.2.2 编写分析输出单元测试（tests/analysis/）
  - 测试报告生成逻辑
  - 验证数值计算正确性
  - 测试可视化图表生成
  - 覆盖率 > 80%

## 4. 工具层实现（Week 10-10.5）⏱️ 节省1周

### 4.1 工具模块（合并optimization到utils）

- [ ] 4.1.1 实现PrecomputeEngine类（utils/precompute.py）
  - 预计算协方差矩阵
  - 预计算数据标准化参数
  - 返回PrecomputedContext对象（简单dataclass）
  - **注意**: 移到utils/而非optimization/，简化目录结构

- [ ] 4.1.2 实现数据工具（utils/data_utils.py）
  - load_data(): Excel数据加载
  - preprocess_data(): 数据预处理和对齐
  - split_train_validation(): 数据集划分
  - verify_alignment(): 数据对齐验证（合并原verify_alignment.py）

- [ ] 4.1.3 编写工具层单元测试（tests/utils/）
  - 验证预计算结果正确性
  - 测试缓存命中逻辑（cache.py已实现）
  - 测试数据加载和预处理
  - 覆盖率 > 80%

## 5. 数值一致性验证（Week 11-13）⏱️ 时间提前

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

- [ ] 5.3.4 实现简化版性能分析工具（可选，utils/performance.py）
  - 基本性能指标收集（PerformanceMetrics）
  - 执行时间统计
  - 内存占用统计
  - 性能对比报告生成
  - **注意**: 仅实现必要功能，不需要完整复现train_model的performance_benchmark.py

## 6. UI层迁移（Week 14-15）⏱️ 时间提前

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

## 7. 文档更新（Week 16）⏱️ 时间提前

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

## 8. 部署上线（Week 17-18）⏱️ 时间提前

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

## 9. 清理与合并（Week 19）⏱️ 总时间19周，节省3周

### 9.1 代码清理

- [ ] 9.1.1 **删除train_model模块**（仅在Phase 8完全完成后执行）
  - ⚠️ 前置条件：Phase 8所有测试和部署已完成
  - ⚠️ 前置条件：用户验收测试通过，无阻塞性问题
  - ⚠️ 前置条件：生产环境运行稳定至少1周
  - 删除`dashboard/DFM/train_model/`目录（15,049行）
  - 删除相关导入（检查所有文件）
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

### 9.4 合并feature分支到main

- [ ] 9.4.1 合并前最终检查
  - 运行完整测试套件（单元测试 + 集成测试）
  - 运行数值一致性验证
  - 运行性能基准测试
  - 确认train_model已完全删除
  - 确认所有文档已更新

- [ ] 9.4.2 代码审查
  - 提交完整的代码审查请求
  - 审查所有新增和修改的代码
  - 确保符合项目编码规范
  - 修复审查中发现的问题

- [ ] 9.4.3 合并到main分支
  - 将feature/refactor-train-model合并到main
  - 创建合并提交（包含完整的变更说明）
  - 打标签：v2.0.0-train-ref
  - 推送到远程仓库
  - 删除feature分支（本地和远程）

- [ ] 9.4.4 发布后验证
  - 从main分支重新部署
  - 运行冒烟测试
  - 确认所有功能正常
  - 监控系统运行状态

## 验收标准（完全重构）

所有任务完成后，必须满足以下标准：

**Phase 1-7 验收标准**：
1. 所有单元测试通过（总覆盖率 > 80%，核心层 > 90%）
2. 对比测试数值差异在容差内（RMSE < 1e-4, Hit Rate < 1%）
3. 性能测试执行时间增加 < 10%
4. UI功能完全一致，无回归问题
5. train_model模块保持完整，仅作参考

**Phase 8 验收标准**：
6. 用户验收测试通过，无阻塞性问题
7. 生产环境运行稳定至少1周
8. 所有功能与train_model完全一致

**Phase 9 验收标准**：
9. train_model代码完全删除，无遗留引用
10. 文档完整更新，无train_model相关内容
11. 代码审查通过，符合项目规范
12. feature分支成功合并到main
13. 发布标签v2.0.0-train-ref已创建

## 时间估算（精简重构 - 方案B）

| 阶段 | 任务 | 原计划 | 方案B | 节省 |
|------|------|-------|-------|------|
| 0 | 前置准备（Baseline生成） | 1周 | 1周 | - |
| 1 | 变量选择层实现 | 3周 | 2.5周 | 0.5周 |
| 2 | 训练协调层实现（合并evaluation+pipeline） | 4周 | 3周 | 1周 |
| 3 | 分析输出层实现（合并generate_report） | 4周 | 3.5周 | 0.5周 |
| 4 | 工具层实现（合并optimization，删除interfaces） | 2周 | 1周 | 1周 |
| 5 | 数值一致性验证 | 3周 | 3周 | - |
| 6 | UI层迁移 | 2周 | 2周 | - |
| 7 | 文档更新 | 1周 | 1周 | - |
| 8 | 部署上线 | 2周 | 2周 | - |
| 9 | 清理与合并 | 1周 | 1周 | - |
| **总计** | | **23周** | **19周** | **4周** |

**节省时间说明**：
- ✅ Phase 1: 删除selection_engine.py（省0.5周）
- ✅ Phase 2: 合并evaluation+pipeline到trainer.py（省1周）
- ✅ Phase 3: 合并generate_report到reporter.py（省0.5周）
- ✅ Phase 4: 删除interfaces/wrapper，合并optimization到utils（省1周）
- ✅ 累计节省：3周代码编写 + 1周测试和集成

## 关键成功因素

1. **数值一致性**：必须与baseline结果完全一致（误差 < 1e-6）
2. **高测试覆盖**：核心算法层 > 90%，保证质量
3. **分阶段验证**：每完成一个模块都要进行对比测试
4. **保留参考代码**：train_model在Phase 1-8期间保持完整，仅作参考，不被调用
5. **分支隔离**：所有开发在feature分支进行，验证完成后才合并
6. **谨慎删除**：只有在生产环境稳定运行后才删除train_model
