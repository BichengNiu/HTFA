# DFM训练模块完全重构设计文档

## Context

当前HTFA项目中存在两个DFM训练实现：
- `train_model/`：原始实现（15,049行，24个文件）
- `train_ref/`：重构版本（2,673行，实际功能仅覆盖40%）

经过深度代码分析发现，train_ref的核心算法层已完成，但关键功能模块缺失：
- 变量选择层（selection/）- 1,200行缺失
- 训练协调层（training/trainer.py, pipeline.py）- 3,500行缺失
- 分析输出层（analysis/）- 3,300行缺失
- 预计算引擎（optimization/precompute.py）- 800行缺失

**采用完全重构**：重新实现所有缺失模块，彻底解决架构问题。

**train_model处理策略**：
- ⚠️ 在所有重构工作、一致性验证、用户测试完成前，**不得删除**train_model模块
- 📖 train_model代码仅作为**参考**，不得被调用或修改
- ✅ 只有在生产环境稳定运行后，才在Phase 9删除train_model

**分支管理策略**：
- 🌿 所有重构工作在`feature/refactor-train-model`分支进行
- 🚫 重构期间不得合并到main分支
- ✅ 只有在Phase 9（清理与合并）完全完成后，才合并到main

## Goals / Non-Goals

### Goals

1. **功能等价**：重新实现与train_model完全相同的功能
2. **数值一致**：相同输入下，输出结果数值误差 < 1e-6
3. **架构清晰**：分层明确，职责单一，易于维护
4. **代码质量**：遵循KISS、DRY、YAGNI、SOC、SRP原则
5. **高测试覆盖**：核心算法层 > 90%，其他层 > 80%
6. **彻底消除技术债**：不保留任何遗留代码

### Non-Goals

1. **保留兼容**：不提供train_model与train_ref的兼容层或回退机制
2. **性能优化**：不引入超出train_model的新优化（如GPU加速）
3. **功能扩展**：不添加train_model不支持的新功能（如增量训练）
4. **更改数据格式**：保持输入输出格式与train_model一致

## Decisions

### 1. 分层架构保持不变

采用train_ref已确立的分层设计：

```
train_ref/
├── core/           # 核心算法（卡尔曼滤波、DFM模型、EM估计）
├── evaluation/     # 评估层（评估器、指标、验证器）
├── selection/      # 变量选择层（后向选择器、选择引擎）
├── optimization/   # 优化层（缓存、预计算）
├── training/       # 训练协调层（训练器、流程编排、配置）
├── analysis/       # 分析输出层（报告生成、可视化）
├── utils/          # 工具层（数据工具、日志、可重现性）
└── facade.py       # 统一API入口
```

**理由**：
- 清晰的职责分离，降低耦合
- 单向依赖关系，易于测试和维护
- 已验证有效（核心层已稳定运行）

### 2. 变量选择实现策略

**决策**：重新实现后向逐步变量选择算法

**核心算法**：
```python
# selection/backward_selector.py
class BackwardSelector:
    def __init__(self, evaluator: ModelEvaluator, precompute_engine: PrecomputeEngine):
        self.evaluator = evaluator
        self.precompute = precompute_engine

    def select(self,
               data: pd.DataFrame,
               target_col: str,
               initial_variables: List[str],
               criterion: str = 'rmse',
               progress_callback: Optional[Callable] = None) -> SelectionResult:
        """后向逐步变量选择

        算法流程：
        1. 从全部变量开始
        2. 逐个尝试剔除每个变量
        3. 评估剔除后的模型性能
        4. 选择性能提升最大的变量剔除
        5. 重复直到无法提升
        """
        current_vars = initial_variables.copy()
        selection_history = []

        # 预计算协方差矩阵加速评估
        precomputed_ctx = self.precompute.compute(data[current_vars + [target_col]])

        while len(current_vars) > 1:
            best_score = -np.inf
            best_var_to_remove = None

            # 尝试剔除每个变量
            for var in current_vars:
                candidate_vars = [v for v in current_vars if v != var]
                score = self._evaluate_subset(
                    data, candidate_vars, target_col, criterion, precomputed_ctx
                )

                if score > best_score:
                    best_score = score
                    best_var_to_remove = var

            # 检查是否停止
            baseline_score = self._evaluate_subset(
                data, current_vars, target_col, criterion, precomputed_ctx
            )

            if best_score <= baseline_score:
                break

            # 剔除变量并记录
            current_vars.remove(best_var_to_remove)
            selection_history.append({
                'removed_variable': best_var_to_remove,
                'score': best_score,
                'remaining_vars': current_vars.copy()
            })

            if progress_callback:
                progress_callback(f"[SELECTION] 剔除: {best_var_to_remove}, "
                                f"剩余: {len(current_vars)}, 得分: {best_score:.4f}")

        return SelectionResult(
            selected_variables=current_vars,
            selection_history=selection_history,
            final_score=best_score
        )
```

**理由**：
- 完全控制算法实现，易于调试
- 依赖注入便于测试
- 使用预计算引擎优化性能

**实现要点**：
1. 参考train_model/variable_selection.py的核心逻辑
2. 支持RMSE和Hit Rate双目标优化
3. 完整的单元测试（>85%覆盖率）

### 3. 训练流程编排

**决策**：使用Pipeline设计模式实现两阶段训练流程

```python
# training/pipeline.py
class TrainingPipeline:
    def __init__(self, selection_engine, factor_model, evaluator):
        self.selection_engine = selection_engine
        self.factor_model = factor_model
        self.evaluator = evaluator

    def run(self, config: TrainingConfig, progress_callback=None) -> TrainingResult:
        """完整两阶段训练流程"""
        # 阶段1：变量选择
        stage1_result = self._run_stage1(config, progress_callback)

        # 阶段2：因子数选择
        stage2_result = self._run_stage2(config, stage1_result, progress_callback)

        # 最终训练
        final_result = self._run_final_training(config, stage2_result, progress_callback)

        return self._merge_results(stage1_result, stage2_result, final_result)

    def _run_stage1(self, config, progress_callback):
        """阶段1：变量选择（固定k=块数）"""
        if not config.selection.enable:
            return Stage1Result(selected_variables=config.data.selected_indicators)

        data = load_data(config.data.data_path)
        result = self.selection_engine.run(
            selection_method=config.selection.method,
            data=data,
            config=config,
            progress_callback=progress_callback
        )
        return Stage1Result(selected_variables=result.selected_variables,
                          selection_history=result.selection_history)

    def _run_stage2(self, config, stage1_result, progress_callback):
        """阶段2：因子数选择（PCA/Elbow/Fixed）"""
        data = load_data(config.data.data_path)[stage1_result.selected_variables]

        if config.model.factor_selection_method == 'fixed':
            return Stage2Result(k_factors=config.model.k_factors)

        elif config.model.factor_selection_method == 'cumulative':
            pca_analysis = self._perform_pca_analysis(data)
            k = self._select_k_by_cumulative_variance(
                pca_analysis.eigenvalues,
                threshold=config.model.pca_threshold
            )
            return Stage2Result(k_factors=k, pca_analysis=pca_analysis)

        elif config.model.factor_selection_method == 'elbow':
            pca_analysis = self._perform_pca_analysis(data)
            k = self._select_k_by_elbow(
                pca_analysis.eigenvalues,
                threshold=config.model.elbow_threshold
            )
            return Stage2Result(k_factors=k, pca_analysis=pca_analysis)

    def _run_final_training(self, config, stage2_result, progress_callback):
        """最终模型训练"""
        model = self.factor_model.fit(
            data=data,
            k_factors=stage2_result.k_factors,
            max_iter=config.model.max_iterations,
            progress_callback=progress_callback
        )

        # 评估
        metrics = self.evaluator.evaluate(model, data, config)

        return FinalResult(model=model, metrics=metrics)
```

**理由**：
- 清晰的两阶段流程分离
- 每个阶段可独立测试
- 支持未来扩展（如添加阶段3）

### 4. 结果分析层实现

**决策**：完整重新实现分析报告和可视化功能

```python
# analysis/reporter.py
class AnalysisReporter:
    def generate_full_report(self, model, data, config) -> AnalysisReport:
        """生成完整分析报告"""
        return AnalysisReport(
            pca_analysis=self.generate_pca_report(model),
            contribution_analysis=self.generate_contribution_report(model, data),
            r2_analysis=self.generate_r2_report(model, data),
            factor_loadings=self.calculate_factor_loadings(model)
        )

    def generate_pca_report(self, model) -> pd.DataFrame:
        """PCA方差贡献分析"""
        eigenvalues = model.get_eigenvalues()
        variance_ratios = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(variance_ratios)

        return pd.DataFrame({
            '因子索引': range(1, len(eigenvalues) + 1),
            '特征值': eigenvalues,
            '方差贡献率': variance_ratios,
            '累积方差贡献率': cumulative_variance
        })

    def generate_contribution_report(self, model, data) -> pd.DataFrame:
        """因子贡献度分解（按指标、按行业）"""
        factors = model.get_factors()
        loadings = model.get_loadings()

        # 计算贡献度 = 因子载荷 * 因子值 * 标准差
        contributions = {}
        for i, var in enumerate(data.columns):
            contrib = loadings[i, :] * factors * data[var].std()
            contributions[var] = contrib.sum(axis=0)

        return pd.DataFrame(contributions)

    def generate_r2_report(self, model, data) -> pd.DataFrame:
        """个体R²和行业R²计算"""
        # 实现个体变量R²计算
        # 实现行业聚合R²计算
        pass
```

**理由**：
- 与训练逻辑解耦
- 便于扩展新的分析类型
- 支持多种输出格式（Excel、CSV、JSON）

### 5. 数值一致性保证机制

**决策**：采用严格的对比测试框架

```python
# tests/consistency/test_end_to_end.py
class TestNumericConsistency:
    def test_parameter_estimation(self):
        """验证EM算法参数估计结果一致"""
        # 加载相同数据和配置
        data = load_test_data()
        config = create_test_config()

        # 运行train_ref（在删除train_model前保留对比）
        from dashboard.DFM.train_ref import DFMTrainer
        new_result = DFMTrainer(config).train()

        # 加载train_model的基准结果
        baseline_result = load_baseline_results('test_case_1.pkl')

        # 验证参数矩阵
        assert np.linalg.norm(new_result.params.A - baseline_result.params.A) < 1e-6
        assert np.linalg.norm(new_result.params.Q - baseline_result.params.Q) < 1e-6
        assert np.linalg.norm(new_result.params.H - baseline_result.params.H) < 1e-6
        assert np.linalg.norm(new_result.params.R - baseline_result.params.R) < 1e-6

    def test_forecast_consistency(self):
        """验证预测结果一致"""
        # 对比每个时间点的预测值
        for t in range(len(new_result.forecast_oos)):
            assert abs(new_result.forecast_oos[t] - baseline_result.forecast_oos[t]) < 1e-6

    def test_metrics_consistency(self):
        """验证评估指标一致"""
        assert abs(new_result.metrics.rmse_oos - baseline_result.metrics.rmse_oos) < 1e-4
        assert abs(new_result.metrics.hit_rate_oos - baseline_result.metrics.hit_rate_oos) < 0.01
```

**实施步骤**：
1. 在删除train_model前，运行train_model并保存所有结果作为baseline
2. 实现train_ref新模块时，持续对比baseline验证数值一致性
3. 所有对比测试通过后，才可删除train_model

### 6. UI层直接切换

**决策**：直接修改UI使用train_ref，不保留版本切换

```python
# dashboard/ui/pages/dfm/model_training_page.py

# 旧代码（删除）
# from dashboard.DFM.train_model.tune_dfm import run_tuning
# results = run_tuning(...)

# 新代码
from dashboard.DFM.train_ref import DFMTrainer, TrainingConfig

# 构建配置
config = TrainingConfig(
    data=DataConfig(
        data_path=selected_file,
        target_variable=target_var,
        ...
    ),
    model=ModelConfig(
        k_factors=k_factors,
        max_iter=max_iterations,
        ...
    ),
    selection=SelectionConfig(
        enable=enable_selection,
        method='backward',
        criterion='rmse'
    )
)

# 训练
trainer = DFMTrainer(config)
results = trainer.train(progress_callback=lambda msg: update_progress(msg))
```

**理由**：
- 简化代码，无需维护版本切换逻辑
- 用户体验完全一致（接口保持不变）
- 彻底移除遗留代码

## Risks / Trade-offs

### 风险1：数值精度差异

**描述**：重新实现算法可能引入微小数值误差

**缓解**：
- 参考train_model实现，确保算法逻辑正确
- 建立全面的对比测试（误差 < 1e-6）
- 使用相同的随机种子和数值库版本

**接受度**：RMSE差异 < 1e-4可接受

### 风险2：开发周期长

**描述**：22周开发周期可能影响其他项目

**缓解**：
- 分阶段开发，每个阶段都有可交付成果
- 并行开发多个模块（选择层、分析层）
- 持续集成和测试，及早发现问题

### 风险3：复杂算法实现难度高

**描述**：变量选择、结果分析等算法复杂度高

**缓解**：
- 深入研究train_model实现
- 分步实现，先实现核心逻辑再优化
- 高测试覆盖率（>90%）

### 关键权衡

1. **开发时间 vs 长期维护**：22周换取60%维护成本降低
2. **风险 vs 收益**：高风险换取彻底解决技术债务
3. **完美 vs 实用**：不引入新优化，专注功能等价

## Migration Plan

### Phase 1：准备阶段（1周）

1. 运行train_model生成所有baseline结果
2. 保存baseline到tests/consistency/baseline/
3. 创建feature分支进行开发

### Phase 2：核心开发（16周）

**Week 1-3**：变量选择层
- 实现BackwardSelector
- 实现SelectionEngine
- 单元测试（覆盖率 > 85%）

**Week 4-7**：训练协调层
- 实现TrainingPipeline
- 实现DFMTrainer
- 补充配置类
- 单元测试（覆盖率 > 80%）

**Week 8-11**：分析输出层
- 实现AnalysisReporter
- 实现ResultVisualizer
- 实现分析工具函数
- 单元测试（覆盖率 > 80%）

**Week 12-13**：优化层
- 实现PrecomputeEngine
- 实现优化评估器
- 性能测试

**Week 14-16**：集成测试
- 端到端对比测试
- 数值一致性验证
- 性能基准测试

### Phase 3：UI迁移（2周）

1. 修改model_training_page.py
2. 修改相关组件
3. 全面UI测试

### Phase 4：上线（2周）

1. 代码审查
2. 部署到测试环境
3. 用户验收测试
4. 部署到生产环境

### Phase 5：清理与合并（1周）

**前置条件**：
- Phase 4所有测试和部署已完成
- 用户验收测试通过
- 生产环境稳定运行至少1周

**任务**：
1. 删除train_model目录（满足前置条件后）
2. 更新文档（CLAUDE.md）
3. 发布变更日志
4. 合并feature分支到main
5. 创建发布标签v2.0.0-train-ref

**总计**：22周

## Open Questions

1. **是否需要保留train_model的性能分析模块**？
   - **决策**：暂不实现，作为可选功能后续添加

2. **是否需要支持并行训练**？
   - **决策**：保留串行实现，性能可接受

3. **测试覆盖率最低标准**？
   - **决策**：核心算法层 > 90%，其他层 > 80%
