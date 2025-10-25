# dfm-model-training Specification (Delta)

## ADDED Requirements

### Requirement: 训练结果文件导出

系统SHALL在train_ref模块训练完成后自动导出结果文件，确保格式和内容与train_model完全一致，支持离线分析和下游功能集成。

#### Scenario: 导出所有结果文件

**GIVEN** DFM模型训练已完成，返回TrainingResult对象
**WHEN** 调用train()方法时enable_export=True
**THEN** 系统应导出3个文件：
- 模型文件（final_dfm_model_{timestamp}.joblib）
- 元数据文件（final_dfm_metadata_{timestamp}.pkl）
- Excel报告（final_report_{timestamp}.xlsx）

**AND** 系统应将文件路径保存到TrainingResult.export_files字典

**AND** 系统应在progress_callback中通知导出进度

**文件路径格式**:
```python
{
    'final_model_joblib': '/tmp/dfm_results_abc123/final_dfm_model_20251024_153045.joblib',
    'metadata': '/tmp/dfm_results_abc123/final_dfm_metadata_20251024_153045.pkl',
    'excel_report': '/tmp/dfm_results_abc123/final_report_20251024_153045.xlsx'
}
```

#### Scenario: 导出模型文件

**GIVEN** TrainingResult包含有效的model_result
**WHEN** 系统导出模型文件
**THEN** 系统应：
- 使用joblib.dump序列化model_result
- 文件命名包含时间戳（YYYYMMDD_HHMMSS格式）
- 验证文件写入成功且可读取
- 返回文件绝对路径

**AND** 文件应包含完整的DFMModelResult对象：
- params（Z, A, Q, R, x0, P0）
- loglikelihood（对数似然值）
- convergence（收敛信息）

#### Scenario: 导出元数据文件

**GIVEN** TrainingResult和TrainingConfig可用
**WHEN** 系统导出元数据文件
**THEN** 系统应：
- 构建元数据字典（调用MetadataBuilder）
- 使用pickle.dump序列化字典
- 文件命名包含时间戳
- 验证文件写入成功
- 返回文件绝对路径

**AND** 元数据应包含以下必需字段（与train_model一致）：
```python
{
    'timestamp': str,  # ISO格式时间戳
    'target_variable': str,  # 目标变量名
    'best_variables': List[str],  # 选中的变量
    'best_params': {'k': int, 'max_em_iter': int, 'tolerance': float},
    'train_end_date': str,
    'validation_start_date': str,  # 可选
    'validation_end_date': str,  # 可选
    'target_mean_original': float,
    'target_std_original': float,
    'is_rmse': float,
    'oos_rmse': float,
    'is_hit_rate': float,
    'oos_hit_rate': float,
    'is_correlation': float,
    'oos_correlation': float,
    'total_runtime_seconds': float,
    'var_type_map': Dict[str, str],  # 变量→行业映射
    'complete_aligned_table': pd.DataFrame,  # 完整对齐数据表
    'factor_loadings_df': pd.DataFrame  # 因子载荷矩阵
}
```

#### Scenario: 导出Excel报告

**GIVEN** TrainingResult和TrainingConfig可用
**WHEN** 系统导出Excel报告
**THEN** 系统应：
- 生成多sheet Excel文件
- 文件命名包含时间戳
- 使用openpyxl库写入
- 验证文件写入成功
- 返回文件绝对路径

**AND** Excel应包含以下sheet（与train_model格式一致）：

**Sheet 1: 模型参数**
- 因子数（k）
- EM最大迭代次数（max_em_iter）
- 收敛容差（tolerance）
- 训练结束日期
- 验证开始/结束日期
- 选中变量数
- 选中变量列表（逗号分隔）

**Sheet 2: 评估指标**
| 指标类型 | 样本内 | 样本外 |
|---------|-------|--------|
| RMSE | is_rmse（4位小数） | oos_rmse（4位小数） |
| Hit Rate (%) | is_hit_rate（2位小数） | oos_hit_rate（2位小数）或N/A |
| 相关系数 | is_correlation（4位小数） | oos_correlation（4位小数） |

**Sheet 3: 因子载荷**
- 行：变量名
- 列：Factor_1, Factor_2, ..., Factor_k
- 值：载荷系数（3位小数）

**Sheet 4: 状态转移矩阵**（可选，如params.A存在）
- k×k矩阵
- 值：转移系数（4位小数）

#### Scenario: 元数据字段完整性

**GIVEN** 系统需要构建元数据字典
**WHEN** 某些字段在TrainingResult或TrainingConfig中缺失
**THEN** 系统应提供合理的默认值：
- validation_start_date/validation_end_date: None（如未指定）
- var_type_map: 所有变量映射到"未分类"
- complete_aligned_table: 从原始数据和预测结果重建（必需）
- factor_loadings_df: 从params.Z提取（必需）

**AND** 系统不应因缺失可选字段而失败

#### Scenario: 临时文件目录管理

**GIVEN** 系统需要创建临时目录存储导出文件
**WHEN** output_dir参数为None
**THEN** 系统应：
- 使用tempfile.mkdtemp(prefix='dfm_results_')创建临时目录
- 目录路径包含随机字符串（如/tmp/dfm_results_abc123/）
- 记录目录路径到日志

**AND** 系统应依赖OS的临时文件清理机制

**WHEN** output_dir参数指定具体目录
**THEN** 系统应：
- 使用指定目录
- 如目录不存在则创建
- 验证目录可写

#### Scenario: 文件导出错误处理

**GIVEN** 系统正在导出多个文件
**WHEN** 单个文件导出失败（如磁盘满、权限不足）
**THEN** 系统应：
- 记录错误日志（logger.error）
- 该文件路径值设为None
- 继续导出其他文件
- 不抛出异常中断训练结果返回

**示例**:
```python
export_files = {
    'final_model_joblib': '/tmp/.../model.joblib',  # 成功
    'metadata': None,  # 失败
    'excel_report': '/tmp/.../report.xlsx'  # 成功
}
```

**WHEN** 所有文件导出都失败
**THEN** 系统应：
- 记录警告日志（logger.warning）
- export_files为None或空字典
- TrainingResult仍然有效并返回

#### Scenario: UI文件下载集成

**GIVEN** 训练完成且export_files不为None
**WHEN** UI读取dfm_model_results_paths状态
**THEN** UI应显示"文件下载"区域

**AND** UI应为每个非None的文件路径创建下载按钮：
- final_model_joblib: 显示名称"[PACKAGE] 模型文件"，MIME="application/octet-stream"
- metadata: 显示名称"元数据文件"，MIME="application/octet-stream"
- excel_report: 显示名称"[DATA] Excel报告"，MIME="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

**WHEN** 用户点击下载按钮
**THEN** 浏览器应触发文件下载

#### Scenario: 格式兼容性验证

**GIVEN** 系统生成的Excel报告和元数据文件
**WHEN** news_analysis或其他下游功能读取这些文件
**THEN** 系统应：
- 成功加载文件无错误
- 所有必需字段可访问
- 数据类型和格式与train_model一致
- 分析功能正常运行

**验证方式**:
- 使用train_model和train_ref生成文件
- 逐字段对比元数据结构
- 逐sheet对比Excel格式
- 运行news_analysis兼容性测试

#### Scenario: 禁用文件导出

**GIVEN** 系统需要快速训练而不导出文件（如测试场景）
**WHEN** 调用train()方法时enable_export=False
**THEN** 系统应：
- 跳过所有文件导出逻辑
- export_files保持None
- 训练速度不受影响

#### Scenario: 自定义导出目录

**GIVEN** 系统需要将文件导出到特定目录
**WHEN** 调用train()方法时指定export_dir='/path/to/custom/dir'
**THEN** 系统应：
- 使用指定目录而非临时目录
- 如目录不存在则创建
- 验证目录可写
- 将文件写入该目录

#### Scenario: 导出性能要求

**GIVEN** 正常数据量场景（k=2-5, 10-15个变量, 200-500行数据）
**WHEN** 系统执行export_all()
**THEN** 系统应：
- 总耗时 < 5秒
- 内存峰值增量 < 200MB
- 不阻塞训练主流程

**WHEN** 大数据量场景（k=10, 30个变量, 1000行数据）
**THEN** 系统应：
- 总耗时 < 15秒
- 内存峰值增量 < 500MB
- 提供导出进度反馈

#### Scenario: 文件命名规范

**GIVEN** 系统生成导出文件
**WHEN** 创建文件名
**THEN** 文件名应遵循以下格式：
- 模型文件: `final_dfm_model_{YYYYMMDD_HHMMSS}.joblib`
- 元数据文件: `final_dfm_metadata_{YYYYMMDD_HHMMSS}.pkl`
- Excel报告: `final_report_{YYYYMMDD_HHMMSS}.xlsx`

**示例**:
- final_dfm_model_20251024_153045.joblib
- final_dfm_metadata_20251024_153045.pkl
- final_report_20251024_153045.xlsx

**AND** 同一次训练生成的3个文件应使用相同的时间戳

#### Scenario: 日志记录

**GIVEN** 系统执行文件导出
**WHEN** 导出过程的各个阶段
**THEN** 系统应记录以下日志：

**INFO级别**:
- "开始导出结果文件到: {output_dir}"
- "导出模型文件: {filename} ({file_size})"
- "导出元数据文件: {filename} ({file_size})"
- "导出Excel报告: {filename} ({file_size})"
- "文件导出完成，共{count}/3个文件"

**ERROR级别**:
- "导出模型文件失败: {error}"
- "导出元数据文件失败: {error}"
- "导出Excel报告失败: {error}"

**WARNING级别**:
- "文件导出全部失败，训练结果仍可用"
- "缺少必需字段: {field_name}，使用默认值"

---

## MODIFIED Requirements

无修改的现有requirement。
