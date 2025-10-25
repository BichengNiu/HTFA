# train_ref文件导出功能设计

## 架构概览

### 分层设计

```
UI Layer (Streamlit)
    ↓ 调用
Training Coordinator (DFMTrainer)
    ↓ 完成后调用
Export Layer (NEW)
    ├── TrainingResultExporter (协调器)
    ├── MetadataBuilder (元数据构建)
    ├── ExcelReportGenerator (报告生成)
    └── File I/O (joblib, pickle, openpyxl)
```

**设计原则**:
- **单向依赖**: Export Layer不依赖Training Layer之上的层
- **职责分离**: 每个类职责单一
- **可测试性**: 所有组件可独立测试
- **向后兼容**: 不修改现有TrainingResult字段（只增不改）

## 核心组件设计

### 1. TrainingResultExporter (export/exporter.py)

**职责**: 协调所有导出操作，管理临时文件

**接口设计**:
```python
class TrainingResultExporter:
    """训练结果导出器"""

    def export_all(
        self,
        result: TrainingResult,
        config: TrainingConfig,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        导出所有结果文件

        工作流程:
        1. 创建临时目录（或使用指定目录）
        2. 生成时间戳
        3. 依次导出：模型 → 元数据 → Excel报告
        4. 验证所有文件存在
        5. 返回文件路径字典

        Args:
            result: 训练结果对象
            config: 训练配置对象
            output_dir: 输出目录（None=创建临时目录）

        Returns:
            {
                'final_model_joblib': '/tmp/dfm_results_xxx/final_dfm_model_20251024_153045.joblib',
                'metadata': '/tmp/dfm_results_xxx/final_dfm_metadata_20251024_153045.pkl',
                'excel_report': '/tmp/dfm_results_xxx/final_report_20251024_153045.xlsx'
            }

        异常处理:
        - 单个文件导出失败不阻塞其他文件
        - 失败的文件路径值为None
        - 记录详细错误日志
        """

    def export_model(self, result: TrainingResult, path: str) -> str:
        """
        导出模型文件

        保存内容:
        - result.model_result (DFMModelResult对象)
        - 包含params, loglikelihood, convergence等

        格式: joblib (与train_model一致)
        """

    def export_metadata(
        self,
        result: TrainingResult,
        config: TrainingConfig,
        path: str
    ) -> str:
        """
        导出元数据文件

        保存内容:
        - 调用MetadataBuilder.build_metadata生成字典
        - 使用pickle序列化

        格式: pickle (与train_model一致)
        """

    def export_excel_report(
        self,
        result: TrainingResult,
        config: TrainingConfig,
        path: str
    ) -> str:
        """
        导出Excel报告

        保存内容:
        - 调用ExcelReportGenerator.generate
        - 多sheet详细报告

        格式: Excel (.xlsx)
        """

    def _create_temp_dir(self) -> str:
        """创建临时目录"""

    def _generate_timestamp(self) -> str:
        """生成时间戳字符串（格式: 20251024_153045）"""

    def _verify_file_exists(self, path: str) -> bool:
        """验证文件是否存在"""
```

**关键设计决策**:
1. **临时目录管理**: 使用`tempfile.mkdtemp(prefix='dfm_results_')`创建，由OS负责最终清理
2. **错误隔离**: 单个文件失败不影响其他文件，返回字典中该键值为None
3. **文件命名**: 统一使用时间戳后缀，便于追溯和排序

### 2. MetadataBuilder (export/metadata_builder.py)

**职责**: 从TrainingResult和TrainingConfig构建元数据字典

**接口设计**:
```python
class MetadataBuilder:
    """元数据构建器"""

    def build_metadata(
        self,
        result: TrainingResult,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        构建元数据字典

        输出字段（与train_model完全一致）:
        {
            # 基本信息
            'timestamp': str,  # ISO格式时间戳
            'target_variable': str,  # 目标变量名
            'best_variables': List[str],  # 选中的变量（result.selected_variables）

            # 模型参数
            'best_params': {
                'k': int,  # 因子数（result.k_factors）
                'max_em_iter': int,  # EM最大迭代（config.max_iterations）
                'tolerance': float  # 收敛容差（config.tolerance）
            },

            # 训练配置
            'train_end_date': str,  # config.train_end_date
            'validation_start_date': str,  # config.validation_start_date（可选）
            'validation_end_date': str,  # config.validation_end_date（可选）

            # 标准化参数
            'target_mean_original': float,  # result.target_mean_original
            'target_std_original': float,  # result.target_std_original

            # 评估指标
            'is_rmse': float,  # result.metrics.is_rmse
            'oos_rmse': float,  # result.metrics.oos_rmse
            'is_hit_rate': float,  # result.metrics.is_hit_rate
            'oos_hit_rate': float,  # result.metrics.oos_hit_rate
            'is_correlation': float,  # result.metrics.is_correlation
            'oos_correlation': float,  # result.metrics.oos_correlation

            # 训练统计
            'total_runtime_seconds': float,  # result.training_time

            # 变量类型映射（用于下游分析）
            'var_type_map': Dict[str, str],  # 变量名 → 行业/类型

            # 完整对齐数据表（用于news_analysis）
            'complete_aligned_table': pd.DataFrame,  # 时间序列对齐表

            # 因子载荷（用于Excel报告和分析）
            'factor_loadings_df': pd.DataFrame  # 变量 × 因子
        }
        """

    def _extract_basic_info(self, result, config) -> Dict:
        """提取基本信息字段"""

    def _extract_model_params(self, result, config) -> Dict:
        """提取模型参数字段"""

    def _extract_training_config(self, config) -> Dict:
        """提取训练配置字段"""

    def _extract_evaluation_metrics(self, result) -> Dict:
        """提取评估指标字段"""

    def _build_var_type_map(self, result, config) -> Dict[str, str]:
        """构建变量类型映射"""

    def _build_aligned_table(self, result, config) -> pd.DataFrame:
        """构建完整对齐数据表"""

    def _extract_factor_loadings(self, result) -> pd.DataFrame:
        """提取因子载荷矩阵"""
```

**关键设计决策**:
1. **字段完整性优先**: 确保包含train_model的所有字段，即使某些字段train_ref暂不使用
2. **默认值策略**: 缺失字段提供合理默认值（如var_type_map默认为"未分类"）
3. **数据表构建**: complete_aligned_table从原始数据和预测结果重建（用于news_analysis）

### 3. ExcelReportGenerator (export/report_generator.py)

**职责**: 生成与train_model格式完全一致的Excel报告

**接口设计**:
```python
class ExcelReportGenerator:
    """Excel报告生成器"""

    def generate(
        self,
        result: TrainingResult,
        config: TrainingConfig,
        output_path: str
    ) -> str:
        """
        生成Excel报告

        Sheet结构:
        1. 模型参数
        2. 评估指标
        3. 因子载荷
        4. 状态转移矩阵（可选）

        返回: 生成的文件路径
        """

    def _generate_params_sheet(self, writer, result, config):
        """生成模型参数sheet"""

    def _generate_metrics_sheet(self, writer, result):
        """生成评估指标sheet"""

    def _generate_loadings_sheet(self, writer, result):
        """生成因子载荷sheet"""

    def _generate_transition_matrix_sheet(self, writer, result):
        """生成状态转移矩阵sheet（如果可用）"""
```

**Sheet 1: 模型参数**
```
字段名          | 值
-----------------+------------------
因子数          | 3
EM最大迭代      | 30
收敛容差        | 1e-06
训练结束日期    | 2024-06-30
验证开始日期    | 2024-07-01
验证结束日期    | 2024-12-31
选中变量数      | 8
选中变量列表    | var1, var2, ...
```

**Sheet 2: 评估指标**
```
指标类型        | 样本内     | 样本外
----------------+------------+------------
RMSE            | 5.0278     | 4.9508
Hit Rate (%)    | 65.3       | N/A
相关系数        | 0.89       | 0.87
```

**Sheet 3: 因子载荷**
```
变量名                  | Factor_1 | Factor_2 | Factor_3
------------------------+----------+----------+----------
铁矿石疏港量            | 0.856    | -0.123   | 0.034
中厚板产量              | 0.723    | 0.456    | -0.089
...
```

**Sheet 4: 状态转移矩阵** (如果params.A存在)
```
       | F1     | F2     | F3
-------+--------+--------+--------
F1     | 0.95   | 0.02   | 0.01
F2     | 0.03   | 0.92   | 0.04
F3     | 0.01   | 0.05   | 0.94
```

**关键设计决策**:
1. **格式复用**: 直接参考train_model/generate_report.py的实现
2. **数值精度**: RMSE保留4位小数，Hit Rate保留2位小数，载荷保留3位小数
3. **错误处理**: 缺失数据显示"N/A"，不抛出异常

## 集成设计

### DFMTrainer集成

**修改点**: `dashboard/DFM/train_ref/training/trainer.py`

**修改1: TrainingResult扩展**
```python
@dataclass
class TrainingResult:
    # ... 现有字段 ...

    # 新增：导出文件路径
    export_files: Optional[Dict[str, str]] = None
```

**修改2: train方法扩展**
```python
def train(
    self,
    progress_callback=None,
    enable_export: bool = True,  # 新增
    export_dir: Optional[str] = None  # 新增
) -> TrainingResult:
    """训练模型"""

    # ... 现有训练逻辑 ...

    # 新增：导出文件（在返回前）
    if enable_export:
        if progress_callback:
            progress_callback("[EXPORT] 正在导出结果文件...")

        try:
            from dashboard.DFM.train_ref.export import TrainingResultExporter
            exporter = TrainingResultExporter()
            result.export_files = exporter.export_all(
                result=result,
                config=self.config,
                output_dir=export_dir
            )

            if progress_callback:
                file_count = sum(1 for v in result.export_files.values() if v is not None)
                progress_callback(f"[SUCCESS] 已导出 {file_count}/3 个文件")

        except Exception as e:
            logger.warning(f"文件导出失败（不影响训练结果）: {e}")
            result.export_files = None

    return result
```

**设计原则**:
- **非阻塞**: 导出失败不影响训练结果返回
- **可选**: 可以通过enable_export=False禁用导出（用于快速测试）
- **灵活**: 可以指定export_dir或使用临时目录

### UI集成

**修改点**: `dashboard/ui/pages/dfm/model_training_page.py`

**修改1: 训练调用（第1519-1520行）**
```python
# 修改前
trainer = DFMTrainer(training_config)
result: TrainingResult = trainer.train(progress_callback=progress_callback)

# 修改后
trainer = DFMTrainer(training_config)
result: TrainingResult = trainer.train(
    progress_callback=progress_callback,
    enable_export=True,  # 启用文件导出
    export_dir=None  # 使用临时目录
)
```

**修改2: 状态保存（第1536-1539行）**
```python
# 修改前
set_dfm_state('dfm_training_result', result_summary)
set_dfm_state('dfm_training_status', '训练完成')

# 修改后
set_dfm_state('dfm_training_result', result_summary)
set_dfm_state('dfm_model_results_paths', result.export_files)  # 新增
set_dfm_state('dfm_training_status', '训练完成')
```

**现有UI下载逻辑无需修改**（第1676-1738行）:
- UI已实现完整的文件下载逻辑
- 从`dfm_model_results_paths`读取文件路径字典
- 自动显示3个下载按钮
- 只需确保`result.export_files`格式正确即可

## 数据流

```
TrainingConfig + 数据
    ↓
DFMTrainer.train()
    ↓ 训练完成
TrainingResult (内存对象)
    ↓ enable_export=True
TrainingResultExporter.export_all()
    ├→ export_model() → final_dfm_model.joblib
    ├→ export_metadata() → final_dfm_metadata.pkl
    │   └→ MetadataBuilder.build_metadata()
    └→ export_excel_report() → final_report.xlsx
        └→ ExcelReportGenerator.generate()
    ↓ 返回
result.export_files = {
    'final_model_joblib': '/tmp/.../xxx.joblib',
    'metadata': '/tmp/.../xxx.pkl',
    'excel_report': '/tmp/.../xxx.xlsx'
}
    ↓ 保存到状态
set_dfm_state('dfm_model_results_paths', result.export_files)
    ↓ UI读取
UI下载按钮显示（现有逻辑）
    ↓ 用户点击
文件下载到本地
```

## 错误处理策略

### 分层错误处理

**Level 1: 单个文件导出失败**
```python
# export_model, export_metadata, export_excel_report
try:
    # 导出逻辑
    return file_path
except Exception as e:
    logger.error(f"导出xxx失败: {e}")
    return None  # 返回None，不抛出异常
```

**Level 2: export_all整体失败**
```python
# export_all
result_files = {}
result_files['final_model_joblib'] = self.export_model(...)  # 可能为None
result_files['metadata'] = self.export_metadata(...)  # 可能为None
result_files['excel_report'] = self.export_excel_report(...)  # 可能为None
return result_files  # 总是返回字典，即使所有值都是None
```

**Level 3: DFMTrainer.train中的导出失败**
```python
# DFMTrainer.train
try:
    result.export_files = exporter.export_all(...)
except Exception as e:
    logger.warning(f"文件导出失败（不影响训练结果）: {e}")
    result.export_files = None  # 训练结果仍然有效
```

**Level 4: UI层处理**
```python
# UI中的逻辑（现有代码）
if training_results:
    available_files = []
    for file_key, file_path in training_results.items():
        if file_path and os.path.exists(file_path):
            available_files.append(...)

    if available_files:
        # 显示下载按钮
    else:
        st.warning("未找到可用的结果文件")
```

**错误处理原则**:
1. **训练优先**: 导出失败不影响训练结果
2. **部分成功**: 1个文件失败，其他文件仍可用
3. **清晰日志**: 每层记录详细错误信息
4. **用户友好**: UI显示明确的错误提示

## 性能考虑

### 文件大小估算

**典型场景**（k=3, 10个变量, 300行数据）:
- **模型文件**（.joblib）: ~2-5MB
  - 包含params（Z, A, Q, R, x0, P0）
  - 包含loglikelihood历史
  - 包含convergence信息
- **元数据文件**（.pkl）: ~0.5-1MB
  - 主要是complete_aligned_table（DataFrame）
  - 其他字段很小
- **Excel报告**（.xlsx）: ~0.1-0.3MB
  - 多sheet，主要是因子载荷和参数表

**总计**: ~3-6MB（合理范围）

### 导出耗时分析

**预期耗时**（正常数据量）:
- export_model: ~0.5秒（joblib序列化）
- export_metadata: ~1秒（包含DataFrame序列化）
- export_excel_report: ~2秒（Excel写入）
- **总计**: ~3-4秒

**优化策略**:
1. **延迟导出**: 不在训练中导出，而是训练完成后
2. **异步导出**（未来）: 可以考虑后台异步导出
3. **增量写入**: Excel使用openpyxl的write_only模式（如需要）

### 内存占用

**峰值分析**:
- 训练过程峰值: ~200-300MB（现有）
- 导出过程额外峰值: ~100-200MB（临时DataFrame + Excel buffer）
- **总峰值**: ~300-500MB（可接受）

**优化策略**:
1. **及时释放**: 每个文件写入后立即释放临时对象
2. **流式写入**: 大数据表使用流式写入（如需要）
3. **分批处理**: complete_aligned_table分批写入元数据（如很大）

## 测试策略

### 单元测试

**MetadataBuilder测试**:
- 测试所有必需字段存在
- 测试字段类型正确
- 测试默认值逻辑
- 测试缺失字段处理

**ExcelReportGenerator测试**:
- 测试每个sheet生成
- 测试sheet名称正确
- 测试列名和数据类型
- 测试格式化规则

**TrainingResultExporter测试**:
- 测试export_all返回3个文件路径
- 测试文件存在性
- 测试文件可读取
- 测试错误处理

### 集成测试

**格式对比测试**:
```python
def test_format_alignment():
    # 1. 使用相同数据训练train_model和train_ref
    # 2. 生成两份Excel报告
    # 3. 逐sheet对比：
    #    - Sheet名称一致
    #    - 列名一致
    #    - 数据类型一致
    # 4. 对比元数据字段完整性
```

**下游兼容性测试**:
```python
def test_downstream_compatibility():
    # 1. 使用train_ref生成文件
    # 2. 尝试用news_analysis读取
    # 3. 验证所有必需字段可用
    # 4. 验证分析功能正常
```

### 性能测试

```python
def test_export_performance():
    # 1. 测量export_all耗时
    # 2. 监控内存峰值
    # 3. 验证临时文件清理
    # 4. 不同数据量场景测试
```

## 未来扩展

### 可选增强

1. **异步导出**: 后台线程导出，不阻塞UI
2. **导出格式选项**: 支持CSV、JSON等格式
3. **压缩打包**: 自动打包为.zip下载
4. **云存储集成**: 上传到S3/OSS等
5. **导出模板**: 自定义Excel模板

### 向后兼容

**如果train_model完全移除**:
- 保留generate_report.py作为参考
- 确保train_ref格式永久兼容
- 提供格式迁移工具（如需要）
