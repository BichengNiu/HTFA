# DFM模型训练后端测试接口使用指南

这是DFM模型训练模块的后端测试接口，允许在不启动UI前端的情况下进行模型训练和测试。

## 快速开始

### 方式1：快速测试模式（推荐）

使用`dashboard/models/DFM/tests/data/`目录下的测试数据快速运行训练：

```bash
# 从项目根目录运行
python dashboard/models/DFM/tests/code/test_train.py --quick-test

# 或从code目录运行
cd dashboard/models/DFM/tests/code
python test_train.py --quick-test
```

这将自动：
- 加载测试数据 (`dashboard/models/DFM/tests/data/dfm_prepared_output.csv`)
- 自动划分训练集和验证集（80%/20%）
- 使用合理的默认参数
- 显示完整的训练过程和结果

### 方式2：使用配置文件

使用提供的示例配置文件：

```bash
# 从code目录运行
cd dashboard/models/DFM/tests/code
python test_train.py --config example_config.json

# 从项目根目录运行
python dashboard/models/DFM/tests/code/test_train.py \
  --config dashboard/models/DFM/tests/code/example_config.json
```

或者创建自己的配置文件：

```bash
# 1. 进入code目录
cd dashboard/models/DFM/tests/code

# 2. 复制模板
cp config_template.json my_config.json

# 3. 编辑配置文件（根据需要修改参数）
# 编辑 my_config.json

# 4. 运行训练
python test_train.py --config my_config.json
```

### 方式3：使用命令行参数

完全通过命令行参数指定所有配置：

```bash
# 从code目录运行（推荐）
cd dashboard/models/DFM/tests/code
python test_train.py \
  --data-path "../data/dfm_prepared_output.csv" \
  --target "规模以上工业增加值:当月同比" \
  --k-factors 3 \
  --train-end "2023-12-31" \
  --validation-start "2024-01-01" \
  --validation-end "2024-06-30"

# 从项目根目录运行
python dashboard/models/DFM/tests/code/test_train.py \
  --data-path "dashboard/models/DFM/tests/data/dfm_prepared_output.csv" \
  --target "规模以上工业增加值:当月同比" \
  --k-factors 3 \
  --train-end "2023-12-31" \
  --validation-start "2024-01-01" \
  --validation-end "2024-06-30"
```

## 命令行参数详解

### 数据配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data-path` | 数据文件路径 | `--data-path "data/经济数据.csv"` |
| `--target` | 目标变量名称 | `--target "规模以上工业增加值:当月同比"` |
| `--indicators` | 选中的指标列表 | `--indicators "指标1" "指标2" "指标3"` |

### 日期配置

| 参数 | 说明 | 示例 |
|------|------|------|
| `--train-end` | 训练期结束日期 | `--train-end "2023-12-31"` |
| `--validation-start` | 验证期开始日期 | `--validation-start "2024-01-01"` |
| `--validation-end` | 验证期结束日期 | `--validation-end "2024-06-30"` |
| `--target-freq` | 目标频率 | `--target-freq "W-FRI"` |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--k-factors` | 3 | 因子数量 |
| `--max-iter` | 30 | EM算法最大迭代次数 |
| `--tolerance` | 1e-6 | 收敛容差 |

### 变量选择

| 参数 | 说明 |
|------|------|
| `--enable-selection` | 启用变量选择 |
| `--min-variables` | 变量选择后的最少变量数 |

### 因子选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--factor-method` | fixed | 因子数选择方法（fixed/cumulative） |
| `--pca-threshold` | 0.9 | PCA累积方差阈值 |

### 输出配置

| 参数 | 说明 |
|------|------|
| `--output-dir` | 指定输出目录 |
| `--no-export` | 不导出结果文件（仅显示结果） |

### 其他选项

| 参数 | 说明 |
|------|------|
| `--verbose` | 显示详细输出 |
| `--show-config` | 显示配置并退出（不执行训练） |
| `--help` | 显示帮助信息 |

## 使用示例

以下示例均从code目录运行，如需从其他目录运行，请相应调整路径。

```bash
cd dashboard/models/DFM/tests/code
```

### 示例1：基本训练

最简单的训练命令：

```bash
python test_train.py --quick-test
```

### 示例2：指定输出目录

```bash
python test_train.py \
  --quick-test \
  --output-dir "../result/my_experiment"
```

### 示例3：启用变量选择

```bash
python test_train.py \
  --config example_config.json \
  --enable-selection \
  --min-variables 10
```

### 示例4：使用PCA自动选择因子数

```bash
python test_train.py \
  --quick-test \
  --factor-method cumulative \
  --pca-threshold 0.85
```

### 示例5：仅查看配置

检查配置是否正确，但不执行训练：

```bash
python test_train.py \
  --config my_config.json \
  --show-config
```

### 示例6：使用自定义数据

```bash
python test_train.py \
  --data-path "path/to/your/data.csv" \
  --target "我的目标变量" \
  --train-end "2023-12-31" \
  --validation-start "2024-01-01" \
  --validation-end "2024-06-30" \
  --k-factors 4 \
  --output-dir "../result/experiment_001"
```

## 配置文件说明

### 配置文件格式

配置文件使用JSON格式，包含以下主要部分：

1. **数据配置** - 数据文件路径和变量选择
2. **日期配置** - 训练/验证期划分
3. **模型参数** - DFM模型参数
4. **变量选择配置** - 是否进行变量选择
5. **因子选择配置** - 因子数确定方法
6. **输出配置** - 结果保存位置

### 配置文件示例

#### 最小配置

```json
{
  "data_path": "../data/dfm_prepared_output.csv",
  "target_variable": "规模以上工业增加值:当月同比",
  "train_end": "2023-12-31",
  "validation_start": "2024-01-01",
  "validation_end": "2024-06-30"
}
```

注意：`data_path` 是相对于配置文件所在目录（code目录）的路径。

#### 完整配置

参考 `config_template.json` 文件查看所有可用参数。

## 输出说明

### 控制台输出

训练过程中会在控制台显示：

1. **训练配置摘要** - 包括日期范围、样本数、变量数、因子数
2. **变量选择过程**（如果启用）
   - 基线模型RMSE
   - 每轮剔除的变量和RMSE改善
   - 进度条
3. **最终模型摘要** - 最终变量数、因子数、RMSE等
4. **详细评估指标** - 训练期和验证期的各项指标

### 导出文件

默认会导出以下文件到输出目录（或临时目录）：

1. **模型文件** - `final_dfm_model_<timestamp>.joblib`
   - 训练好的DFM模型对象
   - 可用于后续预测

2. **元数据文件** - `final_dfm_metadata_<timestamp>.pkl`
   - 完整的训练元数据
   - 包含变量列表、评估指标、PCA结果等

### 禁用文件导出

如果只想查看结果而不保存文件：

```bash
python test_train.py --quick-test --no-export
```

## 测试数据说明

项目提供的测试数据位于 `dashboard/models/DFM/tests/data/` 目录：

- **dfm_prepared_output.csv** - 预处理后的时间序列数据
  - 第一列：Date（日期）
  - 第二列：规模以上工业增加值:当月同比（目标变量）
  - 其他列：各类经济指标（约89个）

- **dfm_prepared_output_industry_map.csv** - 行业映射表
  - 将各个指标映射到对应的行业分类

## 常见问题

### Q1: 如何选择合适的因子数？

A:
- 如果不确定，可以使用PCA方法：`--factor-method cumulative --pca-threshold 0.85`
- 经验值：3-5个因子通常能解释大部分方差
- 可以通过多次实验比较不同因子数的效果

### Q2: 训练很慢怎么办？

A:
- 减少变量数：使用 `--indicators` 只选择关键指标
- 减少迭代次数：`--max-iter 20`
- 不启用变量选择（变量选择会进行多次模型训练）

### Q3: 如何提高模型准确度？

A:
- 启用变量选择：`--enable-selection`
- 调整训练/验证期划分
- 尝试不同的因子数
- 检查数据质量和缺失值处理

### Q4: 如何调试配置问题？

A:
```bash
# 先查看配置是否正确（从code目录运行）
python test_train.py --config my_config.json --show-config

# 使用verbose模式查看详细信息
python test_train.py --config my_config.json --verbose
```

### Q5: 输出文件在哪里？

A:
- 如果指定了 `--output-dir`，文件在该目录下
- 否则在系统临时目录（训练完成后会显示具体路径）
- 文件名包含时间戳，便于区分不同的训练结果

## 进阶使用

### 批量实验

在code目录创建一个shell脚本进行批量实验：

```bash
#!/bin/bash
# experiment.sh - 放在 dashboard/models/DFM/tests/code/ 目录下

# 实验1：不同因子数
for k in 2 3 4 5; do
  python test_train.py \
    --quick-test \
    --k-factors $k \
    --output-dir "../result/exp_k${k}"
done

# 实验2：有无变量选择对比
python test_train.py \
  --quick-test \
  --output-dir "../result/no_selection"

python test_train.py \
  --quick-test \
  --enable-selection \
  --output-dir "../result/with_selection"
```

运行方式：
```bash
cd dashboard/models/DFM/tests/code
chmod +x experiment.sh
./experiment.sh
```

### Python脚本调用

也可以直接在Python脚本中调用：

```python
from dashboard.models.DFM.train import DFMTrainer, TrainingConfig

# 创建配置
config = TrainingConfig(
    data_path="dashboard/models/DFM/tests/data/dfm_prepared_output.csv",
    target_variable="规模以上工业增加值:当月同比",
    selected_indicators=[],
    train_end="2023-12-31",
    validation_start="2024-01-01",
    validation_end="2024-06-30",
    k_factors=3
)

# 创建训练器并训练
trainer = DFMTrainer(config)
result = trainer.train(
    progress_callback=lambda msg: print(msg),
    enable_export=True
)

# 查看结果
print(f"验证期RMSE: {result.metrics.oos_rmse:.4f}")
```

## 技术支持

如有问题，请参考：
- 主模块README: `dashboard/models/DFM/train/README.md`
- 配置模板: `config_template.json`
- 示例配置: `example_config.json`
