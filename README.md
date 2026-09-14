# ProEssenLLM

ProEssenLLM 是一个基于蛋白质大语言模型的原核生物必需性二分类训练框架。项目提供两种严格分离的实验协议、物种均衡采样、面向类别不平衡的损失、逐物种评价，以及带泄漏审计的 validation-only 模型选择流程。

训练阶段只接受 frozen LMDB residue features。仓库提供独立的 `build_esm_lmdb.py`，可在训练前将蛋白质序列转换为兼容的特征 LMDB。默认特征生成和训练截断长度均为 **1000** 个残基。

## 功能概览

- `within_species`：在每个符合条件的物种内部按标签拆分 train、validation 和 test，用于已知物种预测。
- `species_holdout`：按物种拆分，target species 只用于最终 test，用于未知物种 zero-shot 评价。
- 独立的序列预处理脚本：从 FASTA 或表格批量生成 ESM residue embeddings 与配套 metadata。
- species-label-balanced batch sampler：控制每批物种数以及每个物种的正负样本比例。
- focal loss 与 global/within-species pairwise AUC loss。
- pooled 与逐物种 AUC、AUPR、F1、balanced accuracy、q25 AUC 和 worst-species AUC。
- validation-only checkpoint、early stopping 和分类阈值选择。
- 自动生成样本/物种泄漏审计、训练历史、逐物种指标和测试预测。

## 工程结构

```text
ProEssenLLM/
├── train.py                         # 统一训练入口
├── build_esm_lmdb.py                # 蛋白质序列转 frozen-feature LMDB
├── requirements-feature-builder.txt # 离线特征生成的可选依赖
├── configs/                         # 参数解析和两种实验协议的 YAML 示例
├── datasets/                        # metadata、拆分、LMDB Dataset 与 DataLoader
├── models/                          # residue-feature Transformer 与分类器
├── trainers/                        # 共享训练引擎及两个协议入口
├── losses/                          # focal 与 pairwise AUC 损失
├── samplers/                        # species-label-balanced batch sampler
├── evaluation/                      # 指标、checkpoint、泄漏审计和运行比较
├── tests/                           # 单元测试与端到端 smoke test
└── docs/                            # 架构、实验安全和性能比较说明
```

## 环境要求与安装

- Python 3.9 或更高版本
- PyTorch 2.0 或更高版本
- 可选 NVIDIA CUDA GPU；没有可用 CUDA 时，程序会回退到 CPU

建议在虚拟环境中安装：

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要特定 CUDA 版本，请先按照 PyTorch 官方说明安装匹配的 PyTorch，再安装其余依赖。

如果需要从蛋白质序列生成 LMDB，额外安装离线特征构建依赖：

```bash
python -m pip install -r requirements-feature-builder.txt
```

## 输入数据格式

### 从序列生成 LMDB

`build_esm_lmdb.py` 支持 FASTA、pickle DataFrame、CSV、TSV 和 TXT。表格输入默认需要 `ID`、`group`、`essential` 和 `sequence` 四列：

```bash
python build_esm_lmdb.py \
  --data_path ./data/proteins.csv \
  --output_lmdb ./data/features.lmdb \
  --model_name ./pretrained_model/esm2_t33_650M_UR50D.pt \
  --truncation_seq_length 1000 \
  --device cuda:0
```

`--model_name` 可以是本地 fair-esm checkpoint 路径，也可以是 fair-esm 支持的模型名称。默认使用 `./pretrained_model/esm2_t33_650M_UR50D.pt`；本地模型权重不会被 Git 跟踪。

FASTA 输入要求 header 的第一个 token 至少符合：

```text
>index_ID_group_label
```

其中 `label` 必须以 `0` 或 `1` 开头。表格列名可通过 `--id_column`、`--group_column`、`--label_column` 和 `--sequence_column` 修改。如果希望使用某一列作为 LMDB key，可指定：

```bash
--key_source column --index_column index
```

构建脚本会：

- 将序列去除空白并转为大写；
- 移除末端终止符，对内部终止符和非法字符执行 `mask`、`remove` 或 `skip` 策略；
- 检测重复 LMDB key，并汇总重复序列、跨物种重复和标签冲突；
- 对相同序列只计算一次 embedding，再为每个 sample 写入独立记录；
- 从实际模型自动读取 feature dimension，而不是写死维度；
- 写入训练代码可读取的 `__metadata__` 和 `feature_length`。

默认还会在 LMDB 旁生成：

```text
features.lmdb.metadata.json
features.metadata_table.pkl
features.metadata_table.csv
```

companion table 包含显式 `lmdb_key`，可直接作为训练 metadata：

```bash
python train.py \
  --mode within_species \
  --data_path ./data/features.metadata_table.pkl \
  --feature_dir ./data/features.lmdb
```

显存不足时降低 `--toks_per_batch`。已存在的输出默认不会覆盖；只有显式传入 `--overwrite` 才会替换目标 LMDB 文件及其 lock 文件。完整参数可通过 `python build_esm_lmdb.py --help` 查看。

### Metadata

`--data_path` 支持 `.pkl`、`.pickle`、`.csv`、`.tsv` 和 `.txt`。表格至少包含：

| 字段 | 默认名称 | 要求 |
|---|---|---|
| 标签 | `essential` | 二分类整数，只能为 `0` 或 `1` |
| 物种 | `group` | species ID；数值或字符串均可 |
| LMDB key | `lmdb_key`（可选） | 未提供时使用 DataFrame index 的字符串形式 |

Metadata index 转为字符串后必须唯一，LMDB key 也必须唯一。可以用 `--label_name` 和 `--group_column` 指定不同的标签列和物种列。

### Frozen-feature LMDB

`--feature_dir` 指向 LMDB 文件或目录。每个 sample value 应为 pickle 序列化的映射，并包含二维 `feature` 数组：

```python
{
    "feature": numpy.ndarray(shape=(protein_length, feature_dimension))
}
```

项目会优先从 LMDB 的 `__metadata__` key 读取 `feature_length`：

```python
{"feature_length": 1280}
```

如果旧 LMDB 没有该 metadata，程序会从训练集首个可用样本推断特征维度。所有样本的第二维必须一致。`--input_size` 可用于显式校验该维度，不指定时自动解析。

超过 `--max_length` 的残基特征会按 `--truncate_strategy` 截断。支持 `head`、`tail` 和默认的 `head_tail`；默认 `max_length` 为 **1000**。短序列在 batch 内动态补零，并通过 mask 排除 padding。

> 仅加载自己生成或可信来源的 pickle/LMDB 文件；pickle 反序列化不适合不可信输入。

## 快速开始

### 已知物种：within_species

编辑路径后直接使用示例配置：

```bash
python train.py --config configs/within_species.yaml
```

等价的常用命令行示例：

```bash
python train.py \
  --mode within_species \
  --data_path ./data/metadata.pkl \
  --feature_dir ./data/features.lmdb \
  --save_path ./results/within_species \
  --max_length 1000 \
  --sampling_strategy species_balanced \
  --species_per_batch 8 \
  --samples_per_species 16 \
  --positive_fraction 0.4 \
  --batch_size 128
```

每个物种的正负类分别拆分，从而在 train、validation 和 test 中维持同物种覆盖。`within_species` 不接受 `target_species` 或 `validation_species`。

### 未知物种：species_holdout

先将 `configs/species_holdout.yaml` 中的占位 target species 替换为真实 ID，然后运行：

```bash
python train.py --config configs/species_holdout.yaml
```

命令行示例：

```bash
python train.py \
  --mode species_holdout \
  --data_path ./data/metadata.pkl \
  --feature_dir ./data/features.lmdb \
  --save_path ./results/species_holdout \
  --target_species 722 \
  --validation_species_ratio 0.2
```

`target_species` 完全排除在 train 和 validation 之外。默认从 source species 中按固定随机种子选择 validation species；如需固定集合，可使用 `--validation_species 101 205`。target、validation 和 exclude species 不能重叠。

### 配置文件与命令行覆盖

配置文件支持 YAML 或 JSON，键名与命令行参数名一致。命令行中显式提供的值会覆盖配置文件值：

```bash
python train.py \
  --config configs/within_species.yaml \
  --device cuda:1 \
  --num_epochs 200
```

查看全部命令行参数；主要默认值见下一节：

```bash
python train.py --help
```

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--max_length` | `1000` | 保留的最大残基数 |
| `--truncate_strategy` | `head_tail` | 超长特征的截断方式 |
| `--hidden_size` | `320` | 下游 Transformer 隐藏维度 |
| `--num_layers` | `3` | Transformer block 数量 |
| `--num_heads` | `4` | attention heads 数量 |
| `--batch_size` | `128` | 普通 batch 大小；均衡采样时须满足下式 |
| `--sampling_strategy` | `species_balanced` | `random` 或 `species_balanced` |
| `--learning_rate` | `1e-4` | AdamW 初始学习率 |
| `--num_epochs` | `150` | 最大训练 epoch 数 |
| `--early_stopping_patience` | `15` | early stopping patience |
| `--device` | `cuda:0` | 训练设备；CUDA 不可用时自动回退 CPU |

使用 `species_balanced` 时必须满足：

```text
batch_size = species_per_batch × samples_per_species
```

每个 batch 中抽取互不重复的 species，并按 `positive_fraction` 控制每个 species 的正负样本构成。使用 `--sampling_strategy random` 可改为普通随机采样。

## Checkpoint 选择

`--checkpoint_metric` 支持 `micro_auc`、`macro_auc`、`q25_species_auc`、`worst_species_auc` 和默认的 `composite`。默认复合分数为：

```text
0.55 × micro AUC
+ 0.20 × macro AUC
+ 0.15 × q25 species AUC
+ 0.10 × worst species AUC
```

四项权重可通过对应的 `--*_auc_weight` 参数调整，但总和必须为 1。分类阈值只使用 validation balanced accuracy 优化，并保存在最佳 checkpoint 中；test/target 不参与模型选择或阈值优化。

## 输出文件

默认输出目录为 `results/<mode>`，也可通过 `--save_path` 修改。

| 文件 | 内容 |
|---|---|
| `config.json` | 完整运行配置、解析后的输入维度和物种集合 |
| `best_validation_model.pt` | 带格式版本、架构指纹和 validation 来源的 checkpoint |
| `split_statistics.json` | 各 split 的样本、标签和物种统计 |
| `leakage_audit.json` | 样本/物种重叠、target 隔离和选择纪律检查 |
| `model_statistics.json` | 模型总参数量与可训练参数量 |
| `species_positive_weights.json` | 训练集计算的逐物种正类权重 |
| `training_history.json` | 每个 epoch 的 train/validation 指标和学习率 |
| `sampler_statistics.json` | 每个 epoch 的采样统计 |
| `species_performance.csv` | 每个 epoch 及最终评价的逐物种指标 |
| `validation_results*.json` | 最佳 checkpoint 的 validation 汇总及逐物种结果 |
| `test_results*.json` | 最终 test 汇总及逐物种结果 |
| `test_predictions.csv` | sample-level test 概率、预测和标签 |
| `run_summary.json` | 运行摘要及最终结果路径 |

`species_holdout` 还会生成 `target_species_results.json` 和 `target_species_results_per_species.json`。

## 测试

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

无需真实数据的端到端 smoke test 会创建临时 metadata 和 LMDB，各训练一个 epoch：

```bash
python tests/smoke_training.py --mode within_species
python tests/smoke_training.py --mode species_holdout
```

## 比较实验

在相同数据、物种集合、随机种子和训练预算下完成两组运行后，可以生成指标差异表：

```bash
python evaluation/compare_runs.py \
  --mode within_species \
  --baseline_dir ../baseline_results \
  --proessenllm_dir ./results/within_species \
  --output_dir ./comparison
```

比较工具输出 AUC、AUPR、balanced accuracy、F1、macro AUC 和 worst-species AUC 的 CSV 与 Markdown。公平比较约束见 [docs/performance_comparison.md](docs/performance_comparison.md)。

## 复现与实验安全

- Python、NumPy、PyTorch、CUDA、DataLoader 和 sampler 都使用显式随机种子。
- 默认启用 deterministic algorithms；具体 GPU、CUDA、驱动和算子版本仍可能影响 bitwise 复现。
- checkpoint 和阈值只根据 validation 选择。
- test DataLoader 仅在训练结束并重新载入最佳 checkpoint 后用于最终推理。
- `species_holdout` 会显式断言 train、validation、target species 三者不相交。
- 代码不会自动执行生物学同源聚类；如研究设计要求 cluster-level isolation，应在制作 metadata 时先完成同源簇划分。

更多内容见 [架构说明](docs/code_architecture_analysis.md) 和 [实验安全检查](docs/safety_review.md)。

## 从早期版本迁移

版本 1.1.0 删除了训练阶段的在线 ESM 微调路径及其命令行参数，只保留预计算 LMDB 特征输入。离线 ESM 仅由 `build_esm_lmdb.py` 用于训练前的数据准备。旧训练配置中的编码模式、ESM 模型路径、ESM 专用学习率、梯度检查点和低秩适配器参数需要删除。Checkpoint 格式已升级为 v2，旧格式 checkpoint 不会被静默载入。

## GitHub 发布注意事项

`.gitignore` 默认排除本地数据、LMDB、模型 checkpoint、虚拟环境和生成结果。提交前建议：

```bash
python -m unittest discover -s tests -v
git status
```

本仓库当前未声明开源许可证。公开发布前请根据实际授权要求添加合适的 `LICENSE`；没有许可证时，GitHub 上的代码默认不代表允许他人复制、修改或再分发。
