# ProEssenLLM

ProEssenLLM 是统一的蛋白必需性预测训练框架。它共享同一套模型、损失、采样与指标实现，但严格保留两种不同的实验协议：

- `within_species`：每个符合条件的物种内部按标签分层拆分 train/validation/test，用于已知物种预测。
- `species_holdout`：按物种拆分，target species 仅进入最终 test，用于未知物种 zero-shot 评价。

两种模式不会自动混合。`within_species` 拒绝 target species 参数；`species_holdout` 强制 target species，并保证 target 不参与训练、checkpoint 选择或阈值优化。

## 工程结构

```text
ProEssenLLM/
├── train.py                         # 统一入口
├── models/                          # 共享模型、ESM-LoRA
├── datasets/                        # 元数据、两种拆分、LMDB/序列 Dataset 与 DataLoader
├── trainers/                        # 共享引擎及两个模式入口
├── losses/                          # 物种正类权重、focal、全局/物种内 AUC 排序损失
├── samplers/                        # species-label-balanced batch sampler
├── evaluation/                      # 指标、泄漏审计、checkpoint、运行比较
├── configs/                         # 命令行配置和 YAML 示例
├── docs/                            # 分析、安全与性能说明
└── tests/                           # 切分、采样和指标测试
```

## 安装

```bash
python -m pip install -r requirements.txt
```

如需 `--encoder_mode esm_lora`：

```bash
python -m pip install -r requirements-esm.txt
```

默认输入是 metadata pickle 与 frozen-feature LMDB。metadata 至少包含：

- `essential`：二分类标签，取值只能为 0/1；
- `group`：species ID；
- 可选 `lmdb_key`：当 LMDB key 不等于 DataFrame index 时使用。

LMDB sample 至少包含二维 `feature` 数组。框架优先读取 `__metadata__` 中的 `feature_length`，也兼容通过首个训练样本推断旧 LMDB 的特征维度。

## 已知物种训练

```bash
python train.py \
  --mode within_species
```

常用显式参数：

```bash
python train.py \
  --mode within_species \
  --data_path ./data/metadata.pkl \
  --feature_dir ./data/features.lmdb \
  --save_path ./results/within_species \
  --sampling_strategy species_balanced \
  --species_per_batch 8 \
  --samples_per_species 16 \
  --positive_fraction 0.4
```

`species_balanced` 要求 `batch_size == species_per_batch × samples_per_species`。每个 batch 中的 species 不重复，每个 species 分别按 `positive_fraction` 抽取正负样本。选择 `--sampling_strategy random` 可恢复普通随机采样。

## 未知物种 zero-shot 训练

```bash
python train.py \
  --mode species_holdout \
  --target_species 722
```

推荐显式指定路径：

```bash
python train.py \
  --mode species_holdout \
  --data_path ./data/metadata.pkl \
  --feature_dir ./data/features.lmdb \
  --save_path ./results/species_holdout \
  --target_species 722 \
  --validation_species_ratio 0.2
```

如需固定 source validation species，可使用 `--validation_species 101 205`。target species 与 validation species 不得重叠。

## Checkpoint 选择

`--checkpoint_metric` 支持：

- `micro_auc`
- `macro_auc`
- `q25_species_auc`
- `worst_species_auc`
- `composite`（默认）

默认复合分数为：

```text
0.55 × micro AUC
+ 0.20 × macro AUC
+ 0.15 × q25 species AUC
+ 0.10 × worst species AUC
```

权重可通过四个 `--*_auc_weight` 参数调整，但必须合计为 1。阈值只根据 validation 的 balanced accuracy 优化，并随最佳 checkpoint 固化。

## 主要输出

- `best_validation_model.pt`：带版本、模式、架构指纹和选择来源的 checkpoint；
- `species_performance.csv`：每个 epoch 的 train/validation 逐物种指标及最终 test；
- `leakage_audit.json`：样本、物种、模型输入与选择纪律审计；
- `split_statistics.json`：完整拆分样本与物种统计；
- `training_history.json`、`sampler_statistics.json`；
- `validation_results.json`、`test_results.json` 及逐物种文件；
- `test_predictions.csv`。

## 比较原 722 运行

先用相同数据、物种集合、随机种子和训练预算分别完成运行，再执行：

```bash
python evaluation/compare_runs.py \
  --mode within_species \
  --baseline_dir ../baseline_722_results \
  --proessenllm_dir ./results/within_species
```

该工具输出 AUC、AUPR、Balanced Accuracy、F1、Macro AUC 和 Worst species AUC 的 CSV 与 Markdown 对比。详细公平比较规则见 `docs/performance_comparison.md`。

## 测试

```bash
python -m unittest discover -s tests -v
```

无需真实数据的端到端 smoke test：

```bash
python tests/smoke_training.py --mode within_species
python tests/smoke_training.py --mode species_holdout
```

完整设计审计见 `docs/code_architecture_analysis.md`，文件级修改记录见 `changed_files.md`。
