# ProEssenLLM 架构说明

## 设计目标

ProEssenLLM 面向多物种蛋白质必需性二分类，使用预计算的 residue-level embeddings 作为唯一训练输入。独立的 `build_esm_lmdb.py` 在训练前负责将蛋白质序列转换为 frozen-feature LMDB；它不是训练模型的一部分。工程将“已知物种内预测”和“未知物种 zero-shot 预测”实现为两个明确的实验协议，并复用模型、损失、采样、评价和 checkpoint 基础设施。

核心约束如下：

1. 模型只接收蛋白质 residue features 与 padding mask，不接收 species ID。
2. checkpoint、early stopping 和分类阈值只由 validation 数据决定。
3. `species_holdout` 的 target species 在最佳 checkpoint 锁定前不得进入模型评价。
4. 所有数据拆分、采样与训练随机源均可由 `random_seed` 重放。

## 数据流

```text
FASTA/table --(offline feature builder)--> metadata table + feature LMDB
            │
            ▼
  normalize and validate
            │
            ▼
  protocol-specific split
   ├─ within_species: 每个 species 内按标签拆分
   └─ species_holdout: train/validation/test 的 species 不相交
            │
            ▼
  FrozenLMDBDataset → dynamic padding + valid_mask
            │
            ▼
  residue feature encoder → three-way pooling → classifier
            │
            ▼
  validation-only selection → reload best checkpoint → final test
```

## 配置层

`configs/proessenllm_config.py` 负责：

- 合并 YAML/JSON 配置与命令行参数；
- 检查实验模式要求和物种集合冲突；
- 检查 hidden size 与 attention heads 的整除关系；
- 检查 species-balanced batch 的大小约束；
- 检查损失和 checkpoint 组合权重；
- 检查长度、epoch、梯度累积和 early stopping 等参数范围。

项目只支持 frozen-feature LMDB 输入，因此不存在运行时 encoder 分支。`max_length` 默认值为 1000，可在配置文件或命令行中显式覆盖。

## Metadata 与拆分

### 离线特征构建

`build_esm_lmdb.py` 加载预训练 ESM 模型，以长度感知的 token budget 组织推理 batch，并从指定 representation layer 提取逐残基特征。默认最多编码 1000 个残基，与训练默认长度一致。

特征宽度从所加载模型的 `embed_dim` 自动获取，并写入 LMDB 的 `__metadata__`。脚本还会生成带显式 `lmdb_key` 的 companion metadata，保证跳过无效序列或使用自定义 key 后，训练表仍能准确找到对应 LMDB 记录。重复序列只推理一次，但每个样本仍保留独立 key 和来源信息。

### 训练 Metadata

`datasets/proessenllm_dataset.py` 将外部字段规范化为内部列：

| 内部列 | 作用 |
|---|---|
| `_sample_index` | 字符串化的唯一 sample ID |
| `_lmdb_key` | LMDB 查询键 |
| `_species_id` | 规范化后的 species ID |
| `_species_code` | 仅供 loss 使用的整数 species code |
| `_label` | 0/1 标签 |

`within_species` 对每个 species 的正负类分别拆分。单类或小样本 species 根据配置执行 `exclude`、`train_only` 或 `error` 策略。

`species_holdout` 先将 target species 固定为 test，再从其余 source species 中选择 validation species，最后将剩余 species 用于训练。拆分结束后会验证 sample 集合两两不相交；holdout 协议还会验证 active split 的 species 集合两两不相交。

## LMDB Dataset 与 DataLoader

`FrozenLMDBDataset` 延迟打开 LMDB，适配多 worker DataLoader。每个样本读取后都会检查：

- `feature` 是否为二维数组；
- 特征维度是否与解析出的 `feature_length` 一致；
- residue 长度是否非零。

超长样本支持 head、tail 和 head-tail 截断，默认最多保留 1000 个残基。`collate_frozen` 只补到当前 batch 的最大长度，并生成布尔 `valid_mask`。

训练 DataLoader 支持普通随机采样或 `SpeciesLabelBalancedBatchSampler`。后者在每个 batch 中选择互不重复的 species，并从每个 species 按固定正负配额取样；`set_epoch()` 使用 `seed + epoch` 保证可复现且不同 epoch 顺序不同。

## 模型结构

`models/proessenllm_model.py` 包含三个主要部分：

1. `ProEssenLLMFeatureEncoder`：输入投影、多层 pre-norm Transformer block 与最终 LayerNorm。
2. 三路 pooling：attention pooling、masked mean pooling 和 masked max pooling。
3. `ProEssenLLMClassifier`：投影、残差 MLP、LayerNorm 和单 logit 输出。

Transformer block 在 attention 与 feed-forward 残差分支中使用 dropout 和 stochastic depth。所有 block 之后都会再次清零 padding 位置，避免 padding 表示在残差路径中累积。

模型 forward 签名只包含 `residue_features` 和 `valid_mask`。`MODEL_INPUT_KEYS` 被泄漏审计复用，用于证明 species 元数据没有进入模型。

## 损失与优化

训练损失由以下部分组成：

- 按训练集逐物种估计正类权重的 focal binary classification loss；
- global pairwise AUC ranking loss；
- within-species pairwise AUC ranking loss。

global 与 within-species 项按配置比例组合，再由 `auc_weight` 与 focal loss 合并。optimizer 使用 AdamW，学习率调度为 warmup 后 cosine decay；支持 AMP、梯度累积和梯度范数裁剪。

## 指标与模型选择

每轮训练都会计算 train 和 validation 的 pooled/逐物种指标，并写入统一 CSV。单类 species 的 ROC-AUC 记为不可用，而不是填充人为数值。

模型选择支持单一 AUC 指标或由 micro、macro、q25 和 worst-species AUC 组成的加权复合指标。分数接近时使用 macro balanced accuracy 和 validation loss 作为 tie-break。

分类阈值通过 validation balanced accuracy 选择，并与 checkpoint 一起保存。训练结束后重新加载最佳 checkpoint 及其阈值，随后才运行最终 test。

## Checkpoint 兼容性

Checkpoint 格式版本为 2，包含：

- 项目标识、框架版本、实验模式与 frozen LMDB 输入来源；
- 完整模型 state、optimizer、scheduler 与 AMP scaler state；
- 模型架构参数及其 SHA-256 指纹；
- validation 指标、阈值、选择来源与 early-stopping 状态；
- target/validation species 和完整运行配置。

加载时验证项目、格式版本、实验模式、架构指纹、选择来源和物种集合。旧 checkpoint 不会被静默加载到不同架构。

## 模块职责

| 模块 | 职责 |
|---|---|
| `configs` | 配置解析与跨参数验证 |
| `build_esm_lmdb.py` | 可选的离线序列清洗、ESM 推理与 LMDB 构建 |
| `datasets` | metadata、拆分、LMDB Dataset、collator 与 DataLoader |
| `samplers` | species-label-balanced batch 生成及统计 |
| `models` | residue-feature Transformer 与分类头 |
| `losses` | 类别不平衡与 AUC 排序损失 |
| `evaluation` | 指标、泄漏审计、checkpoint 与运行比较 |
| `trainers` | 训练循环、模型选择、最终评价和结果落盘 |

## 已知边界

- 工程不会生成 residue embeddings；输入特征必须提前计算并写入 LMDB。
- 代码无法仅凭 metadata 自动识别同源蛋白或系统发育相关性；如实验要求，应使用 cluster-aware 的预处理和拆分。
- 逐物种 AUC 需要该物种同时包含正负样本；单类物种只能报告可定义的指标。
- CUDA、驱动和算子版本可能影响 GPU 的完全 bitwise 复现。
