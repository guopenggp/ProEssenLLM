# 文件夹 1 与文件夹 722 代码架构分析报告

本报告是在创建和修改 ProEssenLLM 前完成的只读分析结论。原始文件夹 `1` 与 `722` 未被修改。

## 文件夹 1

### 模型结构

模型接收预计算的逐残基 1280 维特征，经过两层输入投影、LayerNorm、通道重标定和可学习位置编码，再进入多层 pre-norm Transformer。序列聚合同时使用 attention、masked mean 和 masked max，并通过可学习 gate 合并，最终进入残差分类头。模型还包含监督对比头和隐式序列比对组件；主训练脚本使用分类与对比特征，但隐式序列比对没有进入常规训练/验证路径。

优势是对 padding、全 mask、截断、噪声和梯度 checkpoint 有较多防御；不足是模型、实验性 zero-shot 组件和常规分类接口耦合，存在未被主流程验证的功能。

### 数据读取、Dataset 与 DataLoader

metadata 通过 pickle 读取，标签来自指定列，species 来自 `group`。切分以 species 为循环单位，再对每个 species 的正负样本分别随机拆分，因此 train/validation/test 是同物种覆盖、样本级隔离。代码另行处理排除物种、仅正类物种、低样本物种和临时 zero-shot 列表。

Dataset 从 LMDB 读取固定 key，校验二维特征维度并转换为 float32；短序列补零，长序列支持 head、tail 和 head-tail 截断。DataLoader 初始为随机 shuffle，随后主脚本可能重建 sampler 或 batch sampler。

主要风险是数据规则散落在数据模块与训练脚本两处；positive-only 与临时 zero-shot 逻辑使 “within species” 协议边界不够单一；worker 随机种子没有完整显式管理。

### Sampler

共有四类训练方式：普通 shuffle、样本加权的 species-balanced、全局标签平衡 batch，以及 species-label-balanced batch。最后一种先选择多个 species，再为每个 species 固定抽取正负样本，是最符合本项目要求的实现。

可迁移点是物种与标签双层采样思想。需修复点是 sampler 内部持有持续推进的 NumPy RNG，而没有标准 `set_epoch()`；当物种数少于请求数时允许有放回抽 species，不能严格保证 batch 内 distinct species 数量。

### Loss

主损失组合 species 正类权重 focal、全局 pairwise AUC、soft-Fbeta 和可选监督对比损失；positive-only 数据还可使用额外 BCE。数值稳定性处理较好，pair 数设有上限。

优势是适合不平衡分类；风险是目标较多、超参数耦合，且全局 pairwise AUC 没有明确区分跨物种排序与物种内排序。

### Training loop、validation 与 evaluation

训练支持 AMP、梯度裁剪、cosine scheduler、SWA、label smoothing 和 loss 分量记录。validation 每个 epoch 分别优化 F1 与 balanced-accuracy 阈值，并保存 AUC、BA、composite 三类 checkpoint。最终 test 使用 validation 阈值。逐物种评价只保留同时含正负类的 species，单类 species 被静默跳过。

优点是训练功能完整、阈值来自 validation；不足是 checkpoint 逻辑重复、保存格式缺少架构/版本兼容校验，逐物种监控只在最终阶段出现，early stopping 没有利用物种分布尾部表现。

## 文件夹 722

### 模型结构

主体支持两种编码方式：读取 frozen LMDB residue features，或离线加载 fair-esm 并只训练 LoRA adapter。下游特征编码器包含输入投影、多层 pre-norm Transformer、attention/mean/max 三路 pooling 和残差分类头。LoRA 注入支持限定最后 N 层、可选 LayerNorm 训练和 ESM gradient checkpointing。

相较文件夹 1，模型边界更清楚，ESM base 参数冻结检查和双学习率 optimizer 更适合大模型训练。

### 数据读取、Dataset 与 DataLoader

metadata 支持 pickle、CSV 和 TSV，并将 sample index、LMDB key、species ID、species code 和标签规范化为内部列。Dataset 分成 frozen LMDB 与 raw sequence 两种，collator 统一输出标签、species 元数据与模型输入。

原实现要求 LMDB 必须包含 `__metadata__`，对旧 LMDB 兼容性不足；序列截断只有 head 截断。DataLoader 的 worker seed 与 generator 管理优于文件夹 1。

### within_species 流程

每个 species 的正负类分别拆分，保持同 species 覆盖。单类 species 支持 exclude、train-only 或 error。训练 sampler 只有普通 shuffle 和基于 species 样本量幂次的 WeightedRandomSampler；它能减弱大物种主导，但不能保证 batch 内同时出现多个 species，也不直接控制正负比例。

Loss 使用训练集内计算的 species 正类权重 focal，并把 pairwise AUC 显式拆成 global 与 within-species 两部分，这是文件夹 722 的重要优势。Metrics 同时计算 pooled micro AUC、species macro AUC、q25 AUC 和尾部 AUC，并排除无法计算 AUC 的单类 species。

### species_holdout 流程

target species 首先从 source 数据移除；validation 使用独立 source species，train 使用剩余 source species。代码对样本集合做两两不相交检查，并在 holdout 模式进一步检查 train、validation、target 三者的 species 集合不相交。自动 validation species 由固定随机种子选择，支持用户显式指定。

zero-shot 逻辑的核心是共享模型仅从蛋白序列/特征学习，不输入 species ID；species ID 只进入 loss 和 metrics。target species 不参与训练或 epoch checkpoint 选择。原代码最终加载 checkpoint 后重新用 validation 优化阈值，虽然仍未触碰 target，但未直接使用 checkpoint 已保存阈值，削弱了实验可追溯性。

### Checkpoint 与 early stopping

checkpoint 基于 validation 的 micro/macro/q25 AUC 加权分数，macro balanced accuracy 和 validation loss 作为 tie-break；early stopping 使用同一候选规则。该逻辑优于只看 pooled AUC。

不足包括：所谓尾部指标是最差 5 个 species 的均值，并非真正 worst-species AUC；checkpoint 只保存训练参数但未记录状态类型、架构指纹和格式版本；optimizer 已保存但缺少 scheduler/scaler；最终阈值未直接复用 checkpoint 中的值；训练和 validation 没有统一逐物种 CSV。

## 优势归纳

### 文件夹 1 优势

1. species-label-balanced batch 的采样思想最符合已知物种性能均衡目标。
2. head-tail 截断、特征形状校验和 padding 防御更完整。
3. species 正类权重、AMP、梯度裁剪和较丰富的训练诊断可复用。
4. 已有逐物种指标与阈值优化基础。

### 文件夹 722 优势

1. species_holdout 的 species 级拆分与显式泄漏断言更严谨。
2. frozen-feature 与 ESM-LoRA 共用下游模型，适合统一框架。
3. global/within-species pairwise AUC 的损失分解更清晰。
4. micro、macro、q25 分布指标驱动 checkpoint，关注跨 species 稳健性。
5. metadata、Dataset、collator 和训练配置的职责边界更清楚。

## 迁移与重构决策

| 模块 | 融合来源 | ProEssenLLM 决策 |
|---|---|---|
| 模型与 ESM-LoRA | 722 | 保留双编码模式，统一类名，增加 checkpoint 状态类型与架构指纹 |
| frozen LMDB Dataset | 1 + 722 | 保留动态 padding，补入 head/tail/head-tail，支持无 metadata 的旧 LMDB |
| within-species split | 1 + 722 | 每 species、每标签分层；协议中不再混入 target species 临时分支 |
| species-holdout split | 722 | target 直接成为最终 test；train/validation/test species 三集合严格不交 |
| batch sampler | 1 | 改为 `set_epoch()` 可复现实现；每批 distinct species；每 species 固定正负配额 |
| Loss | 722 + 1 | species positive focal + global/within-species pairwise ranking，去除未验证的多辅助目标 |
| Metrics | 722 + 1 | 补 AUPR、F1、balanced accuracy、真实 worst-species AUC；所有 species 均报告 |
| Checkpoint | 722 | 支持单指标或四项 AUC composite；validation-only；保存 threshold/source/版本/架构 |
| Monitoring | 新增 | 每 epoch 输出 train/validation 每 species 六项统计，统一写入 CSV |
| 安全审计 | 722 + 新增 | 写出 sample/species overlap、target isolation、模型输入和 test 访问时机 |

## 需要重点重构的模块

1. 将单文件训练脚本拆成 `datasets`、`samplers`、`losses`、`evaluation`、`trainers` 和 `models`。
2. 把 `split_mode` 提升为统一入口的 `--mode`，让两种协议在参数验证层即互斥。
3. 把样本级 species 权重采样升级为能保证 batch 结构的 species-label-balanced batch sampler。
4. 用同一个版本化 checkpoint manager 替代重复保存与加载逻辑。
5. 将 target test 的首次模型推理推迟到 checkpoint 已锁定并重新加载之后。
6. 将逐物种指标从最终附加报告升级为 train/validation 的 epoch 级核心监控。
7. 删除旧项目名称在代码文件、类、变量和配置中的命名痕迹。
