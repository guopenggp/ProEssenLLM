# 代码安全与实验纪律检查

## 模式独立性

- `within_species` 只执行同物种内样本级分层，拒绝 `target_species` 和 `validation_species`。
- `species_holdout` 强制提供 `target_species`，并执行 train/validation/test 的 species 级不相交检查。
- 两种模式共用模型实现，但不共用拆分协议或 target 评价逻辑。

## Species leakage

每次运行生成 `leakage_audit.json`。审计覆盖所有 active split 的 sample index 两两交集。holdout 模式额外覆盖 species ID 两两交集、target 是否缺席 train/validation，以及 test species 是否恰好等于 target 集合。

## Species 是否进入模型

模型 forward 只允许 residue feature/mask 或 token/attention mask。species ID/code 仅用于 batch sampler、正类权重、物种内 pairwise loss 和指标分组。泄漏审计会检查模型输入键中不存在 species 字段。

## 训练、选择与测试隔离

训练循环只访问 train 与 validation DataLoader。checkpoint 和阈值都来自 validation。test DataLoader 在训练结束、最佳 checkpoint 重新加载以后才首次传入模型。holdout 模式下 test 就是 target species。

## 随机种子与复现

Python、NumPy、PyTorch、CUDA、DataLoader generator 和 worker 都显式设种子。species-balanced sampler 使用 `seed + epoch` 并实现 `set_epoch()`。默认开启 deterministic algorithm；如底层算子只能使用非确定实现，PyTorch 会发出警告。

## Checkpoint 兼容性

checkpoint 包含项目标识、格式版本、模式、encoder 类型、架构参数与 SHA-256 指纹、model state 类型、optimizer/scheduler/scaler、validation threshold、选择来源和配置。加载时验证项目、格式、模式、架构和选择来源。frozen 模式保存完整模型；ESM-LoRA 模式保存所有可训练参数，并要求同一冻结 backbone 可用。

## 已知边界

- 输入 metadata 和 LMDB 的生物学同源关系去重不属于代码可自动推断的范围；如果论文协议要求按同源簇隔离，应在 metadata 中提前形成 cluster-aware split。
- GPU 完全 bitwise 复现仍受 CUDA、驱动和具体算子版本影响；配置与软件环境应随论文归档。
- 单类 species 的 AUC 标记为不可用，不用 0.5 伪造；它仍会报告 AUPR、F1 和 balanced accuracy 中可定义的部分。
