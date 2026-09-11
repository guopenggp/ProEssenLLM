# 代码安全与实验纪律检查

## 实验协议隔离

- `within_species` 只执行同物种内样本级分层，拒绝 `target_species` 和 `validation_species`。
- `species_holdout` 强制提供 `target_species`，并执行 train、validation 和 test 的 species 级不相交检查。
- 两种协议共用模型实现，但不共用拆分规则或 target 评价逻辑。

## 输入边界

- 模型输入仅为 `residue_features` 和 `valid_mask`。
- species ID/code 只用于拆分、batch sampler、正类权重、物种内 pairwise loss 和指标分组。
- LMDB 特征必须是非空二维数组，并且所有样本的特征维度一致。
- Metadata 标签必须可转为 0/1，字符串化后的 sample index 和 LMDB key 必须唯一。
- 项目使用 pickle 读取 metadata 和 LMDB payload，因此只应加载可信数据。

离线特征构建器会拒绝空输入、非 0/1 标签、空 species、重复 LMDB key 和无可用序列的输入。非法字符与内部终止符的处理策略会写入 metadata JSON；正式实验应归档该文件。

## Species leakage

每次运行生成 `leakage_audit.json`。审计覆盖所有 active split 的 sample index 两两交集。holdout 协议额外覆盖 species ID 两两交集、target 是否缺席 train/validation，以及 test species 是否与 target 集合完全一致。

这类检查只能发现 ID 层面的直接重叠，不能自动发现同源蛋白、重复序列或系统发育近邻造成的间接泄漏。若论文协议要求按同源簇隔离，应在生成 metadata 前完成聚类，并以 cluster 为拆分单位。

## 训练、选择与测试隔离

训练循环只用 train 更新参数，只用 validation 选择 checkpoint、执行 early stopping 和优化阈值。test DataLoader 虽在训练前构建，但直到最佳 checkpoint 重新加载以后才首次传给评价函数。holdout 协议下 test 即 target species。

最终 `leakage_audit.json` 会记录 checkpoint epoch、阈值、两者的 validation 来源，以及 test 是否在 checkpoint 加载后执行。

## 随机性与复现

Python、NumPy、PyTorch、CUDA、DataLoader generator 和 worker 都显式设置随机种子。species-balanced sampler 使用 `seed + epoch`，并实现 `set_epoch()`。

默认启用 deterministic algorithms；底层只能使用非确定性实现时，PyTorch 会给出警告而不是静默忽略。GPU 完全 bitwise 复现仍依赖 PyTorch、CUDA、cuDNN、驱动、硬件和具体算子版本，正式实验应归档完整软件环境。

## Checkpoint 完整性

Checkpoint 格式版本为 2，保存项目标识、框架版本、实验模式、frozen LMDB 输入来源、架构参数与 SHA-256 指纹、完整模型 state、optimizer/scheduler/scaler、validation threshold、选择来源、物种集合和配置。

加载时验证：

1. 项目标识与 checkpoint 格式版本；
2. 当前实验模式；
3. 架构指纹；
4. checkpoint 与 threshold 是否来自 validation；
5. target 和 validation species 是否与当前拆分一致。

任何不一致都会终止加载，避免把旧模型或不同实验协议的权重混入当前结果。

## 输出与隐私

特征构建器生成的 companion table、metadata JSON、`split_statistics.json` 和 `test_predictions.csv` 会保存 sample ID 或源文件路径；如果其中含内部标识或敏感信息，应在共享结果前脱敏。Git 默认忽略 `data/`、LMDB、模型 checkpoint 和 `results/`，但上传前仍应执行 `git status` 检查暂存内容。

## 已知边界

- 输入数据的生物学正确性、标签来源和授权范围需要由数据提供者确认。
- 自动泄漏审计无法替代研究者对同源性、批次效应与系统发育偏差的检查。
- 单类 species 的 AUC 标记为不可用；不会用 0.5 或其他常数伪造。
- `species_balanced` 会有放回地重复抽样稀有类别；报告结果时应同时保留 sampler 统计。
