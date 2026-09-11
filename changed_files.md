# 修改记录

## 版本 1.1.0（2026-09-11）

本次修改将项目收敛为单一的 frozen-feature LMDB 输入流程，便于维护和公开发布。

### 代码

- 新增 `build_esm_lmdb.py`，用于从 FASTA/表格蛋白质序列离线生成训练兼容的 ESM residue-feature LMDB 与 companion metadata。
- 特征构建器自动记录实际 feature dimension、显式 `lmdb_key`、序列清洗统计和重复序列审计，并将默认生成长度设为 1000。
- 删除在线 ESM 编码、微调、低秩适配器注入和相应 gradient-checkpointing 实现。
- 删除 raw-sequence Dataset、序列 collator 和双输入 DataLoader 分支。
- 删除编码模式、ESM 路径、适配器超参数和双学习率 optimizer 参数。
- 将 `--max_length` 默认值从 768 调整为 1000，并增加正数校验。
- 简化模型 forward 为 `residue_features + valid_mask`，checkpoint 始终保存完整模型 state。
- 将框架版本升级为 1.1.0，将 checkpoint 格式升级为 2。
- 删除不再需要的可选 ESM 依赖文件。
- 新增 `requirements-feature-builder.txt`，将离线 ESM 特征生成依赖与核心训练依赖分开。

### 配置与测试

- 两个示例 YAML 都显式记录 `max_length: 1000`。
- 增加默认长度和已删除参数不再出现在 CLI 中的回归测试。
- 扩展 `.gitignore`，排除本地数据、LMDB、checkpoint、虚拟环境和编辑器文件。

### 文档

- 重写 README，补充环境、数据格式、运行方式、核心参数、输出文件、测试、复现与 GitHub 发布说明。
- 重写架构文档，使模块职责、数据流和 checkpoint v2 与当前代码一致。
- 更新安全审计，说明 pickle 信任边界、ID 泄漏检查范围和输出隐私风险。

## 主要文件职责

| 文件 | 职责 |
|---|---|
| `train.py` | 统一训练入口 |
| `build_esm_lmdb.py` | FASTA/表格序列清洗、ESM embedding 与 LMDB 构建 |
| `configs/proessenllm_config.py` | 命令行/YAML/JSON 配置与参数验证 |
| `datasets/proessenllm_dataset.py` | Metadata、拆分、LMDB Dataset 与 DataLoader |
| `models/proessenllm_model.py` | Residue-feature Transformer 和分类器 |
| `trainers/base_trainer.py` | 训练、validation-only 选择与最终评价 |
| `evaluation/checkpoint.py` | 版本化 checkpoint 与架构兼容检查 |
| `README.md` | 用户安装、数据、运行和输出指南 |
| `docs/code_architecture_analysis.md` | 当前工程架构说明 |
| `docs/safety_review.md` | 数据泄漏、复现和输入安全检查 |
