# 修改日志

原始文件夹 `1` 和 `722` 均未修改。所有新增与重构均位于本工程。

| 文件 | 修改原因 | 融合来源 |
|---|---|---|
| `train.py` | 新增统一 `--mode` 入口 | 722 主训练流程，文件夹 1 参数思想 |
| `__init__.py` | 声明工程包和版本 | 新增 |
| `models/proessenllm_model.py` | 统一 frozen/ESM-LoRA 模型，移除旧命名，增加安全 checkpoint 状态接口 | 722 模型主体，文件夹 1 的 padding 防御 |
| `datasets/proessenllm_dataset.py` | 分离两种拆分协议，统一 Dataset/DataLoader，增加旧 LMDB 兼容与 head-tail 截断 | 722 数据主体，文件夹 1 截断与 within-species 逻辑 |
| `samplers/species_label_balanced.py` | 保证每批多个 distinct species，并控制每 species 正负比例 | 文件夹 1 的 species-label-balanced 思想 |
| `losses/species_losses.py` | 训练集限定的 species 正类权重与 global/within-species ranking loss | 722 loss 主体，文件夹 1 数值稳定处理 |
| `evaluation/metrics.py` | 增加 AUPR、逐物种 CSV、q25 与真正 worst-species AUC | 722 分布指标，文件夹 1 的 F1/BA 报告 |
| `evaluation/checkpoint.py` | 单/复合指标选择、early stopping、格式版本和架构指纹 | 722 validation candidate 逻辑 |
| `evaluation/leakage.py` | 样本、物种、target、模型输入和选择来源审计 | 722 断言扩展 |
| `evaluation/compare_runs.py` | 自动输出原 722 与新框架六项性能比较 | 新增 |
| `trainers/base_trainer.py` | 统一训练引擎；test 延迟到 checkpoint 锁定后；逐物种 epoch 监控 | 722 主体，文件夹 1 训练诊断 |
| `trainers/within_species_trainer.py` | 明确已知物种协议入口 | 新增 |
| `trainers/species_holdout_trainer.py` | 明确未知物种 zero-shot 协议入口 | 722 holdout 逻辑重构 |
| `configs/proessenllm_config.py` | 参数验证、模式互斥、可复现默认值 | 两工程配置整合 |
| `configs/*.yaml` | 提供两种模式示例 | 新增 |
| `tests/test_framework.py` | 验证拆分、target 隔离、batch 结构与尾部指标 | 新增 |
| `tests/smoke_training.py` | 用合成 LMDB 对两种模式执行一轮端到端训练 | 新增 |
| `README.md` | 工程结构、安装、运行与输出说明 | 新增 |
| `requirements*.txt`、`.gitignore` | 区分基础/ESM 依赖并排除生成文件 | 新增 |
| `docs/code_architecture_analysis.md` | 保存修改前架构分析与迁移决策 | 两工程只读审计 |
| `docs/safety_review.md` | 保存论文实验安全检查 | 新增 |
| `docs/performance_comparison.md` | 定义无数据时不造数及公平比较协议 | 新增 |
