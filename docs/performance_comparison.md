# 原 722 与 ProEssenLLM 性能比较

当前工作区只有代码，没有可用于训练的 metadata、LMDB、原 722 checkpoint 或结果 JSON，因此不能诚实地产出数值型性能结论。本工程不填造 AUC 或 F1。

## 公平比较协议

两套代码必须使用：

1. 相同 metadata 版本、标签列和 feature LMDB；
2. 相同 target/validation species（species_holdout）或相同随机种子与拆分比例（within_species）；
3. 相同模型隐藏维度、层数、batch 预算、epoch 上限和 early-stopping patience；
4. 相同 threshold 来源：只允许 validation；
5. target/test 只运行一次最终评价，不参与调参。

建议至少运行 5 个预注册随机种子，报告均值、标准差和逐 seed 配对差。species_holdout 应按 target species 重复实验，而不是只汇总一个随机物种。

## 目标指标

| 指标 | 原 722 | ProEssenLLM | 说明 |
|---|---:|---:|---|
| AUC | 待运行 | 待运行 | pooled/micro ROC-AUC |
| AUPR | 待运行 | 待运行 | pooled average precision |
| Balanced Accuracy | 待运行 | 待运行 | 使用 validation 固化阈值 |
| F1 | 待运行 | 待运行 | 使用同一 validation 阈值 |
| Macro AUC | 待运行 | 待运行 | 仅聚合可计算 AUC 的 species |
| Worst species AUC | 待运行 | 待运行 | 所有有效 species AUC 的最小值 |

## 自动比较

```bash
python evaluation/compare_runs.py \
  --mode species_holdout \
  --baseline_dir ../baseline_722_results \
  --proessenllm_dir ./results/species_holdout \
  --output_dir ./comparison
```

工具会兼容原 722 的结果文件命名，并在原汇总缺少 true worst-species AUC 时从逐物种 JSON 重新计算。输出包含 CSV、Markdown 和绝对变化。
