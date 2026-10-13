# 候选逐项结果与创新整理

日期：2026-10-13。所有增量均为百分点 pp。
投入标准见 `960_candidate_confirmation_series_20261013.md`；
原注册文件、历史 STOP、权重和源快照保持不变。
本文件按每项完整评价结束后追加。

## 实验 1：相对定位收益监督

1610 张 test-dev，既有 epoch20 权重和原归一化。
原普通校准数组和官方 TXT 全量精确重放，官方工具及运行时核验通过。
未训练检测器、未重训本实验头、未用 test 标注生成预测。

| 分支 | 官方 AP (%) | 官方 AR500 (%) |
|---|---:|---:|
| calibration | 30.894269 | 46.018180 |
| absolute | 30.878382 | 45.642782 |
| gain | 30.974892 | 45.758578 |

gain 对 calibration：AP +0.080623 pp，AR500 -0.259603 pp。
gain 对 absolute：AP +0.096510 pp，AR500 +0.115796 pp。

固定判断：SMALL_POSITIVE_SIGNAL。超过匹配头，召回在预算内；
相对校准增量小于登记的 +0.10 pp，但不能称没有效果。
val 的 +0.254031 pp 回落到 test-dev 的 +0.080623 pp，
目前不宜单独当作有说服力的主线贡献，不增加 epoch 或调整 margin。

辅助分尺寸口径是 valid GT + 未做忽略区过滤的 no-GT 预测，reference640 COCO-style：
gain 对校准 tiny -0.012291 pp，small_all +0.013294 pp，
不能声称已证明小目标特有收益。它不是官方 VisDrone tiny/small AP。
新增 CPU1 菜单/归一化/头/选择平均 4.774 ms，P95 7.833 ms，排除共享检测器与校准。

48 项相关测试通过；两图链路预检通过。版本管理记录：
`docs/experiments/candidate1_relative_gain_20261013.json`。
完整产物：`output/analysis/candidate-relative-fixed-20261013/`。

## 分尺寸标签澄清

新系列脚本原输出把分尺寸口径标成 COCO-style/IoF-ignore，但实际调用
`coco_size_metrics` 时没有在预测之后执行忽略区掩码。
因此真实计算是 valid GT 配合全部 no-GT 预测，不能按原标签解释。
每项独立保存 `size_diagnostic_scope_corrected.json`，绑定原报告 SHA256；
保留原报告和执行源快照。此澄清不改变数值、预测、官方 AP/AR 或主要判定。
后续阈值实验的同类辅助统计同样明确这个实际口径。
