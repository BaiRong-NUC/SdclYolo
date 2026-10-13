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

## 实验 2：保留原框机制

同一组 gain 权重、同一动作菜单和同一 1610 张 test-dev。
gain 的官方 MAT 从实验 1 精确核验后复用，只新增 gain_always 官方评价。
没有再次推理检测器或训练，完整性核验通过。

| 分支 | 官方 AP (%) | 官方 AR500 (%) |
|---|---:|---:|
| 始终替换 gain_always | 30.800018 | 44.453996 |
| 可保留原框 gain | 30.974892 | 45.758578 |

回退相对始终替换：AP +0.174874 pp，AR500 +1.304582 pp。
固定判断：PROMISING_POINT_ESTIMATE，两项方向均通过温和机制确认标准。

可整理的机制结论是允许保留原框有作用，避免无条件替换造成损伤。
这不是回退策略首创，也不是相对普通校准的召回增益；
gain 相对校准仍为 AP +0.080623 pp、AR500 -0.259603 pp。
将回退保留为后续定位决策的必要设计和消融项，不单独包装成论文核心创新。

版本管理记录：`docs/experiments/candidate2_retention_20261013.json`。
完整产物：`output/analysis/candidate-retention-fixed-20261013/`。

## 实验 3：按错误代价加权的动作排序

同一完整 test-dev、同一缓存、既有 matched uniform/regret epoch20 权重。
普通校准复用全量精确重放后的旧官方结果，两个排序分支完成新官方评价和核验。

| 分支 | 官方 AP (%) | 官方 AR500 (%) |
|---|---:|---:|
| calibration | 30.894269 | 46.018180 |
| uniform | 31.110990 | 45.784917 |
| regret | 31.127982 | 45.651197 |

regret 对 calibration：AP +0.233713 pp，AR500 -0.366984 pp。
regret 对 uniform：AP +0.016992 pp，AR500 -0.133720 pp。
固定判断：PROMISING_POINT_ESTIMATE，通过本轮温和的预算筛选。

不能把 +0.233713 pp 全部归给加权机制：普通排序已经获得 +0.216721 pp。
“匹配普通方法的点估计 >0”只是筛选要求，不是显著性证明；
这个 +0.016992 pp 尚未被种子重复或 AP 区间确认。
保留排序任务与回退策略作为目前最值得复核的完整方法候选，
额外 regret 权重暂时作为消融/扩展，不能写成已证明的核心创新。

分尺寸解释性统计（valid GT、未过滤预测，非官方尺寸 AP）：
regret 对 calibration，tiny +0.138411 pp、small_all +0.174554 pp；
regret 对 uniform，tiny +0.055404 pp、small_all +0.041657 pp。
有小目标倾向的正向点估计，不能因整体增量很小就断言权重完全无效，
也不能将这些辅助指标升级为正式独立确认。

新增缓存后处理 CPU1 平均 4.761 ms、P95 7.765 ms，排除共享检测器和校准。
另从两个冻结配方作事后解释性比较：uniform 对 absolute 的官方
AP +0.232608 pp、AR500 +0.142135 pp。这不是原登记的主比较，
不能事后把它替换成实验 3 的成功标准。

当前研究叙述应重点讨论“将固定框身份下的跨视图坐标选择作为排序任务”，
而不是声称成对排序、代价加权或回退本身原创。
IoU-Net、Rank & Sort、学习型 NMS 等的任务、输入和输出仍需要更近的全文对照。

版本管理记录：`docs/experiments/candidate3_ranking_20261013.json`。
完整产物：`output/analysis/candidate-ranking-fixed-20261013/`。
