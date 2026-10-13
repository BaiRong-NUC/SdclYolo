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

## 实验 4：十个 IoU 阈值的相对收益监督

独立登记先于前面坐标候选的 test-dev 结果。复用 6471 张 train 动作缓存，
同初始化、容量与批次的 threshold_absolute、threshold_gain 各训练固定 20 epoch。
每头 9,226 参数，只取最终 epoch20，原场景 holdout 只监控。
不重训或推理检测器；1610 张 test-dev 使用共享的预测端动作缓存。

| 分支 | 官方 AP (%) | 官方 AR500 (%) |
|---|---:|---:|
| calibration | 30.894269 | 46.018180 |
| threshold_absolute | 30.914677 | 45.819960 |
| threshold_gain | 30.993026 | 45.931339 |

threshold_gain 对 calibration：AP +0.098757 pp，AR500 -0.086842 pp。
对匹配 threshold_absolute：AP +0.078349 pp，AR500 +0.111379 pp。
对冻结 regret：AP -0.134956 pp，AR500 +0.280142 pp。

固定判断：SMALL_POSITIVE_SIGNAL。相对校准距 +0.10 pp 预算界只有 0.001243 pp，
这种极小差距不是“没有效果”的证据，也不是统计显著性界限。
保留小幅正向与 AP/召回取舍，但它未超过已有排序方法，
目前不再加 epoch、改 margin 或把它换为新强对照。

十阈值标签对同一 GT 的条件成功不等于完整 AP 监督，不能据此宣称
实现了直接 AP 最优化。辅助原始分尺寸报告保留，并另存真实口径澄清。
四项针对性测试通过，完整官方输入、权重、训练缓存及源快照核验通过。
版本管理记录：`docs/experiments/candidate4_threshold_gain_20261013.json`。
完整产物：`output/analysis/threshold-gain-probe-fixed-20261013/`。

## 补评：统一在预测之后过滤忽略区

八个分支完整复用已经固定的预测。只在评价端对各分支使用相同规则：
预测框与单个忽略框的最大 IoF >0.5 时过滤，再算 reference640 COCO-style 指标。
不重训、不更改预测、不更改官方指标或前面主要判断。
这是项目分尺寸诊断，过滤实现与官方忽略区预处理不能等同。

| 分支 | tiny AP50–95 对校准 (pp) | small_all AP50–95 对校准 (pp) |
|---|---:|---:|
| absolute | -0.062217 | -0.051737 |
| gain | -0.012105 | +0.012291 |
| gain_always | -0.097433 | -0.082509 |
| uniform | +0.083667 | +0.131067 |
| regret | +0.132560 | +0.173394 |
| threshold_absolute | +0.020012 | +0.010848 |
| threshold_gain | +0.023581 | +0.053570 |

regret 对 uniform 的 tiny +0.048894 pp、small_all +0.042327 pp；
仍是很小的单种子辅助信号。阈值方案的小目标增量更弱，
补评没有推翻优先确认排序路线的判断。
原始未过滤报告及其真实口径澄清全部保留。
原预测、记录和计算源核验通过；ignore 元数据未在运行前单独绑定 hash，
因此不把这一补评包装成与正式官方输入同等级的全量注释字节审计。

版本管理记录：`docs/experiments/candidate_size_postignore_supplement_20261013.json`。
完整产物：`output/analysis/candidate-size-postignore-fixed-20261013/`。

## 实验 5：两个新头种子的排序确认

seed1/seed2 的 absolute、uniform、regret 各完成固定 20 epoch。
三头容量、初始化和批次均匹配；没有更改原 mean/std 或动作菜单。
train 为 630,619 个采样原框，train-scene holdout 为 36,247 个原框；
holdout 只监控。seed0 复用实验 1/3 的已核验 MAT，不挑种子或 checkpoint。
每个新种子的三个分支全部完成 1610 张 test-dev 官方评价及完整性核验。

| 头种子 | absolute AP (%) | uniform AP (%) | regret AP (%) | regret 对 uniform AP (pp) |
|---|---:|---:|---:|---:|
| 0 | 30.878382 | 31.110990 | 31.127982 | +0.016992 |
| 1 | 30.859895 | 31.083358 | 31.121627 | +0.038268 |
| 2 | 30.852787 | 31.076261 | 31.138151 | +0.061891 |
| 等权均值 | 30.863688 | 31.090203 | 31.129253 | +0.039050 |

uniform 对 calibration：平均 AP +0.195934 pp，平均 AR500 -0.231194 pp。
uniform 对 absolute：平均 AP +0.226515 pp，平均 AR500 +0.174385 pp；
三个种子的 AP 方向全部为正。固定判断 PROMISING_HEAD_SEED_REPEAT。

regret 对 uniform：平均 AP +0.039050 pp，三个种子全部为正；
平均 AR500 -0.121914 pp。regret 对 calibration 的平均 AP +0.234985 pp、
平均 AR500 -0.353108 pp。固定判断 POSITIVE_EXTRA_HEAD_SEED_SIGNAL。
AP 增量在头种子间重复，不能再简单视为只有 seed0 的偶然正点估计；
但权重额外收益仍小，并有召回代价。

这些重复只改变小头初始化和批次，不改变冻结检测器/校准器，
数据也仍是已查看的同一 test-dev，不能称完整三种子检测器验证、独立数据确认，
不能把头种子标准差当成数据采样 AP 置信区间。
六项新增的匹配初始化、损失与汇总门槛测试通过。

近邻论文的方法部分已进一步核对；RankDetNet 式 6 已在排序项内引入 IoU 差，
Rank-NMS 已按 IoU 学习框排序，IoU-Net 已学习定位质量并保护定位退化。
因此，继续执行事先条件登记的实验 6 普通已有方法对照，
不直接把当前加权形式或稳定的工程收益当作原创。
核对说明与条件登记：`960_ranking_nearest_prior_and_next_screen_20261013.md`。

版本管理记录：`docs/experiments/candidate5_ranking_seeds_20261013.json`。
完整产物：`output/analysis/ranking-seeds-fixed-20261013/`。

### 排序收益应如何命名

根据实验 5 冻结的三头种子平均官方结果，uniform 对校准的 AP50 +0.023253 pp、
AP75 +0.324337 pp；regret 的 AP50 +0.054697 pp、AP75 +0.334431 pp。
平均 AR500 分别 -0.231194/-0.353108 pp。
因此，目前支持的是小幅坐标质量收益，不能称解决了大量小目标漏检，
不能把 “小目标召回增强” 当成已经验证的方法贡献。
主任务描述宜为“冻结检测器后的跨视图定位动作选择”，
它是否小目标特有、能否迁移独立数据、是否超过更近的完整已有方法仍待确认。

## 实验 6：已有 IoU 差排序项的匹配适配对照

只借鉴 RankDetNet 正框指数排序项形状，alpha1，FP 保留项沿用当前配方。
同一三种子/输入/容量/初始化/批次/归一化/动作/20 epoch 预算，
只新增三个小头，不训练或推理检测器。完整 train 与 1610 张 test-dev 已核验。

| 头种子 | prior_gap 官方 AP (%) | prior_gap 官方 AR500 (%) | AP 对对应 uniform (pp) |
|---|---:|---:|---:|
| 0 | 30.903276 | 46.010336 | -0.207714 |
| 1 | 30.977233 | 46.011631 | -0.106125 |
| 2 | 31.003508 | 46.008051 | -0.072753 |
| 等权均值 | 30.961339 | 46.010006 | -0.128864 |

对 calibration：平均 AP +0.067070 pp，AR500 -0.008174 pp。
对 regret：平均 AP -0.167914 pp，AR500 +0.344933 pp。
固定判断 NO_REPEATABLE_ADVANTAGE_OVER_UNIFORM，三个种子都未超过匹配排序，
不替换当前强对照；仍保留相对校准的小正向信号。
停止这个固定指数适配配方，不事后调整 alpha/margin/epoch。

独立、无标注的预测诊断核验了 572,056 个原框的分数/类别/数量/顺序不变。
prior_gap 三头分别只改坐标 0.116422%、0.070098%、0.076566%；
uniform 三头为 3.375718%、3.097249%、2.878390%；
regret 为 5.074154%、4.778378%、4.756877%。
新对照较少替换的现象与其接近校准的 AP/召回结果相符，
但这不是导致效果的因果证明。输出尺度、梯度强度、现有 FP 保留项的平衡
可能影响它的活动程度，尚未被本实验隔离。
因此，未胜出的只是这个固定适配管线，不是对完整 RankDetNet 的胜利，
更不是当前损失已经足够新颖的证据。

四项指数项、掩码、FP 保留与汇总判定的针对性测试通过。
版本记录：`docs/experiments/candidate6_prior_gap_20261013.json`；
坐标诊断：`docs/experiments/candidate_coordinate_changes_20261013.json`。
完整官方产物：`output/analysis/prior-gap-fixed-20261013/`。

接下来只运行已独立登记的现有三头坐标共识回退，不新增训练；
它检查既有保留策略能否改善 AP/召回取舍，作为标准工程对照。

## 实验 7：三个既有头的坐标共识回退

只复用 seed0/1/2 的既有预测，三头坐标相同才替换，否则保留原框。
uniform_consensus 和 regret_consensus 都完成 1610 张官方评价及输入/源核验。
不训练、不推理检测器、不设浮点容差、不调 2/3 投票或按 AP 选种子。

| 分支 | 官方 AP (%) | 官方 AR500 (%) |
|---|---:|---:|
| calibration | 30.894269 | 46.018180 |
| 固定 seed0 uniform | 31.110990 | 45.784917 |
| 固定 seed0 regret | 31.127982 | 45.651197 |
| uniform_consensus | 31.024678 | 45.880001 |
| regret_consensus | 31.037583 | 45.723982 |

regret_consensus 对 calibration：AP +0.143314 pp，AR500 -0.294199 pp。
对固定 seed0 regret：AP -0.090399 pp，AR500 +0.072785 pp。
对 uniform_consensus：AP +0.012905 pp，AR500 -0.156020 pp。
固定判断 PROMISING_CONSENSUS_TRADEOFF，保持原登记判定。

但是，与已存在的单头普通 uniform 比较，regret_consensus 的 AP 与 AR500
两个指标都更低：seed0 分别低 0.073406/0.060935 pp，seed1/seed2 也都如此。
这是对已冻结方法的事后实用比较，不是新增门槛或改写原判断。
因此，不升级加权共识为新的强对照，不继续这个固定共识配方。
通过温和预算筛选，不能自动等同“优于现有普通方法”。
普通 uniform_consensus 本身是另一种较保守的 AP/召回取舍，保留普通对照结果，
不将标准集成当作论文创新。

部署共识需要三个小头，单头 uniform 只需一个；此次只评价缓存，
没有测新管线的完整部署延迟，不能按离线生成耗时宣称实时 FPS。
三个新增的精确共识、原字段保护与取舍判定测试通过。
版本记录：`docs/experiments/candidate7_coordinate_consensus_20261013.json`；
完整产物：`output/analysis/coordinate-consensus-fixed-20261013/`。

## 当前保留、停止与创新边界

- 保留普通校准作为原协议参照，保留固定 seed0 uniform 作为更强的普通定位对照；
  不按本轮头种子 AP 重新挑 checkpoint。具体路径、分支、权重 hash 与官方 MAT
  见 `docs/experiments/recommended_ranking_controls_20261013.json`。
- 排序任务对匹配绝对回归的平均 AP +0.226515 pp，三个头种子同向；
  这是当前最可靠的小收益主干。regret 额外 +0.039050 pp 也同向，
  但带来 AR500 -0.121914 pp，仍为取舍候选，不能把总收益全归给权重。
- 十阈值、指数适配及当前共识的完整结果保留；不为增大点估计再调
  epoch、alpha、margin、投票比例或种子。
- 当前可讨论的候选任务是：保持校准分数与数量，用同原 GT 参照学习跨视图
  坐标动作；保留原框是有效设计。任务与近邻方法存在区别，但尚未证明首创、
  小目标特有、独立数据有效或足以支撑论文核心贡献。

不建议由这些屏选否定整个小目标检测领域。建议结束当前纯几何损失形状搜索，
把已有排序作为工程/研究强对照。下一项训练投入应先有新的可观测信号，
例如真实视觉边界证据能否区分当前几何特征无法确定的跨视图定位动作，
并匹配普通视觉 IoU 预测对照。这个具体信号还没有被本轮验证，不当作已成立创新，
在明确新信号之前，不直接启动一轮新的检测器长训练。

后续先做 train 内场景留出的可学习性预检，再锁定方案；
当前已经多次查看的 test-dev 不再当成全新独立确认集。
研究价值判断与发表新颖性判断分开，历史 STOP 和原实验门槛保持不变。

官方 AP/AR 取舍图与其可复现脚本：
`output/analysis/candidate-tradeoffs-20261013/official_ap_ar_tradeoffs.png`，
`scripts/evaluation/plot_candidate_tradeoffs.py`。
图中误差线仅是三个头种子的标准差，不是数据采样 AP 置信区间。
