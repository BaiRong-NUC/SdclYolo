# 960 Soft-NMS 强对照后的可观测性重评分筛选

状态：已实现并通过小规模流程预检；正式全量实验尚未运行。
固定日期：2026-10-13。配方：`configs/experiments/rescue_softnms_960.yaml`。

## 研究假设和边界

候选名称为 View-Conditioned Rescue Rescoring（VCR）。
研究问题是：另一个视图没有给出检测，究竟代表候选可疑，
还是那个视图因为裁剪和放大倍率不同，本来就缺少检测机会？
希望保留切片新增的小目标检出，同时压低重复框和误报。

学习式 NMS、上下文重评分、切片推理与边界处理都有已有工作。
本方案的候选贡献是显式区分“有机会但漏检”和“没有机会”的缺失跨视图证据，
并检验这种区分是否优于普通置信度校准和去掉可观测性信息的版本。
目前这是可证伪的工作假设，不能宣称已经有创新性或已有效。

相关原始文献标识：Learning NMS（arXiv:1705.02950）、
Relation Networks（arXiv:1711.11575）、Adaptive NMS（arXiv:1904.03629）、
SAHI（arXiv:2202.06934）。后续需要进一步核对方法级重合。

## 固定强对照

- 检测器：`output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt`。
- 强对照：`output/analysis/fusion-softnms-fixed-20261013` 的固定 Soft-NMS。
- 原图加 4 个固定 2×2 切片，20% crop-relative overlap。
- 所有视图 960、square、batch 1、FP16；conf 0.001、NMS IoU 0.7、每视图最多 500 框。
- 原坐标映射、IoF > 0.5 忽略过滤、class-aware Gaussian Soft-NMS。
- sigma 0.5、score floor 0.001、最终最多 500 框；连续坐标 IoU，无 `+1`。
- 只修改强对照已保留的候选分数和顺序，不修改框、不恢复被 500 框上限丢弃的候选。

这版 Soft-NMS 已通过之前的三项门槛：
相对全图 960，小目标合组 FPPI≤10 召回 +3.6788 pp，
配对图像 bootstrap 95% CI 为 [3.2061, 4.1564] pp，
整体 AP50–95 +4.3091 pp。
这证明它可以作为较强对照，不代表标准 Soft-NMS 本身是创新。
此前 phase、CPR-TAL、固定 hard-NMS 切片方向的 STOP 记录保留。

## 一个固定实验中的三个版本

训练标签只使用 train GT，验证集不参与拟合或模型选择。
训练候选按强对照输出顺序，在 IoU≥0.5 下进行逐框一对一匹配；
重复框记为 FP。四状态为：

| 状态 | 含义 |
|---|---|
| 0 | 最终 FP，包含匹配竞争后的重复框 |
| 1 | TP，其他视图有同类别 GT-compatible IoU≥0.5 且 conf≥0.25 的支持 |
| 2 | TP，其他视图有检测机会，但没有上述支持 |
| 3 | TP，其他视图都没有检测机会 |

检测机会代理：另一个裁剪覆盖 GT 框面积至少 90%，
且该视图 960 输入下的框几何平均边长至少 8 px。
实际存在检测支持时优先记为状态 1，不能因代理不满足而否定支持。
训练标签根据 GT 判定，推理特征根据候选框计算；这个代理不等于物理上的“不可见”。

| 版本 | 特征与目标 | 推理分数修正 |
|---|---|---|
| calibration | 全部 54 维特征；状态 1/2/3 合并为正确框 | `p(correct)-p(FP)` |
| no_observability | 屏蔽机会、覆盖率、投影尺度及机会汇总列；状态 3 合并到 2 | `p(rescue)+p(sole)-p(FP)` |
| rescue | 全特征、四状态 | `p(rescue)+p(sole)-p(FP)` |

三个版本具有相同 64×64 SiLU MLP 容量、相同初始化、训练预算、采样和优化器，
输出均为四维。前两个版本未使用的状态仍保留对应输出维度。
不承诺消融完全消除从其他尺度信息间接推断机会的可能；
它测试显式可观测性特征与四状态监督的联合价值。

修正在 Soft-NMS 分数的 logit 上加一个 [-1, 1] 内的残差，固定 gain=1。
评分后删除分数低于 0.001 的候选，稳定排序，保持 500 框上限。
支持、裁剪边界、候选类别、Soft-NMS 分数衰减等特征不会读取验证 GT。
验证忽略标注仅沿用强对照的诊断过滤协议。

## 固定训练预算

train 共 6471 张。场景名取 stem 的第一个下划线前字段，
SHA256 前 8 字节 little-endian 整数模 10 等于 0 的场景保留作训练内部监测。
当前为 20/208 个场景、340 张图像；其余 6131 张用于梯度更新。
此保留规则是场景前缀代理，不保证不同场景之间没有视觉相关性。
不依据内部监测或 val 挑选 epoch，统一使用第 20 轮。

每图每状态最多取 32 个候选，按图名确定采样种子；
使用原状态候选数/采样数作为重要性权重。
标准化只在拟合子集估计，std 最小 0.05，特征截断至 ±6。
AdamW，lr=0.002、weight decay=0.0001、batch=2048、seed=0、20 轮。
一次训练脚本同时训练三个评分器；不重新训练 YOLO。

## 预先声明的九项门槛

AP 均为 AP50–95，召回指标均为 small_all、IoU 0.5。
pp 表示百分点。所有九项必须同时通过：

| 比较 | 指标 | 要求 |
|---|---|---:|
| rescue vs Soft-NMS | FPPI≤10 召回增量 | ≥ +0.50 pp |
| rescue vs Soft-NMS | 上述增量的 95% CI 下界 | > 0 |
| rescue vs Soft-NMS | tiny AP 增量 | ≥ +0.30 pp |
| rescue vs Soft-NMS | 整体 AP 增量 | ≥ −0.10 pp |
| rescue vs Soft-NMS | FPPI≤5 召回增量 | ≥ −0.10 pp |
| rescue vs calibration | FPPI≤10 召回增量 | ≥ +0.30 pp |
| rescue vs calibration | 上述增量的 95% CI 下界 | > 0 |
| rescue vs no_observability | FPPI≤10 召回增量 | ≥ +0.20 pp |
| rescue vs no_observability | 上述增量的 95% CI 下界 | > 0 |

bootstrap 为 2000 次配对图像重采样，固定所选阈值；
区间不覆盖训练种子、阈值选择或数据域变化带来的不确定性。
这些是一次方向筛选条件，不是多重比较校正后的科学定论。
统计特征提取及评分耗时，不把新增处理成本隐藏成免费。

如果 `STOP`，停止当前重评分配方，保留标准 Soft-NMS 强对照。
不改机会阈值、残差强度、epoch 或增加另一种救回模块来挽救这轮结果。
如果 `CONTINUE`，才值得安排多种子、独立数据和官方 VisDrone 协议验证。
现在的指标是 COCO-style / IoF-ignore 诊断，不能称为官方 VisDrone AP。

## 整理后的执行指令

在 `D:\Python\YOLO` 的 PowerShell 中按顺序执行；任一步报错就停止，
不要在失败后继续训练或评估。

先生成全量 train 特征；中断后重复此条会核验配方、源码、数据和已写分块再续做：

```powershell
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/data/prepare_rescue.py --device 0
```

再在一次固定运行中训练三个评分器：

```powershell
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/training/train_rescue.py --device 0
```

若评分器训练中断，已有完整 epoch 的 `last.pt` 可用以下指令恢复：

```powershell
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/training/train_rescue.py --device 0 --resume
```

若在第一轮 checkpoint 产生之前中断，保留该输出目录，用新的 `--output` 运行，
并把该路径下的 `last.pt` 传给评估的 `--checkpoint`，不要覆盖原目录。

完成 20 轮后评估所有 548 张 val，并自动判定：

```powershell
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/evaluation/diagnose_rescue.py --output output/analysis/rescue-softnms-fixed-20261013
```

评估目录必须不存在；若已存在，改为一个新目录。
训练期间不要修改 `sdcl/`、`scripts/` 或固定配方。
源码改变时缓存会拒绝复用，需使用新的准备和训练输出路径；
评估必须使用 checkpoint 对应的完整源码版本。

输出：

- 特征：`output/rescue/train_features_seed0/`，包含 train provenance、源码 ZIP 和分块哈希。
- 训练：`output/runs/rescue_softnms_960_seed0/`，包含 `last.pt`、三个模型、损失、计划和源码 ZIP。
- 评估：指定目录下的 `decision.json`、`decision.md`、`verification.json`、
  `rescue_vs_soft_nms/report.json`、两个学习对照报告、预测和处理耗时。

## 本轮预检证据

`output/verification/rescue_20261013/organized/`：
32 张随机 train 图像生成 2198 条候选样本，三个评分器完成 20 轮，
所有训练损失有限；2 张 val 的评估完成，状态为 `SMOKE_ONLY`。
该随机 train 小样本未抽到内部保留场景，监测 loss 为 null；
全量数据的保留场景计数已独立核验。小样本不能说明研究效果。
重建候选与冻结 Soft-NMS 逐元素一致，评估完整性检查通过。
另外，20 个入口的 `--help` 通过，旧 baseline/phase/coverage 权重都能加载。
在相同小样本上模拟第 10 轮中断后恢复到第 20 轮，
三个模型的权重和损失记录与连续训练完全一致，证据为
`output/verification/rescue_20261013/organized/migration-and-resume.json`。
相关测试 199 项通过；旧训练队列测试另有 11 项既存失败，
仍在调用旧接口或要求删除中断目录。本轮保留队列的归档保护行为，
未以恢复删除行为的方式消除这些失败。

整理后的实现位置：

| 位置 | 功能 |
|---|---|
| `sdcl/postprocess/rescue.py` | 视图机会、特征、四状态标签、评分模型和分数修正 |
| `sdcl/training/rescue.py` | train-only 准备、缓存恢复、三个模型训练 |
| `sdcl/evaluation/rescue_screen.py` | 冻结验证缓存、配对评价、九项判定 |
| `sdcl/runtime/` | 递归源码哈希、环境记录、Windows 原子替换重试 |
| `scripts/data/prepare_rescue.py` | 特征准备入口 |
| `scripts/training/train_rescue.py` | 固定训练和恢复入口 |
| `scripts/evaluation/diagnose_rescue.py` | 全量评估入口 |

历史 checkpoint 保留 `sdcl.trainer` 兼容导入。
原来其他脚本分别迁入 `scripts/data/`、`training/`、`evaluation/`、`verification/`。
旧实验输出和归档源码 ZIP 保留，迁移源码不会冒充历史评估版本。
