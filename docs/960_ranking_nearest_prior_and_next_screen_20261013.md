# 排序候选的近邻方法核对与实验 6 条件登记

日期：2026-10-13。本文件先于实验 5 新种子官方结果及实验 6 训练。
原始 PDF、逐页文本和方法页渲染位于
`output/verification/ranking_nearest_prior_20261013/`，字节 hash 见 manifest.json。
这次只核对与当前任务直接相关的方法部分，不声称穷尽文献或复现论文整套系统。

## 核对到的已有方法

| 原始论文 | 已核对的方法位置 | 已有机制 | 当前实现的区别及限制 |
|---|---|---|---|
| IoU-Net / Acquisition of Localization Confidence for Accurate Object Detection，ECCV 2018 | 第 3.1–3.3 节、算法 1/2，PDF 第 7–9 页 | 用 FPN/Precise RoI Pooling 预测 IoU；按定位质量而非分类分数做 NMS；梯度优化框坐标，并设置定位退化停止条件 | 当前仅用框特征选择六个现成坐标动作，保持原框分数与数量，不做 RoI 特征或梯度坐标优化。质量预测和防退化已有先例 |
| Learning to Rank Proposals for Object Detection，ICCV 2019 | 第 3.1–3.3 节、式 1–3、算法 1，PDF 第 3–4 页 | RoI 特征预测框的排序分数，成对 hinge 排序；IoU 按 0.05 步长分组采样，排序分数与分类分数结合用于 NMS | 当前排序只用于同原框动作的坐标选择，不改检测分数或 NMS；但“按 IoU 学框排序”明显不是新思想 |
| RankDetNet: Delving into Ranking Constraints for Object Detection，CVPR 2021 | 第 3.1/3.3/3.4 节、式 6，PDF 第 4–5 页 | 原文 L_rank 使用 exp；IoU-guided 项为 exp(-alpha * 分数差 * IoU差)，alpha=1；同类、同 FPN 层正框配对，并联合优化分数和部分坐标路径 | 当前 regret 将绝对 IoU 差放在 softplus 外作为归一化权重，IoU 来自冻结动作标签，输出仅用于选择坐标。放在损失内外的差异不自动构成创新 |

来源：

- https://arxiv.org/abs/1807.11590
- https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html
- https://openaccess.thecvf.com/content/CVPR2021/html/Liu_RankDetNet_Delving_Into_Ranking_Constraints_for_Object_Detection_CVPR_2021_paper.html

## 当前可以写与不能写的内容

可以作为候选任务描述：固定普通校准后的检测分数/类别/数量，用跨视图框特征
决定保持、替换或投票；训练全部动作相对同一个原 GT，不借更换 GT 制造收益。
这种拆分让坐标选择贡献更容易隔离，但还不能保证推理实例身份，也不能证明首创。

已有实验支持小幅后处理收益及保留原框的消融作用。
不能把分辨率、Soft-NMS、普通校准、IoU 排序、代价加权或退化保护列为原创贡献。
阈值净收益监督没有超过冻结排序；当前还没有足够证据提出新的论文核心机制。

## 实验 6：匹配已有 IoU 差排序原则的低成本对照

条件：只有实验 5 排序任务达到 PROMISING_HEAD_SEED_REPEAT，才运行本实验。
若实验 5 未通过，不为凑实验继续训练。

目的不是提出另一个新损失，而是检查当前结果能否被更近的已有排序原则解释。
将 RankDetNet 式 6 的正框排序形状适配到现有动作任务：

`exp(-(IoU_i - IoU_j) * (utility_i - utility_j))`

alpha 固定 1；不使用额外温度、裁剪、IoU 重分组或超参数扫描。
只在原框拥有 GT 的有效非平手动作对上使用该项，对每个原框按有效对数平均。
FP/重复框保留动作项完全沿用原 uniform/regret 的 softplus/0.05。
采样权重、mean/std、三种子 0/1/2、同种子的初始化和批次、
68->64->64->1、20 epoch、batch2048、AdamW lr0.002/wd0.0001、margin0.01 全不变。
新增三个单头，复用 train/test-dev 缓存，不训练/推理检测器。
只取 epoch20；非有限损失直接记录失败，不事后裁剪或重设 alpha。

主要比较固定为这个普通 prior_gap 对实验 5 对应种子的 uniform 官方 AP。
等权平均相对 uniform >0 且至少 2/3 种子为正，相对校准平均 AP >=+0.10 pp、
平均 AR500 >=-0.50 pp，才作为有价值的普通强对照。
另外固定报告相对 regret 的差异；不将相对回归头的收益全归于新的权重形式。
正向但不足预算记小正向；不改实验 1–5 的结果或门槛。

此处只借鉴原文正框排序项形状，保留了当前动作、特征、FP 项及训练预算，
不是 RankDetNet、Rank-NMS 或 IoU-Net 整套方法的公平复现，不宣称打败原论文。
不同排序损失的输出不是已校准的 IoU 收益；固定相同数值 margin 不保证
相同的有效替换强度。比较的是各自固定完整配方，不能把差异完全归于损失形状。
即使它有效，也只是标准原则适配后的更强普通对照，不能改名包装成创新。
若普通方法已经解释当前收益，停止把现有损失形状当创新继续搜索；
下一项投入必须来自具体新失败模式和 train-only 证据，或验证独立数据迁移。

复现指令（训练脚本会检查实验 5 完整判定，未通过就拒绝启动）：

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python scripts/training/train_prior_gap_confirmation.py --device 0
python scripts/evaluation/confirm_prior_gap.py --workers 2
```

固定训练输出 `output/runs/prior_gap_confirmation_960_20261013`。
同公式、无效动作掩码、FP 项与三种子门槛的四项针对性测试已通过；
测试成功仅证明实现和判断逻辑，不代表效果。正式产物拒绝覆盖。
