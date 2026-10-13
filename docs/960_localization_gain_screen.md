# 普通校准强对照上的相对定位增益筛选

日期：2026-10-13。本方案只做一次固定 seed0 筛选，正式结果出来后不调门槛、邻框条件、
回退阈值、训练轮数或种子。正式训练尚未自动执行。

## 为什么换到定位

普通校准版已被选为新的工程强对照。冻结记录在
`output/analysis/calibration-control-supplement-20261013/control_manifest.json`：

- YOLO11s 960、原图加四个固定切片、Gaussian Soft-NMS。
- 使用 `rescue_softnms_960_seed0/last.pt` 的 `calibration` 分支，固定 epoch20。
- 冻结检测器、校准权重及其 mean/std、候选集、类别、分数和顺序。
- 本机补评：all AP50–95 33.577313%，tiny 26.317716%，small_all 31.467569%；
  small_all Recall@FPPI5 52.150613%、Recall@FPPI10 62.204483%。
- 这些是项目 COCO-style / IoF-ignore 诊断，分组使用 reference640，不是官方 VisDrone AP。
- 这是在已观察的验证结果上选择的强对照，后续通过筛选也仍需独立数据验证。

此前 VCR 四状态评分方案保持 STOP，不重开其配方。
本次假设是：在校准分数已经有效的前提下，预测“替换框相对原框的定位增益”
可能进一步改善严格 IoU 下的 AP。

## 候选贡献及边界

候选方案称为“保持原框参照的跨视图定位增益选择”：

1. 对每个校准后候选保留六个动作：保持原框、其他四个视图各一个同类框、
   原框与这些邻框的置信度加权坐标投票。
2. 邻框原始置信度至少 0.05，与原框 IoU 至少 0.5。各视图选 IoU 最大者，
   并列时选置信度更高者，再按原顺序。没有邻框的动作标为无效。
3. 训练时按原校准预测的分数降序、同类、IoU>=0.5，一对一固定 GT 归属。
   每个动作的 IoU 均相对这个同一 GT 计算，避免在拥挤场景切换到邻近目标。
4. 增益头直接学习 `IoU(action, owned_GT) - IoU(keep, owned_GT)`。
   原框匹配失败或属于重复 FP 的候选，所有动作目标均为零。
5. 推理只读取预测特征。最佳可用动作的预测增益严格大于 0.01 才改坐标；
   否则保留原框。不新增检测框，不改分数、类别或顺序，不追加 NMS。

待检验的贡献是“相对增益监督 + 固定原框参照 + 增益不足时回退”的组合效果。
它目前是可证伪的候选机制，不是已证明首创；不能把框投票或 IoU 质量预测本身
写成创新点。正式论文前仍需对框融合、IoU 质量预测、学习型框修正进行完整原始文献对照。
本次在线检索未返回可核验文献，因此尚未完成优先权核查。

这里的固定投票是逐原框建立的坐标平均，不是进行聚类、合并数量和重算分数的标准 WBF。
标签不监督从原框 IoU<0.5 的漏检中恢复，也不声称提高所有 GT 的定位质量。
选错动作仍可能降低 IoU；预测增益大于阈值并不提供实际增益保证。

## 一个训练运行包含的对照

| 分支 | 训练内容 | 推理规则 |
|---|---|---|
| calibration | 无新训练 | 冻结普通校准预测 |
| vote | 无新训练 | 有邻框就使用固定坐标投票 |
| absolute | 学习各动作绝对 IoU | `Q(action)-Q(keep)>0.01` 才替换 |
| gain_always | 复用 gain 权重 | 总是选最佳可用非 keep 动作，不设增益回退 |
| gain | 学习相对原框的 IoU 增益 | 最佳非 keep 增益 `>0.01` 才替换 |

只训练 absolute 和 gain 两个 `68 -> 64 -> 64 -> 1` SiLU 评分头；
初始化、容量、样本、归一化、优化器、批次和轮数完全匹配。
每个头 8,641 个参数。68 维输入由原校准的 54 维特征和 14 维动作特征组成：
相对坐标差、原框 IoU、来源置信度、五视图来源、投票标记、邻框数量、校准分数。

- train 全部 6,471 张图生成预测；按已有 scene 前缀 SHA256 modulo10 划分拟合/监控集。
  全数据下拟合 6,131 张、监控 340 张；监控集不选 epoch。
- 每图 TP/FP 各最多采样 64 个原框，使用逆采样权重；原框内按有效动作平均损失。
- mean/std 只来自拟合集有效动作；std 下限 0.05，归一化截断到 ±6。
- AdamW，lr=0.002，weight_decay=0.0001，batch=2048，SmoothL1 beta=0.1。
- seed0，20 epoch，仅评估最后 epoch20。不选择最佳验证轮次。
- detector 预测为 FP16、960、conf=0.001、IoU=0.7、max500、rect=False、batch1，
  与冻结五视图策略一致。

## 提前固定的 11 项门槛

下列所有项同时通过才输出 CONTINUE；否则 STOP。AP 和 Recall 差值为百分点，
IoU 差值也乘以 100。

| 检查 | 相对对照 | 门槛 |
|---|---|---:|
| tiny AP50–95 | calibration | >= +0.50 pp |
| small_all AP50–95 | calibration | >= +0.30 pp |
| all AP50–95 | calibration | >= +0.20 pp |
| small_all Recall@FPPI5 | calibration | >= -0.10 pp |
| small_all Recall@FPPI10 | calibration | >= -0.10 pp |
| small_all AP50–95 | vote | >= +0.20 pp |
| small_all AP50–95 | absolute | >= +0.20 pp |
| small_all AP50–95 | gain_always | >= +0.20 pp |
| 原校准匹配成功的 small_all GT 平均 IoU | absolute，同 GT 归属 | >= +0.50 pp |
| 上一项配对图像 bootstrap 95% CI 下界 | absolute | > 0 |
| 新定位菜单、归一化、gain 头和坐标选择平均耗时 | calibration 之上的增量 | <= 15 ms/图 |

bootstrap 固定 2,000 次、seed0、按图像配对；FP 操作点选择后固定阈值。
不包含训练种子或阈值选择的不确定性，也不进行场景级 bootstrap。
耗时为 CPU 单线程、预热后的缓存后处理，排除图像解码和五视图检测器耗时。
不把其他实验头的时间计入 gain 分支，也不将该数字写成部署 FPS。

## 直接执行

在 PowerShell 顺序执行；上一条成功后再运行下一条：

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python scripts/data/prepare_localization.py --device 0
python scripts/training/train_localization.py --device 0
python scripts/evaluation/diagnose_localization.py
```

第一步需要重新生成 **训练集** 五视图几何。旧 train 特征缓存没有这些框坐标，
不能直接复用；此步是冻结权重的推理，不是检测器训练。
验证集已有 `raw_views.jsonl` 会复用，不重复跑检测器。

默认训练数据在 `output/localization/train_actions_seed0`，
两个新头在 `output/runs/localization_gain_960_seed0/last.pt`。
准备中断：重新执行原准备指令，它会检查并跳过完成的图像。
训练中断：恢复最后完整保存的 epoch，重新执行未完成的 epoch：

```powershell
python scripts/training/train_localization.py --device 0 --resume
```

每轮原子保存权重、优化器状态和日志。完成的 epoch20 使用 `--resume` 不再训练。
不手工删除已有运行、换目录重复正式实验或修改配方。评估每次生成新目录，
拒绝覆盖已有结果。

## 核验与产物

新流程核验普通校准 manifest、冻结检测器/校准头/验证预测、历史源文件及 ZIP。
允许增加独立定位模块，但任何原有重放依赖变更都拒绝。
不修改旧流程的严格全源检查规则。旧历史评估重放仍应使用它的源快照。

本实验单独保存完整 Python 源文件和配方 ZIP，缓存记录每图内容 SHA256，
绑定 train 记录 hash、采样集、scene 划分和 full_training_data 标记。
评估重建普通校准并与已冻结数组逐项精确比较；正式全量还核验保存预测的字节 hash。
最终检查所有分支的分数、类别、顺序与数量一致。

评估目录 `output/analysis/localization-<时间戳>/` 包含：

- `plan.json`、`source.zip`：固定配方、门槛与源快照。
- `predictions/{calibration,vote,absolute,gain_always,gain}/`：预测及内容 hash。
- `coco_reference.json`：五个分支的 reference640 分尺寸 AP50–95。
- `matched_fp_budgets.json`、`fp_curves.json`：FPPI5/10 配对结果与操作点。
- `owned_gt_mechanism.json`、`owned_gt_per_image.csv`：同 GT 归属机制诊断。
- `timings.csv`、`processing_times.json`：公共校准与新增后处理分项耗时。
- `action_counts.csv`、`verification.json`、`decision.json`、`decision.md`。

只有完整训练集、完整 548 张验证集和 epoch20 才允许正式 CONTINUE/STOP。
任何 `--limit` 缓存/评估都会输出 SMOKE_ONLY，烟测指标不用于研究判断。
如果正式 STOP，保留普通校准强对照，结束这版定位增益配方。
如果 CONTINUE，下一步才做独立数据、多种子和官方 VisDrone 对齐，
不是直接宣称可发表。

## 本次实际预检

- 210 项相关测试通过；排除了两份已有 API/删除行为不兼容的旧队列测试文件
  `test_training_queue.py` 和 `test_signal_queue.py`，未为此改变旧代码。
- 随机 96 张 train（其中 2 张属于训练 scene 监控集），两个评分头训练 20 轮，
  拟合/监控损失均有限。这里只验证训练和监督链路。
- 模拟保存 epoch10 后中断再恢复：最终两头权重逐位相同，所有逐轮损失完全相同。
  对完成的 epoch20 运行恢复，检查点内容保持不变。
- 合并同类邻框查询的等价计算前后，32 张真实 val 的动作框、有效掩码和特征精确相等。
- 8 张 val 流程测试中，校准预测精确重放，五分支分数/类别/顺序/数量核验通过；
  结果为 `SMOKE_ONLY`。新增 CPU1 缓存后处理平均 7.7048 ms，P95 12.0432 ms。
  这是小样本预检，不能替代正式全量耗时门槛，也不作为研究效果结论。

证据在 `output/verification/localization_20261013/runtime_verification.json`，
以及 `eval_smoke_v2/verification.json`、`eval_smoke_v2/processing_times.json`。
正式默认训练数据与运行目录尚未创建。
