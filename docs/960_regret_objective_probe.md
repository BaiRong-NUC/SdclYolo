# 一次缓存特征排序目标预检：不重训检测器

固定日期：2026-10-13。先完成无训练诊断，再决定做这个独立的可学性预检。
旧相对 IoU 回归/回退配方保持 **STOP（7/11）**，本预检不恢复它、不追加它的 epoch，
不改它的 0.01 回退阈值，也不修改其源文件。

## 为什么还有一个低成本问题可检验

已运行 `scripts/evaluation/diagnose_headroom.py`，用原有五视图预测和普通校准强对照，
完整检查 548 张 val。固定同一原框拥有的 GT，不改类别、分数、排序和数量：

| GT 辅助参考相对普通校准 | 增量 |
|---|---:|
| tiny AP50–95 | +2.2932 pp |
| small_all AP50–95 | +2.4842 pp |
| all AP50–95 | +2.5672 pp |
| 原框已匹配 small_all GT 平均 IoU | +2.5530 pp |
| 上一项配对图像 bootstrap 95% CI | [2.4939, 2.6152] pp |

这五项均超过诊断预先固定的要求。原学习增益头只获得约 6.95% 的上述条件平均 IoU
参考增量；这是同 GT 归属下的 IoU 比例，不是 AP 比例。
GT 辅助参考不是可部署算法，也不是全局最优 AP 上界，不能据此宣称模型一定能学到。

FPPI10 操作点，tiny 漏检中 58.23% 有最终同类 IoU>=0.5 候选，但分数不足。
降低阈值会增加 FP，不能算作固定 FP 预算下的收益。
在 score floor=0.001，tiny 漏检 4,779 个，其中 2,141 个具有其他类有效原始框但
没有同类有效框，1,067 个有同类近邻但定位不足，987 个有同类有效原始框但未保留到
最终候选，414 个在缓存池中没有同类 IoU>=0.1 近邻。
这些为规定优先级下的描述性分类，不能将错类邻框直接归因为分类器错误；它可能属于邻近目标。
缓存已经过每视图 NMS/conf/max500，缺候选不是骨干特征失败的证据。

诊断结果：`output/analysis/headroom-fixed-20261013/decision.json` 为 `HEADROOM_ONLY`，
不是训练成功、不是撤销 STOP、更不是创新证明。
完整诊断的可复现指令：

```powershell
python scripts/evaluation/diagnose_headroom.py
```

## 这次只检验训练目标是否对齐选择任务

原头回归每个动作的相对 IoU。新预检直接训练动作之间的优先级：

- `uniform`：普通成对 logistic 排序，所有有效、有 IoU 差异的动作对等权。
- `regret`：相同排序损失，动作对的权重为 `abs(IoU_i-IoU_j)`，强调错排损失更大的动作。
- 每个原框内部对权重归一化；每个原框仍使用原缓存的逆采样权重。
- 两个头的 FP/重复候选约束完全相同：非 keep 动作不应获得高于 keep 的效用。
- 正样本标签保持原校准一对一 GT 归属，不能把某个动作切换到另一个 GT 来产生增益。
- 无有效邻框或已匹配 GT 的所有动作 IoU 相同，不产生正样本排序梯度。
- 推理读取预测特征，不使用 GT。选效用最大动作，`U(action)-U(keep)>0.01` 才改坐标；
  否则保留原框。效用是排序分数，不再解释为校准的 IoU 增益。

普通成对排序、按错误代价加权都已有一般性研究，不能把它们本身称为首创。
这里检验的是小目标、固定原框身份及既有跨视图动作集上的目标对齐收益。
这是一次缓存特征可学性预检；即使通过，发表前也仍需更接近的文献、独立数据和多种子证据。
它没有提供新的视觉信号；若失败，结束当前缓存特征/动作选择目标路线，
不能再修改损失权重、温度或阈值来追验证指标。

## 计算成本与匹配条件

- **无需准备新数据、无需重复五视图推理、无需重新训练 YOLO。**
- 直接读取已有完整 `output/localization/train_actions_seed0` 的 6,471 张 train 特征缓存。
- 两头都是原来的 `68 -> 64 -> 64 -> 1` SiLU MLP，每头 8,641 参数；完全相同的初始权重。
- 原缓存拟合行 630,619、训练内监控行 36,247；场景划分保持 6,131/340 张。
- 归一化沿用原拟合集 mean/std，不能从 val 或训练内监控集估计。
- 固定 seed0、20 epoch、batch2048、AdamW lr0.002/wd0.0001；排序温度 0.05。
- 两个头的初始化、训练样本、归一化、优化器、批次、轮数完全匹配，仅排序代价不同。
- 固定最后 epoch20；监控集损失只记录，不选择最优 epoch。
- 每轮原子保存模型、优化器和日志。中断后恢复最后完整轮次，无 dropout/随机增强。
- 本机已验证小规模训练和恢复；正式全量预检不自动启动，不据小样本指标定成败。

配方：`configs/experiments/regret_probe_960.yaml`。
新源快照和运行不会改写旧报告、旧权重、原 STOP 或旧源检查。
允许添加独立模块，但逐文件核验历史源快照和 ZIP；任何旧重放依赖变更都会拒绝。
完整训练只接受此前正式缓存的内容 hash；不重写历史缓存为当前版本。

## 一次实验固定十项门槛

新预检与旧 11 项筛选分开登记，旧结果保持 STOP。
以下全部通过才输出 CONTINUE，缺一项就是 STOP：

| 检查 | 门槛 |
|---|---:|
| regret 的 tiny AP，相对普通校准 | >= +0.50 pp |
| regret 的 small_all AP，相对普通校准 | >= +0.30 pp |
| regret 的整体 AP，相对普通校准 | >= +0.20 pp |
| small_all Recall@FPPI5，相对普通校准 | >= -0.10 pp |
| small_all Recall@FPPI10，相对普通校准 | >= -0.10 pp |
| small_all AP，相对已停止的旧增益头 | >= +0.20 pp |
| small_all AP，相对普通排序 uniform 头 | >= +0.20 pp |
| 同原 GT small_all 平均 IoU，相对 uniform | >= +0.50 pp |
| 上一项配对图像 95% CI 下界 | > 0 |
| 新增缓存后处理平均耗时 | <= 15 ms/图 |

相对旧增益头的门槛意味着 small_all AP 至少达到约 31.9371%，
比普通校准至少高约 0.4695 pp，不是降低旧最低收益要求。
bootstrap 固定按图像 2,000 次、seed0，操作阈值选定后固定；
不包含场景、训练种子或阈值选择的不确定性。
AP 为项目 COCO-style / IoF-ignore、reference640 诊断，不是官方 VisDrone AP。
计时为预热后 CPU1 的菜单、归一化、regret 头及选择，排除其它实验头、校准公共成本、
解码及检测器，不声称部署 FPS。

CONTINUE 只表示值得做独立验证，不允许自动加训练轮数。
STOP 表示停止本次缓存特征/排序目标路线，不等于放弃整个小目标领域。

## 你现在执行的指令

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python scripts/training/train_regret_probe.py --device 0
python scripts/evaluation/diagnose_regret_probe.py
```

上一条成功后再执行下一条。不要再执行旧 `train_localization.py --resume`。
不需要调用 `prepare_localization.py`；已有完整缓存就是本次输入。

中断恢复：

```powershell
python scripts/training/train_regret_probe.py --device 0 --resume
```

默认新运行：`output/runs/regret_objective_probe_960_seed0`。
每次评估创建独立 `output/analysis/regret-probe-<时间戳>`，不覆盖历史目录。
看 `decision.json` 和 `decision.md` 判断；同时保留四分支预测、分尺寸 AP、FP 预算、
同 GT 机制诊断、计时和完整性证据。
任何烟测缓存或 `--limit` 评估均为 SMOKE_ONLY，不能做正式判断。

## 为什么不承诺一次实验解决整个研究方向

一次固定实验能否证伪的是这个监督/特征假设，不是整个小目标领域。
这份 val 已被反复用于方案选择；通过也不能替代独立数据验证。
因此本次不把新的损失命名包装成已确定创新，不把 GT 辅助参考写成模型效果，
也不以多次改变门槛、种子或轮数来挽救失败结果。

## 已完成的代码验证

- 232 项相关测试通过；排除了两份此前已有 API/删除行为不兼容的旧队列测试文件，
  未修改旧队列行为。
- 新头在原 96 张烟测缓存上跑满 20 轮；两头训练和训练内监控损失均有限。
- 保存 epoch10 后模拟中断，再恢复：最终权重逐位相同，逐轮损失完全相同。
- 对已完成的 epoch20 执行恢复，检查点字节不变。
- 2 张 val 的四分支评估链路、校准精确重放、分数/类别/顺序/数量及内容 hash 均通过，
  输出 SMOKE_ONLY。这里的指标不用于调配方或判断研究效果。
- 已核验完整正式输入缓存 6,471 张图及训练内监控 340 张；默认正式新运行目录尚未创建。

证据在 `output/verification/regret_probe_20261013/implementation_verification.json`、
`runtime_verification.json` 和 `eval_smoke/verification.json`。
