# 实验 5：排序任务与代价权重的两个新种子确认

登记日期：2026-10-13，先于 seed1/seed2 训练及其结果。已知 seed0 的 test-dev 结果，
本实验是同一数据上的头初始化/批次顺序重复，不是独立测试集确认。

## 固定对照和成本

复用 6471 张 train 动作缓存、场景 holdout、GT 身份、采样权重和历史 mean/std。
冻结检测器、普通校准、六动作菜单、框分数/类别/数量/顺序。
新种子固定为 1、2，每个种子同时训练 absolute、uniform、regret 三头。
absolute 是已有 SmoothL1 绝对 IoU 回归；uniform/regret 精确复用既有排序损失。
三头共用初始化和批次顺序，容量均为 68 -> 64 -> 64 -> 1。
固定 20 epoch、batch2048、AdamW lr0.002、wd0.0001、margin0.01。
只取 epoch20；训练场景 holdout 仅监控，不选 checkpoint、不调温度或 margin。
使用 seed+epoch 决定批次顺序，与原 seed0 配方一致。

1610 张已查看的 test-dev 仅用于评估，复用动作缓存；不重新推理/训练检测器。
每个新种子的三头完成官方评价，种子 0 使用全量已核验的冻结结果。
不把 3 次头重复叫作 3 次完整检测器训练，不挑最高 AP 的种子。

## 提前固定的温和预算判断

先分别列出每个种子的 AP、AR500，然后汇总 seed0/1/2 等权均值、标准差、极差。

排序任务有继续投入价值的三个条件：

- uniform 相对普通校准的平均官方 AP 至少 +0.10 pp；
- uniform 相对匹配 absolute 的平均官方 AP >0，且至少 2/3 种子方向为正；
- uniform 相对普通校准的平均官方 AR500 >=-0.50 pp。

正向但不足 +0.10 pp 记录为小正向信号，不写成没效果。
权重额外贡献单独判断：regret 对 uniform 的平均 AP >0，且至少 2/3 种子为正，
regret 对校准的平均 AR500 >=-0.50 pp。没有要求额外 +0.30/+0.50 pp。
达标只是预算筛选；非常小的权重收益仍不能独立支撑创新或实用结论。
不以任一 seed 的结果修改旧实验 1–4 的判定。

本轮不做 test-dev 超参数扫描、种子选择或集成。分尺寸与耗时只能解释结果，
不增加一组同时必须通过的硬门槛。不启动新的检测器长训练。
若排序不稳定，停止当前坐标头配方；若只有普通排序稳定，将它作为强对照，
不把普通成对排序、回退或代价权重宣称原创。

## 复现

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python scripts/training/train_ranking_seed_confirmation.py --device 0
python scripts/evaluation/confirm_ranking_seeds.py --workers 2
```

固定训练输出 `output/runs/ranking_seed_confirmation_960_20261013`；
已存在的训练/评估产物拒绝覆盖。完整结果结束后单独 Git 提交。
