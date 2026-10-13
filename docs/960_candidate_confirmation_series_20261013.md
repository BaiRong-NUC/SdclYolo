# 候选逐项验证：固定权重 test-dev 确认

登记日期：2026-10-13。用户要求候选逐项验证，每次实验完成后提交代码并以实验写提交信息，
不将收益门槛设得过高。本登记先于任何本轮坐标方案的 test-dev 结果。

## 验证顺序

1. 相对收益监督：calibration、absolute、gain，主要比较 gain 对 absolute 的官方 AP。
2. 保留原框机制：gain、gain_always，主要比较官方 AR500，复用实验 1 的 gain 结果。
3. 加权排序：calibration、uniform、regret，主要比较 regret 对 uniform 的官方 AP。

当前六个分支全部使用已有固定 epoch20 权重、原归一化和原动作菜单。
只有分支 1、2、3 的评价逐次运行；准备阶段一次保存所有共享预测，不做提前候选选择。
候选“身份竞争模块”已被上一轮零邻框复用、极少疑似错位的条件诊断暂时否定投入价值，
不为凑齐候选启动训练。面向 IoU 阈值净收益的监督仍是独立新假设，不能把前三项的
成功自动当成它的效果；先完成固定权重确认，再判断一次训练预检是否值得。

## 温和的低成本投入标准

实验 1 与 3：

- 相对普通校准，官方 AP 至少增加 0.10 pp。
- 相对匹配普通方法，官方 AP 增量 >0。
- 相对普通校准，官方 AR500 变化 >=-0.50 pp。

满足以上输出 PROMISING_POINT_ESTIMATE，表示值得下一步确认，不表示显著或可发表。
正向但不足 0.10 pp 输出 SMALL_POSITIVE_SIGNAL；不能写成没有效果。
未超过匹配普通方法，输出 NO_TRANSFERRED_MECHANISM_ADVANTAGE，不能把普通方法收益
全部归因于专门机制。召回超过预算单独标为取舍，不据此否定小目标领域。

实验 2：相对始终替换，AR500 增量 >0 且 AP 不下降，即为正向机制信号。
不要求回退必须额外增加 0.30/0.50 pp AP。

0.10 pp 是探索预算标准，未被种子波动估计证明为噪声界；
-0.50 pp 是此次低成本探索容忍，尚不是具体应用要求。
没有将统计显著性、工程意义、机制指标混为一套“全部必须通过”的 10/11 项门槛。
尺寸 AP、条件 IoU 与计时作为解释性证据，不再一票否决。
所有旧门槛、STOP 与源快照保留。

## 数据和成本

1610 张 VisDrone2019-DET-test-dev，使用原始官方标注、固定工具及验证过的 Octave 辅助函数。
此前只看过普通校准/Soft-NMS/单视图结果，不能再称整个 test-dev 完全未查看；
本轮坐标机制比较未查看，没有 test 标签拟合、训练、阈值调整或 epoch 选择。
文件名/字节无交集不等于地理独立证明。一套训练种子不能证明稳定收益。

此前 test-dev 没存原始五视图几何，本轮补做一次冻结检测器推理；
必须逐图精确重放先前普通校准数组及官方 TXT，差异会直接拒绝评价。
共享保存原始框、预测、动作框/有效掩码/归一化特征和计时，以便后续候选复用。
不读取 test 忽略区来生成预测；GT 只参与评价、内容核验及独立诊断。
普通校准官方指标在全量预测精确重放后复用已验证 MAT，不重复跑它的官方评价。
尺寸结果为 reference640 COCO-style/IoF-ignore 诊断，不是官方 tiny/small AP。

## 复现指令

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python scripts/evaluation/confirm_coordinate_candidates.py prepare --device 0
python scripts/evaluation/confirm_coordinate_candidates.py evaluate --candidate relative
python scripts/evaluation/confirm_coordinate_candidates.py evaluate --candidate retention --reuse <实验1结果目录>
python scripts/evaluation/confirm_coordinate_candidates.py evaluate --candidate ranking
```

已完成的缓存/报告拒绝覆盖。默认缓存为
`output/candidate_series/testdev_actions_20261013`。
小样本 prepare --limit 只能检查链路，不允许正式评价或研究判断。

每项完整官方评价和核验结束后，将源代码、登记说明、精简指标及产物 hash 纳入独立 Git 提交。
不把下一项未完成的结果写成实验成功，不改上一次登记的门槛。
