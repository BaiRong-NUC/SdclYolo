# SDCL-YOLO

**基于尺度与难度协同学习的无人机航拍小目标检测**

Scale and Difficulty Collaborative Learning for UAV Small Object Detection.

当前为研究实现框架，已完成真实数据 30 轮对照与分尺寸诊断，尚未证实稳定收益。不要把候选方法称为已验证创新，也不要引用烟测的合成数据指标。

## 1. 已实现

- VisDrone 转换与数据检查：保留原始标注、忽略区域、遮挡字段和 SHA256 清单。
- YOLO11 检测模型／训练器扩展：不修改全局 `site-packages`。
- 目标级尺度—难度权重：支持 joint、scale、difficulty、additive 四种信号。
- 有界且保持正质量总量的权重调节、渐进启用。
- 正分类监督和逐正样本回归／DFL 加权；支持分类、回归、联合三个分支，不改变原始分配与软目标。
- 忽略区域同步进入增强流程，再从有效真值中拆分。
- 基线／SDCL 配置、诊断验证、VisDrone 格式预测导出、合成烟测与测试。

## 2. 安装与环境检查

使用 Conda `yolo`，Python 3.12.4。PyTorch／CUDA 安装见 `docs/实验环境安装说明.md`。

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python -m pip install -e . --no-deps
python -m sdcl env
python -m pytest
```

框架固定适配 `ultralytics==8.4.172`。本地项目以 editable 模式安装后可直接修改 `sdcl`。

脚本包装入口也可以直接运行，不依赖 editable 安装。加载本项目保存的自定义模型检查点时，需要本项目可导入。

## 3. 第一步：烟测，不需要下载数据

```powershell
python scripts/smoke_test.py --device 0
```

这会在新的 `output/smoke/` 子目录生成 4 张训练图与 2 张验证图，分别运行基线与 SDCL 的 YOLO11n 一个 epoch，然后保存、重新加载与验证。

模型由内置 YAML 初始化，不下载预训练模型。Ultralytics 首次运行可能自动获取字体等小型资源。无 GPU 时可用 `--device cpu`。

输出包含：

```text
output/smoke/<run>/
  dataset/
  runs/{baseline,sdcl}/
    weights/{best,last}.pt
    experiment.json
    sdcl_statistics.jsonl
  smoke_report.json
```

烟测只验证链路；随机初始化训练一个 epoch 的 AP 不用于方法判断。

## 4. 第二步：准备 VisDrone

官方入口与数据说明：

- https://github.com/VisDrone/VisDrone-Dataset
- https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip
- https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
- https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip

遵循数据使用条件，完整下载并解压到：

```text
data/raw/
  VisDrone2019-DET-train/{images,annotations}/
  VisDrone2019-DET-val/{images,annotations}/
  VisDrone2019-DET-test-dev/{images,annotations}/  # 可选
```

转换和检查：

```powershell
python scripts/prepare_visdrone.py --raw-root data/raw --output data/visdrone
python scripts/audit_dataset.py --dataset data/visdrone
```

输出包含 `data/visdrone/dataset.yaml`，两个实验配置直接使用它。转换器拒绝覆盖已有输出目录；先核查失败原因，不自动清空目录。

2026-10-04 本机已转换并审计真实数据：train 6471 张、val 548 张、test-dev 1610 张，
审计无错误。转换器支持解压后的同名嵌套目录，跳过并统计 3 条零高框，同时保留原始标注。
当前数据已可直接用于训练，无需重复转换。详细数量见 `data/README.md`。

## 5. 第三步：先跑短程基线

先做 5 epoch 的真实数据试跑，检查读取、损失和显存，不直接租卡跑完整实验：

```powershell
python scripts/train.py --config configs/experiments/baseline.yaml --epochs 5 --batch 8
```

确认后分别跑完整基线与 SDCL：

```powershell
python scripts/train.py --config configs/experiments/baseline.yaml
python scripts/train.py --config configs/experiments/sdcl.yaml
```

默认 YOLO11s、640 输入、200 epoch、batch 16、单 GPU、Windows workers 0。这是起点配置，不保证 12 GB 显存下所有密集图像都能装入；OOM 时对所有对照统一减小 batch。

模型首次使用 `yolo11s.pt` 时可能自动下载权重。训练前应先完成数据转换和检查，生成 `data/visdrone/dataset.yaml`。

## 6. 恢复训练与验证

恢复尚未完成、包含优化器状态的检查点：

```powershell
python scripts/train.py --config configs/experiments/sdcl.yaml --resume output/runs/<run>/weights/last.pt
```

恢复时 SDCL 设置与忽略协议必须和 `experiment.json` 相同。按保存的训练进度恢复调度；已完成且被上游剥离优化器的检查点，不用于继续恢复。

诊断验证：

```powershell
python scripts/validate.py --model output/runs/<run>/weights/best.pt
```

**这里的指标是带简化 IoF 忽略过滤的 Ultralytics 指标，不是官方 VisDrone AP。** IoF 过滤不完整实现官方的有效目标优先匹配、忽略区域合并等逻辑，只用于统一监控与调试。

比较两个最佳检查点的尺寸收益与漏检：

```powershell
python scripts/analyze_sizes.py --output output/analysis/size_comparison_new
```

默认比较 30 轮 baseline 与 SDCL，生成 COCO 风格分尺寸 AP、固定阈值 Recall、
逐目标新增检出／丢失记录和可视化。尺寸、匹配规则与限制见
[按尺寸评估与漏检分析](docs/按尺寸评估与漏检分析.md)。

导出真实原图坐标预测：

```powershell
python scripts/export_visdrone.py --model output/runs/<run>/weights/best.pt --images data/visdrone/images/val --output output/predictions/<run>
```

导出每图最多 500 个检测框、VisDrone 八字段文本。最终论文使用官方评价工具：

https://github.com/VisDrone/VisDrone2018-DET-toolkit

框架没有伪造一个“官方评价完成”入口；调用官方工具和对齐评价结果属于下一阶段工作。

## 7. 消融如何开始

分类／回归分支消融已完成 30 轮。复现实验时，两组配置默认 30 epoch、batch 8、640 输入、seed 0，
与已完成的 30 轮对照使用相同训练配方。逐条执行，第一组成功结束后再运行第二组：

```powershell
python scripts/train.py --config configs/experiments/sdcl_cls.yaml --epochs 30 --batch 8
python scripts/train.py --config configs/experiments/sdcl_reg.yaml --epochs 30 --batch 8
```

两组输出名分别为 `sdcl_cls_yolo11s_seed0` 与 `sdcl_reg_yolo11s_seed0`。
`sdcl.apply_to` 支持 `classification`（仅正分类监督）、`regression`（框回归与 DFL）
和 `both`（联合加权，旧配置默认值）。权重信号与渐进调度保持一致。
完整指令、评估入口和判断标准见
[分类与回归加权消融](docs/分类与回归加权消融.md)。

2026-10-07 下一组只修改分类加权形式，运行：

```powershell
python scripts/train.py --config configs/experiments/sdcl_cls_full_bce.yaml --epochs 30 --batch 8
```

`classification_weighting: full_bce` 对已匹配真实类别的完整软标签 BCE 加权，
其他类别与背景监督保持原处理；旧配置默认 `positive_term`，仅对 BCE 正项加权。
本组 `apply_to: classification`，与已完成的分类消融仅在分类加权形式上不同。
这是一组机制验证实验，尚未证明效果提升。

后续再拆尺度／难度信号：

复制 `configs/experiments/sdcl.yaml`，给每个 run 设置不同 `name`：

```yaml
sdcl:
  enabled: true
  signal: scale  # joint / scale / difficulty / additive
```

设置 `difficulty_alpha: 1.0` 或 `0.0` 比较分类／定位难度；这不控制权重作用的损失分支。
设置 `lambda_max: 0.0` 检查零强度等价性。参数、种子和训练预算保持一致。

当前框架只实现首版核心方法。位置级、非守恒、随机权重、非单调难度和背景扩展仍待实现，不要把它们写成已完成消融。

## 8. 目录

```text
configs/experiments/  # baseline 与 SDCL 配方
sdcl/                # 方法、数据、训练器与 CLI
scripts/             # 可直接运行的入口
tests/               # 张量公式、损失等价、数据增强、CUDA AMP
data/                # 本地真实数据，不提交
docs/                # 实现方案、文献分析、安装说明
reference/           # 参考方法，与主训练依赖隔离
output/              # 检查点、日志、烟测，不提交
```

## 9. 验证记录与限制

初次验证环境：Python 3.12.4、Torch 2.10.0+cu128、TorchVision 0.25.0+cu128、Ultralytics 8.4.172，RTX 4070 Ti。

已验证内容见 `docs/代码框架与开始步骤.md`。当前测试为 71 项通过，包含两种分类加权形式、
三个加权分支的损失／梯度对照、固定软目标处的梯度检查、零强度等价性、CUDA AMP
与旧配置兼容检查；已完成真实数据全量审计、
三个分集的画框抽样、YOLO11s 基线／SDCL 的真实训练与 30 轮对照检查，
并增加分尺寸评价与漏检分析入口。尚未完成第二数据集、官方指标评价或多卡训练。

当前限制：

- 只支持 YOLO11 常规水平框检测头，拒绝端到端检测头。
- 只支持单设备，不支持 DDP；不支持类别过滤／单类别重映射。
- 忽略框采用增强哨兵类别，需要保证不在格式化后泄漏，已有翻转／Mosaic 测试。
- 部分复杂混合增强与真实密集数据仍需进一步验证。
- 权重有界、总质量守恒不等于损失／梯度守恒，也不意味着能识别错标。
- 方法、代码有效性与投稿创新性是三个不同验收层级。

Ultralytics 具有自身许可要求；公开、分发或商业使用前核查上游 AGPL／商业许可条件。本框架暂不添加未经选择的项目许可证。
