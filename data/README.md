# 实验数据目录

本目录中的真实图像、标注、缓存和权重不提交到 Git。

先下载并解压 VisDrone2019-DET，推荐结构：

```text
data/raw/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
  VisDrone2019-DET-test-dev/  # 可选，需包含标注
```

然后执行：

```powershell
python scripts/prepare_visdrone.py --raw-root data/raw --output data/visdrone
python scripts/audit_dataset.py --dataset data/visdrone
```

转换器保留原始标注、忽略区域 metadata 和 SHA256 清单，并生成 `data/visdrone/dataset.yaml`。输出目录已存在时拒绝覆盖。

也支持压缩包解压后出现的同名嵌套目录，例如
`data/raw/VisDrone2019-DET-train/VisDrone2019-DET-train/{images,annotations}/`，
无需移动原始数据。零宽／零高标注跳过并计入 `outside_or_zero_area`；负尺寸和其他非法标注仍报错。

2026-10-04 本机已完成三个分集的转换与审计：

| 分集 | 图像 | 有效框 | 忽略框 |
|---|---:|---:|---:|
| train | 6471 | 343204 | 10343 |
| val | 548 | 38759 | 1410 |
| test-dev | 1610 | 75102 | 2445 |

训练集有 3 条零高框，已跳过并记录，原始标注完整保留。上述有效框数量为转换后的数量；
Ultralytics 加载训练集时另去除 4 条重复标签，转换文件与原始文件不因此改写。

现有 `data/visdrone/` 可直接用于训练，不要重复执行转换。检查记录见
`output/real-data-check/dataset_audit.json`，抽样画框见同目录的 `preview_{train,val,test}.jpg`。
绿色表示有效目标，红色表示忽略区域。

代码烟测的合成样本位于 `output/smoke/`，只验证程序，不用于论文指标。
