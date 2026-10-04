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

代码烟测的合成样本位于 `output/smoke/`，只验证程序，不用于论文指标。
