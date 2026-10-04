import json

import numpy as np
import pytest
import torch
from ultralytics.cfg import DEFAULT_CFG, get_cfg

from sdcl.data import audit_dataset, convert_annotation, make_toy_visdrone, prepare_visdrone
from sdcl.geometry import anchor_ignore_mask
from sdcl.trainer import make_dataset


def test_conversion_clip_ignore_and_invalid():
    labels, ignores, objects, stats = convert_annotation(
        ["-2,10,12,10,1,4,0,1", "0,0,10,10,0,0,0,0", "10,10,5,5,1,11,0,0"], 100, 100
    )
    assert labels[0].startswith("3 ")
    assert objects[0]["box_xyxy"] == [0, 10, 10, 20]
    assert len(ignores) == 2
    assert stats["valid"] == 1
    with pytest.raises(ValueError):
        convert_annotation(["bad"], 100, 100)
    with pytest.raises(ValueError):
        convert_annotation(["0,0,10,10,1,12,0,0"], 100, 100)


@pytest.fixture
def converted(tmp_path):
    raw = make_toy_visdrone(tmp_path / "raw")
    yaml = prepare_visdrone(raw, tmp_path / "converted")
    return yaml.parent


def dataset_for(root, augment=False, **settings):
    args = get_cfg(DEFAULT_CFG, overrides={
        "imgsz": 128, "workers": 0, "mosaic": 0.0, "degrees": 0.0,
        "translate": 0.0, "scale": 0.0, "fliplr": 0.0, "flipud": 0.0,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, **settings,
    })
    data = {"nc": 10, "names": {index: str(index) for index in range(10)}, "channels": 3}
    return make_dataset(args, str(root / "images" / "train"), data, "train" if augment else "val", 2, 32, True)


def test_prepare_and_audit(converted):
    report = audit_dataset(converted)
    assert report["ok"]
    assert report["splits"]["train"]["objects"] == 8
    assert report["splits"]["train"]["ignored"] == 4
    metadata = json.loads((converted / "metadata/train/toy_train_0.json").read_text())
    assert metadata["ignore_boxes_xyxy"] == [[0, 0, 16, 16]]
    with pytest.raises(FileExistsError):
        prepare_visdrone(converted, converted)


def test_dataset_separates_and_collates(converted):
    dataset = dataset_for(converted)
    sample = dataset[0]
    assert sample["cls"].shape == (2, 1)
    assert sample["ignore_bboxes"].shape == (1, 4)
    batch = dataset.collate_fn([sample, dataset[1]])
    assert batch["bboxes"].shape == (4, 4)
    assert batch["ignore_batch_idx"].tolist() == [0, 1]


def test_ignore_horizontal_flip(converted):
    dataset = dataset_for(converted, augment=True, fliplr=1.0)
    sample = dataset[0]
    assert sample["ignore_bboxes"][0, 0] == pytest.approx(1 - 8 / 128)
    assert sample["ignore_bboxes"][0, 1] == pytest.approx(8 / 128)


def test_mosaic_sentinel_does_not_reach_labels(converted):
    np.random.seed(0)
    dataset = dataset_for(converted, augment=True, mosaic=1.0)
    for index in range(4):
        sample = dataset[index]
        assert (sample["cls"] < 10).all()
        assert (sample["ignore_bboxes"] >= 0).all()
        assert (sample["ignore_bboxes"] <= 1).all()


def test_anchor_mask():
    points = torch.tensor([[4, 4], [40, 40], [99, 99]]).float()
    boxes = torch.tensor([[0.05, 0.05, 0.1, 0.1]])
    mask = anchor_ignore_mask(points, boxes, torch.tensor([0]), 2, (100, 100))
    assert mask.tolist() == [[True, False, False], [False, False, False]]
