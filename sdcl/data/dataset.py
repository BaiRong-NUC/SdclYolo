"""Keep ignored boxes inside the same geometric augmentation pipeline as GT."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import xyxy2xywh


@lru_cache(maxsize=20000)
def read_metadata(path: str):
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Missing ignore metadata: {file}. Run prepare_visdrone first.")
    with file.open(encoding="utf-8") as stream:
        return json.load(stream)


class IgnoreDataset(YOLODataset):
    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        image = Path(label["im_file"])
        metadata = read_metadata(str(image.parents[2] / "metadata" / image.parent.name / f"{image.stem}.json"))
        height, width = label["ori_shape"]
        if metadata["original_size"] != [width, height]:
            raise ValueError(f"Metadata/image size mismatch: {image}")
        boxes = np.asarray(metadata.get("ignore_boxes_xyxy", []), dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            boxes = xyxy2xywh(boxes)
            boxes /= np.array([width, height, width, height], dtype=np.float32)
            instances = label["instances"]
            ignored = Instances(
                boxes, np.zeros((0, instances.segments.shape[1], 2), dtype=np.float32),
                bbox_format="xywh", normalized=True,
            )
            label["instances"] = Instances.concatenate([instances, ignored])
            # This sentinel exists only during augmentation and never reaches the model.
            sentinel = np.full((len(boxes), 1), self.data["nc"], dtype=np.float32)
            label["cls"] = np.concatenate([label["cls"], sentinel], axis=0)
        return label

    def __getitem__(self, index):
        result = super().__getitem__(index)
        ignored = result["cls"].flatten() == self.data["nc"]
        result["ignore_bboxes"] = result["bboxes"][ignored]
        result["ignore_batch_idx"] = result["batch_idx"][ignored]
        result["bboxes"] = result["bboxes"][~ignored]
        result["cls"] = result["cls"][~ignored]
        result["batch_idx"] = result["batch_idx"][~ignored]
        return result

    @staticmethod
    def collate_fn(samples):
        ordinary = [{key: value for key, value in item.items() if key not in {"ignore_bboxes", "ignore_batch_idx"}}
                    for item in samples]
        batch = YOLODataset.collate_fn(ordinary)
        batch["ignore_bboxes"] = torch.cat([item["ignore_bboxes"] for item in samples], 0)
        batch["ignore_batch_idx"] = torch.cat(
            [item["ignore_batch_idx"] + index for index, item in enumerate(samples)], 0
        )
        return batch

