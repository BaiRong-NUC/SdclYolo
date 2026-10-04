"""VisDrone conversion, auditing and a small local smoke-test fixture."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import yaml

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
RAW_SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_annotation(lines, width, height):
    labels, ignored, objects = [], [], []
    stats = Counter()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            values = [float(value.strip()) for value in line.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError(f"Non-numeric annotation at line {number}") from exc
        if len(values) != 8 or not np.isfinite(values).all():
            raise ValueError(f"Expected 8 finite annotation fields at line {number}")
        x, y, bw, bh, score, category, truncation, occlusion = values
        if category != int(category) or score < 0 or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid annotation at line {number}")
        if not 0 <= int(category) <= 11:
            raise ValueError(f"Unknown VisDrone category at line {number}: {category}")
        x1, y1 = max(0.0, min(width, x)), max(0.0, min(height, y))
        x2, y2 = max(0.0, min(width, x + bw)), max(0.0, min(height, y + bh))
        if x2 <= x1 or y2 <= y1:
            stats["outside_or_zero_area"] += 1
            continue
        box = [x1, y1, x2, y2]
        if score == 0 or int(category) in (0, 11):
            ignored.append(box)
            stats["ignored"] += 1
            continue
        cls = int(category) - 1
        norm = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                (x2 - x1) / width, (y2 - y1) / height]
        labels.append(f"{cls} " + " ".join(f"{value:.9f}" for value in norm))
        objects.append({
            "object_id": number, "class_id": cls, "box_xyxy": box,
            "occlusion": int(occlusion), "truncation": int(truncation),
        })
        stats["valid"] += 1
    return labels, ignored, objects, dict(stats)


def prepare_visdrone(raw_root, output):
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    splits = {}
    # Check completeness before writing anything.
    for split, folder in RAW_SPLITS.items():
        source = raw_root / folder
        if not source.is_dir():
            if split in {"train", "val"}:
                raise FileNotFoundError(f"Missing required raw split: {source}")
            continue
        images = sorted((source / "images").glob("*.jpg"))
        if not images:
            raise ValueError(f"No JPG images found: {source}")
        missing = [image.stem for image in images if not (source / "annotations" / f"{image.stem}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing annotations in {source}: {missing[:5]}")
        splits[split] = (source, images)
    for split, (source, images) in splits.items():
        for kind in ("images", "labels", "metadata", "original_annotations"):
            (output / kind / split).mkdir(parents=True)
        manifest = []
        for image in images:
            annotation = source / "annotations" / f"{image.stem}.txt"
            with Image.open(image) as decoded:
                width, height = decoded.size
                decoded.verify()
            text = annotation.read_text(encoding="utf-8")
            labels, ignores, objects, stats = convert_annotation(text.splitlines(), width, height)
            shutil.copy2(image, output / "images" / split / image.name)
            shutil.copy2(annotation, output / "original_annotations" / split / annotation.name)
            (output / "labels" / split / f"{image.stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )
            metadata = {
                "image_id": image.stem, "original_size": [width, height],
                "ignore_boxes_xyxy": ignores, "valid_objects": objects,
            }
            (output / "metadata" / split / f"{image.stem}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest.append({
                "image": image.name, "image_sha256": sha256(image),
                "annotation_sha256": sha256(annotation), **stats,
            })
        (output / "manifests").mkdir(exist_ok=True)
        (output / "manifests" / f"{split}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = {
        "path": str(output), "train": "images/train", "val": "images/val",
        "names": dict(enumerate(VISDRONE_NAMES)),
    }
    if "test" in splits:
        config["test"] = "images/test"
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def audit_dataset(root):
    root = Path(root).resolve()
    stats, errors = {}, []
    for split in ("train", "val", "test"):
        directory = root / "images" / split
        if not directory.is_dir():
            continue
        images = sorted(file for file in directory.iterdir() if file.suffix.lower() in {".jpg", ".png", ".jpeg"})
        counts = Counter(images=len(images))
        for image in images:
            try:
                with Image.open(image) as decoded:
                    size = list(decoded.size)
                    decoded.verify()
                label = root / "labels" / split / f"{image.stem}.txt"
                metadata = json.loads((root / "metadata" / split / f"{image.stem}.json").read_text(encoding="utf-8"))
                if metadata["original_size"] != size:
                    raise ValueError("Image/metadata dimensions differ")
                for line in label.read_text(encoding="utf-8").splitlines():
                    values = [float(value) for value in line.split()]
                    if len(values) != 5 or not np.isfinite(values).all():
                        raise ValueError("Invalid YOLO label")
                    cls, cx, cy, width, height = values
                    if cls != int(cls) or not 0 <= cls < len(VISDRONE_NAMES):
                        raise ValueError("Invalid class ID")
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("Invalid normalized box")
                    if min(cx - width / 2, cy - height / 2) < -1e-6 or max(cx + width / 2, cy + height / 2) > 1 + 1e-6:
                        raise ValueError("Box extends beyond image")
                    counts["objects"] += 1
                for box in metadata["ignore_boxes_xyxy"]:
                    x1, y1, x2, y2 = box
                    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
                        raise ValueError("Invalid ignore box")
                counts["ignored"] += len(metadata["ignore_boxes_xyxy"])
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{image}: {exc}")
        stats[split] = dict(counts)
    if not stats:
        errors.append(f"No dataset splits found: {root}")
    return {"root": str(root), "splits": stats, "errors": errors, "ok": not errors}


def make_toy_visdrone(raw_root, size=128):
    """Synthetic fixtures only: these are not research/evaluation data."""
    root = Path(raw_root)
    if root.exists():
        raise FileExistsError(root)
    generator = np.random.default_rng(2026)
    for split, count in (("train", 4), ("val", 2)):
        directory = root / RAW_SPLITS[split]
        (directory / "images").mkdir(parents=True)
        (directory / "annotations").mkdir()
        for index in range(count):
            image = Image.fromarray(generator.integers(0, 60, (size, size, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 36, 36), fill=(230, 200, 20))
            draw.rectangle((75, 60, 95, 80), fill=(20, 180, 220))
            image.save(directory / "images" / f"toy_{split}_{index}.jpg")
            (directory / "annotations" / f"toy_{split}_{index}.txt").write_text(
                "20,20,16,16,1,1,0,0\n75,60,20,20,1,4,0,0\n0,0,16,16,0,0,0,0\n", encoding="utf-8"
            )
    return root

