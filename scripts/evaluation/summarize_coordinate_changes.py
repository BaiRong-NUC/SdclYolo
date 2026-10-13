"""Prediction-only coordinate-change diagnostics across the frozen seed controls."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


def summarize(output):
    paths = [project_path(p) for p in (
        "output/candidate_series/testdev_actions_20261013/predictions.jsonl",
        "output/analysis/ranking-seeds-fixed-20261013/predictions.jsonl",
        "output/analysis/prior-gap-fixed-20261013/predictions.jsonl",
    )]
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite independent diagnostic evidence.")
    hashes = {str(p): file_hash(p) for p in [*paths, Path(__file__).resolve()]}
    sums, images, rows = {}, [], 0
    with paths[0].open() as original, paths[1].open() as repeats, paths[2].open() as priors:
        for first, second, third in zip(original, repeats, priors, strict=True):
            baseline, repeat, prior = map(json.loads, (first, second, third))
            stem = baseline["image"]
            if stem != repeat["image"] or stem != prior["image"] or stem in images:
                raise ValueError("Different image order or duplicated image.")
            images.append(stem)
            c = np.asarray(baseline["arms"]["calibration"]).reshape(-1, 6)
            arms = {f"{a}_seed0": baseline["arms"][a] for a in ("absolute", "uniform", "regret")}
            arms.update(repeat["arms"])
            arms.update(prior["arms"])
            if len(arms) != 12:
                raise ValueError("Missing coordinate diagnostic arms.")
            scale = np.maximum(c[:, 2:4] - c[:, :2], 1e-6)
            scale = np.column_stack((scale, scale))
            rows += len(c)
            for arm, values in arms.items():
                p = np.asarray(values).reshape(-1, 6)
                if p.shape != c.shape or not np.array_equal(p[:, 4:], c[:, 4:]):
                    raise ValueError("Coordinates must be the only changed prediction fields.")
                moved = (p[:, :4] != c[:, :4]).any(1)
                distance = (np.abs(p[:, :4] - c[:, :4]) / scale).mean(1)
                aggregate = sums.setdefault(arm, {
                    "coordinate_changed_count": 0, "normalized_l1_sum": 0.,
                    "changed_normalized_l1_sum": 0.,
                })
                aggregate["coordinate_changed_count"] += int(moved.sum())
                aggregate["normalized_l1_sum"] += float(distance.sum())
                aggregate["changed_normalized_l1_sum"] += float(distance[moved].sum())
            if len(images) == 1 or len(images) % 400 == 0:
                print(f"Prediction-only coordinate diagnostic: {len(images)}/1610", flush=True)
    if len(images) != 1610:
        raise ValueError("This diagnostic requires all1610 images.")
    if any(file_hash(p) != h for p, h in hashes.items()):
        raise ValueError("Diagnostic captures or source changed.")
    summary = {}
    for arm, aggregate in sums.items():
        count = aggregate["coordinate_changed_count"]
        summary[arm] = {
            "coordinate_changed_count": count, "coordinate_changed_percent": count / rows * 100,
            "mean_normalized_l1_all_boxes": aggregate["normalized_l1_sum"] / rows,
            "mean_normalized_l1_changed_boxes": (
                aggregate["changed_normalized_l1_sum"] / count if count else 0.
            ),
        }
    report = {
        "images": len(images), "calibration_boxes": rows, "inputs_source_sha256": hashes,
        "summary": summary,
        "verification": {
            "same_images_order_rows_scores_classes": True, "all1610images": True,
            "captures_source_unchanged": True, "no_annotation_read": True,
        },
        "scope": "Post-hoc prediction-only coordinate-change frequency and displacement. "
                 "Equal coordinates count unchanged even if a different action was selected. "
                 "No AP, IoU, efficacy, GT identity or new registered decision claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default="output/analysis/prior-gap-coordinate-changes-20261013.json")
    print(json.dumps(summarize(**vars(parser.parse_args())), indent=2))
