"""Exploratory owned-GT crossings; no prediction generation, fitting or AP claim."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import (
    box_areas, box_iou_matrix, file_hash, load_records, match_at_threshold, record_hash,
)


def analyze(capture, output):
    capture, output = project_path(capture), project_path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite diagnostic evidence.")
    records = load_records(project_path("data/visdrone"), "val")
    expected = {r["stem"]: r for r in records}
    thresholds = np.arange(10) * .05 + .5
    arms = ("vote", "absolute", "gain_always", "gain", "uniform", "regret")
    totals = {arm: {group: {
        "owned": 0, "changed": 0, "improved_iou": 0, "worsened_iou": 0,
        "gained_thresholds": [0] * 10, "lost_thresholds": [0] * 10,
    } for group in ("all", "tiny", "small_all")} for arm in arms}
    digest = file_hash(capture)
    seen = set()
    with capture.open() as stream:
        for line in stream:
            item = json.loads(line)
            stem = item["image"]
            if stem not in expected or stem in seen:
                raise ValueError("Unexpected or duplicate captured image.")
            seen.add(stem)
            r = expected[stem]
            gt = np.asarray([o["box_xyxy"] for o in r["objects"]]).reshape(-1, 4)
            classes = np.asarray([o["class_id"] for o in r["objects"]])
            c = np.asarray(item["arms"]["calibration"]).reshape(-1, 6)
            matched, _ = match_at_threshold(box_iou_matrix(gt, c[:, :4]), classes, c, .001)
            ids = np.flatnonzero(matched >= 0)
            rows = matched[ids]
            old = np.diag(box_iou_matrix(gt[ids], c[rows, :4]))
            areas = box_areas(gt[ids]) * (640 / max(r["size"]))**2
            for arm in arms:
                pred = np.asarray(item["arms"][arm]).reshape(-1, 6)
                if pred.shape != c.shape or not np.array_equal(pred[:, 4:], c[:, 4:]):
                    raise ValueError("Coordinate audit invariants changed.")
                new = np.diag(box_iou_matrix(gt[ids], pred[rows, :4]))
                changed = np.any(pred[rows, :4] != c[rows, :4], axis=1)
                for group, mask in (
                    ("all", np.ones(len(ids), dtype=bool)),
                    ("tiny", areas < 16**2), ("small_all", areas < 32**2),
                ):
                    result = totals[arm][group]
                    result["owned"] += int(mask.sum())
                    result["changed"] += int(changed[mask].sum())
                    result["improved_iou"] += int((new[mask] > old[mask] + 1e-9).sum())
                    result["worsened_iou"] += int((new[mask] < old[mask] - 1e-9).sum())
                    before = old[mask, None] >= thresholds[None]
                    after = new[mask, None] >= thresholds[None]
                    for key, values in (
                        ("gained_thresholds", ((~before) & after).sum(0)),
                        ("lost_thresholds", (before & (~after)).sum(0)),
                    ):
                        result[key] = [x + int(y) for x, y in zip(result[key], values)]
    if seen != set(expected) or file_hash(capture) != digest:
        raise ValueError("Incomplete or changed prediction capture.")
    for groups in totals.values():
        for result in groups.values():
            result["net_thresholds"] = [
                a - b for a, b in zip(result["gained_thresholds"], result["lost_thresholds"])
            ]
    result = {
        "scope": "Post-hoc same-owned-GT counts, not AP; excludes unmatched GT and rematching.",
        "reference_size": 640, "thresholds": thresholds.tolist(),
        "prediction_capture_sha256": digest, "records_sha256": record_hash(records),
        "analysis_source_sha256": file_hash(__file__), "arms": totals,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"output": str(output), "images": len(seen), "scope": result["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(analyze(**vars(parser.parse_args())), indent=2))
