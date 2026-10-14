"""Rebuild candidate54 geometry and native-support gates without model forwards."""

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


OUTPUT = "output/analysis/augmentation-censoring-fixed-20261015"
LIMITS = dict(all=math.inf, tiny=256., small_all=1024.)


def contained(box, bounds):
    return (box[2] > box[0] and box[3] > box[1] and
            box[0] >= bounds[0] and box[1] >= bounds[1] and
            box[2] <= bounds[2] and box[3] <= bounds[3])


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


def source_boxes(stem, layout):
    folder = project_path("data/visdrone")
    rows = np.loadtxt(folder/f"labels/train/{stem}.txt", dtype=np.float32).reshape(-1, 5)
    _, indices = np.unique(rows, axis=0, return_index=True)
    if len(indices) < len(rows):
        rows = rows[indices]
    metadata = json.loads((folder/f"metadata/train/{stem}.json").read_text())
    ow, oh = metadata["original_size"]
    ratio = 960/max(ow, oh)
    np.testing.assert_array_equal(layout["resized_shape"], [min(math.ceil(oh*ratio), 960), min(math.ceil(ow*ratio), 960)])
    ignored = np.array(metadata.get("ignore_boxes_xyxy", []), np.float32).reshape(-1, 4)
    if len(ignored):
        xywh = np.empty_like(ignored)
        xywh[:, 0] = (ignored[:, 0]+ignored[:, 2])/2
        xywh[:, 1] = (ignored[:, 1]+ignored[:, 3])/2
        xywh[:, 2:] = ignored[:, 2:]-ignored[:, :2]
        xywh /= np.array([ow, oh, ow, oh], np.float32)
        rows = np.concatenate((rows, np.column_stack((np.full(len(ignored), 10, np.float32), xywh))))
    normalized = rows[:, 1:]
    boxes = np.empty_like(normalized)
    boxes[:, :2] = normalized[:, :2]-normalized[:, 2:]/2
    boxes[:, 2:] = normalized[:, :2]+normalized[:, 2:]/2
    h, w = layout["resized_shape"]
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    full = np.array([contained(b, layout["crop"]) for b in boxes], bool)
    boxes[:, [0, 2]] += layout["offset"][0]
    boxes[:, [1, 3]] += layout["offset"][1]
    assert len(boxes) == layout["count"]
    return boxes, rows[:, 0], full


def verify():
    torch.set_num_threads(1)
    output = project_path(OUTPUT)
    read = lambda path: json.loads(path.read_text())
    result, plan = read(output/"result.json"), read(output/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest
    with zipfile.ZipFile(output/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            assert file_hash(project_path(name)) == digest
        for name, item in plan["installed_sources"].items():
            assert hashlib.sha256(z.read("installed/"+name)).hexdigest() == item["sha256"]
            assert file_hash(Path(item["path"])) == item["sha256"]
    assert file_hash(project_path(plan["checkpoint"])) == plan["checkpoint_sha256"]
    excluded = set()
    for name, digest in plan["exclusions"].items():
        path = project_path(name)
        assert file_hash(path) == digest
        excluded.update(re.findall(r"\d{7}_\d{5}_[a-z]_\d{7}", path.read_text(encoding="utf-8")))
    assert len(excluded) == plan["excluded_unique_stems"]
    all_paths = sorted(project_path("data/visdrone/images/train").glob("*.jpg"),
                       key=lambda p: hashlib.sha256(("candidate54-visible-censor-v1:"+p.stem).encode()).hexdigest())
    expected = [p.stem for p in all_paths if p.stem not in excluded][:256]
    assert expected == plan["images"] and len(set(expected)) == 256 and not (set(expected) & excluded)
    for stem, item in plan["sources"].items():
        for key, relative in (("image", f"images/train/{stem}.jpg"), ("label", f"labels/train/{stem}.txt"),
                              ("metadata", f"metadata/train/{stem}.json")):
            assert file_hash(project_path("data/visdrone")/relative) == item[key]
    layouts, manifest = read(output/"layouts.json"), read(output/"frames.json")
    traces, native_raw_support, max_coordinate_error = [], 0, 0.
    assigner = TaskAlignedAssigner(topk=10, num_classes=10, alpha=.5, beta=6., stride=[8., 16., 32.])
    for index in range(64):
        seed = int.from_bytes(hashlib.sha256(f"candidate54-warp-v1:{index}".encode()).digest()[:4], "little")
        assert seed == plan["seeds"][index]
        items = layouts[index]
        assert [item["stem"] for item in items] == expected[4*index:4*index+4]
        path = output/manifest[str(index)]["file"]
        assert file_hash(path) == manifest[str(index)]["sha256"]
        with np.load(path, allow_pickle=False) as z:
            frame = {k: z[k].copy() for k in z.files}
        parts = [source_boxes(item["stem"], item) for item in items]
        originals, classes, visible = [np.concatenate([p[k] for p in parts]) for k in range(3)]
        for key, actual in (("source_boxes", originals), ("source_classes", classes), ("source_fully_visible", visible)):
            np.testing.assert_allclose(actual, frame[key], atol=2e-4, rtol=0)
        clipped = np.clip(originals, 0, 1920)
        good = (clipped[:, 2:] > clipped[:, :2]).all(1)
        np.testing.assert_array_equal(good, frame["mosaic_good"])
        np.testing.assert_array_equal(classes[good], frame["mosaic_classes"])
        np.testing.assert_array_equal(visible[good], frame["mosaic_fully_visible"])
        boxes = frame["mosaic_boxes"]
        np.testing.assert_allclose(clipped[good], boxes, atol=2e-4, rtol=0)
        corners = np.ones((len(boxes), 4, 3), np.float32)
        for k, (x, y) in enumerate(((0, 1), (2, 1), (2, 3), (0, 3))):
            corners[:, k, 0], corners[:, k, 1] = boxes[:, x], boxes[:, y]
        transformed = corners @ frame["matrix"].T
        unclipped = np.column_stack((transformed[:, :, :2].min(1), transformed[:, :, :2].max(1)))
        np.testing.assert_allclose(unclipped, frame["affine_unclipped"], atol=2e-4, rtol=0)
        clipped = np.clip(unclipped, 0, 960)
        np.testing.assert_allclose(clipped, frame["transformed_boxes"], atol=2e-4, rtol=0)
        max_coordinate_error = max(max_coordinate_error, float(np.max(np.abs(clipped-frame["transformed_boxes"]), initial=0)))
        # Use saved float32 coordinates at discontinuous threshold boundaries.
        b = frame["transformed_boxes"]
        reference = boxes.copy()
        reference *= float(frame["scale"])
        np.testing.assert_allclose(reference, frame["scaled_reference"], atol=2e-4, rtol=0)
        full = np.array([bool(v) and contained(box, [0, 0, 960, 960])
                         for v, box in zip(frame["mosaic_fully_visible"], frame["affine_unclipped"])])
        valid = (frame["mosaic_classes"] >= 0) & (frame["mosaic_classes"] < 10)
        width, height = (b[:, 2:]-b[:, :2]).T
        rw, rh = (frame["scaled_reference"][:, 2:]-frame["scaled_reference"][:, :2]).T
        size_bad, area_bad = (width <= 2) | (height <= 2), width*height/(rw*rh+1e-16) <= .1
        aspect_bad = np.maximum(width/(height+1e-16), height/(width+1e-16)) >= 100
        keep = ~(size_bad | area_bad | aspect_bad)
        for key, actual in (("fully_visible", full), ("valid_gt", valid), ("keep", keep),
                            ("size_rejected", size_bad), ("area_rejected", area_bad), ("aspect_rejected", aspect_bad)):
            np.testing.assert_array_equal(actual, frame[key])
        areas = width*height*(640/960)**2
        np.testing.assert_array_equal(areas, frame["areas"])
        trace, supported = {}, np.zeros(len(b), bool)
        if result["decision"]["native_stage_measured"]:
            anchors, probability, predictions = frame["native_anchors"], frame["native_probability"], frame["native_boxes"]
            assert anchors.shape == (18900, 2) and probability.shape == (18900, 10)
            assert np.isfinite(probability).all() and np.isfinite(predictions).all()
            retained = keep & valid
            tensor = lambda a: torch.tensor(a, dtype=torch.float32)
            with torch.no_grad():
                _, _, _, foreground, _ = assigner(
                    tensor(probability)[None], tensor(predictions)[None], tensor(anchors),
                    tensor(frame["mosaic_classes"][retained])[None, :, None], tensor(b[retained])[None],
                    torch.ones((1, int(retained.sum()), 1), dtype=torch.bool))
            np.testing.assert_array_equal(foreground[0].numpy(), frame["native_foreground"])
            ignored = np.zeros(len(anchors), bool)
            for ignore in b[keep & (frame["mosaic_classes"] == 10)]:
                ignored |= ((anchors[:, 0] >= ignore[0]) & (anchors[:, 0] <= ignore[2]) &
                            (anchors[:, 1] >= ignore[1]) & (anchors[:, 1] <= ignore[3]))
            np.testing.assert_array_equal(ignored, frame["native_ignored"])
            argmax = np.argmax(probability, 1)
            eligible = ~foreground[0].numpy() & ~ignored
            pred_area = (predictions[:, 2]-predictions[:, 0])*(predictions[:, 3]-predictions[:, 1])
            for row in np.flatnonzero(full & valid & ~keep):
                target, cls = b[row], int(frame["mosaic_classes"][row])
                iw = np.maximum(np.minimum(target[2], predictions[:, 2])-np.maximum(target[0], predictions[:, 0]), 0)
                ih = np.maximum(np.minimum(target[3], predictions[:, 3])-np.maximum(target[1], predictions[:, 1]), 0)
                overlap = iw*ih
                iou = overlap/np.maximum(pred_area+(target[2]-target[0])*(target[3]-target[1])-overlap, 1e-12)
                raw_support = (iou >= .5) & (argmax == cls) & (probability[:, cls] >= .05)
                native_raw_support += int(raw_support.any())
                supported[row] = (raw_support & eligible).any()
            np.testing.assert_array_equal(supported, frame["native_supported"])
        for group, limit in LIMITS.items():
            selected = full & valid & (areas < limit)
            rejected = selected & ~keep
            trace[group] = dict(fully_visible_gt=int(sum(selected)), rejected_gt=int(sum(rejected)),
                affected_mosaics=int(any(rejected)), size_rejected_gt=int(sum(rejected & size_bad)),
                area_rejected_gt=int(sum(rejected & area_bad)), aspect_rejected_gt=int(sum(rejected & aspect_bad)))
            if result["decision"]["native_stage_measured"]:
                trace[group].update(supported_gt=int(sum(selected & supported)), supported_mosaics=int(any(selected & supported)))
        traces.append(trace)
    for actual, expected_trace in zip(traces, read(output/"per_image.json"), strict=True):
        compare(actual, expected_trace)
    metrics, checks, native_checks = {}, {}, {}
    p, q = plan["registration"]["first_stage_policy"], plan["registration"]["second_stage_policy"]
    for half, items in (("screen", traces[:32]), ("confirmation", traces[32:])):
        metrics[half] = {}
        for group in LIMITS:
            total = {key: sum(t[group][key] for t in items) for key in items[0][group]}
            denominator = total["fully_visible_gt"]
            total["rejected_fraction"] = total["rejected_gt"]/denominator if denominator else 0.
            if "supported_gt" in total:
                total["supported_fraction"] = total["supported_gt"]/denominator if denominator else 0.
            metrics[half][group] = total
        tiny = metrics[half]["tiny"]
        checks[half] = dict(denominator=tiny["fully_visible_gt"] >= p["fully_visible_tiny_gt_per_half_min"],
                            rejected=tiny["rejected_gt"] >= p["fully_visible_discarded_tiny_per_half_min"],
                            mosaics=tiny["affected_mosaics"] >= p["affected_mosaics_per_half_min"],
                            fraction=tiny["rejected_fraction"] >= p["discarded_tiny_fraction_per_half_min"])
        if result["decision"]["native_stage_measured"]:
            native_checks[half] = dict(supported=tiny["supported_gt"] >= q["supported_discarded_tiny_per_half_min"],
                                      mosaics=tiny["supported_mosaics"] >= q["supported_mosaics_per_half_min"],
                                      fraction=tiny["supported_fraction"] >= q["supported_tiny_fraction_per_half_min"])
    compare(metrics, result["metrics"])
    assert checks == result["decision"]["geometry_checks"]
    assert (native_checks or None) == result["decision"]["native_checks"]
    assert result["decision"]["decision"] == "STOP_THIS_FULLY_VISIBLE_AUGMENTATION_CENSORING_ROUTE"
    assert not (all(all(v.values()) for v in checks.values()) and
                native_checks and all(all(v.values()) for v in native_checks.values()))
    audit = dict(all_sources_archives_256_source_images_labels_metadata_and64_frames_hashed=True,
                 exclusions_selection_source_GT_and_crop_visibility_rebuilt=True,
                 mosaic_affine_coordinates_maximum_absolute_error=max_coordinate_error,
                 all_filter_reasons_groups_both_halves_and_fixed_gates_rebuilt=True,
                 retained_GT_TAL_foreground_replayed_CPU_from_saved_predictions=64 if native_checks else 0,
                 raw_native_support_before_background_filter=native_raw_support,
                 all_ignore_masks_and_native_support_recomputed=True,
                 extra_detector_forwards=0, extra_JPEG_decodes=0, extra_optimizer_updates=0,
                 verifier_sha256=file_hash(Path(__file__)),
                 verifier_repair="First attempt used a NumPy float64 0d scale instead of original Python float; corrected scalar type, retaining original2e-4pixel tolerance.",
                 limits="Coordinate recomputation tolerance2e-4pixels; threshold decisions use saved float32 coordinates. TAL replay shares installed assigner source. Raw detector tensors and image warps are hashed, not independently re-extracted/rewarped. No AP, historical training replay or novelty proof.")
    destination = output/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
