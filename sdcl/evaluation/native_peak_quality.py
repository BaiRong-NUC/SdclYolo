"""Frozen inventory diagnostic of score-selected native localization quality."""

import json
from time import perf_counter
import zipfile

import numpy as np
from scipy.special import expit

from sdcl.config import project_path
from sdcl.evaluation.native_capacity import checked_prior, map_boxes
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash


REGISTRATION = "docs/experiments/candidate60_native_peak_quality_registration_20261015.json"
OUTPUT = "output/analysis/native-peak-quality-fixed-20261015"
FIELDS = ("count", "best_iou", "peak_iou", "peak_min_iou", "peak_count", "max_logit",
          "uniform_mean_iou", "uniform_high_iou_fraction")
GROUPS = {"all": (0., float("inf")), "tiny": (0., 256.),
          "small_all": (0., 1024.), "medium_plus": (256., float("inf"))}


def pool_statistics(overlap, logits, classes, valid):
    overlap, logits = np.asarray(overlap, float), np.asarray(logits)
    result = {k: np.zeros(len(classes)) for k in FIELDS}
    best_class = logits.argmax(1)
    high = expit(logits[np.arange(len(logits)), best_class].astype(float)) >= .05
    for row, cls in enumerate(classes):
        pool = np.flatnonzero(valid & high & (best_class == cls) & (overlap[row] >= .5))
        if not len(pool):
            continue
        quality, score = overlap[row, pool], logits[pool, cls]
        maximum = score.max()
        tied = quality[score == maximum]
        result["count"][row] = len(pool)
        result["best_iou"][row] = quality.max()
        result["peak_iou"][row] = tied.max()
        result["peak_min_iou"][row] = tied.min()
        result["peak_count"][row] = len(tied)
        result["max_logit"][row] = maximum
        result["uniform_mean_iou"][row] = quality.mean()
        result["uniform_high_iou_fraction"][row] = (quality >= .75).mean()
    return result


def frame_statistics(areas, values):
    supported = (values["best_iou"] >= .75).any(1)
    selected = (values["peak_iou"] >= .75).any(1)
    loss = supported & ~selected
    assert not np.any(selected & ~supported)
    result = {}
    for group, (low, high) in GROUPS.items():
        keep = (areas >= low) & (areas < high)
        result[group] = dict(gt=int(keep.sum()), high_quality_supported=int((keep & supported).sum()),
            peak_high_quality_selected=int((keep & selected).sum()),
            peak_loss=int((keep & loss).sum()), peak_loss_images=int((keep & loss).any()))
    return result


def aggregate(traces, policy):
    metrics, checks = {}, {}
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    for split, rows in traces.items():
        assert len(rows) == 64
        m = {g: {k: sum(row[g][k] for row in rows) for k in rows[0][g]} for g in GROUPS}
        for value in m.values():
            value["peak_loss_fraction_all_GT"] = value["peak_loss"]/value["gt"] if value["gt"] else 0.
        count = draws@np.array([row["tiny"]["gt"] for row in rows])
        numerator = draws@np.array([row["tiny"]["peak_loss"] for row in rows])
        ci = np.percentile(numerator/count, [2.5, 97.5]).tolist() if (count > 0).all() else None
        m["tiny_peak_loss_fraction_bootstrap95ci"] = ci
        tiny = m["tiny"]
        checks[split] = dict(gt=tiny["gt"] >= policy["tiny_gt_min"],
            supported=tiny["high_quality_supported"] >= policy["tiny_high_quality_supported_min"],
            loss=tiny["peak_loss"] >= policy["tiny_peak_loss_min"],
            images=tiny["peak_loss_images"] >= policy["tiny_peak_loss_images_min"],
            fraction=tiny["peak_loss_fraction_all_GT"] >= policy["tiny_peak_loss_fraction_all_GT_min"],
            interval=ci is not None and ci[0] > policy["tiny_peak_loss_fraction_bootstrap_lower_min_exclusive"])
        metrics[split] = m
    passed = all(all(row.values()) for row in checks.values())
    return metrics, dict(checks=checks,
        decision="REPLICATED_ORACLE_NATIVE_PEAK_QUALITY_GAP_SUPPORTS_DISTINCT_LOW_COST_FEASIBILITY" if passed
        else "STOP_THIS_NATIVE_SCORE_PEAK_HIGH_IOU_ROUTE_ON_FIXED_INVENTORY",
        scope="GT-conditioned optimistic native ranking gap only; no deployable/AP/innovation or detector-training claim.")


def run():
    tick = perf_counter()
    root = project_path(OUTPUT)
    if root.exists():
        raise FileExistsError(root)
    prior, previous, old_manifest, previous_result = checked_prior()
    registration = json.loads(project_path(REGISTRATION).read_text(encoding="utf-8"))
    selection = {s: previous["image_selection"][s] for s in ("screen", "confirmation")}
    assert all(len(stems) == 64 for stems in selection.values())
    assert len({stem[:7] for stems in selection.values() for stem in stems}) == 128
    sources = {p: file_hash(project_path(p)) for p in (
        REGISTRATION, "sdcl/evaluation/native_peak_quality.py", "scripts/evaluation/diagnose_native_peak_quality.py",
        "tests/test_native_peak_quality.py", "sdcl/evaluation/native_capacity.py",
        "sdcl/evaluation/size_analysis.py", "sdcl/config.py")}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(project_path(name), name)
    plan = dict(registration=registration, image_selection=selection, source_hashes=sources,
        source_archive_sha256=file_hash(root/"source.zip"), prior_result_sha256=file_hash(previous_result),
        prior_plan_sha256=file_hash(prior/"plan.json"),
        prior_frames_manifest_sha256=file_hash(prior/"frames.json"))
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    save("plan.json", plan)
    manifest, traces = {}, {s: [] for s in selection}
    for split, stems in selection.items():
        for stem in stems:
            entry = old_manifest[stem]
            assert file_hash(prior/entry["file"]) == entry["sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == previous["inputs"][stem]["metadata_sha256"]
            metadata = json.loads(meta_path.read_text())
            gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
            areas = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)*(640/max(metadata["original_size"]))**2
            with np.load(prior/entry["file"], allow_pickle=False) as z:
                boxes = map_boxes(z["native_boxes"], entry["windows"]).reshape(5, 18900, 4)
                logits = z["native_logits"].copy()
            values = {k: np.zeros((len(gt), 5)) for k in FIELDS}
            for view in range(5):
                valid = (np.isfinite(boxes[view]).all(1) & np.isfinite(logits[view]).all(1) &
                         (boxes[view, :, 2:] > boxes[view, :, :2]).all(1))
                for start in range(0, len(gt), 32):
                    overlap = box_iou_matrix(gt[start:start+32], boxes[view])
                    block = pool_statistics(overlap, logits[view], classes[start:start+32], valid)
                    for key, data in block.items():
                        values[key][start:start+len(data), view] = data
            traces[split].append(frame_statistics(areas, values))
            path = root/"frames"/(stem+".npz")
            np.savez_compressed(path, gt=gt, classes=classes, areas=areas, **values)
            manifest[stem] = dict(split=split, file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                prior_frame_sha256=entry["sha256"], metadata_sha256=file_hash(meta_path))
            if len(manifest) % 32 == 0:
                print(f"Native peak quality: {len(manifest)}/128", flush=True)
    metrics, decision = aggregate(traces, registration["policy_per_half"])
    for name, value in (("frames.json", manifest), ("per_image.json", traces),
                        ("metrics.json", metrics), ("decision.json", decision)):
        save(name, value)
    assert all(file_hash(project_path(p)) == h for p, h in sources.items())
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(root),
        metrics=metrics, decision=decision,
        costs=dict(cached_frames=128, JPEG_decodes=0, detector_forwards=0, fits=0, gradients=0, updates=0,
                   AP_evaluations=0, elapsed_seconds=perf_counter()-tick),
        artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save("result.json", result)
    print(json.dumps(result, indent=2))
    return result
