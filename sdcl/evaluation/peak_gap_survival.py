"""Measure whether the native oracle peak gap survives the frozen control."""

import hashlib
import json
from time import perf_counter
import zipfile

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.native_capacity import checked_prior
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash, match_at_threshold
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator


REGISTRATION = "docs/experiments/candidate61_peak_gap_survival_registration_20261015.json"
OUTPUT = "output/analysis/peak-gap-survival-fixed-20261015"
PEAK = "output/analysis/native-peak-quality-fixed-20261015"
GROUPS = {"all": (0., float("inf")), "tiny": (0., 256.),
          "small_all": (0., 1024.), "medium_plus": (256., float("inf"))}


def support_scores(gt, classes, predictions):
    predictions = np.asarray(predictions, float).reshape(-1, 6)
    assert np.isfinite(predictions).all()
    overlap = box_iou_matrix(gt, predictions[:, :4])
    same = classes[:, None] == predictions[None, :, 5]
    mask = same & (overlap >= .75)
    maximum = np.where(mask, predictions[None, :, 4], 0.).max(1) if len(predictions) else np.zeros(len(gt))
    return maximum, overlap


def partition(peak_gap, raw001, calibrated001, calibrated025):
    assert np.all(~calibrated025 | calibrated001)
    result = np.full(len(peak_gap), -1, np.int8)
    result[peak_gap & calibrated025] = 0
    result[peak_gap & ~calibrated025 & calibrated001] = 1
    result[peak_gap & ~calibrated001 & raw001] = 2
    result[peak_gap & ~calibrated001 & ~raw001] = 3
    assert np.array_equal(result > 0, peak_gap & ~calibrated025)
    return result


def frame_statistics(frame):
    result = {}
    for group, (low, high) in GROUPS.items():
        keep = (frame["areas"] >= low) & (frame["areas"] < high)
        result[group] = dict(gt=int(keep.sum()),
            native_high=int((keep & frame["native_high"]).sum()),
            peak_gap=int((keep & frame["peak_gap"]).sum()),
            remaining_gap=int((keep & frame["remaining_gap"]).sum()),
            remaining_gap_images=int((keep & frame["remaining_gap"]).any()),
            partition=[int((keep & (frame["partition"] == k)).sum()) for k in range(4)],
            raw001_support=int((keep & frame["raw001"]).sum()),
            raw005_support=int((keep & frame["raw005"]).sum()),
            calibrated001_support=int((keep & frame["calibrated001"]).sum()),
            calibrated025_support=int((keep & frame["calibrated025"]).sum()),
            calibrated025_greedy_matched=int((keep & (frame["greedy_owner"] >= 0)).sum()),
            calibrated001_without_raw001=int((keep & frame["calibrated001"] & ~frame["raw001"]).sum()))
    return result


def aggregate(traces, policy):
    metrics, checks = {}, {}
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    for split, rows in traces.items():
        assert len(rows) == 64
        m = {}
        for group in GROUPS:
            m[group] = {k: sum(row[group][k] for row in rows) for k in rows[0][group] if k != "partition"}
            m[group]["partition"] = [sum(row[group]["partition"][j] for row in rows) for j in range(4)]
            assert sum(m[group]["partition"]) == m[group]["peak_gap"]
            assert sum(m[group]["partition"][1:]) == m[group]["remaining_gap"]
            m[group]["remaining_gap_fraction_all_GT"] = m[group]["remaining_gap"]/m[group]["gt"] if m[group]["gt"] else 0.
        denominator = draws@np.array([row["tiny"]["gt"] for row in rows])
        numerator = draws@np.array([row["tiny"]["remaining_gap"] for row in rows])
        ci = np.percentile(numerator/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        m["tiny_remaining_gap_fraction_bootstrap95ci"] = ci
        tiny = m["tiny"]
        checks[split] = dict(gt=tiny["gt"] >= policy["tiny_gt_min"],
            original_gap=tiny["peak_gap"] >= policy["tiny_original_peak_gap_min"],
            remaining=tiny["remaining_gap"] >= policy["tiny_remaining_gap_min"],
            images=tiny["remaining_gap_images"] >= policy["tiny_remaining_gap_images_min"],
            fraction=tiny["remaining_gap_fraction_all_GT"] >= policy["tiny_remaining_gap_fraction_all_GT_min"],
            interval=ci is not None and ci[0] > policy["tiny_remaining_gap_fraction_bootstrap_lower_min_exclusive"])
        metrics[split] = m
    passed = all(all(c.values()) for c in checks.values())
    return metrics, dict(checks=checks,
        decision="REPLICATED_POSTCONTROL_PEAK_GAP_SUPPORTS_SEPARATE_SMALL_LEARNABILITY_PILOT" if passed else
        "STOP_THIS_POSTCONTROL_PEAK_GAP_ROUTE_ON_FIXED_INVENTORY",
        scope="Fixed cached support gaps only, not recoverableTP,AP,falsepositive-control,training orinnovation evidence.")


def run():
    started = perf_counter()
    root, peak = project_path(OUTPUT), project_path(PEAK)
    if root.exists():
        raise FileExistsError(root)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    old = read(project_path("docs/experiments/candidate60_native_peak_quality_result_20261015.json"))
    for name, digest in old["artifacts_sha256"].items():
        assert file_hash(peak/name) == digest
    previous, peak_manifest = read(peak/"plan.json"), read(peak/"frames.json")
    for name, digest in previous["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    prior, prior_plan, prior_manifest, prior_result = checked_prior()
    assert file_hash(prior_result) == previous["prior_result_sha256"]
    control = frozen_control()
    calibrator = load_calibrator(control)
    selection = previous["image_selection"]
    assert len({s[:7] for stems in selection.values() for s in stems}) == 128
    sources = {p: file_hash(project_path(p)) for p in (
        REGISTRATION, "sdcl/evaluation/peak_gap_survival.py", "scripts/evaluation/diagnose_peak_gap_survival.py",
        "tests/test_peak_gap_survival.py", "sdcl/evaluation/native_capacity.py", "sdcl/evaluation/slicing_screen.py",
        "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py",
        "sdcl/evaluation/size_analysis.py", "sdcl/config.py")}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(project_path(name), name)
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    registration = read(project_path(REGISTRATION))
    plan = dict(registration=registration, image_selection=selection, source_hashes=sources,
        source_archive_sha256=file_hash(root/"source.zip"), prior60_result_sha256=file_hash(peak/"result.json"),
        prior60_closeout_sha256=file_hash(project_path("docs/experiments/candidate60_native_peak_quality_result_20261015.json")),
        prior60_plan_sha256=file_hash(peak/"plan.json"), prior60_manifest_sha256=file_hash(peak/"frames.json"),
        prior56_result_sha256=file_hash(prior_result), control_manifest_sha256=prior_plan["control_manifest_sha256"],
        literature_audit_sha256=file_hash(project_path("docs/literature/native_peak_quality_prior_audit_20261015.json")))
    save("plan.json", plan)
    manifest, traces = {}, {split: [] for split in selection}
    for split, stems in selection.items():
        for stem in stems:
            pe, entry = peak_manifest[stem], prior_manifest[stem]
            assert file_hash(peak/pe["file"]) == pe["sha256"]
            assert file_hash(prior/entry["file"]) == entry["sha256"] == pe["prior_frame_sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == pe["metadata_sha256"]
            metadata = read(meta_path)
            with np.load(peak/pe["file"], allow_pickle=False) as z:
                frame = {k: z[k].copy() for k in ("gt", "classes", "areas")}
                frame["native_high"] = (z["best_iou"] >= .75).any(1)
                frame["peak_gap"] = frame["native_high"] & ~(z["peak_iou"] >= .75).any(1)
            with np.load(prior/entry["file"], allow_pickle=False) as z:
                reference = z["reference"].copy()
                views = [z["local_"+str(v)].copy() for v in range(5)]
            windows = entry["windows"]
            mapped = [map_tile_predictions(v, w) if i else v.copy()
                      for i, (v, w) in enumerate(zip(views, windows, strict=True))]
            replay, _, _, _ = calibrated_candidates(mapped, windows, metadata["original_size"], [], calibrator)
            np.testing.assert_array_equal(reference, replay)
            raw_scores = np.column_stack([support_scores(frame["gt"], frame["classes"], v)[0] for v in mapped])
            scores, overlap = support_scores(frame["gt"], frame["classes"], reference)
            frame.update(raw_max_support_score=raw_scores, calibrated_max_support_score=scores,
                raw001=(raw_scores >= .001).any(1), raw005=(raw_scores >= .05).any(1),
                calibrated001=scores >= .001, calibrated025=scores >= .25)
            frame["greedy_owner"], _ = match_at_threshold(overlap, frame["classes"], reference, .25, .75)
            frame["partition"] = partition(frame["peak_gap"], frame["raw001"], frame["calibrated001"], frame["calibrated025"])
            frame["remaining_gap"] = frame["peak_gap"] & ~frame["calibrated025"]
            traces[split].append(frame_statistics(frame))
            path = root/"frames"/(stem+".npz")
            np.savez_compressed(path, **frame)
            manifest[stem] = dict(split=split, file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                prior60_frame_sha256=pe["sha256"], prior56_frame_sha256=entry["sha256"],
                metadata_sha256=file_hash(meta_path),
                replay_reference_array_sha256=hashlib.sha256(replay.tobytes()).hexdigest())
            if len(manifest) % 32 == 0:
                print(f"Peak gap after control: {len(manifest)}/128", flush=True)
    metrics, decision = aggregate(traces, registration["policy_per_half"])
    for split in selection:
        for group in GROUPS:
            assert metrics[split][group]["peak_gap"] == old["metrics"][split][group]["peak_loss"]
    for name, value in (("frames.json", manifest), ("per_image.json", traces),
                        ("metrics.json", metrics), ("decision.json", decision)):
        save(name, value)
    assert all(file_hash(project_path(p)) == d for p, d in sources.items())
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(root),
        metrics=metrics, decision=decision,
        costs=dict(cached_frames=128, exact_full_reference_replays=128, CPU_calibrator_MLP_calls=128,
                   JPEG_decodes=0, detector_encoder_forwards=0, fits=0, gradients=0, parameter_updates=0,
                   AP_evaluations=0, elapsed_seconds=perf_counter()-started),
        artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save("result.json", result)
    print(json.dumps(result, indent=2))
    return result
