"""Independent kept-array support, partition and matching audit for61."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


def independent_support(gt, classes, predictions):
    scores, overlaps = [], np.zeros((len(gt), len(predictions)))
    for j, box in enumerate(gt):
        intersections = (np.maximum(0, np.minimum(predictions[:, 2], box[2])-np.maximum(predictions[:, 0], box[0]))*
                         np.maximum(0, np.minimum(predictions[:, 3], box[3])-np.maximum(predictions[:, 1], box[1])))
        area = np.maximum(0, predictions[:, 2]-predictions[:, 0])*np.maximum(0, predictions[:, 3]-predictions[:, 1])
        union = (box[2]-box[0])*(box[3]-box[1])+area-intersections
        overlaps[j] = intersections/np.maximum(union, 1e-12)
        eligible = (predictions[:, 5] == classes[j]) & (overlaps[j] >= .75)
        scores.append(float(max(predictions[eligible, 4])) if eligible.any() else 0.)
    return np.array(scores), overlaps


def verify():
    root = project_path("output/analysis/peak-gap-survival-fixed-20261015")
    peak = project_path("output/analysis/native-peak-quality-fixed-20261015")
    prior = project_path("output/analysis/texture-deletion-fixed-20261015")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    assert not (root/"independent_verification.json").exists()
    result, plan = read(root/"result.json"), read(root/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
    for number, directory, key in ((60, peak, "prior60_closeout_sha256"), (56, prior, "prior56_result_sha256")):
        name = ("candidate60_native_peak_quality_result_20261015.json" if number == 60
                else "candidate56_texture_deletion_result_20261015.json")
        closeout = project_path("docs/experiments/"+name)
        assert file_hash(closeout) == plan[key]
        for artifact, digest in read(closeout)["artifacts_sha256"].items():
            assert file_hash(directory/artifact) == digest
        for source, digest in read(directory/"plan.json")["source_hashes"].items():
            assert file_hash(project_path(source)) == digest
    assert file_hash(peak/"result.json") == plan["prior60_result_sha256"]
    assert file_hash(peak/"plan.json") == plan["prior60_plan_sha256"]
    assert file_hash(peak/"frames.json") == plan["prior60_manifest_sha256"]
    assert file_hash(project_path("docs/literature/native_peak_quality_prior_audit_20261015.json")) == plan["literature_audit_sha256"]
    control_path = project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")
    assert file_hash(control_path) == plan["control_manifest_sha256"]
    control = read(control_path)
    for key in ("scorer", "detector", "prediction_capture"):
        assert file_hash(Path(control[key]["path"])) == control[key]["sha256"]
    assert file_hash(Path(control["source_archive"])) == control["source_archive_sha256"]
    with zipfile.ZipFile(control["source_archive"]) as z:
        for source, digest in control["source_hashes"].items():
            assert file_hash(project_path(source)) == hashlib.sha256(z.read(source)).hexdigest() == digest
    selection = plan["image_selection"]
    assert selection == read(peak/"plan.json")["image_selection"]
    assert all(len(stems) == 64 for stems in selection.values())
    assert len({s[:7] for stems in selection.values() for s in stems}) == 128
    manifest, pm, old = read(root/"frames.json"), read(peak/"frames.json"), read(prior/"frames.json")
    traces, queries = {}, 0
    groups = {"all": (0., math.inf), "tiny": (0., 256.), "small_all": (0., 1024.),
              "medium_plus": (256., math.inf)}
    for split, stems in selection.items():
        traces[split] = []
        for stem in stems:
            entry, pe, original = manifest[stem], pm[stem], old[stem]
            assert entry["split"] == split
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(peak/pe["file"]) == entry["prior60_frame_sha256"] == pe["sha256"]
            assert file_hash(prior/original["file"]) == entry["prior56_frame_sha256"] == original["sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == entry["metadata_sha256"]
            metadata = read(meta_path)
            gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
            areas = (gt[:, 2]-gt[:, 0])*(gt[:, 3]-gt[:, 1])*(640/max(metadata["original_size"]))**2
            with np.load(peak/pe["file"], allow_pickle=False) as z:
                np.testing.assert_array_equal(gt, z["gt"])
                np.testing.assert_array_equal(classes, z["classes"])
                np.testing.assert_allclose(areas, z["areas"], atol=1e-12, rtol=1e-14)
                high = np.any(z["best_iou"] >= .75, axis=1)
                gap = high & ~np.any(z["peak_iou"] >= .75, axis=1)
            raw_scores = []
            with np.load(prior/original["file"], allow_pickle=False) as z:
                reference = z["reference"].copy()
                assert hashlib.sha256(reference.tobytes()).hexdigest() == entry["replay_reference_array_sha256"]
                for view, window in enumerate(original["windows"]):
                    v = z["local_"+str(view)].copy()
                    if view:
                        x1, y1, x2, y2 = window
                        v[:, 0] = np.clip(v[:, 0], 0, x2-x1)+np.float32(x1)
                        v[:, 2] = np.clip(v[:, 2], 0, x2-x1)+np.float32(x1)
                        v[:, 1] = np.clip(v[:, 1], 0, y2-y1)+np.float32(y1)
                        v[:, 3] = np.clip(v[:, 3], 0, y2-y1)+np.float32(y1)
                    raw_scores.append(independent_support(gt, classes, v.astype(float))[0])
            raw_scores = np.column_stack(raw_scores)
            scores, overlap = independent_support(gt, classes, reference)
            queries += len(gt)*6
            raw001, raw005, c001, c025 = (raw_scores >= .001).any(1), (raw_scores >= .05).any(1), scores >= .001, scores >= .25
            category = np.full(len(gt), -1, np.int8)
            for j in np.flatnonzero(gap):
                category[j] = 0 if c025[j] else 1 if c001[j] else 2 if raw001[j] else 3
            remaining, owners, taken = gap & ~c025, np.full(len(gt), -1), set()
            for row in sorted(range(len(reference)), key=lambda r: (-reference[r, 4], r)):
                if reference[row, 4] < .25:
                    continue
                available = [j for j in range(len(gt)) if j not in taken and classes[j] == reference[row, 5] and overlap[j, row] >= .75]
                if available:
                    chosen = max(available, key=lambda j: (overlap[j, row], -j))
                    taken.add(chosen)
                    owners[chosen] = row
            expected = dict(gt=gt, classes=classes, areas=areas, native_high=high, peak_gap=gap,
                raw_max_support_score=raw_scores, calibrated_max_support_score=scores,
                raw001=raw001, raw005=raw005, calibrated001=c001, calibrated025=c025,
                partition=category, remaining_gap=remaining, greedy_owner=owners)
            with np.load(root/entry["file"], allow_pickle=False) as z:
                assert set(expected) == set(z.files)
                for name, array in expected.items():
                    if name == "areas":
                        np.testing.assert_allclose(array, z[name], atol=1e-12, rtol=1e-14)
                    else:
                        np.testing.assert_array_equal(array, z[name])
            summary = {}
            for group, (lo, hi) in groups.items():
                keep = (areas >= lo) & (areas < hi)
                summary[group] = dict(gt=int(sum(keep)), native_high=int(sum(keep & high)),
                    peak_gap=int(sum(keep & gap)), remaining_gap=int(sum(keep & remaining)),
                    remaining_gap_images=int(any(keep & remaining)),
                    partition=[int(sum(keep & (category == k))) for k in range(4)],
                    raw001_support=int(sum(keep & raw001)), raw005_support=int(sum(keep & raw005)),
                    calibrated001_support=int(sum(keep & c001)), calibrated025_support=int(sum(keep & c025)),
                    calibrated025_greedy_matched=int(sum(keep & (owners >= 0))),
                    calibrated001_without_raw001=int(sum(keep & c001 & ~raw001)))
            traces[split].append(summary)
    assert traces == read(root/"per_image.json")
    metrics, checks = {}, {}
    policy = plan["registration"]["policy_per_half"]
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    for split, rows in traces.items():
        m = {}
        for group in groups:
            m[group] = {k: sum(row[group][k] for row in rows) for k in rows[0][group] if k != "partition"}
            m[group]["partition"] = [sum(row[group]["partition"][j] for row in rows) for j in range(4)]
            assert sum(m[group]["partition"][1:]) == m[group]["remaining_gap"]
            m[group]["remaining_gap_fraction_all_GT"] = m[group]["remaining_gap"]/m[group]["gt"] if m[group]["gt"] else 0.
        ratios = []
        for draw in draws:
            denominator = sum(int(w)*row["tiny"]["gt"] for w, row in zip(draw, rows, strict=True))
            assert denominator > 0
            ratios.append(sum(int(w)*row["tiny"]["remaining_gap"] for w, row in zip(draw, rows, strict=True))/denominator)
        ci = np.percentile(ratios, [2.5, 97.5]).tolist()
        m["tiny_remaining_gap_fraction_bootstrap95ci"] = ci
        tiny = m["tiny"]
        checks[split] = dict(gt=tiny["gt"] >= policy["tiny_gt_min"],
            original_gap=tiny["peak_gap"] >= policy["tiny_original_peak_gap_min"],
            remaining=tiny["remaining_gap"] >= policy["tiny_remaining_gap_min"],
            images=tiny["remaining_gap_images"] >= policy["tiny_remaining_gap_images_min"],
            fraction=tiny["remaining_gap_fraction_all_GT"] >= policy["tiny_remaining_gap_fraction_all_GT_min"],
            interval=ci[0] > policy["tiny_remaining_gap_fraction_bootstrap_lower_min_exclusive"])
        metrics[split] = m
    assert metrics == result["metrics"] and checks == result["decision"]["checks"]
    expected = ("REPLICATED_POSTCONTROL_PEAK_GAP_SUPPORTS_SEPARATE_SMALL_LEARNABILITY_PILOT"
                if all(all(c.values()) for c in checks.values()) else
                "STOP_THIS_POSTCONTROL_PEAK_GAP_ROUTE_ON_FIXED_INVENTORY")
    assert expected == result["decision"]["decision"]
    audit = dict(all_source_archive_prior_control_weight_metadata_frame_hashes_verified=True,
        fixed128images_and_sequence_disjointness_verified=True,
        independently_rebuilt_GT_view_or_reference_support_queries=queries,
        all_raw_calibrated_support_scores_flags_partition_and_greedy_owners_reconstructed=True,
        all_groups_traces_bootstrap_and_twelve_gates_rebuilt=True,
        added_JPEG_decodes=0, added_detector_encoder_forwards=0, added_calibrator_calls=0,
        added_fits=0, added_updates=0, verifier_sha256=file_hash(Path(__file__)),
        limits="Uses same cached detector/reference arrays. Main replayed all128references exactly through frozen calibration; verifier does not independently reimplement calibration MLP orSoftNMS. Support may reuse detections acrossGT; noignore/officialAP/FPbudget/causality/learnability/innovation proof.")
    (root/"independent_verification.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
