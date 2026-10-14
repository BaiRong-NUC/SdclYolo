"""Preregistered deployment screen for shared-site cross-class contradiction."""

import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import box_iou_matrix, coco_size_metrics, file_hash, match_at_threshold, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.class_competition import ARMS, competition_evidence, suppress_scores
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.training.transport_quality import diagnostic_filter


REGISTRATION = "docs/experiments/candidate64_shared_site_class_competition_registration_20261015.json"
AUDIT = "docs/literature/shared_site_class_competition_prior_audit_20261015.json"
PRIOR = "output/analysis/transport-quality-fixed-20261015"
OUTPUT = "output/analysis/shared-site-class-competition-fixed-20261015"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def checked_prior():
    prior = project_path(PRIOR)
    closeout = project_path("docs/experiments/candidate62_transport_quality_result_20261015.json")
    previous, old = read(closeout), read(prior/"plan.json")
    for name, digest in previous["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    with zipfile.ZipFile(prior/"source.zip") as archive:
        for name, digest in old["source_hashes"].items():
            assert file_hash(project_path(name)) == digest
            import hashlib
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    for name, digest in old["installed_source_hashes"].items():
        assert file_hash(name) == digest
    return prior, closeout, old, read(prior/"frames.json")


def cluster_interval(records, maps, comparison):
    counts, differences = [], []
    thresholds = {k: v["confidence"] for k, v in comparison["models"].items()}
    for record in records:
        gt = np.asarray([o["box_xyxy"] for o in record["objects"]], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in record["objects"]], int)
        area = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)*(640/max(record["size"]))**2
        keep = area < 1024.
        tp = []
        for label, arm in (("baseline", "strong"), ("sdcl", "shared")):
            dt = maps[arm][record["stem"]]
            matching, _ = match_at_threshold(box_iou_matrix(gt, dt[:, :4]), classes, dt, thresholds[label])
            tp.append(int((keep & (matching >= 0)).sum()))
        counts.append(int(keep.sum()))
        differences.append(tp[1]-tp[0])
    rows = [dict(stem=r["stem"], gt=g, TP_difference=d) for r, g, d in zip(records, counts, differences)]
    prefixes = sorted({r["stem"][:7] for r in rows})
    gt = np.array([sum(r["gt"] for r in rows if r["stem"][:7] == s) for s in prefixes])
    delta = np.array([sum(r["TP_difference"] for r in rows if r["stem"][:7] == s) for s in prefixes])
    draws = np.random.default_rng(0).multinomial(len(prefixes), [1/len(prefixes)]*len(prefixes), size=400)
    denominator = draws@gt
    ci = np.percentile((draws@delta)/denominator*100, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
    return dict(sequence_units=len(prefixes), interval_pp=ci, per_image=rows)


def decide(reports, policy):
    checks = {}
    for split, report in reports.items():
        ap = report["AP"]
        tests = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"],
                     changed_rows=report["changed_rows"]["shared"] >= policy["changed_rows_min"])
        for group, arm in (("tiny", "strong"), ("small_all", "strong"), ("all", "strong"),
                           ("tiny", "marginal"), ("tiny", "affinity"), ("tiny", "geometry")):
            x, y = ap["shared"][group]["ap50_95"], ap[arm][group]["ap50_95"]
            key = f"{group}_ap_vs_{arm}"
            tests[key] = x is not None and y is not None and (x-y)*100 >= policy[key+"_min_pp"]
        for index, budget in enumerate((5, 10)):
            value = report["fixed_fp"]["shared"][index]["delta_recall_pp"]["small_all"]
            key = f"small_all_recall_fppi{budget}_vs_strong"
            tests[key] = value is not None and value >= policy[key+"_min_pp"]
        ci = report["small_all_fppi10_sequence_bootstrap"]["interval_pp"]
        tests["sequence_interval"] = ci is not None and ci[0] > policy[
            "small_all_recall_fppi10_sequence_interval_lower_min_exclusive_pp"]
        tests["processing"] = report["mean_evidence_plus_four_arm_processing_ms"] <= policy[
            "mean_evidence_plus_four_arm_processing_ms_max"]
        checks[split] = tests
    passed = all(all(v.values()) for v in checks.values())
    return dict(checks=checks, decision=(
        "REPEATED_DEPLOYABLE_SHARED_SITE_CLASS_COMPETITION_SUPPORTS_SEPARATE_OFFICIAL_EXTERNAL_AUDIT" if passed else
        "STOP_THIS_SHARED_SITE_CROSS_CLASS_CONTRADICTION_RECIPE"),
        scope="Deterministic score-only deployment pilot on reused detector-training data. Not official AP,external generalization orinnovation; no full training recommendation.")


def run():
    started = perf_counter()
    root = project_path(OUTPUT)
    if root.exists():
        raise FileExistsError(root)
    prior, previous_path, old, manifest = checked_prior()
    selection = {s: old["image_selection"][s] for s in ("screen", "confirmation")}
    assert len(selection["screen"]) == 32 and len(selection["confirmation"]) == 17
    assert len({s[:7] for s in selection["screen"]}) == 32
    assert len({s[:7] for s in selection["confirmation"]}) == 5
    assert not {s[:7] for s in selection["screen"]} & {s[:7] for s in selection["confirmation"]}
    registration, audit = read(project_path(REGISTRATION)), read(project_path(AUDIT))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    control = frozen_control()
    calibrator = load_calibrator(control)
    sources = (REGISTRATION, AUDIT, "sdcl/postprocess/class_competition.py", "sdcl/evaluation/class_competition.py",
        "scripts/evaluation/diagnose_class_competition.py", "tests/test_class_competition.py",
        "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py", "sdcl/evaluation/size_analysis.py",
        "sdcl/data/geometry.py", "sdcl/training/transport_quality.py", "sdcl/evaluation/slicing_screen.py",
        "sdcl/evaluation/official_details.py", "sdcl/config.py")
    hashes = {p: file_hash(project_path(p)) for p in sources}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(project_path(name), name)
    plan = dict(registration=registration, image_selection=selection, source_hashes=hashes,
        source_archive_sha256=file_hash(root/"source.zip"), prior62_closeout_sha256=file_hash(previous_path),
        prior62_plan_sha256=file_hash(prior/"plan.json"), prior62_frames_sha256=file_hash(prior/"frames.json"),
        strong_control_manifest_sha256=file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")),
        detector_sha256=control["detector"]["sha256"], calibrator_sha256=control["scorer"]["sha256"],
        evidence="No GT read or outcomes computed before source/selection freeze.")
    save(root/"plan.json", plan)
    output_manifest, reports = {}, {}
    for split, stems in selection.items():
        records, timings, changes = [], [], {a: 0 for a in ARMS}
        maps = {a: {} for a in ("strong", *ARMS)}
        pair_count, active_pair_count = 0, 0
        for count, stem in enumerate(stems, 1):
            entry = manifest[stem]
            assert entry["split"] == split
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == entry["image_sha256"]
            reference = prior/entry["reference"]["file"]
            assert file_hash(reference) == entry["reference"]["sha256"]
            strong = np.load(reference, allow_pickle=False)
            native_views, local_views = [], []
            for view, snapshot in enumerate(entry["views"]):
                path = prior/snapshot["file"]
                assert file_hash(path) == snapshot["sha256"]
                with np.load(path, allow_pickle=False) as archive:
                    native_views.append((archive["boxes"].copy(), archive["decoded"][4:14].T.copy()))
                    original = archive["original_local"].copy()
                local_views.append(map_tile_predictions(original, entry["windows"][view]) if view else original)
            replay = calibrated_candidates(local_views, entry["windows"], entry["size"], [], calibrator)[0]
            np.testing.assert_array_equal(replay, strong)
            tick = perf_counter()
            evidence = competition_evidence(strong, native_views)
            arrays = {"strong": strong, **evidence}
            for arm in ARMS:
                output, rows = suppress_scores(strong, evidence["penalty_"+arm])
                arrays[arm], arrays["rows_"+arm] = output, rows
                changes[arm] += int(np.count_nonzero(evidence["penalty_"+arm] > 0))
                np.testing.assert_array_equal(output[:, :4], strong[rows, :4])
                np.testing.assert_array_equal(output[:, 5], strong[rows, 5])
            timings.append((perf_counter()-tick)*1000)
            pair_count += len(evidence["pairs"])
            active_pair_count += int(np.count_nonzero(evidence["contrast_shared"] > 0))
            # Outputs are complete before annotation access.
            metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(metadata_path) == entry["metadata_sha256"]
            metadata = read(metadata_path)
            assert metadata["original_size"] == entry["size"]
            records.append(dict(id=count, stem=stem, size=entry["size"], objects=metadata["valid_objects"]))
            for arm in maps:
                arrays["diagnostic_"+arm] = diagnostic_filter(arrays[arm], metadata)
                maps[arm][stem] = arrays["diagnostic_"+arm]
            path = root/"frames"/(stem+".npz")
            np.savez_compressed(path, **arrays)
            output_manifest[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                split=split, metadata_sha256=entry["metadata_sha256"],
                reference_sha256=entry["reference"]["sha256"], processing_ms=timings[-1])
            if count == 1 or count % 8 == 0 or count == len(stems):
                print(f"Class competition {split}: {count}/{len(stems)}", flush=True)
        ap = {a: coco_size_metrics(records, m, dict(enumerate(NAMES)), reference_size=640) for a, m in maps.items()}
        comparisons = {}
        for arm in ARMS:
            comparisons[arm] = recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[arm]),
                budgets=(5, 10), bootstrap_samples=0)[0]
        reports[split] = dict(AP=ap, fixed_fp=comparisons,
            small_all_fppi10_sequence_bootstrap=cluster_interval(records, maps, comparisons["shared"][1]),
            changed_rows=changes, competing_pairs=pair_count, shared_active_pairs=active_pair_count,
            mean_evidence_plus_four_arm_processing_ms=float(np.mean(timings)))
    decision = decide(reports, registration["policy_per_half"])
    for name, value in (("frames.json", output_manifest), ("metrics.json", reports), ("decision.json", decision)):
        save(root/name, value)
    assert all(file_hash(project_path(name)) == digest for name, digest in hashes.items())
    frozen_control()
    result = dict(experiment=registration["experiment"], decision=decision, metrics=reports,
        costs=dict(cached_images=49, cached_native_views=245, strong_output_replays=49,
            JPEG_decodes=0, detector_forwards=0, fits=0, gradients=0, parameter_updates=0,
            official_AP_evaluations=0, elapsed_seconds=perf_counter()-started),
        artifacts_sha256={p.name: file_hash(p) for p in root.iterdir() if p.is_file()})
    save(root/"result.json", result)
    print(json.dumps(result["decision"], indent=2), flush=True)
    return result
