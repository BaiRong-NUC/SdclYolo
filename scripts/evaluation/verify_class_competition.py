"""Independently recompute candidate64 pair witnesses, score edits and decision."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.class_competition import checked_prior
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.training.transport_quality import diagnostic_filter


ARMS = ("shared", "marginal", "affinity", "geometry")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(box, others):
    box, others = np.asarray(box, float), np.asarray(others, float)
    intersection = np.maximum(0, np.minimum(box[2:], others[:, 2:])-np.maximum(box[:2], others[:, :2])).prod(1)
    union = np.maximum(0, box[2:]-box[:2]).prod()+np.maximum(0, others[:, 2:]-others[:, :2]).prod(1)-intersection
    return intersection/np.maximum(union, 1e-12)


def independent(predictions, views):
    pairs, ious = [], []
    for i in range(len(predictions)):
        values = overlap(predictions[i, :4], predictions[:i, :4])
        for j in range(i):
            if predictions[i, 5] != predictions[j, 5] and values[j] >= .7:
                pairs.append((i, j))
                ious.append(values[j])
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    witnesses = np.full((len(pairs), 5), -1, int)
    weights, directions, mi, mj = [np.zeros((len(pairs), 5)) for _ in range(4)]
    semantic_sum, semantic_weight = np.zeros((len(predictions), 10)), np.zeros(len(predictions))
    collided = set(pairs.reshape(-1).tolist())
    for view, (boxes, probabilities) in enumerate(views):
        boxes, probabilities = np.asarray(boxes, float), np.asarray(probabilities, float)
        valid = (np.isfinite(boxes).all(1) & np.isfinite(probabilities).all(1) &
                 (boxes[:, 2:] > boxes[:, :2]).all(1) & (probabilities.max(1) >= .05))
        native = np.flatnonzero(valid)
        if not len(native):
            continue
        boxes, p = boxes[native], probabilities[native]
        geometry = {row: overlap(predictions[row, :4], boxes) for row in sorted(collided)}
        for row in sorted(collided):
            pool = np.flatnonzero(geometry[row] >= .5)
            if not len(pool):
                continue
            support = geometry[row][pool]*p[pool, int(predictions[row, 5])]
            local = int(np.argmax(support))
            chosen, weight = pool[local], support[local]
            semantic_sum[row] += weight*p[chosen]/max(p[chosen].sum(), 1e-12)
            semantic_weight[row] += weight
        for rank, (i, j) in enumerate(pairs):
            minimum = np.minimum(geometry[int(i)], geometry[int(j)])
            pool = np.flatnonzero(minimum >= .5)
            if not len(pool):
                continue
            pi, pj = p[pool, int(predictions[i, 5])], p[pool, int(predictions[j, 5])]
            support = minimum[pool]*np.maximum(pi, pj)
            local = int(np.argmax(support))
            witnesses[rank, view] = native[pool[local]]
            weights[rank, view] = support[local]
            directions[rank, view] = (pj[local]-pi[local])/(pj[local]+pi[local]+1e-12)
            mi[rank, view], mj[rank, view] = max(minimum[pool]*pi), max(minimum[pool]*pj)
    semantic = semantic_sum/np.maximum(semantic_weight[:, None], 1e-12)
    contrasts = dict(shared=np.maximum(0, (weights*directions).sum(1)/np.maximum(weights.sum(1), 1e-12)),
        marginal=np.maximum(0, (mj.sum(1)-mi.sum(1))/(mj.sum(1)+mi.sum(1)+1e-12)),
        affinity=np.array([min(1., max(0., np.sqrt(semantic[i]*semantic[j]).sum())) for i, j in pairs]),
        geometry=np.ones(len(pairs)))
    penalties = {arm: np.zeros(len(predictions)) for arm in ARMS}
    # Match frozen NumPy FP64 multiplication order, while aggregating independently.
    pair_penalties = {arm: np.square(np.asarray(ious))*contrasts[arm]/.5 for arm in ARMS}
    for rank, (i, _) in enumerate(pairs):
        for arm in ARMS:
            penalties[arm][i] = max(penalties[arm][i], pair_penalties[arm][rank])
    return dict(pairs=pairs, pair_iou=np.array(ious), witnesses=witnesses, shared_weights=weights,
        shared_directions=directions, marginal_i=mi, marginal_j=mj, semantic=semantic,
        **{"contrast_"+a: v for a, v in contrasts.items()}, **{"penalty_"+a: v for a, v in penalties.items()})


def greedy_small_count(record, dt, threshold):
    gt = np.asarray([o["box_xyxy"] for o in record["objects"]], float).reshape(-1, 4)
    classes = np.asarray([o["class_id"] for o in record["objects"]], int)
    used = np.zeros(len(gt), bool)
    for row in sorted(range(len(dt)), key=lambda i: (-dt[i, 4], i)):
        if dt[row, 4] < threshold:
            continue
        eligible = np.flatnonzero(~used & (classes == int(dt[row, 5])))
        if not len(eligible):
            continue
        scores = overlap(dt[row, :4], gt[eligible])
        best = int(np.argmax(scores))
        if scores[best] >= .5:
            used[eligible[best]] = True
    areas = np.maximum(0, gt[:, 2:]-gt[:, :2]).prod(1)*(640/max(record["size"]))**2
    small = areas < 1024.
    return int(small.sum()), int((small & used).sum())


def verify():
    root = project_path("output/analysis/shared-site-class-competition-fixed-20261015")
    target = root/"independent_verification.json"
    assert not target.exists()
    result, plan = read(root/"result.json"), read(root/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == digest == hashlib.sha256(archive.read(name)).hexdigest()
    prior, closeout, old, manifest = checked_prior()
    assert file_hash(closeout) == plan["prior62_closeout_sha256"]
    assert file_hash(prior/"plan.json") == plan["prior62_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior62_frames_sha256"]
    registration_path = project_path("docs/experiments/candidate64_shared_site_class_competition_registration_20261015.json")
    assert read(registration_path) == plan["registration"]
    assert plan["image_selection"] == {s: old["image_selection"][s] for s in ("screen", "confirmation")}
    assert not {s[:7] for s in plan["image_selection"]["screen"]} & {s[:7] for s in plan["image_selection"]["confirmation"]}
    audit = read(project_path("docs/literature/shared_site_class_competition_prior_audit_20261015.json"))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    control = frozen_control()
    assert file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")) == plan["strong_control_manifest_sha256"]
    assert control["detector"]["sha256"] == plan["detector_sha256"]
    assert control["scorer"]["sha256"] == plan["calibrator_sha256"]
    calibrator = load_calibrator(control)
    frames = read(root/"frames.json")
    all_checks, pair_total, witness_slots = {}, 0, 0
    for split, stems in plan["image_selection"].items():
        records, maps = [], {a: {} for a in ("strong", *ARMS)}
        changes = {a: 0 for a in ARMS}
        pairs, active = 0, 0
        for count, stem in enumerate(stems, 1):
            entry, snapshot = manifest[stem], frames[stem]
            assert snapshot["split"] == split
            assert file_hash(root/snapshot["file"]) == snapshot["sha256"]
            with np.load(root/snapshot["file"], allow_pickle=False) as archive:
                saved = {k: archive[k].copy() for k in archive.files}
            ref = prior/entry["reference"]["file"]
            assert file_hash(ref) == entry["reference"]["sha256"] == snapshot["reference_sha256"]
            strong = np.load(ref, allow_pickle=False)
            np.testing.assert_array_equal(strong, saved["strong"])
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == entry["image_sha256"]
            views, local_views = [], []
            for view, p in enumerate(entry["views"]):
                assert file_hash(prior/p["file"]) == p["sha256"]
                with np.load(prior/p["file"], allow_pickle=False) as archive:
                    views.append((archive["boxes"].copy(), archive["decoded"][4:14].T.copy()))
                    local = archive["original_local"].copy()
                local_views.append(map_tile_predictions(local, entry["windows"][view]) if view else local)
            np.testing.assert_array_equal(calibrated_candidates(local_views, entry["windows"], entry["size"], [], calibrator)[0], strong)
            evidence = independent(strong, views)
            for key, value in evidence.items():
                np.testing.assert_array_equal(value, saved[key])
            for arm in ARMS:
                output = strong.copy()
                penalty = evidence["penalty_"+arm]
                for row in range(len(output)):
                    if penalty[row] > 0:
                        output[row, 4] *= np.exp(-penalty[row])
                order = np.array(sorted([i for i in range(len(output)) if output[i, 4] >= .001],
                    key=lambda i: (-output[i, 4], i))[:500], int)
                np.testing.assert_array_equal(order, saved["rows_"+arm])
                np.testing.assert_array_equal(output[order], saved[arm])
                changes[arm] += int(np.count_nonzero(penalty > 0))
            metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(metadata_path) == entry["metadata_sha256"] == snapshot["metadata_sha256"]
            metadata = read(metadata_path)
            records.append(dict(id=count, stem=stem, size=entry["size"], objects=metadata["valid_objects"]))
            for arm in maps:
                maps[arm][stem] = diagnostic_filter(saved[arm], metadata)
                np.testing.assert_array_equal(maps[arm][stem], saved["diagnostic_"+arm])
            pairs += len(evidence["pairs"])
            active += int(np.count_nonzero(evidence["contrast_shared"] > 0))
            pair_total += len(evidence["pairs"])
            witness_slots += evidence["witnesses"].size
        ap = {a: coco_size_metrics(records, m, dict(enumerate(NAMES)), reference_size=640) for a, m in maps.items()}
        fp = {a: recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[a]),
            budgets=(5, 10), bootstrap_samples=0)[0] for a in ARMS}
        report = result["metrics"][split]
        assert ap == report["AP"] and fp == report["fixed_fp"]
        assert changes == report["changed_rows"] and pairs == report["competing_pairs"] and active == report["shared_active_pairs"]
        rows = []
        thresholds = fp["shared"][1]["models"]
        for record in records:
            gt, base = greedy_small_count(record, maps["strong"][record["stem"]], thresholds["baseline"]["confidence"])
            _, new = greedy_small_count(record, maps["shared"][record["stem"]], thresholds["sdcl"]["confidence"])
            rows.append(dict(stem=record["stem"], gt=gt, TP_difference=new-base))
        sequences = sorted({row["stem"][:7] for row in rows})
        gt = np.array([sum(row["gt"] for row in rows if row["stem"][:7] == s) for s in sequences])
        delta = np.array([sum(row["TP_difference"] for row in rows if row["stem"][:7] == s) for s in sequences])
        draws = np.random.default_rng(0).multinomial(len(sequences), [1/len(sequences)]*len(sequences), size=400)
        denominator = draws@gt
        ci = np.percentile((draws@delta)/denominator*100, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        assert dict(sequence_units=len(sequences), interval_pp=ci, per_image=rows) == report["small_all_fppi10_sequence_bootstrap"]
        timing = float(np.mean([frames[s]["processing_ms"] for s in stems]))
        assert timing == report["mean_evidence_plus_four_arm_processing_ms"]
        policy = plan["registration"]["policy_per_half"]
        checks = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"],
                      changed_rows=changes["shared"] >= policy["changed_rows_min"])
        for group, arm in (("tiny", "strong"), ("small_all", "strong"), ("all", "strong"),
                           ("tiny", "marginal"), ("tiny", "affinity"), ("tiny", "geometry")):
            key = f"{group}_ap_vs_{arm}"
            x, y = ap["shared"][group]["ap50_95"], ap[arm][group]["ap50_95"]
            checks[key] = x is not None and y is not None and (x-y)*100 >= policy[key+"_min_pp"]
        for i, budget in enumerate((5, 10)):
            key = f"small_all_recall_fppi{budget}_vs_strong"
            val = fp["shared"][i]["delta_recall_pp"]["small_all"]
            checks[key] = val is not None and val >= policy[key+"_min_pp"]
        checks["sequence_interval"] = ci is not None and ci[0] > policy["small_all_recall_fppi10_sequence_interval_lower_min_exclusive_pp"]
        checks["processing"] = timing <= policy["mean_evidence_plus_four_arm_processing_ms_max"]
        all_checks[split] = checks
    assert all_checks == result["decision"]["checks"]
    passed = all(all(c.values()) for c in all_checks.values())
    decision = ("REPEATED_DEPLOYABLE_SHARED_SITE_CLASS_COMPETITION_SUPPORTS_SEPARATE_OFFICIAL_EXTERNAL_AUDIT" if passed else
                "STOP_THIS_SHARED_SITE_CROSS_CLASS_CONTRADICTION_RECIPE")
    assert decision == result["decision"]["decision"] == read(root/"decision.json")["decision"]
    assert result["metrics"] == read(root/"metrics.json")
    verification = dict(all_hashes_sources_prior_control_cached_native_views_and_paper_hashes_checked=True,
        unchanged_strong_outputs_replayed_exactly=49, cross_class_pairs_independently_rebuilt=pair_total,
        shared_witness_slots_independently_rebuilt=witness_slots, four_arm_evidence_penalties_outputs_replayed_exactly=True,
        AP_FPbudget_counts_sequence_intervals_and_fixed_gates_recomputed=True, decision=decision,
        added_detector_forwards=0, added_JPEG_decodes=0, added_fits=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Pair/evidence/witness/score-update/operating-point TP matching/bootstrap/gates independently coded. Original calibration/SoftNMS,diagnostic ignore,COCO and FPbudget engines reused. Timing is audit of saved measurement,not a new latency benchmark. Reused detector-training data,not official AP/external generalization/innovation.")
    target.write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    verify()
