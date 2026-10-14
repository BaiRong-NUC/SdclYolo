"""Independent per-GT vector IoU/pool reconstruction for candidate60."""

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


def verify():
    root = project_path("output/analysis/native-peak-quality-fixed-20261015")
    prior = project_path("output/analysis/texture-deletion-fixed-20261015")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result, plan = read(root/"result.json"), read(root/"plan.json")
    assert not (root/"independent_verification.json").exists()
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
    prior_result_path = project_path("docs/experiments/candidate56_texture_deletion_result_20261015.json")
    assert file_hash(prior_result_path) == plan["prior_result_sha256"]
    for name, digest in read(prior_result_path)["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_manifest_sha256"]
    old_plan, old_manifest = read(prior/"plan.json"), read(prior/"frames.json")
    for name, digest in old_plan["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for value in old_plan["installed_sources"].values():
        assert file_hash(Path(value["path"])) == value["sha256"]
    for split, stems in plan["image_selection"].items():
        assert stems == old_plan["image_selection"][split] and len(stems) == 64
    assert len({s[:7] for stems in plan["image_selection"].values() for s in stems}) == 128
    manifest, traces, gt_view_pools, positions = read(root/"frames.json"), {}, 0, 0
    groups = {"all": (0., math.inf), "tiny": (0., 256.),
              "small_all": (0., 1024.), "medium_plus": (256., math.inf)}
    fields = ("count", "best_iou", "peak_iou", "peak_min_iou", "peak_count",
              "max_logit", "uniform_mean_iou", "uniform_high_iou_fraction")
    for split, stems in plan["image_selection"].items():
        traces[split] = []
        for stem in stems:
            entry, old_entry = manifest[stem], old_manifest[stem]
            assert entry["split"] == split
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(prior/old_entry["file"]) == entry["prior_frame_sha256"] == old_entry["sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == entry["metadata_sha256"] == old_plan["inputs"][stem]["metadata_sha256"]
            metadata = read(meta_path)
            gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
            cls = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
            areas = ((gt[:, 2]-gt[:, 0])*(gt[:, 3]-gt[:, 1])*
                     (640/max(metadata["original_size"]))**2)
            rebuilt = {name: np.zeros((len(gt), 5)) for name in fields}
            with np.load(prior/old_entry["file"], allow_pickle=False) as z:
                local, logits = z["native_boxes"].copy(), z["native_logits"].copy()
            for view, window in enumerate(old_entry["windows"]):
                x1, y1, x2, y2 = window
                mapped = np.empty_like(local[view], dtype=np.float32)
                for column, offset, limit in ((0, x1, x2-x1), (1, y1, y2-y1),
                                               (2, x1, x2-x1), (3, y1, y2-y1)):
                    mapped[:, column] = np.clip(local[view, :, column], 0, limit)+np.float32(offset)
                boxes = mapped.astype(float)
                width, height = boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]
                valid = (np.isfinite(boxes).all(1) & np.isfinite(logits[view]).all(1) &
                         (width > 0) & (height > 0))
                classes = np.argmax(logits[view], axis=1)
                maximum = logits[view, np.arange(len(boxes)), classes].astype(float)
                eligible = valid & (maximum >= math.log(.05/.95))
                positions += len(boxes)
                for j, box in enumerate(gt):
                    intersection = (np.maximum(np.minimum(boxes[:, 2], box[2])-np.maximum(boxes[:, 0], box[0]), 0)*
                                    np.maximum(np.minimum(boxes[:, 3], box[3])-np.maximum(boxes[:, 1], box[1]), 0))
                    union = (box[2]-box[0])*(box[3]-box[1])+np.maximum(width, 0)*np.maximum(height, 0)-intersection
                    iou = intersection/np.maximum(union, 1e-12)
                    pool = eligible & (classes == cls[j]) & (iou >= .5)
                    quality = iou[pool]
                    gt_view_pools += 1
                    if not quality.size:
                        continue
                    scores = logits[view, pool, cls[j]]
                    max_score = max(scores)
                    ties = quality[scores == max_score]
                    row = (len(quality), max(quality), max(ties), min(ties), len(ties),
                           max_score, sum(quality)/len(quality), sum(quality >= .75)/len(quality))
                    for name, value in zip(fields, row, strict=True):
                        rebuilt[name][j, view] = value
            with np.load(root/entry["file"], allow_pickle=False) as z:
                np.testing.assert_array_equal(gt, z["gt"])
                np.testing.assert_array_equal(cls, z["classes"])
                np.testing.assert_allclose(areas, z["areas"], rtol=1e-14, atol=1e-12)
                for name in fields:
                    np.testing.assert_allclose(rebuilt[name], z[name], rtol=1e-12, atol=1e-12)
            supported = np.any(rebuilt["best_iou"] >= .75, axis=1)
            peak = np.any(rebuilt["peak_iou"] >= .75, axis=1)
            loss = supported & ~peak
            summary = {}
            for group, (lo, hi) in groups.items():
                keep = (areas >= lo) & (areas < hi)
                summary[group] = dict(gt=int(keep.sum()), high_quality_supported=int(sum(keep & supported)),
                    peak_high_quality_selected=int(sum(keep & peak)), peak_loss=int(sum(keep & loss)),
                    peak_loss_images=int(any(keep & loss)))
            traces[split].append(summary)
    assert traces == read(root/"per_image.json")
    metrics, checks = {}, {}
    policy = plan["registration"]["policy_per_half"]
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    for split, rows in traces.items():
        metrics[split] = {g: {k: sum(row[g][k] for row in rows) for k in rows[0][g]} for g in groups}
        for m in metrics[split].values():
            m["peak_loss_fraction_all_GT"] = m["peak_loss"]/m["gt"] if m["gt"] else 0.
        ratios = []
        for draw in draws:
            count = sum(int(w)*row["tiny"]["gt"] for w, row in zip(draw, rows, strict=True))
            assert count > 0
            ratios.append(sum(int(w)*row["tiny"]["peak_loss"] for w, row in zip(draw, rows, strict=True))/count)
        ci = np.percentile(ratios, [2.5, 97.5]).tolist()
        metrics[split]["tiny_peak_loss_fraction_bootstrap95ci"] = ci
        m = metrics[split]["tiny"]
        checks[split] = dict(gt=m["gt"] >= policy["tiny_gt_min"],
            supported=m["high_quality_supported"] >= policy["tiny_high_quality_supported_min"],
            loss=m["peak_loss"] >= policy["tiny_peak_loss_min"],
            images=m["peak_loss_images"] >= policy["tiny_peak_loss_images_min"],
            fraction=m["peak_loss_fraction_all_GT"] >= policy["tiny_peak_loss_fraction_all_GT_min"],
            interval=ci[0] > policy["tiny_peak_loss_fraction_bootstrap_lower_min_exclusive"])
    assert metrics == result["metrics"] and checks == result["decision"]["checks"]
    expected = ("REPLICATED_ORACLE_NATIVE_PEAK_QUALITY_GAP_SUPPORTS_DISTINCT_LOW_COST_FEASIBILITY"
                if all(all(c.values()) for c in checks.values()) else
                "STOP_THIS_NATIVE_SCORE_PEAK_HIGH_IOU_ROUTE_ON_FIXED_INVENTORY")
    assert expected == result["decision"]["decision"]
    audit = dict(all_source_archive_prior_frame_metadata_hashes_verified=True,
        fixed128images_and_disjoint_sequence_prefixes_verified=True,
        native_positions_remapped_independently=positions,
        per_GT_view_pools_independently_recomputed=gt_view_pools,
        all_eight_statistics_and_GT_area_groups_reconstructed=True,
        all_image_traces_aggregates_bootstrap_and_twelve_gates_rebuilt=True,
        added_JPEG_decodes=0, added_detector_forwards=0, added_fits=0, added_updates=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Reuses same saved FP16logits/FP32boxes, not an independent detector run. GT-conditioned pools,oracleviewselection,optimistic ties,reusedpositions,noNMS/FPbudget/ignoreconstraints. NoAP,deployability,novelty ortraining authorization. Original registration's GFLV2identifier is corrected only in separate literature audit.")
    (root/"independent_verification.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
