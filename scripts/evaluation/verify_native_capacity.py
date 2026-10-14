"""Independently rebuild native graphs and prove optimum with cover certificates."""

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


ROOT = "output/analysis/native-capacity-fixed-20261015"
PRIOR = "output/analysis/texture-deletion-fixed-20261015"
GROUPS = dict(all=math.inf, tiny=256., small_all=1024.)
FAMILIES = ("geometry", "argmax005")


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            compare(a, b)
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


def independently_mapped_boxes(saved, windows):
    output = []
    for local, (left, top, right, bottom) in zip(saved, windows, strict=True):
        b = local.copy().astype(np.float32)
        b[:, 0] = np.clip(b[:, 0], 0, right-left)+np.float32(left)
        b[:, 2] = np.clip(b[:, 2], 0, right-left)+np.float32(left)
        b[:, 1] = np.clip(b[:, 1], 0, bottom-top)+np.float32(top)
        b[:, 3] = np.clip(b[:, 3], 0, bottom-top)+np.float32(top)
        output.append(b.astype(float))
    return np.concatenate(output)


def independently_constructed_edges(gt, classes, boxes, logits):
    valid = np.isfinite(logits).all(1) & np.isfinite(boxes).all(1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    classes_best = np.argmax(logits, axis=1)
    # The logit threshold is mathematically equivalent to sigmoid(logit)>=.05.
    high = logits[np.arange(len(logits)), classes_best].astype(float) >= math.log(.05/.95)
    areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    pairs = {f: [] for f in FAMILIES}
    for g, target in enumerate(gt):
        intersection = (np.maximum(0, np.minimum(boxes[:, 2], target[2])-np.maximum(boxes[:, 0], target[0])) *
                        np.maximum(0, np.minimum(boxes[:, 3], target[3])-np.maximum(boxes[:, 1], target[1])))
        area = max(target[2]-target[0], 0)*max(target[3]-target[1], 0)
        iou = intersection/np.maximum(area+areas-intersection, 1e-12)
        geometric = np.flatnonzero(valid & (iou >= .5))
        semantic = geometric[high[geometric] & (classes_best[geometric] == classes[g])]
        for f, columns in (("geometry", geometric), ("argmax005", semantic)):
            pairs[f].extend((g, int(column)) for column in columns)
    return {f: np.array(p, np.int32).reshape(-1, 2).T for f, p in pairs.items()}


def verify_certificate(n_left, edges, matched, cover_left, cover_right):
    rows, columns = edges
    assert matched.shape == (n_left,)
    assert (matched >= -1).all()
    assert np.unique(cover_left).size == len(cover_left)
    assert np.unique(cover_right).size == len(cover_right)
    assert ((cover_left >= 0) & (cover_left < n_left)).all()
    assert (cover_right >= 0).all()
    assigned = [(g, int(p)) for g, p in enumerate(matched) if p >= 0]
    assert len({p for _, p in assigned}) == len(assigned)
    edge_set = set(zip(rows.tolist(), columns.tolist()))
    assert all((g, p) in edge_set for g, p in assigned)
    left_set, right_set = set(cover_left.tolist()), set(cover_right.tolist())
    assert all(g in left_set or p in right_set for g, p in edge_set)
    # Feasible matching + covering set with equal size certifies the exact optimum.
    assert len(assigned) == len(left_set)+len(right_set)
    support = np.zeros(n_left, bool)
    support[rows] = True
    gap = int(support.sum())-len(assigned)
    assert gap >= 0
    return support, dict(gt=n_left, independently_supported_gt=int(support.sum()),
                        matching_capacity=len(assigned), capacity_deficit=gap, deficit_images=int(gap > 0))


def control_owners(gt, classes, reference, metadata):
    ignored = np.zeros(len(reference), bool)
    dt_areas = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)
    for box in metadata["ignore_boxes_xyxy"]:
        intersection = np.maximum(np.minimum(reference[:, 2:4], box[2:])-
                                  np.maximum(reference[:, :2], box[:2]), 0).prod(1)
        ignored |= intersection/np.maximum(dt_areas, 1e-12) > .5
    used = np.zeros(len(gt), bool)
    gt_areas = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)
    for row in sorted(range(len(reference)), key=lambda j: (-reference[j, 4], j)):
        if reference[row, 4] < .25 or ignored[row]:
            continue
        intersection = np.maximum(np.minimum(gt[:, 2:], reference[row, 2:4])-
                                  np.maximum(gt[:, :2], reference[row, :2]), 0).prod(1)
        iou = intersection/np.maximum(gt_areas+dt_areas[row]-intersection, 1e-12)
        eligible = np.flatnonzero(~used & (classes == reference[row, 5]) & (iou >= .5))
        if len(eligible):
            used[eligible[np.argmax(iou[eligible])]] = True
    return used


def verify():
    root, prior = project_path(ROOT), project_path(PRIOR)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result, plan, manifest = read(root/"result.json"), read(root/"plan.json"), read(root/"frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        assert file_hash(root/"source.zip") == plan["source_archive_sha256"]
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
    assert file_hash(Path(plan["installed_matching_binary"]["path"])) == plan["installed_matching_binary"]["sha256"]
    prior_result_path = project_path("docs/experiments/candidate56_texture_deletion_result_20261015.json")
    assert file_hash(prior_result_path) == plan["prior_result_sha256"]
    prior_result = read(prior_result_path)
    for name, digest in prior_result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_manifest_sha256"]
    old, old_manifest = read(prior/"plan.json"), read(prior/"frames.json")
    selection = {h: old["image_selection"][h] for h in ("screen", "confirmation")}
    assert selection == plan["image_selection"] and all(len(v) == 64 for v in selection.values())
    assert len({s[:7] for v in selection.values() for s in v}) == 128
    for name, digest in old["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for installed in old["installed_sources"].values():
        assert file_hash(Path(installed["path"])) == installed["sha256"]
    control = read(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json"))
    assert file_hash(Path(control["detector"]["path"])) == plan["checkpoint_sha256"] == old["checkpoint_sha256"]
    traces, certificates, edge_count = {h: [] for h in selection}, 0, 0
    for half, stems in selection.items():
        for stem in stems:
            entry, previous = manifest[stem], old_manifest[stem]
            assert entry["half"] == half and file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(prior/previous["file"]) == entry["prior_frame_sha256"] == previous["sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == entry["metadata_sha256"] == old["inputs"][stem]["metadata_sha256"]
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == old["inputs"][stem]["image_sha256"]
            metadata = read(meta_path)
            gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
            areas = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)*(640/max(metadata["original_size"]))**2
            with np.load(prior/previous["file"], allow_pickle=False) as z:
                boxes = independently_mapped_boxes(z["native_boxes"], previous["windows"])
                logits = z["native_logits"].reshape(-1, 10)
                reference = z["reference"].copy()
            edges = independently_constructed_edges(gt, classes, boxes, logits)
            edge_count += sum(e.shape[1] for e in edges.values())
            owners = control_owners(gt, classes, reference, metadata)
            trace = {}
            with np.load(root/entry["file"], allow_pickle=False) as z:
                for name, values in (("gt", gt), ("classes", classes), ("areas", areas), ("control_gt", owners)):
                    np.testing.assert_array_equal(values, z[name])
                for family, full in edges.items():
                    np.testing.assert_array_equal(full, z[family+"_edges"])
                    trace[family] = {}
                    for scope in ("union", "view0", "view1", "view2", "view3", "view4"):
                        trace[family][scope] = {}
                        relation = full if scope == "union" else full[:, full[1]//18900 == int(scope[-1])]
                        for group, ceiling in GROUPS.items():
                            included = np.flatnonzero(areas < ceiling)
                            lookup = {int(original): j for j, original in enumerate(included)}
                            induced = [(lookup[int(g)], int(p)) for g, p in relation.T if int(g) in lookup]
                            induced = np.array(induced, int).reshape(-1, 2).T
                            key = "_".join((family, scope, group))
                            support, statistics = verify_certificate(len(included), induced, z[key+"_matched"],
                                z[key+"_cover_left"], z[key+"_cover_right"])
                            np.testing.assert_array_equal(support, z[key+"_support"])
                            trace[family][scope][group] = statistics
                            certificates += 1
            trace["ordinary_calibration_matches"] = {g: int(owners[areas < limit].sum()) for g, limit in GROUPS.items()}
            gt_keys = [(*box.tolist(), int(c)) for box, c in zip(gt, classes)]
            trace["exact_duplicate_GT_annotations"] = len(gt_keys)-len(set(gt_keys))
            traces[half].append(trace)
    compare(traces, read(root/"per_image.json"))
    metrics, checks = {}, {}
    policy = plan["registration"]["policy_per_half"]
    for half, items in traces.items():
        totals = {f: {scope: {g: {k: sum(t[f][scope][g][k] for t in items) for k in items[0][f][scope][g]}
                              for g in GROUPS} for scope in items[0][f]} for f in FAMILIES}
        for family in FAMILIES:
            for groups in totals[family].values():
                for values in groups.values():
                    values["capacity_deficit_fraction_all_GT"] = values["capacity_deficit"]/values["gt"] if values["gt"] else 0.
        draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
        fractions = []
        for draw in draws:
            denominator = sum(int(w)*t["argmax005"]["union"]["tiny"]["gt"] for w, t in zip(draw, items))
            if not denominator:
                fractions = None
                break
            fractions.append(sum(int(w)*t["argmax005"]["union"]["tiny"]["capacity_deficit"] for w, t in zip(draw, items))/denominator)
        ci = np.percentile(fractions, [2.5, 97.5]).tolist() if fractions is not None else None
        totals["primary_tiny_union_deficit_fraction_bootstrap95ci"] = ci
        totals["ordinary_calibration_matches"] = {g: sum(t["ordinary_calibration_matches"][g] for t in items) for g in GROUPS}
        totals["exact_duplicate_GT_annotations"] = sum(t["exact_duplicate_GT_annotations"] for t in items)
        metrics[half] = totals
        tiny = totals["argmax005"]["union"]["tiny"]
        checks[half] = dict(gt=tiny["gt"] >= policy["tiny_gt_min"],
            supported=tiny["independently_supported_gt"] >= policy["tiny_union_independently_supported_min"],
            deficit=tiny["capacity_deficit"] >= policy["tiny_union_capacity_deficit_min"],
            images=tiny["deficit_images"] >= policy["tiny_union_capacity_deficit_images_min"],
            fraction=tiny["capacity_deficit_fraction_all_GT"] >= policy["tiny_union_capacity_deficit_fraction_min"],
            interval=ci is not None and ci[0] > policy["tiny_union_capacity_deficit_fraction_bootstrap_lower_min_exclusive"])
    compare(metrics, result["metrics"])
    assert checks == result["decision"]["checks"]
    passed = all(all(c.values()) for c in checks.values())
    expected = "REPLICATED_NATIVE_CAPACITY_DEFICIT_SUPPORTS_SEPARATE_INSTANCE_DIVERSIFICATION_FEASIBILITY" if passed else "STOP_THIS_NATIVE_SHARED_CAPACITY_ROUTE_ON_FIXED_INVENTORY"
    assert expected == result["decision"]["decision"]
    audit = dict(all_source_archive_matching_binary_prior_data_frames_image_metadata_hashes_verified=True,
        fixed128image_selection_and_disjoint_sequence_prefixes_verified=True,
        all_native_coordinates_probability_conditions_and_edges_independently_reconstructed=edge_count,
        all_feasible_matchings_and_equal_size_cover_certificates_independently_proved=certificates,
        all_GT_groups_control_matching_perimage_aggregates_bootstrap_and_twelve_gates_rebuilt=True,
        added_JPEG_decodes=0, added_detector_forwards=0, added_matching_solver_calls=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Capacity uses same saved original detector tensors, not new detector evidence. Certificates prove graph optimality but not AP, distinct physical hypotheses after NMS, learning potential or novelty. Tiny-only graph omits larger-GT competition. Cached float32 local box mapping and float64 sigmoid conventions differ from raw rounded FP16 probability evaluation. Prior scene diagnostics may contain other frames; detector/calibrator trained on these images. Historical individual-ignore diagnostic is not official VisDrone.")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
