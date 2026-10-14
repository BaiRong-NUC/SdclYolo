"""Audit candidate50 from saved tensors, labels and independent support geometry."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


OUTPUT = "output/analysis/assignment-reachability-fixed-20261014"
LIMITS = {"all": np.inf, "tiny": 256., "small_all": 1024.}


def bound(box, point):
    # Independent scalar area formula using the four outward extension lengths.
    box, point = [float(x) for x in box], [float(x) for x in point]
    w, h = float(box[2]-box[0]), float(box[3]-box[1])
    extra_x = max(float(box[0]-point[0]), 0)+max(float(point[0]-box[2]), 0)
    extra_y = max(float(box[1]-point[1]), 0)+max(float(point[1]-box[3]), 0)
    return w*h/((w+extra_x)*(h+extra_y))


def grid_bound(box):
    candidates = []
    for stride in (8, 16, 32):
        x = (float(box[0])+float(box[2]))/(2*stride)-.5
        y = (float(box[1])+float(box[3]))/(2*stride)-.5
        for ix in (math.floor(x), math.floor(x)+1):
            for iy in (math.floor(y), math.floor(y)+1):
                ix, iy = min(max(ix, 0), 960//stride-1), min(max(iy, 0), 960//stride-1)
                candidates.append(bound(box, [(ix+.5)*stride, (iy+.5)*stride]))
    return max(candidates)


def verify():
    torch.set_num_threads(1)
    output = project_path(OUTPUT)
    read = lambda name: json.loads((output/name).read_text())
    result, plan, manifest = read("result.json"), read("plan.json"), read("frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest, name
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
        assert file_hash(project_path(name)) == digest
        excluded.update(stem for stems in json.loads(project_path(name).read_text())["image_selection"].values() for stem in stems)
    images = sorted(project_path("data/visdrone/images/train").glob("*.jpg"),
                    key=lambda p: hashlib.sha256(("candidate50-support-v1:"+p.stem).encode()).hexdigest())
    selected = [p for p in images if p.stem not in excluded][:64]
    assert [p.stem for p in selected] == plan["images"] and len(excluded) == plan["excluded_images"]
    assigner = TaskAlignedAssigner(topk=plan["actual_assigner"]["topk"], num_classes=10, alpha=.5,
                                  beta=6., stride=[8, 16, 32], eps=1e-9,
                                  topk2=plan["actual_assigner"]["topk2"])
    assert all(getattr(assigner, k) == v for k, v in plan["actual_assigner"].items())
    expected_anchors, expected_stride = [], []
    for stride in (8, 16, 32):
        for y in range(960//stride):
            for x in range(960//stride):
                expected_anchors.append(((x+.5)*stride, (y+.5)*stride))
                expected_stride.append([stride])
    actual_metrics, actual_trace = read("metrics.json"), read("per_image.json")
    totals = {g: dict.fromkeys((k for k in m if k not in ("low_ceiling_mass_fraction", "weighted_unattainable_iou_gap")), 0)
              for g, m in actual_metrics.items()}
    max_label_error, pairs_checked = 0., 0
    for n, image in enumerate(selected):
        stem, item = image.stem, manifest[image.stem]
        assert file_hash(output/item["file"]) == item["sha256"]
        for suffix, key in ((f"images/train/{stem}.jpg", "image_sha256"),
                            (f"metadata/train/{stem}.json", "metadata_sha256"),
                            (f"labels/train/{stem}.txt", "label_sha256")):
            assert file_hash(project_path("data/visdrone/"+suffix)) == item[key]
        with np.load(output/item["file"], allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(f["anchors"], expected_anchors)
        np.testing.assert_array_equal(f["stride"], expected_stride)
        assert np.all(f["decoded_boxes"][:, :2] <= f["anchors"]+1e-4)
        assert np.all(f["decoded_boxes"][:, 2:] >= f["anchors"]-1e-4)
        metadata = json.loads(project_path(f"data/visdrone/metadata/train/{stem}.json").read_text())
        width, height = metadata["original_size"]
        labels = np.loadtxt(project_path(f"data/visdrone/labels/train/{stem}.txt"), dtype=np.float32).reshape(-1, 5)
        _, unique = np.unique(labels, axis=0, return_index=True)
        if len(unique) < len(labels):
            labels = labels[unique]
        r = 960/max(width, height)
        rw, rh = min(math.ceil(width*r), 960), min(math.ceil(height*r), 960)
        left, top = round((960-rw)/2-.1), round((960-rh)/2-.1)
        transformed = np.c_[labels[:, 1:3]-labels[:, 3:5]/2, labels[:, 1:3]+labels[:, 3:5]/2].astype(float)
        transformed *= [rw, rh, rw, rh]
        transformed += [left, top, left, top]
        max_label_error = max(max_label_error, float(np.max(np.abs(transformed-f["gt"]))))
        np.testing.assert_allclose(f["gt"], transformed, atol=2e-4, rtol=1e-6)
        np.testing.assert_array_equal(f["classes"], labels[:, 0])
        with torch.no_grad():
            _, boxes, scores, foreground, owner = assigner(
                torch.from_numpy(f["probabilities"])[None], torch.from_numpy(f["decoded_boxes"])[None],
                torch.from_numpy(f["anchors"]), torch.from_numpy(f["classes"])[None, :, None],
                torch.from_numpy(f["gt"])[None], torch.ones((1, len(f["gt"]), 1), dtype=torch.bool))
        np.testing.assert_array_equal(f["owner"], owner[0].numpy())
        np.testing.assert_array_equal(f["foreground"], foreground[0].numpy())
        np.testing.assert_allclose(f["target_mass"], scores[0].sum(-1).numpy(), atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(f["assigned_boxes"], boxes[0].numpy())
        ceiling = np.array([bound(f["gt"][g], p) for g, p in zip(f["owner"], f["anchors"])])
        np.testing.assert_allclose(ceiling, f["ceiling"], atol=1e-12)
        grid_max = np.array([grid_bound(box) for box in f["gt"]])
        np.testing.assert_allclose(grid_max, f["grid_max"], atol=1e-12)
        positive = f["foreground"] & (f["target_mass"] > 0)
        areas = np.array([(float(b[2])-float(b[0]))*(float(b[3])-float(b[1]))*(640/960)**2 for b in f["gt"]])
        assigned = [[] for _ in f["gt"]]
        for row in np.flatnonzero(positive):
            assigned[f["owner"][row]].append(row)
        maximum = np.array([max(ceiling[rows]) if rows else 0. for rows in assigned])
        np.testing.assert_array_equal(maximum, f["assigned_max"])
        for group, limit in LIMITS.items():
            ids = [i for i, a in enumerate(areas) if a < limit]
            rows = [row for g in ids for row in assigned[g]]
            lows = [row for row in rows if ceiling[row] < .75]
            affected = sorted({int(f["owner"][row]) for row in lows})
            inside = lambda row: bool(np.all(f["anchors"][row] >= f["gt"][f["owner"][row], :2]) and
                                      np.all(f["anchors"][row] <= f["gt"][f["owner"][row], 2:]))
            covered_inside = {g for g in affected if any(inside(row) for row in assigned[g])}
            values = dict(gt=len(ids), covered_gt=sum(bool(assigned[g]) for g in ids),
                          productive_positive_pairs=len(rows), target_mass=sum(float(f["target_mass"][row]) for row in rows),
                          outside_pairs=sum(not inside(row) for row in rows), low_ceiling_pairs=len(lows),
                          low_ceiling_mass=sum(float(f["target_mass"][row]) for row in lows),
                          unattainable_iou_gap_mass=sum(float(f["target_mass"][row])*(1-ceiling[row]) for row in rows),
                          low_ceiling_affected_gt=len(affected), low_ceiling_affected_images=int(bool(affected)),
                          affected_with_inside_assigned_alternative=len(covered_inside),
                          affected_with_inside_grid_but_no_inside_assigned=sum(grid_max[g] >= 1-1e-12 and g not in covered_inside for g in affected),
                          affected_without_any_inside_grid_point=sum(grid_max[g] < 1-1e-12 for g in affected),
                          covered_gt_all_assigned_below075=sum(0 < maximum[g] < .75 for g in ids),
                          gt_all_grid_below075=sum(grid_max[g] < .75 for g in ids))
            for k, value in values.items():
                np.testing.assert_allclose(value, actual_trace[n][group][k], atol=2e-5, rtol=2e-7)
                totals[group][k] += value
        pairs_checked += len(f["owner"])
    for g, values in totals.items():
        values["low_ceiling_mass_fraction"] = values["low_ceiling_mass"]/values["target_mass"]
        values["weighted_unattainable_iou_gap"] = values["unattainable_iou_gap_mass"]/values["target_mass"]
        for k, value in values.items():
            np.testing.assert_allclose(value, actual_metrics[g][k], atol=3e-4, rtol=2e-7)
    tiny, policy = totals["tiny"], plan["registration"]["policy"]
    checks = {
        "affected_gt": tiny["low_ceiling_affected_gt"] >= policy["tiny_low_ceiling_affected_gt_min"],
        "affected_images": tiny["low_ceiling_affected_images"] >= policy["tiny_low_ceiling_affected_images_min"],
        "mass_fraction": tiny["low_ceiling_mass_fraction"] >= policy["tiny_low_ceiling_mass_fraction_min"],
        "weighted_gap": tiny["weighted_unattainable_iou_gap"] >= policy["tiny_weighted_unattainable_iou_gap_min"],
    }
    assert checks == result["decision"]["checks"] and not all(checks.values())
    assert result["decision"]["decision"] == "STOP_THIS_NONNEGATIVE_SUPPORT_ROUTE_ON_FIXED_PROXY"
    verified = {
        "all_sources_package_snapshots_checkpoint_selection_and64_image_metadata_label_frame_hashes_verified": True,
        "all_GT_transforms_rebuilt_from_labels_and_metadata_without_image_decode": True,
        "maximum_independent_label_transform_absolute_error_pixels": max_label_error,
        "all64_TAL_owners_foreground_target_mass_and_boxes_replayed_from_saved_native_outputs": True,
        "nonnegative_output_containment_and_scalar_enclosure_bounds_checked_for_pairs": pairs_checked,
        "all_grid_maxima_verified_with_independent12_nearest_center_candidates": True,
        "all_per_image_group_metrics_and_four_gates_independently_recomputed": True,
        "verifier_sha256": file_hash(Path(__file__)),
        "limits": "No new detector forwards. CPU TAL uses the same pinned implementation and numerical tolerances; historical augmented assignments, harmful gradients, AP and novelty remain untested.",
    }
    (output/"independent_verification.json").write_text(json.dumps(verified, indent=2)+"\n", encoding="utf-8")
    return verified


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
