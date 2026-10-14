"""Audit simultaneous native detection support with certified maximum matchings."""

from collections import deque
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import expit
import torch

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash, match_at_threshold
from sdcl.postprocess.localization import frozen_control
from sdcl.postprocess.rescue import save_json


REGISTRATION = "docs/experiments/candidate57_native_capacity_registration_20261015.json"
OUTPUT = "output/analysis/native-capacity-fixed-20261015"
PRIOR = "output/analysis/texture-deletion-fixed-20261015"
GROUPS = dict(all=float("inf"), tiny=256., small_all=1024.)
FAMILIES = ("geometry", "argmax005")


def map_boxes(boxes, windows):
    result = np.asarray(boxes, np.float32).copy()
    for view, (x1, y1, x2, y2) in enumerate(windows):
        result[view, :, [0, 2]] = result[view, :, [0, 2]].clip(0, x2-x1)+x1
        result[view, :, [1, 3]] = result[view, :, [1, 3]].clip(0, y2-y1)+y1
    return result.reshape(-1, 4)


def graph_edges(gt, classes, boxes, logits):
    valid = np.isfinite(boxes).all(1) & np.isfinite(logits).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    best = logits.argmax(1)
    high = expit(logits[np.arange(len(logits)), best].astype(float)) >= .05
    pairs = {family: [[], []] for family in FAMILIES}
    for start in range(0, len(gt), 32):
        overlap = box_iou_matrix(gt[start:start+32], boxes)
        geometric = (overlap >= .5) & valid[None]
        semantic = geometric & high[None] & (classes[start:start+32, None] == best[None])
        for family, edges in (("geometry", geometric), ("argmax005", semantic)):
            rows, columns = np.nonzero(edges)
            pairs[family][0].append(rows+start)
            pairs[family][1].append(columns)
    return {f: np.vstack([np.concatenate(v) for v in p]).astype(np.int32) if p[0] else np.empty((2, 0), np.int32)
            for f, p in pairs.items()}


def certified_capacity(n_left, edges):
    rows, original_columns = np.asarray(edges, int)
    active_columns, columns = np.unique(original_columns, return_inverse=True)
    graph = csr_matrix((np.ones(len(rows), np.int8), (rows, columns)), shape=(n_left, len(active_columns)))
    if len(active_columns):
        matching = maximum_bipartite_matching(graph, perm_type="column")
    else:
        matching = np.full(n_left, -1, int)
    assert len(matching) == n_left
    right_owner = np.full(len(active_columns), -1, int)
    left_matched = np.flatnonzero(matching >= 0)
    right_owner[matching[left_matched]] = left_matched
    reached_left, reached_right = matching < 0, np.zeros(len(active_columns), bool)
    queue = deque(np.flatnonzero(reached_left))
    while queue:
        left = queue.popleft()
        for right in graph.indices[graph.indptr[left]:graph.indptr[left+1]]:
            if matching[left] == right or reached_right[right]:
                continue
            reached_right[right] = True
            mate = right_owner[right]
            if mate >= 0 and not reached_left[mate]:
                reached_left[mate] = True
                queue.append(mate)
    cover_left, cover_right = np.flatnonzero(~reached_left), active_columns[reached_right]
    matched_positions = np.full(n_left, -1, int)
    matched_positions[left_matched] = active_columns[matching[left_matched]]
    assert len(cover_left)+len(cover_right) == len(left_matched)
    assert (np.isin(rows, cover_left) | np.isin(original_columns, cover_right)).all()
    support = np.diff(graph.indptr) > 0
    cardinality = int(len(left_matched))
    deficit = int(support.sum())-cardinality
    return dict(matched=matched_positions, cover_left=cover_left, cover_right=cover_right,
                support=support, statistics=dict(gt=n_left, independently_supported_gt=int(support.sum()),
                matching_capacity=cardinality, capacity_deficit=deficit, deficit_images=int(deficit > 0)))


def frame_statistics(gt, classes, areas, edges, control_tp):
    arrays, stats = {}, {}
    for family, full in edges.items():
        arrays[family+"_edges"] = full
        stats[family] = {}
        for scope in ["union", *["view"+str(i) for i in range(5)]]:
            relation = full if scope == "union" else full[:, full[1]//18900 == int(scope[-1])]
            stats[family][scope] = {}
            for group, ceiling in GROUPS.items():
                selected = np.flatnonzero(areas < ceiling)
                reindex = np.full(len(gt), -1, int)
                reindex[selected] = np.arange(len(selected))
                relevant = reindex[relation[0]] >= 0
                induced = np.vstack((reindex[relation[0, relevant]], relation[1, relevant]))
                result = certified_capacity(len(selected), induced)
                key = "_".join((family, scope, group))
                for name in ("matched", "cover_left", "cover_right", "support"):
                    arrays[key+"_"+name] = result[name]
                stats[family][scope][group] = result["statistics"]
    stats["ordinary_calibration_matches"] = {g: int(control_tp[areas < ceiling].sum()) for g, ceiling in GROUPS.items()}
    duplicate_keys = [tuple([*box.tolist(), int(cls)]) for box, cls in zip(gt, classes)]
    stats["exact_duplicate_GT_annotations"] = len(duplicate_keys)-len(set(duplicate_keys))
    return arrays, stats


def aggregate(traces):
    result = {}
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    for half, items in traces.items():
        total = {family: {scope: {g: {key: sum(t[family][scope][g][key] for t in items)
                                     for key in items[0][family][scope][g]} for g in GROUPS}
                          for scope in items[0][family]} for family in FAMILIES}
        for family in FAMILIES:
            for scope in total[family]:
                for g, m in total[family][scope].items():
                    m["capacity_deficit_fraction_all_GT"] = m["capacity_deficit"]/m["gt"] if m["gt"] else 0.
        counts = np.array([t["argmax005"]["union"]["tiny"]["gt"] for t in items])
        deficits = np.array([t["argmax005"]["union"]["tiny"]["capacity_deficit"] for t in items])
        denominator = draws@counts
        total["primary_tiny_union_deficit_fraction_bootstrap95ci"] = np.percentile((draws@deficits)/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        total["ordinary_calibration_matches"] = {g: sum(t["ordinary_calibration_matches"][g] for t in items) for g in GROUPS}
        total["exact_duplicate_GT_annotations"] = sum(t["exact_duplicate_GT_annotations"] for t in items)
        result[half] = total
    return result


def decide(metrics, policy):
    checks = {}
    for half, values in metrics.items():
        tiny = values["argmax005"]["union"]["tiny"]
        ci = values["primary_tiny_union_deficit_fraction_bootstrap95ci"]
        checks[half] = dict(gt=tiny["gt"] >= policy["tiny_gt_min"],
            supported=tiny["independently_supported_gt"] >= policy["tiny_union_independently_supported_min"],
            deficit=tiny["capacity_deficit"] >= policy["tiny_union_capacity_deficit_min"],
            images=tiny["deficit_images"] >= policy["tiny_union_capacity_deficit_images_min"],
            fraction=tiny["capacity_deficit_fraction_all_GT"] >= policy["tiny_union_capacity_deficit_fraction_min"],
            interval=ci is not None and ci[0] > policy["tiny_union_capacity_deficit_fraction_bootstrap_lower_min_exclusive"])
    return dict(checks=checks,
        decision="REPLICATED_NATIVE_CAPACITY_DEFICIT_SUPPORTS_SEPARATE_INSTANCE_DIVERSIFICATION_FEASIBILITY" if all(all(c.values()) for c in checks.values()) else "STOP_THIS_NATIVE_SHARED_CAPACITY_ROUTE_ON_FIXED_INVENTORY",
        scope="Annotation-assisted optimistic native capacity only; no deployment, training, detection AP or novelty authorization.")


def checked_prior():
    prior = project_path(PRIOR)
    result_path = project_path("docs/experiments/candidate56_texture_deletion_result_20261015.json")
    result = json.loads(result_path.read_text())
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    plan = json.loads((prior/"plan.json").read_text())
    for name, digest in plan["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for value in plan["installed_sources"].values():
        assert file_hash(Path(value["path"])) == value["sha256"]
    return prior, plan, json.loads((prior/"frames.json").read_text()), result_path


def run(output=OUTPUT):
    tick = perf_counter()
    root = project_path(output)
    if root.exists():
        raise FileExistsError(root)
    prior, previous, frames, previous_result = checked_prior()
    control = frozen_control()
    registration = json.loads(project_path(REGISTRATION).read_text())
    selection = {half: previous["image_selection"][half] for half in ("screen", "confirmation")}
    assert all(len(v) == 64 for v in selection.values())
    assert len({stem[:7] for values in selection.values() for stem in values}) == 128
    paths = (REGISTRATION, "sdcl/evaluation/native_capacity.py", "scripts/evaluation/diagnose_native_capacity.py",
             "tests/test_native_capacity.py", "sdcl/data/geometry.py", "sdcl/evaluation/size_analysis.py",
             "sdcl/postprocess/localization.py", "sdcl/config.py")
    sources = {p: file_hash(project_path(p)) for p in paths}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in paths:
            z.write(project_path(name), name)
    import scipy
    import scipy.sparse.csgraph as graph_module
    graph_binary = Path(graph_module._matching.__file__)
    plan = dict(registration=registration, image_selection=selection, source_hashes=sources,
                source_archive_sha256=file_hash(root/"source.zip"), prior_result_sha256=file_hash(previous_result),
                prior_plan_sha256=file_hash(prior/"plan.json"), prior_frames_manifest_sha256=file_hash(prior/"frames.json"),
                checkpoint_sha256=control["detector"]["sha256"], scipy_version=scipy.__version__,
                installed_matching_binary=dict(path=str(graph_binary), sha256=file_hash(graph_binary)),
                selection_saved_before_graph_evaluation=True)
    save_json(root/"plan.json", plan)
    manifest, traces = {}, {half: [] for half in selection}
    for half, stems in selection.items():
        for stem in stems:
            entry = frames[stem]
            assert file_hash(prior/entry["file"]) == entry["sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == previous["inputs"][stem]["metadata_sha256"]
            metadata = json.loads(meta_path.read_text())
            gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
            areas = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)*(640/max(metadata["original_size"]))**2
            with np.load(prior/entry["file"], allow_pickle=False) as z:
                boxes = map_boxes(z["native_boxes"], entry["windows"])
                logits = z["native_logits"].reshape(-1, 10)
                reference = z["reference"].copy()
            ref_iou = box_iou_matrix(gt, reference[:, :4])
            ignored = prediction_ignore_mask(torch.tensor(reference[:, :4]),
                torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float64).reshape(-1, 4)).numpy()
            labeled_reference = reference.copy()
            labeled_reference[ignored, 4] = -1
            owners, _ = match_at_threshold(ref_iou, classes, labeled_reference, .25)
            control_gt = owners >= 0
            edges = graph_edges(gt, classes, boxes, logits)
            arrays, trace = frame_statistics(gt, classes, areas, edges, control_gt)
            path = root/"frames"/(stem+".npz")
            np.savez_compressed(path, gt=gt, classes=classes, areas=areas, control_gt=control_gt, **arrays)
            manifest[stem] = dict(half=half, file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                prior_frame_sha256=entry["sha256"], metadata_sha256=file_hash(meta_path))
            traces[half].append(trace)
            if len(manifest) % 16 == 0:
                print(f"Native capacity graphs: {len(manifest)}/128", flush=True)
    metrics = aggregate(traces)
    decision = decide(metrics, registration["policy_per_half"])
    verification = dict(cached_frames_read=128, JPEG_decodes=0, detector_forwards=0, fitting_steps=0,
        parameter_updates=0, AP_evaluations=0, sparse_matching_problems=128*2*6*3,
        all_matching_vertex_cover_certificates_checked=True,
        source_unchanged=all(file_hash(project_path(p)) == h for p, h in sources.items()),
        installed_matching_binary_unchanged=file_hash(graph_binary) == plan["installed_matching_binary"]["sha256"],
        elapsed_seconds=perf_counter()-tick)
    for name, value in (("frames.json", manifest), ("per_image.json", traces), ("metrics.json", metrics),
                        ("decision.json", decision), ("verification.json", verification)):
        save_json(root/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(root),
                  metrics=metrics, decision=decision, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save_json(root/"result.json", result)
    return result
