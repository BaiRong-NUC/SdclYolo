"""Bounded real-second-observation supervision screen on static DET candidates."""

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.special import expit, logsumexp
import torch
from ultralytics.data import augment as augment_module
from ultralytics.nn.modules import head as head_module
from ultralytics.utils import ops as ops_module

from sdcl.config import project_path
from sdcl.evaluation.neighbor_interference import array_hash, digest, query_pool, state_hash
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash
from sdcl.evaluation.texture_deletion import original_inventory
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews


REGISTRATION = "docs/experiments/candidate68_second_observation_registration_20261015.json"
OUTPUT = "output/analysis/second-observation-fixed-20261015"
ARMS = ("student", "flip_confidence", "source_confidence")


def select_pairs():
    groups = {}
    for path in project_path("data/visdrone/images/train").glob("*.jpg"):
        if not path.stem.startswith("999"):
            groups.setdefault(path.stem[:7], []).append(path.stem)
    valid = [key for key, stems in groups.items() if len({int(s.split("_")[1]) for s in stems}) >= 2]
    chosen = []
    for i, group in enumerate(sorted(valid, key=lambda s: digest("candidate68-group-v1:"+s))[:64]):
        current = min(groups[group], key=lambda s: digest("candidate68-current-v1:"+s))
        frame = int(current.split("_")[1])
        source = min((s for s in groups[group] if int(s.split("_")[1]) != frame),
                     key=lambda s: (abs(int(s.split("_")[1])-frame), digest("candidate68-source-v1:"+s)))
        chosen.append(dict(group=group, current=current, source=source,
                           split="screen" if i < 32 else "confirmation"))
    if len(chosen) != 64:
        raise ValueError("Need64candidate prefix groups; do not relax sampling.")
    return chosen


def project_boxes(boxes, homography):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return boxes.copy()
    points = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]],
                       boxes[:, [2, 3]], boxes[:, [0, 3]]), 1)
    mapped = cv2.perspectiveTransform(points, np.asarray(homography, float))
    return np.column_stack((mapped.min(1), mapped.max(1)))


def mirror_boxes(boxes, width):
    result = np.asarray(boxes).copy()
    result[:, [0, 2]] = width-result[:, [2, 0]]
    return result


def align_images(current, source):
    def gray(image):
        h, w = image.shape[:2]
        ratio = 960/max(h, w)
        value = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                           (round(w*ratio), round(h*ratio)))
        points, descriptors = cv2.SIFT_create(nfeatures=2048).detectAndCompute(value, None)
        return np.array([p.pt for p in points], float).reshape(-1, 2), descriptors, ratio
    cp, cd, cr = gray(current)
    sp, sd, sr = gray(source)
    info = dict(current_keypoints=len(cp), source_keypoints=len(sp), accepted=False, reason="matches")
    if cd is None or sd is None or min(len(cd), len(sd)) < 2:
        return info, {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def matches(a, b):
        return {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(a, b, k=2)
                if m.distance < .75*n.distance}
    forward, reverse = matches(sd, cd), matches(cd, sd)
    pairs = [(s, c) for s, c in sorted(forward.items()) if reverse.get(c) == s]
    info["mutual_matches"] = len(pairs)
    if len(pairs) < 40:
        return info, {}
    source_points, current_points = sp[[p[0] for p in pairs]], cp[[p[1] for p in pairs]]
    cv2.setRNGSeed(68)
    h, mask = cv2.findHomography(source_points, current_points, cv2.RANSAC, 3,
                                 maxIters=2000, confidence=.995)
    if h is None:
        info["reason"] = "homography"
        return info, {}
    mask = mask.ravel().astype(bool)
    mapped = cv2.perspectiveTransform(source_points[None], h)[0]
    error = np.linalg.norm(mapped-current_points, axis=1)
    original_h = np.diag([1/cr, 1/cr, 1])@h@np.diag([sr, sr, 1])
    original_h /= original_h[2, 2]
    sh, sw = source.shape[:2]
    ch, cw = current.shape[:2]
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [sw, 0], [sw, sh], [0, sh]]], float),
                                       original_h)[0].astype(np.float32)
    convex = bool(np.isfinite(corners).all() and cv2.isContourConvex(corners))
    intersection = cv2.intersectConvexConvex(corners, np.array(
        [[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32))[0] if convex else 0.
    overlap = float(intersection/max(min(abs(cv2.contourArea(corners)), cw*ch), 1e-12))
    median = float(np.median(error[mask])) if mask.any() else float("inf")
    info.update(inliers=int(mask.sum()), inlier_fraction=float(mask.mean()),
                median_resized_error=median, overlap_fraction=overlap, convex=convex,
                homography=original_h.tolist(), reason="geometry")
    info["accepted"] = bool(np.isfinite(original_h).all() and mask.sum() >= 40 and mask.mean() >= .4
                            and median <= 1.5 and convex and overlap >= .5)
    return info, dict(source_points=source_points/sr, current_points=current_points/cr,
                      inliers=mask, homography=original_h, current_resize_ratio=np.array(cr))


def correspondences(current, source, homography):
    cg = np.array([o["box_xyxy"] for o in current["valid_objects"]], float).reshape(-1, 4)
    sg = np.array([o["box_xyxy"] for o in source["valid_objects"]], float).reshape(-1, 4)
    cc = np.array([o["class_id"] for o in current["valid_objects"]], int)
    sc = np.array([o["class_id"] for o in source["valid_objects"]], int)
    if not len(cg) or not len(sg):
        return np.empty((0, 2), int), np.zeros((len(cg), len(sg)))
    iou = box_iou_matrix(cg, project_boxes(sg, homography))
    compatible = np.where(cc[:, None] == sc[None], iou, -1.)
    best_source, best_current = compatible.argmax(1), compatible.argmax(0)
    ca, sa = box_areas(cg), box_areas(sg)
    cscale, sscale = 960/max(current["original_size"]), 960/max(source["original_size"])
    matches = []
    for ci, si in enumerate(best_source):
        if best_current[si] != ci or compatible[ci, si] < .6:
            continue
        other_source = np.delete(compatible[ci], si).max(initial=-1)
        other_current = np.delete(compatible[:, si], ci).max(initial=-1)
        if min(compatible[ci, si]-other_source, compatible[ci, si]-other_current) < .15:
            continue
        if ca[ci]*(cscale*2/3)**2 >= 256 or sa[si]*(sscale*2/3)**2 >= 256:
            continue
        ratio = np.sqrt(sa[si])*sscale/max(np.sqrt(ca[ci])*cscale, 1e-12)
        if .8 <= ratio <= 1.25:
            matches.append((ci, int(si)))
    return np.array(matches, int).reshape(-1, 2), iou


def extract_queries(boxes, logits, gt, classes, size):
    queries = []
    for box, cls in zip(gt, classes):
        pool = query_pool(box, *size)
        utility = expit(logits[pool, cls].astype(float))*box_iou_matrix(box[None], boxes[pool])[0]
        queries.append(int(pool[utility.argmax()]))
    return np.array(queries, int)


def measurements(boxes, logits, gt, classes):
    n = len(gt)
    if not n:
        return dict(Q=np.zeros((0, 3)), NLL=np.zeros((0, 3)), success=np.zeros((0, 3), bool),
                    changed=np.zeros((0, 3), bool))
    # Cached views are current, flip and source. Both selectors retain the SAME current reference.
    confidence = expit(logits.astype(float).max(2))
    selected = np.column_stack((np.zeros(n, int),
        np.where(confidence[:, 1] > confidence[:, 0], 1, 0),
        np.where(confidence[:, 2] > confidence[:, 0], 2, 0)))
    rows = np.arange(n)[:, None]
    used_logits, used_boxes = logits[rows, selected], boxes[rows, selected]
    iou = np.array([[box_iou_matrix(gt[i:i+1], used_boxes[i, j:j+1])[0, 0]
                     for j in range(3)] for i in range(n)])
    score = expit(used_logits[np.arange(n), :, classes].astype(float))
    correct = used_logits.argmax(2) == classes[:, None]
    return dict(Q=iou*score*correct, NLL=logsumexp(used_logits.astype(float), axis=2)-
        used_logits[np.arange(n), :, classes].astype(float),
        success=(score >= .25) & correct & (iou >= .5), changed=selected != 0)


def summarize(traces, policy):
    metrics, checks = {}, {}
    for split, frames in traces.items():
        counts = np.array([len(f["Q"]) for f in frames])
        n = int(counts.sum())
        values = {key: np.concatenate([f[key] for f in frames]) for key in frames[0]}
        m = dict(targets=n, groups_with_targets=int((counts > 0).sum()),
                 source_changed=int(values["changed"][:, 2].sum()))
        c = dict(targets=n >= policy["matched_tiny_targets_min"],
                 groups=m["groups_with_targets"] >= policy["groups_with_targets_min"],
                 source_changes=m["source_changed"] >= policy["source_confidence_changed_targets_min"])
        draws = np.random.default_rng(68).multinomial(len(frames), [1/len(frames)]*len(frames), size=2000)
        mass = draws@counts
        for name, arm in (("student", 0), ("flip", 1)):
            sums = np.array([(f["Q"][:, 2]-f["Q"][:, arm]).sum() for f in frames])
            gain = float(sums.sum()/n) if n else None
            ci = np.percentile(draws@sums/mass, [2.5, 97.5]).tolist() if (mass > 0).all() else None
            m["source_Q_gain_vs_"+name] = gain
            m["source_Q_gain_vs_"+name+"_bootstrap95ci"] = ci
            c["Q_vs_"+name] = gain is not None and gain >= policy["source_Q_gain_vs_"+name+"_min"]
            c["interval_vs_"+name] = ci is not None and ci[0] > policy["both_Q_gain_bootstrap_lower_exclusive"]
        m["source_NLL_gain_vs_student"] = float((values["NLL"][:, 0]-values["NLL"][:, 2]).mean()) if n else None
        m["source_success_count_change"] = int(values["success"][:, 2].sum()-values["success"][:, 0].sum())
        m["Q_means"] = {a: float(values["Q"][:, i].mean()) if n else None for i, a in enumerate(ARMS)}
        m["NLL_means"] = {a: float(values["NLL"][:, i].mean()) if n else None for i, a in enumerate(ARMS)}
        c["NLL"] = m["source_NLL_gain_vs_student"] is not None and m["source_NLL_gain_vs_student"] >= policy["source_conditional_NLL_gain_vs_student_min"]
        c["success"] = m["source_success_count_change"] >= policy["source_fixed_query_success_count_change_min"]
        metrics[split], checks[split] = m, c
    passed = all(all(c.values()) for c in checks.values())
    return metrics, dict(passed=passed, checks=checks,
        decision="SECOND_OBSERVATION_PROXY_SUPPORTS_SEPARATELY_REGISTERED_SMALL_TRANSFER_PILOT_WITH_IDENTITY_AUDIT"
        if passed else "STOP_THIS_DET_SECOND_OBSERVATION_PROXY_RECIPE")


def run(device="0", output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = project_path(output)
    if root.exists():
        raise FileExistsError(root)
    registration = json.loads(project_path(REGISTRATION).read_text())
    chosen = select_pairs()
    for pair in chosen:
        pair["inputs"] = {stem: dict(
            image_sha256=file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")),
            metadata_sha256=file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")))
            for stem in (pair["current"], pair["source"])}
    paths = (REGISTRATION, "sdcl/evaluation/second_observation.py",
        "scripts/evaluation/diagnose_second_observation.py", "scripts/evaluation/verify_second_observation.py",
        "tests/test_second_observation.py", "sdcl/evaluation/neighbor_interference.py",
        "sdcl/evaluation/texture_deletion.py", "sdcl/evaluation/size_analysis.py",
        "sdcl/training/class_margin_probe.py", "sdcl/postprocess/class_margin_probe.py",
        "sdcl/postprocess/localization.py", "sdcl/config.py")
    sources = {p: file_hash(project_path(p)) for p in paths}
    installed = {n: Path(inspect.getfile(m)) for n, m in
                 (("augment.py", augment_module), ("head.py", head_module), ("ops.py", ops_module))}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(project_path(p), p)
        for name, p in installed.items():
            archive.write(p, "installed/"+name)
    plan = dict(registration=registration, pairs=chosen, source_hashes=sources,
                source_archive_sha256=file_hash(root/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                selection_saved_before_image_decode=True)
    save_json(root/"plan.json", plan)
    extractor = FrozenViews(device)
    model = extractor.model.predictor.model.model
    before = state_hash(model)
    save_json(root/"detector.json", dict(checkpoint=extractor.control["detector"], state_before=before))
    manifest, traces, forwards, accepted = {}, {"screen": [], "confirmation": []}, 0, 0
    try:
        with torch.inference_mode():
            for index, pair in enumerate(chosen):
                current_stem, source_stem = pair["current"], pair["source"]
                for stem, hashes in pair["inputs"].items():
                    assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == hashes["image_sha256"]
                    assert file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) == hashes["metadata_sha256"]
                images = [cv2.imread(str(project_path(f"data/visdrone/images/train/{stem}.jpg")))
                          for stem in (current_stem, source_stem)]
                metadata = [json.loads(project_path(f"data/visdrone/metadata/train/{stem}.json").read_text())
                            for stem in (current_stem, source_stem)]
                for image, meta in zip(images, metadata):
                    assert [image.shape[1], image.shape[0]] == meta["original_size"]
                alignment, arrays = align_images(*images)
                if alignment["accepted"]:
                    matched, overlap = correspondences(*metadata, arrays["homography"])
                    order = sorted(range(len(matched)), key=lambda i: digest(
                        f"candidate68-object-v1:{current_stem}:{matched[i, 0]}"))[:12]
                    matched = matched[order]
                    arrays.update(matched=matched, correspondence_iou=overlap)
                else:
                    matched = np.empty((0, 2), int)
                measured = measurements(np.zeros((0, 3, 4)), np.zeros((0, 3, 10)),
                                        np.zeros((0, 4)), np.zeros(0, int))
                if alignment["accepted"]:
                    accepted += 1
                    current, source = images
                    flipped = np.ascontiguousarray(current[:, ::-1])
                    inventories = []
                    for name, image in (("current", current), ("flip", flipped), ("source", source)):
                        extractor.predict(image)
                        boxes, logits = original_inventory(extractor, image.shape[1], image.shape[0])
                        arrays[name+"_boxes"], arrays[name+"_logits"] = boxes, logits
                        inventories.append((boxes, logits))
                        forwards += 1
                    extractor.predict(current)
                    rb, rl = original_inventory(extractor, current.shape[1], current.shape[0])
                    np.testing.assert_array_equal(rb, arrays["current_boxes"])
                    np.testing.assert_array_equal(rl, arrays["current_logits"])
                    forwards += 1
                    cg = np.array([o["box_xyxy"] for o in metadata[0]["valid_objects"]], float)
                    sg = np.array([o["box_xyxy"] for o in metadata[1]["valid_objects"]], float)
                    gt, source_gt = cg[matched[:, 0]], sg[matched[:, 1]]
                    classes = np.array([metadata[0]["valid_objects"][i]["class_id"] for i in matched[:, 0]], int)
                    view_gt = (gt, mirror_boxes(gt, current.shape[1]), source_gt)
                    view_size = (metadata[0]["original_size"], metadata[0]["original_size"], metadata[1]["original_size"])
                    queries = [extract_queries(b, l, g, classes, size)
                               for (b, l), g, size in zip(inventories, view_gt, view_size)]
                    boxes = np.stack([b[q] for (b, _), q in zip(inventories, queries)], 1).astype(float)
                    logits = np.stack([l[q] for (_, l), q in zip(inventories, queries)], 1)
                    boxes[:, 1] = mirror_boxes(boxes[:, 1], current.shape[1])
                    boxes[:, 2] = project_boxes(boxes[:, 2], arrays["homography"])
                    measured = measurements(boxes, logits, gt, classes)
                    arrays.update(boxes=boxes, logits=logits, gt=gt, source_gt=source_gt, classes=classes,
                                  queries=np.stack(queries, 1), **measured)
                    alignment["flipped_input_sha256"] = array_hash(flipped)
                traces[pair["split"]].append(measured)
                path = root/"frames"/(pair["group"]+".npz")
                np.savez_compressed(path, **arrays)
                manifest[pair["group"]] = dict(split=pair["split"], file=path.relative_to(root).as_posix(),
                    sha256=file_hash(path), alignment=alignment, targets=len(matched))
                if (index+1) % 8 == 0:
                    print(f"Second observation: {index+1}/64 pairs; {accepted} accepted alignments; {forwards} forwards", flush=True)
        after = state_hash(model)
        assert before == after
    finally:
        extractor.close()
    metrics, decision = summarize(traces, registration["policy_per_half"])
    save_json(root/"frames.json", manifest)
    save_json(root/"metrics.json", metrics)
    save_json(root/"decision.json", decision)
    save_json(root/"detector.json", dict(checkpoint=extractor.control["detector"], state_before=before, state_after=after))
    assert all(file_hash(project_path(p)) == value for p, value in sources.items())
    result = dict(experiment=registration["experiment"], decision=decision, metrics=metrics, output=str(root),
        costs=dict(JPEG_decodes=128, SIFT_extractions=128, accepted_alignments=accepted,
            detector_forwards=forwards, explicit_initialization_prediction_calls=1,
            warmup_scope="Internal backend warmup not separately instrumented.",
            optimizer_steps=0, fits=0, AP_evaluations=0, elapsed_seconds=perf_counter()-started),
        limits="Previously inspected detector-training images;heuristic prefixes not verified sequence identities;GT-assisted correspondence/query proxy;no learned transfer,AP,deployed gains or novelty.",
        artifacts_sha256={p.name: file_hash(p) for p in root.iterdir() if p.is_file()})
    save_json(root/"result.json", result)
    return result
