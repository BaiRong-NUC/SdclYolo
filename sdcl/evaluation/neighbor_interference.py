"""Preregistered nonoverlapping-neighbor input-to-feature interference screen."""

import hashlib
import inspect
import json
from pathlib import Path
import re
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.special import expit
import torch
from ultralytics.data import augment as augment_module
from ultralytics.nn.modules import head as head_module
from ultralytics.utils import ops as ops_module

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash
from sdcl.evaluation.texture_deletion import original_inventory
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews


REGISTRATION = "docs/experiments/candidate67_neighbor_interference_registration_20261015.json"
OUTPUT = "output/analysis/neighbor-interference-fixed-20261015"
ARMS = ("base", "neighbor_mean", "sham_mean", "neighbor_blur", "sham_blur", "self_mean")
STEMS = r"\d{7}_\d{5}_[a-z]_\d{7}"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rectangles_overlap(rectangle, boxes, margin=2):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return False
    return bool((np.minimum(rectangle[2:], boxes[:, 2:]+margin) >
                 np.maximum(rectangle[:2], boxes[:, :2]-margin)).all(1).any())


def sham_rectangles(target, neighbor, boxes, width, height):
    center = (np.asarray(target[:2])+target[2:])/2
    nc = (neighbor[:2]+neighbor[2:])/2
    wh = neighbor[2:]-neighbor[:2]
    result = []
    for sx, sy in ((-1, 1), (1, -1), (-1, -1)):
        sc = center+(nc-center)*[sx, sy]
        lo = np.rint(sc-wh/2).astype(int)
        rect = np.r_[lo, lo+wh]
        if (lo >= 0).all() and (rect[2:] <= [width, height]).all():
            distance_error = abs(np.linalg.norm((rect[:2]+rect[2:])/2-center)-np.linalg.norm(nc-center))
            if distance_error <= 1 and not rectangles_overlap(rect, boxes):
                if not any(np.array_equal(rect, old) for old in result):
                    result.append(rect)
    return result


def eligible_pairs(metadata, stem):
    width, height = metadata["original_size"]
    gt = np.array([obj["box_xyxy"] for obj in metadata["valid_objects"]], float).reshape(-1, 4)
    ignores = np.asarray(metadata["ignore_boxes_xyxy"], float).reshape(-1, 4)
    rects = np.column_stack((np.floor(gt[:, :2]), np.ceil(gt[:, 2:]))).astype(int)
    centers = (gt[:, :2]+gt[:, 2:])/2
    areas = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)
    tiny = (areas*(640/max(width, height))**2 < 256) & ((rects[:, 2:]-rects[:, :2]) >= 4).all(1)
    tiny &= (rects[:, :2] >= 0).all(1) & (rects[:, 2:] <= [width, height]).all(1)
    all_boxes = np.vstack((gt, ignores))
    pairs = []
    for i in np.flatnonzero(tiny):
        distance = np.square(centers-centers[i]).sum(1)
        neighbors = np.flatnonzero(tiny & (np.arange(len(gt)) != i) & (distance <= 16*areas[i]))
        for j in sorted(neighbors, key=lambda j: (distance[j], int(j))):
            other = np.vstack((gt[np.arange(len(gt)) != j], ignores))
            if rectangles_overlap(rects[j], other):
                continue
            shams = sham_rectangles(gt[i], rects[j], all_boxes, width, height)
            if shams:
                pairs.append(dict(target=int(i), neighbor=int(j), target_rect=rects[i].tolist(),
                                  neighbor_rect=rects[j].tolist(), shams=[r.tolist() for r in shams]))
                break
    return sorted(pairs, key=lambda p: digest(f"candidate67-target-v1:{stem}:{p['target']}"))[:4]


def patch_texture(image, rectangle):
    x1, y1, x2, y2 = rectangle
    patch = image[y1:y2, x1:x2].astype(float)
    return np.r_[patch.mean((0, 1)), patch.std((0, 1))]


def perturb(image, rectangle, fill):
    x1, y1, x2, y2 = rectangle
    patch = image[y1:y2, x1:x2]
    if fill == "mean":
        value = np.rint(patch.astype(float).mean((0, 1))).astype(np.uint8)
    elif fill == "blur":
        value = cv2.GaussianBlur(patch, (0, 0), max(1, min(x2-x1, y2-y1)/3),
                                 borderType=cv2.BORDER_REFLECT_101)
    else:
        raise ValueError(fill)
    altered = image.copy()
    altered[y1:y2, x1:x2] = value
    return altered


def input_rectangle(box, width, height):
    ratio = 960/max(width, height)
    pad = np.array([round((960-round(width*ratio))/2-.1),
                    round((960-round(height*ratio))/2-.1)])
    return np.r_[np.floor(np.asarray(box[:2])*ratio+pad),
                 np.ceil(np.asarray(box[2:])*ratio+pad)].astype(int).clip(0, 960)


def query_pool(box, width, height):
    ratio = 960/max(width, height)
    pad = np.array([round((960-round(width*ratio))/2-.1),
                    round((960-round(height*ratio))/2-.1)])
    center = ((np.asarray(box[:2])+box[2:])/2*ratio+pad)/8-.5
    cx, cy = np.rint(center).astype(int)
    return np.array(sorted({int(y*120+x) for y in range(max(0, cy-1), min(120, cy+2))
                            for x in range(max(0, cx-1), min(120, cx+2))}), int)


def select_images():
    excluded, exclusions = set(), {}
    for path in sorted(project_path("output/analysis").glob("*/plan.json")):
        exclusions[path.relative_to(project_path(".")).as_posix()] = file_hash(path)
        excluded.update(re.findall(STEMS, path.read_text(encoding="utf-8")))
    groups = {}
    for path in project_path("data/visdrone/images/train").glob("*.jpg"):
        if path.stem not in excluded:
            groups.setdefault(path.stem[:7], []).append(path)
    chosen, metadata_reads = [], 0
    for sequence in sorted(groups, key=lambda s: digest("candidate67-sequence-v1:"+s)):
        for path in sorted(groups[sequence], key=lambda p: digest("candidate67-image-v1:"+p.stem)):
            meta_path = project_path(f"data/visdrone/metadata/train/{path.stem}.json")
            metadata = json.loads(meta_path.read_text())
            metadata_reads += 1
            pairs = eligible_pairs(metadata, path.stem)
            if len(pairs) >= 2:
                chosen.append(dict(stem=path.stem, sequence=sequence, pairs=pairs,
                    image_sha256=file_hash(path), metadata_sha256=file_hash(meta_path)))
                break
        if len(chosen) == 96:
            break
    return chosen, dict(exclusions=exclusions, excluded_images=len(excluded), metadata_reads=metadata_reads)


def frame_metrics(boxes, logits, gt, classes, neck, features):
    ious = np.array([[box_iou_matrix(gt[i:i+1], boxes[i, a:a+1])[0, 0]
                      for a in range(len(ARMS))] for i in range(len(gt))])
    score = expit(logits[np.arange(len(gt)), :, classes].astype(float))
    quality = score*ious
    successful = (score >= .25) & (logits.argmax(2) == classes[:, None]) & (ious >= .5)
    relative = lambda x: np.linalg.norm(x.astype(float)-x[:, :1].astype(float), axis=2) / np.maximum(
        np.linalg.norm(x[:, :1].astype(float), axis=2), 1e-12)
    return dict(Q=quality, score=score, iou=ious, success=successful,
                class_logit=logits[np.arange(len(gt)), :, classes].astype(float),
                neck_relative_L2=relative(neck), class_feature_relative_L2=relative(features))


def aggregate(frames, policy):
    summary, checks = {}, {}
    for split, rows in frames.items():
        counts = np.array([len(row["Q"]) for row in rows])
        total = int(counts.sum())
        draws = np.random.default_rng(67).multinomial(len(rows), [1/len(rows)]*len(rows), size=2000)
        mass = draws@counts
        joined = {k: np.concatenate([row[k] for row in rows]) for k in rows[0]
                  if k not in ("stem", "same_class")}
        m = dict(pairs=total, sequences=len(rows), fills={},
                 self_mean_Q_drop=float((joined["Q"][:, 0]-joined["Q"][:, 5]).mean()))
        c = dict(pairs=total >= policy["pairs_min"], sequences=len(rows) >= policy["sequences_min"],
                 self_sensitivity=m["self_mean_Q_drop"] >= policy["self_mean_Q_drop_min"])
        both = np.ones(total, bool)
        for fill, ni, si in (("mean", 1, 2), ("blur", 3, 4)):
            vs_base = joined["Q"][:, ni]-joined["Q"][:, 0]
            vs_sham = joined["Q"][:, ni]-joined["Q"][:, si]
            stats = dict(Q_gain_vs_base=float(vs_base.mean()), Q_gain_vs_sham=float(vs_sham.mean()),
                         logit_gain_vs_sham=float((joined["class_logit"][:, ni]-joined["class_logit"][:, si]).mean()),
                         iou_gain_vs_sham=float((joined["iou"][:, ni]-joined["iou"][:, si]).mean()),
                         fixed_query_success_count_change=int(joined["success"][:, ni].sum()-joined["success"][:, 0].sum()),
                         neighbor_neck_relative_L2=float(joined["neck_relative_L2"][:, ni].mean()),
                         sham_neck_relative_L2=float(joined["neck_relative_L2"][:, si].mean()),
                         neighbor_class_feature_relative_L2=float(joined["class_feature_relative_L2"][:, ni].mean()),
                         sham_class_feature_relative_L2=float(joined["class_feature_relative_L2"][:, si].mean()))
            for name, ai in (("base", 0), ("sham", si)):
                sums = np.array([(row["Q"][:, ni]-row["Q"][:, ai]).sum() for row in rows])
                stats[f"Q_gain_vs_{name}_bootstrap95ci"] = np.percentile(draws@sums/mass, [2.5, 97.5]).tolist()
            stats["improved_pairs"] = int((vs_base >= .01).sum())
            stats["harmed_pairs"] = int((vs_base <= -.01).sum())
            both &= (vs_base >= policy["improved_pair_neighbor_Q_gain_min"]) & (
                vs_sham >= policy["improved_pair_Q_gain_vs_sham_min"])
            same = np.concatenate([row["same_class"] for row in rows])
            stats["neighbor_class_subgroups"] = {
                label: dict(pairs=int(mask.sum()), Q_gain_vs_sham=float(vs_sham[mask].mean()) if mask.any() else None)
                for label, mask in (("same", same), ("different", ~same))}
            m["fills"][fill] = stats
            c.update({
                fill+"_vs_base": stats["Q_gain_vs_base"] >= policy["each_fill_neighbor_Q_gain_vs_base_min"],
                fill+"_vs_sham": stats["Q_gain_vs_sham"] >= policy["each_fill_neighbor_Q_gain_vs_sham_min"],
                fill+"_logit": stats["logit_gain_vs_sham"] >= policy["each_fill_neighbor_logit_gain_vs_sham_min"],
                fill+"_base_interval": stats["Q_gain_vs_base_bootstrap95ci"][0] > policy["each_fill_Q_gain_vs_base_bootstrap_lower_exclusive"],
                fill+"_sham_interval": stats["Q_gain_vs_sham_bootstrap95ci"][0] > policy["each_fill_Q_gain_vs_sham_bootstrap_lower_exclusive"]})
        cuts = np.r_[0, counts.cumsum()]
        m["both_fills_improved_pairs"] = int(both.sum())
        m["both_fills_improved_sequences"] = sum(bool(both[cuts[i]:cuts[i+1]].any()) for i in range(len(rows)))
        c["repeat_pairs"] = m["both_fills_improved_pairs"] >= policy["both_fills_improved_pairs_min"]
        c["repeat_sequences"] = m["both_fills_improved_sequences"] >= policy["both_fills_improved_sequences_min"]
        summary[split], checks[split] = m, c
    passed = all(all(c.values()) for c in checks.values())
    return summary, dict(passed=passed, checks=checks,
        decision="REPLICATED_NONOVERLAPPING_NEIGHBOR_INTERFERENCE_SUPPORTS_SEPARATE_SMALL_REPRESENTATION_TRAINING_PILOT"
        if passed else "STOP_THIS_NONOVERLAPPING_NEIGHBOR_TEXTURE_INTERFERENCE_HYPOTHESIS_ON_FIXED_SCREEN")


def run(device="0", output=OUTPUT):
    tick = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = project_path(output)
    if root.exists():
        raise FileExistsError(root)
    registration = json.loads(project_path(REGISTRATION).read_text())
    chosen, selection_audit = select_images()
    if len(chosen) != 96:
        raise ValueError(f"Geometry-only feasibility STOP: {len(chosen)}/96 sequences; do not relax rules.")
    for i, item in enumerate(chosen):
        item["split"] = "screen" if i < 48 else "confirmation"
    paths = (REGISTRATION, "sdcl/evaluation/neighbor_interference.py",
             "scripts/evaluation/diagnose_neighbor_interference.py",
             "scripts/evaluation/verify_neighbor_interference.py", "tests/test_neighbor_interference.py",
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
        for name, path in installed.items():
            archive.write(path, "installed/"+name)
    plan = dict(registration=registration, images=chosen, selection_audit=selection_audit,
        source_hashes=sources, source_archive_sha256=file_hash(root/"source.zip"),
        installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
        selection_saved_before_detector_inference=True)
    save_json(root/"plan.json", plan)
    extractor = FrozenViews(device)
    model = extractor.model.predictor.model.model
    before = state_hash(model)
    save_json(root/"detector.json", dict(checkpoint=extractor.control["detector"], state_before=before))
    captured = {}
    def input_hook(module, args):
        captured["input"] = args[0].detach()
    def neck_hook(module, args):
        captured["neck"] = args[0][0].detach().flatten(2)[0].T
    handles = [model.register_forward_pre_hook(input_hook), extractor.head.register_forward_pre_hook(neck_hook)]
    manifest, traces, forwards = {}, {"screen": [], "confirmation": []}, 0
    try:
        with torch.inference_mode():
            for index, item in enumerate(chosen):
                stem = item["stem"]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                assert file_hash(image_path) == item["image_sha256"] and file_hash(meta_path) == item["metadata_sha256"]
                image = cv2.imread(str(image_path))
                metadata = json.loads(meta_path.read_text())
                height, width = image.shape[:2]
                assert [width, height] == metadata["original_size"]
                gt_all = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float)
                class_all = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
                pairs = item["pairs"]
                for pair in pairs:
                    texture = patch_texture(image, pair["neighbor_rect"])
                    pair["sham_rect"] = min(pair["shams"], key=lambda r: float(np.square(
                        (patch_texture(image, r)-texture)/255).sum()))
                extractor.predict(image)
                base_boxes, base_logits = original_inventory(extractor, width, height)
                base_class = extractor.features[0].flatten(2)[0].T.cpu().numpy().copy()
                base_neck = captured["neck"].cpu().numpy().copy()
                pristine_input = captured["input"][0].cpu().numpy().copy()
                forwards += 1
                extractor.predict(image)
                replay_boxes, replay_logits = original_inventory(extractor, width, height)
                np.testing.assert_array_equal(replay_boxes, base_boxes)
                np.testing.assert_array_equal(replay_logits, base_logits)
                np.testing.assert_array_equal(captured["neck"].cpu().numpy(), base_neck)
                np.testing.assert_array_equal(extractor.features[0].flatten(2)[0].T.cpu().numpy(), base_class)
                forwards += 1
                gt = np.array([gt_all[p["target"]] for p in pairs])
                classes = np.array([class_all[p["target"]] for p in pairs])
                queries = []
                for box, cls in zip(gt, classes):
                    pool = query_pool(box, width, height)
                    utility = expit(base_logits[pool, cls].astype(float))*box_iou_matrix(box[None], base_boxes[pool])[0]
                    queries.append(int(pool[utility.argmax()]))
                n = len(pairs)
                boxes = np.zeros((n, len(ARMS), 4), np.float32)
                logits = np.zeros((n, len(ARMS), 10), base_logits.dtype)
                neck = np.zeros((n, len(ARMS), base_neck.shape[1]), base_neck.dtype)
                features = np.zeros((n, len(ARMS), 128), base_class.dtype)
                boxes[:, 0], logits[:, 0] = base_boxes[queries], base_logits[queries]
                neck[:, 0], features[:, 0] = base_neck[queries], base_class[queries]
                for i, (pair, query) in enumerate(zip(pairs, queries)):
                    x1, y1, x2, y2 = input_rectangle(gt[i], width, height)
                    original_input_patch = pristine_input[:, y1:y2, x1:x2]
                    tx1, ty1, tx2, ty2 = pair["target_rect"]
                    pair["target_tensor_sha256"] = array_hash(original_input_patch)
                    pair["perturbations"] = []
                    for a, arm in enumerate(ARMS[1:], 1):
                        location, fill = arm.split("_")
                        rect = pair[{"neighbor": "neighbor_rect", "sham": "sham_rect", "self": "target_rect"}[location]]
                        altered = perturb(image, rect, fill)
                        if location != "self":
                            np.testing.assert_array_equal(altered[ty1:ty2, tx1:tx2], image[ty1:ty2, tx1:tx2])
                        extractor.predict(altered)
                        altered_boxes, altered_logits = original_inventory(extractor, width, height)
                        tensor_patch = captured["input"][0, :, y1:y2, x1:x2].cpu().numpy()
                        if location != "self":
                            np.testing.assert_array_equal(tensor_patch, original_input_patch)
                        boxes[i, a], logits[i, a] = altered_boxes[query], altered_logits[query]
                        neck[i, a] = captured["neck"][query].cpu().numpy()
                        features[i, a] = extractor.features[0].flatten(2)[0, :, query].cpu().numpy()
                        pair["perturbations"].append(dict(arm=arm, rectangle=rect,
                            altered_image_sha256=array_hash(altered), target_tensor_sha256=array_hash(tensor_patch)))
                        forwards += 1
                metrics = frame_metrics(boxes, logits, gt, classes, neck, features)
                row = dict(stem=stem, same_class=classes == np.array([class_all[p["neighbor"]] for p in pairs]), **metrics)
                traces[item["split"]].append(row)
                path = root/"frames"/(stem+".npz")
                np.savez_compressed(path, base_boxes=base_boxes, base_logits=base_logits, gt=gt, classes=classes,
                    queries=np.array(queries), boxes=boxes, logits=logits, neck=neck, features=features,
                    same_class=row["same_class"], **metrics)
                manifest[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path),
                                     split=item["split"], pairs=pairs)
                if (index+1) % 8 == 0:
                    print(f"Neighbor interference: {index+1}/96 images, {forwards} measured forwards", flush=True)
        after = state_hash(model)
        assert before == after
    finally:
        for handle in handles:
            handle.remove()
        extractor.close()
    save_json(root/"frames.json", manifest)
    summary, decision = aggregate(traces, registration["policy_per_half"])
    save_json(root/"metrics.json", summary)
    save_json(root/"decision.json", decision)
    save_json(root/"per_image.json", {s: [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
                                        for row in rows] for s, rows in traces.items()})
    save_json(root/"detector.json", dict(checkpoint=extractor.control["detector"], state_before=before, state_after=after))
    assert all(file_hash(project_path(p)) == value for p, value in sources.items())
    result = dict(experiment=registration["experiment"], decision=decision, metrics=summary, output=str(root),
        costs=dict(JPEG_decodes=96, detector_forwards=forwards, explicit_initialization_prediction_calls=1,
            initialization_scope="Backend internal warmup not separately instrumented.", optimizer_steps=0,
            fits=0, AP_evaluations=0, elapsed_seconds=perf_counter()-tick),
        limits=registration["limits"],
        artifacts_sha256={p.name: file_hash(p) for p in root.iterdir() if p.is_file()})
    save_json(root/"result.json", result)
    return result
