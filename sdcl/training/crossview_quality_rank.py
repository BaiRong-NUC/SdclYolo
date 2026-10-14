"""Fixed matched projection updates: same-object cross-view quality ranking."""

from copy import deepcopy
import hashlib
import inspect
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import box_iou_matrix, coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.transport_quality import diagnostic_filter, load_cache, native_nms


REGISTRATION = "docs/experiments/candidate66_crossview_quality_rank_registration_20261015.json"
PRIOR = "output/analysis/transport-quality-fixed-20261015"
OUTPUT = "output/analysis/crossview-quality-rank-fixed-20261015"
ARMS = ("bce", "iou", "within", "unlinked", "cross")
LEVELS = np.repeat(np.arange(3), [14400, 3600, 900])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def precision(training):
    torch.backends.cudnn.allow_tf32 = not training
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


class Projection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weight, dtype=torch.float32).clone())
        self.bias = nn.Parameter(torch.as_tensor(bias, dtype=torch.float32).clone())
        assert self.weight.numel() + self.bias.numel() == 3870

    def forward(self, features, levels):
        result = features.new_empty((len(features), 10))
        for level in range(3):
            rows = levels == level
            result[rows] = F.linear(features[rows], self.weight[level], self.bias[level])
        return result


def decoded_with_projection(frame, features, projection, device):
    values = torch.as_tensor(frame["decoded"][None], device=device).clone()
    scores = []
    start = 0
    for level, side in enumerate((120, 60, 30)):
        count = side * side
        x = torch.as_tensor(features[start:start+count].T.copy(), device=device).reshape(1, 128, side, side)
        score = F.conv2d(x, projection.weight[level, :, :, None, None].to(x), projection.bias[level].to(x))
        scores.append(score.reshape(1, 10, count))
        start += count
    logits = torch.cat(scores, -1)
    values[:, 4:14] = logits.sigmoid()
    assert torch.equal(values[:, :4], torch.as_tensor(frame["decoded"][None, :4], device=device))
    return values, logits


def target_rows(frame, metadata, window, size, stem, view):
    """Frozen assignment proxy in original image coordinates, before sampling."""
    from ultralytics.utils.tal import TaskAlignedAssigner
    objects = metadata["valid_objects"]
    boxes = np.asarray([o["box_xyxy"] for o in objects], np.float32).reshape(-1, 4)
    classes = np.asarray([o["class_id"] for o in objects], int)
    x1, y1, x2, y2 = window
    clipped = np.column_stack([np.maximum(boxes[:, :2], [x1, y1]), np.minimum(boxes[:, 2:], [x2, y2])])
    retained = np.prod(np.maximum(clipped[:, 2:]-clipped[:, :2], 0), 1) / np.maximum(
        np.prod(np.maximum(boxes[:, 2:]-boxes[:, :2], 0), 1), 1e-12)
    gt_ids = np.flatnonzero(retained >= .9)
    scale = min(960/(x2-x1), 960/(y2-y1))
    pad = np.array([round((960-round((x2-x1)*scale))/2-.1),
                    round((960-round((y2-y1)*scale))/2-.1)])
    anchors = (frame["anchors"] * frame["strides"][:, None] - pad) / scale + [x1, y1]
    n = len(anchors)
    ignored = prediction_ignore_mask(torch.from_numpy(frame["boxes"].copy()),
        torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)).numpy()
    inside = ((anchors >= [x1, y1]) & (anchors <= [x2, y2])).all(1)
    eligible = ~ignored & inside
    targets = np.zeros((n, 10), np.float32)
    owner, quality = np.full(n, -1, int), np.zeros(n, np.float32)
    if len(gt_ids):
        assigner = TaskAlignedAssigner(topk=10, num_classes=10, alpha=.5, beta=6.,
                                       stride=[8., 16., 32.], topk2=None)
        with torch.no_grad():
            _, _, t, fg, ids = assigner(
                torch.from_numpy(frame["decoded"][4:14].T.astype(np.float32))[None],
                torch.from_numpy(frame["boxes"])[None], torch.tensor(anchors, dtype=torch.float32),
                torch.tensor(classes[gt_ids], dtype=torch.float32)[None, :, None],
                torch.tensor(clipped[gt_ids], dtype=torch.float32)[None],
                torch.ones((1, len(gt_ids), 1), dtype=torch.bool))
        targets = t[0].numpy()
        valid = fg[0].numpy() & (targets.sum(1) > 0)
        owner[valid] = gt_ids[ids[0].numpy()[valid]]
        overlap = box_iou_matrix(clipped, frame["boxes"])
        quality[valid] = overlap[owner[valid], np.flatnonzero(valid)]
    positive = eligible & (owner >= 0)
    rows = list(np.flatnonzero(positive))
    weights = [1.] * len(rows)
    confidence = frame["decoded"][4:14].max(0).astype(np.float32)
    for level in range(3):
        for stratum in (False, True):
            pool = np.flatnonzero(eligible & ~positive & (LEVELS == level) & ((confidence >= .01) == stratum))
            count = min(len(pool), 64)
            if count:
                seed = int(digest(f"candidate66-sample-v1:{stem}:{view}:{level}:{stratum}")[:16], 16)
                selected = np.sort(np.random.default_rng(seed).choice(pool, count, replace=False))
                rows.extend(selected)
                weights.extend([len(pool)/count]*count)
    rows = np.asarray(rows, int)
    ordinary = np.zeros_like(targets)
    pos = np.flatnonzero(positive)
    if len(pos):
        ordinary[pos, classes[owner[pos]]] = quality[pos]
    labels = np.full(len(rows), -1, int)
    good = owner[rows] >= 0
    labels[good] = classes[owner[rows][good]]
    return dict(native=rows, levels=LEVELS[rows], original=targets[rows], ordinary=ordinary[rows],
                weight=np.asarray(weights, np.float32)*.5/max(float(targets.sum()), 1.),
                owner=owner[rows], quality=quality[rows], classes=labels,
                original_score=confidence[rows],
                true_score=np.where(good, frame["decoded"][4:14].T[rows, np.maximum(labels, 0)], 0))


def make_pairs(data):
    """Return count-matched within, linked, and quality-matched unlinked triplets."""
    by_object = {}
    for row in np.flatnonzero(data["owner"] >= 0):
        key = (int(data["image"][row]), int(data["owner"][row]))
        by_object.setdefault(key, {}).setdefault(int(data["view"][row]), []).append(int(row))
    pairsets = []
    all_representatives = []
    for key, views in sorted(by_object.items()):
        reps = []
        for view, rows in sorted(views.items()):
            by_q = sorted(rows, key=lambda i: (-data["quality"][i], i))
            by_p = sorted(rows, key=lambda i: (-data["true_score"][i], i))
            selected = list(dict.fromkeys((by_q[0], by_p[0], by_q[-1], by_p[-1])))
            reps.extend(selected)
        all_representatives.extend(reps)
        within, cross = [], []
        for a, b in combinations(reps, 2):
            if abs(float(data["quality"][a])-float(data["quality"][b])) < .1:
                continue
            hi, lo = (a, b) if data["quality"][a] > data["quality"][b] else (b, a)
            (within if data["view"][a] == data["view"][b] else cross).append((hi, lo))
        keyer = lambda pair: digest(f"candidate66-pair-v1:{key}:{pair}")
        within, cross = sorted(within, key=keyer), sorted(cross, key=keyer)
        pairsets.extend(zip(within[:8], cross[:8]))
    pools = {}
    for row in all_representatives:
        key = (int(data["classes"][row]), int(data["view"][row]), int(data["levels"][row]))
        pools.setdefault(key, []).append(row)
    result = {a: [] for a in ("within", "cross", "unlinked")}
    for within, cross in pairsets:
        hi, lo = cross
        key = (int(data["classes"][lo]), int(data["view"][lo]), int(data["levels"][lo]))
        candidates = [j for j in pools[key]
                      if (data["image"][j], data["owner"][j]) != (data["image"][lo], data["owner"][lo])
                      and abs(float(data["quality"][j])-float(data["quality"][lo])) <= .025
                      and float(data["quality"][hi])-float(data["quality"][j]) >= .1]
        if not candidates:
            continue
        donor = min(candidates, key=lambda j: (abs(float(data["quality"][j])-float(data["quality"][lo])), j))
        result["within"].append(within)
        result["cross"].append(cross)
        result["unlinked"].append((hi, donor))
    return {a: np.asarray(v, np.int64).reshape(-1, 2) for a, v in result.items()}


def extract(root, selection, manifest, device):
    prior = project_path(PRIOR)
    precision(False)
    extractor = FrozenViews(device)
    detector_hash = state_hash(extractor.model.predictor.model.model)
    rows_manifest = {}
    np.savez(root/"initial.npz", weight=extractor.weight.numpy(), bias=extractor.bias.numpy())
    initial = Projection(extractor.weight, extractor.bias).to("cuda:"+str(device))
    (root/"cache").mkdir()
    try:
        with torch.inference_mode():
            for image_no, stem in enumerate([s for split in selection.values() for s in split]):
                entry = manifest[stem]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                assert file_hash(image_path) == entry["image_sha256"]
                assert file_hash(meta_path) == entry["metadata_sha256"]
                metadata = read(meta_path)
                image = cv2.imread(str(image_path))
                assert image is not None and list(image.shape[1::-1]) == entry["size"]
                per_view = []
                for view, window in enumerate(entry["windows"]):
                    frame = load_cache(prior, entry["views"][view])
                    x1, y1, x2, y2 = window
                    outputs, _ = extractor.predict(image[y1:y2, x1:x2].copy())
                    np.testing.assert_array_equal(extractor.raw["scores"][0].T.cpu().numpy(), frame["raw_class"])
                    np.testing.assert_array_equal(extractor.decoded[0].cpu().numpy(), frame["decoded"])
                    np.testing.assert_array_equal(outputs["control"], frame["original_local"])
                    features = np.concatenate([extractor.features[l][0].flatten(1).T.cpu().numpy() for l in range(3)])
                    decoded, logits = decoded_with_projection(frame, features, initial, "cuda:"+str(device))
                    np.testing.assert_array_equal(logits[0].T.cpu().numpy(), frame["raw_class"])
                    np.testing.assert_array_equal(native_nms(decoded, (y2-y1, x2-x1)), frame["original_local"])
                    sampled = target_rows(frame, metadata, window, entry["size"], stem, view)
                    sampled["features"] = features[sampled["native"]]
                    path = root/"cache"/f"{stem}-v{view}.npz"
                    np.savez_compressed(path, features_full=features, **sampled)
                    per_view.append(dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path)))
                rows_manifest[stem] = dict(views=per_view, image_index=image_no, split=entry["split"])
                if image_no == 0 or (image_no+1) % 8 == 0:
                    print(f"Cross-view rank feature replay: {image_no+1}/113", flush=True)
                save(root/"frames.json", rows_manifest)
    finally:
        final_hash = state_hash(extractor.model.predictor.model.model)
        extractor.close()
        save(root/"detector_state.json", dict(before=detector_hash, after=final_hash))
        assert detector_hash == final_hash
    return rows_manifest


def assembled(root, rows_manifest, stems):
    collected = {}
    locations = []
    for stem in stems:
        image = rows_manifest[stem]["image_index"]
        for view, snapshot in enumerate(rows_manifest[stem]["views"]):
            frame = load_cache(root, snapshot)
            n = len(frame["native"])
            frame.pop("features_full")
            frame.update(image=np.full(n, image, int), view=np.full(n, view, int))
            for name, values in frame.items():
                collected.setdefault(name, []).append(values)
            locations.extend([stem] * n)
    data = {k: np.concatenate(v) for k, v in collected.items()}
    data["stems"] = np.array(locations)
    data["weight"] /= len(stems)*5
    pairs = make_pairs(data)
    return data, pairs


def precision_audit(initial, data, device):
    precision(True)
    model = deepcopy(initial)
    n = min(len(data["features"]), 256)
    x = torch.tensor(data["features"][:n], device=device, dtype=torch.float32)
    levels = torch.tensor(data["levels"][:n], device=device, dtype=torch.long)
    target = torch.tensor(data["ordinary"][:n], device=device)
    z = model(x, levels)
    loss = F.binary_cross_entropy_with_logits(z, target, reduction="sum")/n
    gw, gb = torch.autograd.grad(loss, (model.weight, model.bias))
    p, y, xx = z.detach().double().cpu().sigmoid().numpy(), target.double().cpu().numpy(), x.double().cpu().numpy()
    dw, db = np.zeros((3, 10, 128)), np.zeros((3, 10))
    for level in range(3):
        keep = data["levels"][:n] == level
        dz = (p[keep]-y[keep])/n
        dw[level], db[level] = dz.T@xx[keep], dz.sum(0)
    original = np.r_[gw.detach().cpu().numpy().ravel(), gb.detach().cpu().numpy().ravel()]
    reference = np.r_[dw.ravel(), db.ravel()]
    maximum = float(np.max(abs(original-reference)))
    relative = float(np.linalg.norm(original-reference)/max(np.linalg.norm(reference), 1e-30))
    return dict(rows=n, maximum_absolute_error=maximum, relative_L2_error=relative,
                passed=bool(np.all(abs(original-reference) <= 1e-6+1e-4*abs(reference)) and relative <= 1e-4),
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def fit(root, data, pairs, initial, seed, device):
    if not len(pairs["cross"]):
        raise ValueError("No matched training pairs; preserve engineering failure.")
    precision(True)
    x = torch.tensor(data["features"], device=device, dtype=torch.float32)
    levels = torch.tensor(data["levels"], device=device, dtype=torch.long)
    original, ordinary = [torch.tensor(data[k], device=device) for k in ("original", "ordinary")]
    weights = torch.tensor(data["weight"], device=device)
    pair_arrays = {a: torch.tensor(v, device=device) for a, v in pairs.items()}
    cls = torch.tensor(data["classes"], device=device, dtype=torch.long)
    models = {a: deepcopy(initial) for a in ARMS}
    opts = {a: torch.optim.AdamW(m.parameters(), lr=.0002, weight_decay=.0001) for a, m in models.items()}
    history = []
    orders = []
    for epoch in range(8):
        gen = torch.Generator().manual_seed(seed*100+epoch)
        order = torch.randperm(len(x), generator=gen).to(device)
        rankorder = torch.randperm(len(pairs["cross"]), generator=gen).to(device)
        orders.append(dict(epoch=epoch+1, bce_order_sha256=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest(),
                           rank_order_sha256=hashlib.sha256(rankorder.cpu().numpy().tobytes()).hexdigest()))
        losses = {a: dict(bce=0., rank=0.) for a in ARMS}
        for step, start in enumerate(range(0, len(x), 2048)):
            rows = order[start:start+2048]
            pr = rankorder[(torch.arange(256, device=device)+step*256) % len(rankorder)]
            for arm in ARMS:
                m = models[arm]
                logits = m(x[rows], levels[rows])
                target = original[rows] if arm == "bce" else ordinary[rows]
                bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(1)
                       * weights[rows]).sum()*len(x)/len(rows)
                rank = bce.new_zeros(())
                if arm in pair_arrays:
                    pp = pair_arrays[arm][pr]
                    endpoints = pp.flatten()
                    zz = m(x[endpoints], levels[endpoints])
                    zz = zz.gather(1, cls[endpoints, None])[:, 0].reshape(-1, 2)
                    rank = F.softplus(zz[:, 1]-zz[:, 0]).mean()
                loss = bce+.25*rank
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite projection fit.")
                opts[arm].zero_grad(set_to_none=True)
                loss.backward()
                opts[arm].step()
                losses[arm]["bce"] += float(bce.detach())*len(rows)/len(x)
                losses[arm]["rank"] += float(rank.detach())*len(rows)/len(x)
        history.append(dict(epoch=epoch+1, losses=losses))
        print(f"Cross-view rank seed{seed} epoch{epoch+1}/8", flush=True)
    torch.save({a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
               root/f"projections-seed{seed}.pt")
    save(root/f"history-seed{seed}.json", history)
    save(root/f"training-audit-seed{seed}.json", dict(initial_state_sha256=state_hash(initial),
         final_state_sha256={a: state_hash(m) for a, m in models.items()},
         updates_per_arm=8*((len(x)+2047)//2048), rows=len(x), pairs=len(pairs["cross"]),
         orders=orders, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
    return models


def mechanism(data, pairs, models, device):
    p = pairs["cross"]
    values = {}
    with torch.inference_mode():
        for arm, model in models.items():
            chunks = []
            for start in range(0, len(p), 2048):
                endpoints = p[start:start+2048].flatten()
                x = torch.tensor(data["features"][endpoints], device=device, dtype=torch.float32)
                levels = torch.tensor(data["levels"][endpoints], device=device, dtype=torch.long)
                z = model(x, levels).double().cpu().numpy()
                c = data["classes"][endpoints]
                scores = z[np.arange(len(endpoints)), c].reshape(-1, 2)
                chunks.extend(np.logaddexp(0, scores[:, 1]-scores[:, 0]))
            values[arm] = np.asarray(chunks)
    prefixes = np.array([str(s)[:7] for s in data["stems"][p[:, 0]]])
    unique = sorted(set(prefixes))
    counts = np.array([(prefixes == s).sum() for s in unique])
    comparisons = {}
    for control in ("within", "unlinked"):
        delta = values[control]-values["cross"]
        sums = np.array([delta[prefixes == s].sum() for s in unique])
        if len(unique):
            draws = np.random.default_rng(0).multinomial(len(unique), [1/len(unique)]*len(unique), size=400)
            ci = np.percentile((draws@sums)/(draws@counts), [2.5, 97.5]).tolist()
        else:
            ci = None
        comparisons[control] = dict(mean_gain=float(delta.mean()) if len(delta) else None,
                                   sequence95ci=ci)
    return dict(pairs=len(p), sequence_units=len(unique),
                NLL={a: float(v.mean()) if len(v) else None for a, v in values.items()},
                comparisons=comparisons), values


def evaluate(root, selection, prior_manifest, row_manifest, models, initial, device, seed, calibrator):
    precision(False)
    pred_dir = root/f"predictions-seed{seed}"
    pred_dir.mkdir()
    reports, snapshots = {}, {}
    all_models = dict(strong=initial, **models)
    # One unmeasured cached projection/NMS call, without new detector forwards.
    first = selection["screen"][0]
    frame = load_cache(project_path(PRIOR), prior_manifest[first]["views"][0])
    features = load_cache(root, row_manifest[first]["views"][0])["features_full"]
    with torch.inference_mode():
        native_nms(decoded_with_projection(frame, features, initial, device)[0], tuple(reversed(prior_manifest[first]["size"])))
        for split in ("screen", "confirmation"):
            maps = {a: {} for a in all_models}
            timing = {a: [] for a in all_models}
            records = []
            for index, stem in enumerate(selection[split]):
                entry = prior_manifest[stem]
                metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
                assert file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) == entry["metadata_sha256"]
                records.append(dict(id=index+1, stem=stem, size=entry["size"], objects=metadata["valid_objects"]))
                cache = [load_cache(project_path(PRIOR), v) for v in entry["views"]]
                feat = [load_cache(root, v)["features_full"] for v in row_manifest[stem]["views"]]
                arrays = {}
                keys = list(all_models)
                keys = keys[index % len(keys):]+keys[:index % len(keys)]
                for arm in keys:
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    views = []
                    for view, window in enumerate(entry["windows"]):
                        values, _ = decoded_with_projection(cache[view], feat[view], all_models[arm], device)
                        local = native_nms(values, (window[3]-window[1], window[2]-window[0]))
                        if arm == "strong":
                            np.testing.assert_array_equal(local, cache[view]["original_local"])
                        views.append(map_tile_predictions(local, window) if view else local)
                    output = calibrated_candidates(views, entry["windows"], entry["size"], [], calibrator)[0]
                    torch.cuda.synchronize()
                    timing[arm].append((perf_counter()-tick)*1000)
                    if arm == "strong":
                        ref = project_path(PRIOR)/entry["reference"]["file"]
                        assert file_hash(ref) == entry["reference"]["sha256"]
                        np.testing.assert_array_equal(output, np.load(ref, allow_pickle=False))
                    arrays[arm] = output
                    maps[arm][stem] = diagnostic_filter(output, metadata)
                    arrays["diagnostic_"+arm] = maps[arm][stem]
                path = pred_dir/(stem+".npz")
                np.savez_compressed(path, **arrays)
                snapshots[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path))
                if index == 0 or (index+1) % 8 == 0:
                    print(f"Cross-view rank evaluation seed{seed} {split}: {index+1}/{len(selection[split])}", flush=True)
            ap = {a: coco_size_metrics(records, m, dict(enumerate(NAMES)), reference_size=640) for a, m in maps.items()}
            fp = {}
            for arm in ARMS:
                fp[arm] = recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[arm]),
                                              budgets=(10,), bootstrap_samples=0)[0]
            data, pairs = assembled(root, row_manifest, selection[split])
            mech, losses = mechanism(data, pairs, models, device)
            np.savez_compressed(root/f"mechanism-seed{seed}-{split}.npz",
                                pair_stems=data["stems"][pairs["cross"][:, 0]], **losses)
            reports[split] = dict(AP=ap, fixed_fp=fp, mechanism=mech,
                                 timing_ms=timing, rows=len(data["features"]),
                                 pair_counts={a: len(v) for a, v in pairs.items()})
    save(root/f"metrics-seed{seed}.json", reports)
    save(root/f"predictions-seed{seed}.json", snapshots)
    return reports


def decide(reports, policy):
    checks, differences = {}, {}
    for split, report in reports.items():
        ap, mech = report["AP"], report["mechanism"]
        c = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"])
        d = {}
        for group, control in [("tiny", a) for a in ("strong", "bce", "iou", "within", "unlinked")] + [
            ("small_all", "strong"), ("all", "strong")]:
            key = f"{group}_AP_vs_{control}"
            x, y = ap["cross"][group]["ap50_95"], ap[control][group]["ap50_95"]
            d[key] = None if x is None or y is None else (x-y)*100
            c[key] = d[key] is not None and d[key] >= policy[key+"_min_pp"]
        d["recall"] = report["fixed_fp"]["cross"][0]["delta_recall_pp"]["small_all"]
        c["fixed_FP"] = d["recall"] is not None and d["recall"] >= policy["small_all_recall_FPPI10_vs_strong_min_pp"]
        c["mechanism"] = all(v["mean_gain"] is not None and v["mean_gain"] >= policy[
            "crosspair_NLL_gain_vs_each_rankcontrol_min"] for v in mech["comparisons"].values())
        c["sequence_interval"] = all(v["sequence95ci"] is not None and v["sequence95ci"][0] > policy[
            "crosspair_NLL_gain_sequence95_lower_min_exclusive"] for v in mech["comparisons"].values())
        d["added_processing_ms"] = float(np.mean(report["timing_ms"]["cross"])-np.mean(report["timing_ms"]["strong"]))
        c["processing"] = d["added_processing_ms"] <= policy["cached_added_processing_mean_ms_max"]
        checks[split], differences[split] = c, d
    return dict(passed=all(all(c.values()) for c in checks.values()), checks=checks, differences=differences)


def run(device="0"):
    started = perf_counter()
    root = project_path(OUTPUT)
    if root.exists():
        raise FileExistsError(root)
    registration = read(project_path(REGISTRATION))
    prior = project_path(PRIOR)
    prior_plan, manifest = read(prior/"plan.json"), read(prior/"frames.json")
    closeout = read(project_path("docs/experiments/candidate62_transport_quality_result_20261015.json"))
    assert file_hash(prior/"plan.json") == closeout["artifacts_sha256"]["plan.json"]
    assert file_hash(prior/"frames.json") == closeout["artifacts_sha256"]["frames.json"]
    selection = prior_plan["image_selection"]
    control = frozen_control()
    calibrator = load_calibrator(control)
    sources = [REGISTRATION, "docs/literature/crossview_quality_rank_prior_audit_20261015.json",
               "sdcl/training/crossview_quality_rank.py", "scripts/training/run_crossview_quality_rank.py",
               "scripts/evaluation/verify_crossview_quality_rank.py", "tests/test_crossview_quality_rank.py",
               "sdcl/training/class_margin_probe.py", "sdcl/training/transport_quality.py",
               "sdcl/postprocess/localization.py", "sdcl/postprocess/class_margin_probe.py",
               "sdcl/postprocess/rescue.py", "sdcl/data/geometry.py", "sdcl/evaluation/size_analysis.py",
               "sdcl/evaluation/slicing_screen.py", "sdcl/evaluation/official_details.py", "sdcl/config.py"]
    hashes = {p: file_hash(project_path(p)) for p in sources}
    from ultralytics.nn.modules import head
    from ultralytics.utils import tal, ops, nms
    installed = {inspect.getfile(m): file_hash(inspect.getfile(m)) for m in (head, tal, ops, nms)}
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/"source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(project_path(name), name)
    save(root/"plan.json", dict(registration=registration, image_selection=selection, source_hashes=hashes,
         installed_source_hashes=installed, source_archive_sha256=file_hash(root/"source.zip"),
         prior_plan_sha256=file_hash(prior/"plan.json"), prior_frames_sha256=file_hash(prior/"frames.json"),
         detector_sha256=control["detector"]["sha256"], calibrator_sha256=control["scorer"]["sha256"],
         base_commit="9ab98e1", device="cuda:"+str(device)))
    rows_manifest = extract(root, selection, manifest, device)
    data, pairs = assembled(root, rows_manifest, selection["fit"])
    np.savez_compressed(root/"fit-sampled-rows.npz", **data)
    np.savez_compressed(root/"fit-pairs.npz", **pairs)
    with np.load(root/"initial.npz") as z:
        initial = Projection(z["weight"], z["bias"]).to("cuda:"+str(device))
    audit = precision_audit(initial, data, "cuda:"+str(device))
    save(root/"precision_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("Projection gradient precision audit failed; preserve artifacts.")
    decisions = {}
    for seed in (0, 1, 2):
        models = fit(root, data, pairs, initial, seed, "cuda:"+str(device))
        reports = evaluate(root, selection, manifest, rows_manifest, models, initial,
                           "cuda:"+str(device), seed, calibrator)
        decisions[str(seed)] = decide(reports, registration["policy_per_half"])
        save(root/f"decision-seed{seed}.json", decisions[str(seed)])
        if seed == 0:
            # Replication requires the separately executed independent audit, not just main metrics.
            break
    assert all(file_hash(project_path(p)) == h for p, h in hashes.items())
    assert all(file_hash(p) == h for p, h in installed.items())
    frozen_control()
    decision = ("SEED0_PASS_REQUIRES_INDEPENDENT_AUDIT_BEFORE_FIXED_REPLICATION" if decisions["0"]["passed"]
                else "STOP_THIS_SAMEGT_CROSSVIEW_QUALITY_RANK_PROJECTION_RECIPE")
    result = dict(experiment=registration["experiment"], decision=decision, seeds=decisions, output=str(root),
                  costs=dict(JPEG_decodes=113, frozen_detector_forwards=565, detector_warmup_forwards=1,
                             small_projection_fits=5, epochs_each=8, full_detector_training_steps=0,
                             elapsed_seconds=perf_counter()-started),
                  limits="Reused detector-training proxy; COCO-style diagnostics, not official VisDrone AP. "
                         "Only final classification projections updated; not full detector training or innovation proof.",
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save(root/"result.json", result)
    print(json.dumps(dict(decision=decision, decisions=decisions, costs=result["costs"]), indent=2), flush=True)
    return result
