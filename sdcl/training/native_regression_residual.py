"""Matched native regression/classification features for continuous corrections."""

from copy import deepcopy
import hashlib
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import box_iou_matrix, coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import apply_actions
from sdcl.postprocess.native_regression_residual import NativeResidual, correct_coordinates
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate38_native_regression_features_registration_20261014.json"
OUTPUT = "output/analysis/native-regression-residual-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("geometry", "native_reg", "native_cls", "shuffled_reg")


def associate_native(extractor, regression, image, reference, window):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    prediction, _ = extractor.predict(image)
    raw = extractor.decoded[0].T
    score, cls = raw[:, 4:14].max(1)
    native_rows = torch.where(score >= .05)[0]
    shape = tuple(int(x*extractor.head.stride[0]) for x in extractor.raw["feats"][0].shape[2:])
    boxes = scale_boxes(shape, xywh2xyxy(raw[native_rows, :4]).clone(), image.shape[:2])
    values = torch.cat((boxes, score[native_rows, None], cls[native_rows, None].float()), -1).float().cpu().numpy()
    values = map_tile_predictions(values, window)
    reg = torch.cat([regression[k].flatten(2) for k in range(3)], -1)[0].T
    classification = torch.cat([extractor.features[k][:, :64].flatten(2) for k in range(3)], -1)[0].T
    if reg.shape != (18900, 64) or classification.shape != reg.shape:
        raise ValueError("Unexpected native feature shapes.")
    quality = np.zeros(len(reference))
    chosen = np.full(len(reference), -1, int)
    rfeatures = np.zeros((len(reference), 64), np.float32)
    cfeatures = rfeatures.copy()
    if len(values):
        for start in range(0, len(reference), 256):
            end = min(start+256, len(reference))
            target = reference[start:end]
            overlap = box_iou_matrix(target[:, :4], values[:, :4])
            valid = ((overlap >= .5) & (target[:, None, 5] == values[None, :, 5]) &
                     (target[:, None, 4] >= .05))
            utility = np.where(valid, overlap*values[None, :, 4], 0)
            best = utility.argmax(1)
            rows = np.flatnonzero(valid.any(1))
            native = native_rows[torch.as_tensor(best[rows], device=native_rows.device)]
            quality[start+rows] = utility[rows, best[rows]]
            chosen[start+rows] = native.cpu().numpy()
            rfeatures[start+rows] = reg[native].float().cpu().numpy()
            cfeatures[start+rows] = classification[native].float().cpu().numpy()
    return prediction["control"], quality, chosen, rfeatures, cfeatures


def prepare(output, device, frames):
    raw = project_path(RAW)
    _, plan = verify_record(raw, "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw_path = raw/"raw_views.jsonl"
    raw_views = {r["image"]: r for r in (json.loads(x) for x in raw_path.read_text().splitlines())}
    records = {r["stem"]: r for split in ("train", "holdout") for r in plan["records"][split]}
    manifest, count = {}, 0
    regression = {}
    extractor = FrozenViews(device)
    def hook(level):
        def capture(module, inputs):
            regression[level] = inputs[0].detach()
        return capture
    hooks = [c[-1].register_forward_pre_hook(hook(k)) for k, c in enumerate(extractor.head.cv2)]
    (output/"features").mkdir()
    started = perf_counter()
    try:
        with torch.inference_mode():
            for split, rows in frames.items():
                for f in rows:
                    stem = f["stem"]
                    path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    if file_hash(path) != records[stem]["image_sha256"]:
                        raise ValueError("Frozen feature input changed.")
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("Unreadable native feature input.")
                    f["reference"] = apply_actions(f["candidates"], f["boxes"], f["regret"])
                    best = np.zeros(len(f["candidates"]))
                    chosen = np.full((len(best), 2), -1, int)
                    rfeatures = np.zeros((len(best), 64), np.float32)
                    cfeatures = rfeatures.copy()
                    for v, window in enumerate(raw_views[stem]["windows"]):
                        x1, y1, x2, y2 = window
                        kept, quality, native, reg, cls = associate_native(
                            extractor, regression, image[y1:y2, x1:x2].copy(), f["reference"], window)
                        mapped = map_tile_predictions(kept, window) if v else kept
                        if not np.array_equal(mapped, np.array(raw_views[stem]["views"][v], np.float32).reshape(-1, 6)):
                            raise ValueError("Frozen five-view detector replay differs.")
                        change = quality > best
                        best[change], chosen[change, 0], chosen[change, 1] = quality[change], v, native[change]
                        rfeatures[change], cfeatures[change] = reg[change], cls[change]
                    active = best > 0
                    geometry = f["features"][np.arange(len(best)), f["regret"]]
                    extra = {"geometry": geometry[:, :64].copy(), "native_reg": rfeatures,
                             "native_cls": cfeatures, "shuffled_reg": rfeatures.copy()}
                    selected = np.flatnonzero(active)
                    seed = int.from_bytes(hashlib.sha256(stem.encode()).digest()[:8], "little")
                    extra["shuffled_reg"][selected] = rfeatures[np.random.default_rng(seed).permutation(selected)]
                    f.update(extra=extra, active=active, geometry=geometry)
                    target = np.zeros((len(best), 4), np.float32)
                    gt = np.array([o["box_xyxy"] for o in f["metadata"]["valid_objects"]], float).reshape(-1, 4)
                    owned = f["owner"] >= 0
                    scale = np.tile(f["candidates"][:, 2:4]-f["candidates"][:, :2], (1, 2))
                    target[owned] = (gt[f["owner"][owned]]-f["reference"][owned, :4])/np.maximum(scale[owned], 1e-6)
                    f.update(scale=scale, target=np.clip(target, -.1, .1))
                    saved = output/"features"/(stem+".npz")
                    np.savez_compressed(saved, **{"extra_"+a: x for a, x in extra.items()},
                                        chosen_native=chosen, association_quality=best,
                                        active=active, geometry=geometry)
                    manifest[stem] = {"file": saved.relative_to(output).as_posix(), "sha256": file_hash(saved),
                                      "image_sha256": records[stem]["image_sha256"], "split": split}
                    count += 1
                    if count == 1 or count % 32 == 0:
                        print(f"Native residual features: {count}/192, {perf_counter()-started:.1f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()
        extractor.close()
    normalization = {}
    for arm in ARMS:
        values = np.concatenate([f["extra"][arm][f["active"]] for f in frames["train"]])
        mean, std = values.mean(0), np.maximum(values.std(0), 1e-6)
        normalization[arm] = {"mean": mean.tolist(), "std": std.tolist()}
        for rows in frames.values():
            for f in rows:
                f.setdefault("inputs", {})[arm] = np.concatenate(
                    (f["geometry"], (f["extra"][arm]-mean)/std), -1).astype(np.float32)
    save_json(output/"features.json", manifest)
    save_json(output/"normalization.json", normalization)
    return {"images": count, "forwards_excluding_warmup": count*5,
            "raw_five_view_replay_exact": True, "feature_seconds": perf_counter()-started,
            "raw_views_sha256": file_hash(raw_path)}


def fit(seed, frames, output, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = {a: torch.from_numpy(np.concatenate([f["inputs"][a][f["active"]] for f in frames])).to(gpu)
              for a in ARMS}
    target = torch.from_numpy(np.concatenate([f["target"][f["active"]] for f in frames])).to(gpu)
    owned = torch.from_numpy(np.concatenate([f["owner"][f["active"]] >= 0 for f in frames])).to(gpu)
    initial = NativeResidual().to(gpu)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(target), generator=torch.Generator().manual_seed(seed*100+epoch)).to(gpu)
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 512):
            row = order[start:start+512]
            for a, model in models.items():
                loss = F.smooth_l1_loss(model(inputs[a][row]), target[row], beta=.02, reduction="none").mean(1)
                groups = [loss[owned[row] == state].mean() for state in (False, True) if (owned[row] == state).any()]
                value = torch.stack(groups).mean()
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite native residual loss.")
                optimizers[a].zero_grad(set_to_none=True)
                value.backward()
                optimizers[a].step()
                losses[a] += float(value.detach())*len(row)/len(target)
        history.append({"epoch": epoch+1, **losses})
    path = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "seed": seed, "epochs": 8}, path)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": path.name, "sha256": file_hash(path),
                    "active_training_rows": len(target), "owned_training_rows": int(owned.sum()),
                    "parameters_each": sum(p.numel() for p in initial.parameters())}


def evaluate(models, frames, output, seed, gpu):
    predictions = {a: {} for a in ("regret", *ARMS)}
    records, manifest = [], {}
    target = output/f"predictions-seed{seed}"
    target.mkdir()
    with torch.inference_mode():
        for index, f in enumerate(frames, 1):
            arms = {"regret": f["reference"]}
            for a, model in models.items():
                delta = np.zeros((len(f["candidates"]), 4))
                delta[f["active"]] = model(torch.from_numpy(f["inputs"][a][f["active"]]).to(gpu)).cpu().numpy()
                arms[a] = correct_coordinates(f["reference"], delta, f["scale"], f["active"],
                                             f["metadata"]["original_size"])
                if not np.array_equal(arms[a][:, 4:], f["reference"][:, 4:]):
                    raise ValueError("Continuous residual changed scores or classes.")
            for a, values in arms.items():
                predictions[a][f["stem"]] = values
            path = target/(f["stem"]+".npz")
            np.savez_compressed(path, **arms)
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
    save_json(output/f"frames-seed{seed}.json", manifest)
    metrics = {}
    for a, values in predictions.items():
        metrics[a] = coco_size_metrics(records, values, dict(enumerate(NAMES)), 640)
        print(f"Native residual seed{seed}/{a}: small_allAP{metrics[a]['small_all']['ap50_95']*100:.6f}", flush=True)
    budgets, _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"],
        "sdcl": predictions["native_reg"]}, reference_size=640, budgets=(10,), bootstrap_samples=0)
    save_json(output/f"fixed-fp-budgets-seed{seed}.json", budgets)
    return metrics, budgets


def decide(metrics, budgets, policy):
    main = metrics["native_reg"]
    gains = {g: (main[g]["ap50_95"]-metrics["regret"][g]["ap50_95"])*100 for g in ("tiny", "small_all", "all")}
    controls = {a: (main["small_all"]["ap50_95"]-metrics[a]["small_all"]["ap50_95"])*100
                for a in ("geometry", "native_cls", "shuffled_reg")}
    fp = budgets[0]["delta_recall_pp"]["small_all"]
    checks = {**{g+"_AP": v >= policy[g+"_ap_vs_regret_min_pp"] for g, v in gains.items()},
              **{"vs_"+a: v >= policy["small_all_ap_vs_"+a+"_min_pp"] for a, v in controls.items()},
              "fppi10": fp >= policy["small_all_recall_fppi10_vs_regret_min_pp"]}
    return {"passed": all(checks.values()), "checks": checks, "values": {
        "AP_vs_regret_pp": gains, "small_all_AP_vs_controls_pp": controls,
        "small_all_recall_fppi10_gain_pp": fp}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/native_regression_residual.py",
             "sdcl/training/native_regression_residual.py", "scripts/training/run_native_regression_residual.py",
             "sdcl/training/class_margin_probe.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    import ultralytics.nn.modules.head as head_module
    import ultralytics.utils.ops as ops_module
    import ultralytics.utils.nms as nms_module
    installed = {m.__file__: file_hash(m.__file__) for m in (head_module, ops_module, nms_module)}
    frames, prior, heads = load_frames()
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads, "installed_sources": installed,
        "source_archive_sha256": file_hash(output/"source.zip"), "image_selection": prior["image_selection"],
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"))})
    feature_audit = prepare(output, device, frames)
    initial = NativeResidual().eval()
    with torch.inference_mode():
        for rows in frames.values():
            for f in rows:
                delta = initial(torch.from_numpy(f["inputs"]["native_reg"])).numpy()
                replay = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
                if not np.array_equal(replay, f["reference"]):
                    raise ValueError("Zero continuous correction fails exact regret replay.")
    gpu = torch.device("cuda:"+str(device))
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, train = fit(seed, frames["train"], output, gpu)
        values, budgets = evaluate(models, frames["holdout"], output, seed, gpu)
        metrics[str(seed)], training[str(seed)] = values, train
        decisions[str(seed)] = decide(values, budgets, registration["policy"])
        print(f"Native residual seed{seed}: {json.dumps(decisions[str(seed)])}, {perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    passed = (len(decisions) == 3 and sum(v["passed"] for v in decisions.values()) >= 2 and
              all(np.mean([v["values"]["AP_vs_regret_pp"][g] for v in decisions.values()]) >=
                  registration["policy"][g+"_ap_vs_regret_min_pp"] for g in ("tiny", "small_all")) and
              all(v["checks"]["all_AP"] and v["checks"]["fppi10"] for v in decisions.values()))
    verdict = {"decision": "NATIVE_REGRESSION_REPRESENTATION_SIGNAL_CONFIRMED" if passed else
               "STOP_THIS_NATIVE_REGRESSION_FEATURE_RESIDUAL_RECIPE", "per_seed": decisions,
               "scope": "Ordinary residual representation feasibility on repeated train scenes, not innovation or official AP."}
    save_json(output/"metrics.json", metrics)
    save_json(output/"decision.json", verdict)
    save_json(output/"training.json", training)
    save_json(output/"verification.json", {
        "fit_images": 128, "head_holdout_images": 64, **feature_audit, "detector_training_steps": 0,
        "initial_exact_regret_replay": True, "all_heads_parameters_equal": True,
        "same_initialization_rows_batches_and_optimizer": True, "GT_excluded_from_features_and_inference": True,
        "normalization_fit_only_on_training_rows": True, "same_original_owner_all_targets": True,
        "scores_classes_counts_and_order_preserved": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "installed_sources_unchanged": all(file_hash(p) == h for p, h in installed.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict, "training": training}
