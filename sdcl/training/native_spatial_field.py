"""Matched neighboring-grid information test for continuous coordinate residuals."""

from copy import deepcopy
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.native_regression_residual import NativeResidual, correct_coordinates
from sdcl.postprocess.native_spatial_field import control_features, field_features
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.geometry_residual_confirmation import prepare as geometry_frames
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate42_native_spatial_field_registration_20261014.json"
OUTPUT = "output/analysis/native-spatial-field-fixed-20261014"
GEOMETRY = "output/analysis/geometry-residual-confirmation-fixed-20261014"
NATIVE = "output/analysis/native-regression-residual-fixed-20261014"
RAW = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("ordered", "mean", "shuffled")
GROUPS = ("tiny", "small_all", "all")


def decoded_grid(extractor, image, window):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    prediction, _ = extractor.predict(image)
    raw = extractor.decoded[0].T
    if raw.shape != (18900, 14):
        raise ValueError("Frozen square960 native shape differs.")
    shape = tuple(int(x*extractor.head.stride[0]) for x in extractor.raw["feats"][0].shape[2:])
    boxes = scale_boxes(shape, xywh2xyxy(raw[:, :4]).clone(), image.shape[:2]).float()
    x1, y1, x2, y2 = window
    # Preserve historical FP16 scaling then FP32 original-image tile offsets.
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, x2-x1)+x1
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, y2-y1)+y1
    return prediction["control"], boxes.cpu().numpy(), raw[:, 4:14].float().cpu().numpy()


def prepare(output, device, frames):
    base = project_path(RAW)
    _, raw_plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    raw_path = base/"raw_views.jsonl"
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    records = {r["stem"]: r for split in ("train", "holdout") for r in raw_plan["records"][split]}
    native = project_path(NATIVE)
    manifest38 = json.loads((native/"features.json").read_text())
    extractor = FrozenViews(device)
    manifest, count, forwards = {}, 0, 0
    (output/"features").mkdir()
    started = perf_counter()
    try:
        with torch.inference_mode():
            for split, rows in frames.items():
                for f in rows:
                    stem = f["stem"]
                    cache = native/manifest38[stem]["file"]
                    if file_hash(cache) != manifest38[stem]["sha256"]:
                        raise ValueError("Frozen chosen native association changed.")
                    with np.load(cache, allow_pickle=False) as saved:
                        chosen = saved["chosen_native"].copy()
                        if not np.array_equal(saved["active"], f["active"]):
                            raise ValueError("Frozen active row mask differs.")
                    if not np.array_equal((chosen >= 0).all(1), f["active"]):
                        raise ValueError("Active rows and chosen associations disagree.")
                    path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    if file_hash(path) != records[stem]["image_sha256"]:
                        raise ValueError("Frozen spatial feature image changed.")
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("Unreadable spatial feature image.")
                    slots = np.zeros((len(chosen), 8, 8), np.float32)
                    neighbors = np.full((len(chosen), 8), -1, int)
                    for view, window in enumerate(raw[stem]["windows"]):
                        x1, y1, x2, y2 = window
                        kept, coordinates, probabilities = decoded_grid(extractor, image[y1:y2, x1:x2].copy(), window)
                        mapped = map_tile_predictions(kept, window) if view else kept
                        if not np.array_equal(mapped, np.array(raw[stem]["views"][view], np.float32).reshape(-1, 6)):
                            raise ValueError("Frozen five-view FP16 kept detector replay differs.")
                        selected = np.flatnonzero(chosen[:, 0] == view)
                        if len(selected):
                            indices = chosen[selected, 1]
                            slots[selected], neighbors[selected] = field_features(
                                coordinates, probabilities, indices, f["reference"][selected, 5].astype(int), f["scale"][selected])
                        forwards += 1
                    extras, permutations = control_features(slots, stem)
                    f["spatial_extra"] = extras
                    saved = output/"features"/(stem+".npz")
                    np.savez_compressed(saved, slots=slots, neighbors=neighbors, chosen_native=chosen,
                                        permutations=permutations, active=f["active"])
                    manifest[stem] = {"file": saved.relative_to(output).as_posix(), "sha256": file_hash(saved),
                                      "split": split, "image_sha256": records[stem]["image_sha256"],
                                      "association_source_sha256": manifest38[stem]["sha256"]}
                    count += 1
                    if count == 1 or count % 32 == 0:
                        print(f"Native spatial fields: {count}/192, {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    normalization = {}
    for arm in ARMS:
        values = np.concatenate([f["spatial_extra"][arm][f["active"]] for f in frames["train"]])
        mean, std = values.mean(0), np.maximum(values.std(0), 1e-6)
        normalization[arm] = {"mean": mean.tolist(), "std": std.tolist()}
        for rows in frames.values():
            for f in rows:
                f.setdefault("spatial_inputs", {})[arm] = np.concatenate(
                    (f["input"][:, :68], (f["spatial_extra"][arm]-mean)/std), -1).astype(np.float32)
    save_json(output/"features.json", manifest)
    save_json(output/"normalization.json", normalization)
    return {"images": count, "forwards_excluding_warmup": forwards,
            "raw_five_view_replay_exact": True, "feature_seconds": perf_counter()-started,
            "raw_views_sha256": file_hash(raw_path)}


def fit(seed, frames, output, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = {a: torch.from_numpy(np.concatenate([f["spatial_inputs"][a][f["active"]] for f in frames])).to(gpu)
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
            for arm, model in models.items():
                perrow = F.smooth_l1_loss(model(inputs[arm][row]), target[row], beta=.02, reduction="none").mean(1)
                groups = [perrow[owned[row] == state].mean() for state in (False, True) if (owned[row] == state).any()]
                loss = torch.stack(groups).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fixed-recipe spatial loss.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(row)/len(target)
        history.append({"epoch": epoch+1, **losses})
    path = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "seed": seed, "epochs": 8}, path)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": path.name, "sha256": file_hash(path), "active_training_rows": len(target),
                    "owned_training_rows": int(owned.sum()), "parameters_each": sum(p.numel() for p in initial.parameters())}


def evaluate(models, frames, output, seed, gpu, controls):
    base = project_path(GEOMETRY)
    prior_manifest = json.loads((base/f"frames-seed{seed}.json").read_text())
    predictions = {a: {} for a in ("regret", "geometry", *ARMS)}
    records, manifest = [], {}
    destination = output/f"predictions-seed{seed}"
    destination.mkdir()
    with torch.inference_mode():
        for index, f in enumerate(frames, 1):
            path = base/prior_manifest[f["stem"]]["file"]
            if file_hash(path) != prior_manifest[f["stem"]]["sha256"]:
                raise ValueError("Frozen geometry prediction trace changed.")
            with np.load(path, allow_pickle=False) as previous:
                if not np.array_equal(previous["regret"], f["reference"]):
                    raise ValueError("Frozen regret reference differs.")
                arms = {"regret": f["reference"], "geometry": previous["geometry"].copy()}
            for arm, model in models.items():
                delta = np.zeros((len(f["reference"]), 4))
                delta[f["active"]] = model(torch.from_numpy(f["spatial_inputs"][arm][f["active"]]).to(gpu)).cpu().numpy()
                arms[arm] = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
                if not np.array_equal(arms[arm][:, 4:], f["reference"][:, 4:]) or not np.array_equal(
                        arms[arm][~f["active"]], f["reference"][~f["active"]]):
                    raise ValueError("Spatial residual changed frozen score/class/inactive rows.")
            for arm, values in arms.items():
                predictions[arm][f["stem"]] = values
            path = destination/(f["stem"]+".npz")
            np.savez_compressed(path, **arms)
            manifest[f["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
            records.append({"id": index, "stem": f["stem"], "size": f["metadata"]["original_size"],
                            "objects": f["metadata"]["valid_objects"]})
    metrics = {a: controls[str(seed)][a] for a in ("regret", "geometry")}
    for arm in ARMS:
        metrics[arm] = coco_size_metrics(records, predictions[arm], dict(enumerate(NAMES)), 640)
        print(f"Native spatial seed{seed}/{arm}: small_allAP{metrics[arm]['small_all']['ap50_95']*100:.6f}", flush=True)
    budgets, _ = recall_at_fp_budgets(records, {"baseline": predictions["regret"], "sdcl": predictions["ordered"]},
                                     reference_size=640, budgets=(10,), bootstrap_samples=0)
    save_json(output/f"frames-seed{seed}.json", manifest)
    save_json(output/f"fixed-fp-budgets-seed{seed}.json", budgets)
    return metrics, budgets


def decide(metrics, budgets, policy):
    gain = {baseline: {g: (metrics["ordered"][g]["ap50_95"]-metrics[baseline][g]["ap50_95"])*100 for g in GROUPS}
            for baseline in ("regret", "geometry", "mean", "shuffled")}
    fp = budgets[0]["delta_recall_pp"]["small_all"]
    checks = {g+"_vs_regret": gain["regret"][g] >= policy[g+"_ap_vs_regret_min_pp"] for g in GROUPS}
    checks.update({g+"_vs_geometry": gain["geometry"][g] >= policy[g+"_ap_vs_geometry_min_pp"]
                   for g in ("tiny", "small_all")})
    checks.update({"small_all_vs_"+b: gain[b]["small_all"] >= policy["small_all_ap_vs_"+b+"_min_pp"]
                   for b in ("mean", "shuffled")})
    checks["fppi10"] = fp >= policy["small_all_recall_fppi10_vs_regret_min_pp"]
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"AP_gains_pp": gain, "small_all_recall_fppi10_vs_regret_pp": fp}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Frozen geometry replay and feature extraction require available CUDA.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/postprocess/native_spatial_field.py", "sdcl/training/native_spatial_field.py",
             "scripts/training/run_native_spatial_field.py", "sdcl/training/geometry_residual_confirmation.py",
             "sdcl/postprocess/native_regression_residual.py", "sdcl/training/class_margin_probe.py",
             "sdcl/training/side_composition_pilot.py", "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    record39, _ = verify_record(project_path(GEOMETRY), "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json")
    frames, prior, heads, _, seed0 = geometry_frames()
    gpu = torch.device("cuda:"+str(device))
    seed0 = seed0.to(gpu).eval()
    with torch.inference_mode():
        for f in frames["holdout"]:
            delta = np.zeros((len(f["reference"]), 4))
            delta[f["active"]] = seed0(torch.from_numpy(f["input"][f["active"]]).to(gpu)).cpu().numpy()
            replay = correct_coordinates(f["reference"], delta, f["scale"], f["active"], f["metadata"]["original_size"])
            if not np.array_equal(replay, f["seed0_geometry"]):
                raise ValueError("Frozen geometry seed0 GPU replay differs.")
    save_json(output/"plan.json", {"registration": registration, "source_hashes": hashes, "heads": heads,
                                 "source_archive_sha256": file_hash(output/"source.zip"), "image_selection": prior["image_selection"],
                                 "prior_result_sha256": file_hash(project_path(
                                     "docs/experiments/candidate39_geometry_residual_confirmation_result_20261014.json"))})
    features = prepare(output, device, frames)
    controls = json.loads((project_path(GEOMETRY)/"metrics.json").read_text())
    metrics, training, decisions = {}, {}, {}
    for seed in (0, 1, 2):
        models, history = fit(seed, frames["train"], output, gpu)
        values, budgets = evaluate(models, frames["holdout"], output, seed, gpu, controls)
        metrics[str(seed)], training[str(seed)] = values, history
        decisions[str(seed)] = decide(values, budgets, registration["policy"])
        print(f"Native spatial seed{seed}: {json.dumps(decisions[str(seed)])}, {perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    averages = {b: {g: float(np.mean([v["values"]["AP_gains_pp"][b][g] for v in decisions.values()])) for g in GROUPS}
                for b in ("regret", "geometry", "mean", "shuffled")}
    averaged_metrics = {a: {g: {"ap50_95": float(np.mean([v[a][g]["ap50_95"] for v in metrics.values()]))} for g in GROUPS}
                        for a in ("ordered", "regret", "geometry", "mean", "shuffled")}
    min_fp = min(v["values"]["small_all_recall_fppi10_vs_regret_pp"] for v in decisions.values())
    mean_checks = decide(averaged_metrics, [{"delta_recall_pp": {"small_all": min_fp}}], registration["policy"])
    passed = (len(decisions) == 3 and sum(v["passed"] for v in decisions.values()) >=
              registration["policy"]["repeat_min_fully_passing_seeds"] and mean_checks["passed"] and
              all(v["checks"]["all_vs_regret"] and v["checks"]["fppi10"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_ORDERED_NATIVE_FIELD_AUDIT_ALLOWED" if passed else
               "STOP_THIS_ORDERED_NATIVE_SPATIAL_FIELD_RECIPE", "per_seed": decisions, "mean_AP_gains_pp": averages,
               "scope": "Repeated detector-training imagery and ordinary residual heads; no official AP or innovation claim."}
    for name, value in (("metrics", metrics), ("training", training), ("decision", verdict)):
        save_json(output/(name+".json"), value)
    save_json(output/"verification.json", {"features": features, "detector_training_steps": 0,
                                          "heads_per_seed": 3, "geometry_seed0_all64_GPU_replay_exact": True,
                                          "geometry39_STOP_preserved": record39["decision"]["decision"],
                                          "scores_classes_count_order_inactive_preserved": True,
                                          "train_only_normalization_and_original_owners": True,
                                          "neighbors_and_associations_without_GT": True,
                                          "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                                          "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
