"""Fixed native-pixel difference pilot before official accuracy investment."""

from copy import deepcopy
from itertools import combinations
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.censored_retrieval import score_image
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, file_hash
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.native_boundary import action_differences, PixelResidual, shuffle_differences
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate28_native_boundary_registration_20261014.json"
OUTPUT = "output/analysis/native-boundary-pilot-fixed-20261014"
MENU = "output/analysis/action-order-sensitivity-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
ARMS = ("geometry_residual", "interior_difference", "boundary_difference", "boundary_shuffled")
GROUPS = ("all", "tiny", "small_all")
PAIRS = np.array(list(combinations(range(6), 2)))


def objective(utility, ious, valid, owned, reference):
    pair = torch.as_tensor(PAIRS, device=utility.device)
    delta = ious[:, pair[:, 0]] - ious[:, pair[:, 1]]
    mask = valid[:, pair[:, 0]] & valid[:, pair[:, 1]] & (delta.abs() > .01)
    loss = F.softplus(-delta.sign() * (utility[:, pair[:, 0]] - utility[:, pair[:, 1]]) / .05)
    positive = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
    alternative = valid.clone()
    alternative.scatter_(1, reference[:, None], False)
    keep = utility.gather(1, reference[:, None])
    negative = (F.softplus((utility - keep) / .05) * alternative).sum(1) / alternative.sum(1).clamp_min(1)
    active = torch.where(owned, mask.any(1), alternative.any(1))
    values = torch.where(owned, positive, negative)
    return (values * active).sum() / active.sum().clamp_min(1)


def prepare(output):
    prior, base = project_path(MENU), project_path(BASE)
    _, plan = verify_record(prior, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    _, base_plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    heads, _, _, provenance = frozen_heads()
    if provenance != plan["heads"]:
        raise ValueError("Frozen utility head changed.")
    manifest = json.loads((prior / "frames.json").read_text())
    snapshots = {r["stem"]: r for rows in base_plan["records"].values() for r in rows}
    source = {split: [] for split in ("train", "holdout")}
    features = {}
    (output / "pixels").mkdir()
    with torch.inference_mode():
        for split, stems in plan["image_selection"].items():
            for index, stem in enumerate(stems, 1):
                snapshot = manifest[stem]
                path = prior / snapshot["file"]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                if (file_hash(path) != snapshot["sha256"] or
                        file_hash(image_path) != snapshots[stem]["image_sha256"] or
                        file_hash(meta_path) != snapshot["metadata_sha256"]):
                    raise ValueError("Frozen image/menu/annotations changed.")
                with np.load(path, allow_pickle=False) as c:
                    frame = {k: c[k].copy() for k in c.files}
                geometry = torch.from_numpy(frame["features"])
                frame["base_utility"] = heads["uniform"](geometry).numpy()
                if not np.array_equal(select_actions(frame["base_utility"], frame["valid"], "absolute"),
                                      frame["uniform"]):
                    raise ValueError("Zero-residual frozen uniform replay differs.")
                frame["regret"] = select_actions(heads["regret"](geometry).numpy(), frame["valid"], "absolute")
                active = (frame["candidates"][:, 4] >= .05) & (frame["valid"].sum(1) >= 2)
                rows = np.flatnonzero(active)
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Frozen native image unreadable.")
                pixel = {name: action_differences(image, frame["boxes"][rows], frame["uniform"][rows], mode)
                         for name, mode in (("boundary", "boundary"), ("interior", "interior"))}
                frame.update(stem=stem, metadata=json.loads(meta_path.read_text()),
                             active_rows=rows, **pixel)
                target = output / "pixels" / f"{stem}.npz"
                np.savez_compressed(target, active_rows=rows, **{
                    k: v.astype(np.float16) for k, v in pixel.items()})
                # Train/replay the exact serialized representation, not a higher precision shadow.
                with np.load(target, allow_pickle=False) as c:
                    for key in pixel:
                        frame[key] = c[key].astype(np.float32)
                features[stem] = {"file": target.relative_to(output).as_posix(),
                                  "sha256": file_hash(target), "menu_frame_sha256": snapshot["sha256"]}
                source[split].append(frame)
                if index == 1 or index % 32 == 0 or index == len(stems):
                    print(f"Native pixel preparation {split}: {index}/{len(stems)}", flush=True)
    return source, features, plan, provenance


def visual_input(arm, boundary, interior, shuffled):
    if arm == "geometry_residual":
        return torch.zeros_like(boundary)
    return {"boundary_difference": boundary, "interior_difference": interior,
            "boundary_shuffled": shuffled}[arm]


def fit_seed(seed, frames, gpu, output):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    collected = {k: [] for k in ("geometry", "base", "ious", "valid", "owned", "reference",
                                 "boundary", "interior", "shuffled")}
    for frame in frames["train"]:
        rows = frame["active_rows"]
        for key, values in {
            "geometry": frame["features"][rows], "base": frame["base_utility"][rows],
            "ious": frame["ious"][rows, 0], "valid": frame["valid"][rows],
            "owned": frame["owner"][rows] >= 0, "reference": frame["uniform"][rows],
            "boundary": frame["boundary"], "interior": frame["interior"],
            "shuffled": shuffle_differences(frame["boundary"], frame["stem"], rows),
        }.items():
            collected[key].append(values)
    tensors = {k: torch.from_numpy(np.concatenate(v)).to(gpu) for k, v in collected.items()}
    initial = PixelResidual().to(gpu)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.0002, weight_decay=.0001)
                  for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(tensors["owned"]),
            generator=torch.Generator().manual_seed(seed * 100 + epoch)).to(gpu)
        sums = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            for arm, model in models.items():
                pixels = visual_input(arm, tensors["boundary"][rows], tensors["interior"][rows],
                                      tensors["shuffled"][rows])
                utility = tensors["base"][rows] + model(tensors["geometry"][rows], pixels)
                loss = objective(utility, tensors["ious"][rows], tensors["valid"][rows],
                                 tensors["owned"][rows], tensors["reference"][rows])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; do not tune the fixed recipe.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                sums[arm] += float(loss.detach()) * len(rows)
        history.append({"epoch": epoch+1, **{k: v / len(order) for k, v in sums.items()}})
    checkpoint = output / f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                           for a, m in models.items()},
                "seed": seed, "epochs": 8, "registration_sha256": file_hash(project_path(REGISTRATION))},
               checkpoint)
    save_json(output / f"losses-seed{seed}.json", history)
    return models, {"parameters_each": sum(p.numel() for p in initial.parameters()),
                     "active_training_rows": len(tensors["owned"]),
                     "checkpoint": checkpoint.name, "sha256": file_hash(checkpoint)}


def evaluate_heads(models, frames, gpu, output, seed):
    arms = ("uniform", "regret", *ARMS)
    totals = {a: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                  "fp50": 0, "moves": 0} for a in arms}
    choices = {}
    with torch.inference_mode():
        for frame in frames:
            rows = frame["active_rows"]
            geometry = torch.from_numpy(frame["features"][rows]).to(gpu)
            boundary, interior = (torch.from_numpy(frame[k]).to(gpu) for k in ("boundary", "interior"))
            shuffled = torch.from_numpy(shuffle_differences(frame["boundary"], frame["stem"], rows)).to(gpu)
            selected = {"uniform": frame["uniform"], "regret": frame["regret"]}
            for arm, model in models.items():
                utility = frame["base_utility"].copy()
                utility[rows] += model(geometry, visual_input(arm, boundary, interior, shuffled)).cpu().numpy()
                selected[arm] = select_actions(utility, frame["valid"], "absolute")
            choices[frame["stem"]] = {a: s.tolist() for a, s in selected.items()}
            objects = frame["metadata"]["valid_objects"]
            gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
            classes = np.asarray([o["class_id"] for o in objects], int)
            areas = box_areas(gt) * (640 / max(frame["metadata"]["original_size"])) ** 2
            for arm, action in selected.items():
                prediction = apply_actions(frame["candidates"], frame["boxes"], action)
                scored = score_image(prediction, np.zeros(len(prediction), bool), gt, classes, areas)
                for g, counts in scored["groups"].items():
                    for k, v in counts.items():
                        totals[arm]["groups"][g][k] += v
                totals[arm]["fp50"] += scored["fp50"]
                totals[arm]["moves"] += int((action != frame["uniform"]).sum())
    save_json(output / f"choices-seed{seed}.json", choices)
    return {a: {
        **{g + "_ten_threshold_recall_percent": c["tp_ten"] / max(c["gt"] * 10, 1) * 100
           for g, c in counts["groups"].items()},
        "all_fppi50": counts["fp50"] / len(frames), "counts": counts,
    } for a, counts in totals.items()}


def decide(metrics, policy):
    local, key = metrics["boundary_difference"], "small_all_ten_threshold_recall_percent"
    comparisons = {a: local[key] - metrics[a][key]
                   for a in ("uniform", "regret", "geometry_residual", "interior_difference", "boundary_shuffled")}
    tiny = local["tiny_ten_threshold_recall_percent"] - metrics["uniform"]["tiny_ten_threshold_recall_percent"]
    fp = local["all_fppi50"] - metrics["uniform"]["all_fppi50"]
    checks = {
        "tiny_gain": tiny >= policy["tiny_ten_threshold_vs_uniform_min_pp"],
        "small_gain": comparisons["uniform"] >= policy["small_all_ten_threshold_vs_uniform_min_pp"],
        **{name + "_gain": comparisons[arm] > policy[
            "small_all_ten_threshold_vs_" + name + "_min_exclusive_pp"]
           for name, arm in (("regret", "regret"), ("geometry", "geometry_residual"),
                             ("interior", "interior_difference"), ("shuffled", "boundary_shuffled"))},
        "fp_budget": fp <= policy["all_fppi50_vs_uniform_max_increase"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"small_all_gains_pp": comparisons, "tiny_vs_uniform_pp": tiny,
                       "all_fppi50_increase": fp}}


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
    paths = (REGISTRATION, "sdcl/postprocess/native_boundary.py", "sdcl/training/native_boundary_pilot.py",
             "scripts/training/run_native_boundary_pilot.py", "sdcl/postprocess/localization.py",
             "sdcl/evaluation/censored_retrieval.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, manifest, plan, provenance = prepare(output)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": plan["image_selection"],
        "source_archive_sha256": file_hash(output / "source.zip"), "heads": provenance,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")),
    })
    save_json(output / "frames.json", manifest)
    gpu = torch.device("cuda:" + str(device))
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, saved = fit_seed(seed, frames, gpu, output)
        values = evaluate_heads(models, frames["holdout"], gpu, output, seed)
        metrics[str(seed)], training[str(seed)] = values, saved
        decisions[str(seed)] = decide(values, registration["policy"])
        print(f"Native boundary seed{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    mean_gain = float(np.mean([v["values"]["small_all_gains_pp"]["uniform"] for v in decisions.values()]))
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_matched_seeds"] and
              mean_gain >= registration["policy"]["small_all_ten_threshold_vs_uniform_min_pp"] and
              all(v["checks"]["fp_budget"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_NATIVE_BOUNDARY_AUDIT_ALLOWED" if passed else
               "STOP_THIS_NATIVE_PIXEL_DIFFERENCE_PILOT", "per_seed": decisions,
               "seeds_executed": len(decisions), "positive_seeds": positive,
               "scope": "Reused train head-holdout detection proxy, not AP or innovation proof."}
    save_json(output / "metrics.json", metrics)
    save_json(output / "training.json", training)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "fitting_images": len(frames["train"]), "head_holdout_images": len(frames["holdout"]),
        "detector_forwards": 0, "detector_training_steps": 0,
        "initial_uniform_choice_replay_exact": True, "same_residual_initialization": True,
        "no_GT_in_pixel_features": True, "pixel_cache_fp16_used_for_fit_and_evaluation": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
