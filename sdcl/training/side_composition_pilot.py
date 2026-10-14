"""Fixed low-cost same-head whole-box versus composed-side localization pilot."""

from copy import deepcopy
from itertools import combinations
import json
import os
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.censored_retrieval import score_image
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, file_hash
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.postprocess.side_composition import choose_coordinates, SideResidual
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate29_side_composition_registration_20261014.json"
OUTPUT = "output/analysis/side-composition-pilot-fixed-20261014"
MENU = "output/analysis/action-order-sensitivity-fixed-20261014"
ARMS = ("gain_whole", "gain_composed", "rank_whole", "rank_composed")
GROUPS = ("all", "tiny", "small_all")
PAIRS = np.array(list(combinations(range(6), 2)))


def objective(utility, base, errors, valid, owned, reference, mode):
    n = len(utility)
    row = torch.arange(n, device=utility.device)
    anchor_error = errors[row, reference]
    anchor_utility = base[row, reference]
    if mode == "gain":
        targets = anchor_utility[:, None, None] + anchor_error[:, None] - errors
        loss = F.smooth_l1_loss(utility, targets, beta=.1, reduction="none")
        positive = (loss * valid[:, :, None]).sum((1, 2)) / (valid.sum(1) * 4).clamp_min(1)
        informative = valid.any(1)
    elif mode == "rank":
        pairs = torch.as_tensor(PAIRS, device=utility.device)
        difference = errors[:, pairs[:, 1]] - errors[:, pairs[:, 0]]
        comparison = (valid[:, pairs[:, 0], None] & valid[:, pairs[:, 1], None] &
                      (difference.abs() > .01))
        loss = F.softplus(-difference.sign() *
                         (utility[:, pairs[:, 0]] - utility[:, pairs[:, 1]]) / .05)
        positive = (loss * comparison).sum((1, 2)) / comparison.sum((1, 2)).clamp_min(1)
        informative = comparison.any((1, 2))
    else:
        raise ValueError("Unknown fixed side objective.")
    other = valid.clone()
    other.scatter_(1, reference[:, None], False)
    keep = utility[row, reference]
    negative = (F.softplus((utility - keep[:, None]) / .05) * other[:, :, None]).sum((1, 2))
    negative /= (other.sum(1) * 4).clamp_min(1)
    active = torch.where(owned, informative, other.any(1))
    value = torch.where(owned, positive, negative)
    return (value * active).sum() / active.sum().clamp_min(1)


def load_frames():
    prior = project_path(MENU)
    _, plan = verify_record(prior, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    models, _, _, provenance = frozen_heads()
    if provenance != plan["heads"]:
        raise ValueError("Historical head provenance differs.")
    manifest = json.loads((prior / "frames.json").read_text())
    result = {"train": [], "holdout": []}
    with torch.inference_mode():
        for split, stems in plan["image_selection"].items():
            for stem in stems:
                snapshot = manifest[stem]
                frame_path = prior / snapshot["file"]
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                if file_hash(frame_path) != snapshot["sha256"] or file_hash(meta_path) != snapshot["metadata_sha256"]:
                    raise ValueError("Frozen coordinate frame/labels changed.")
                with np.load(frame_path, allow_pickle=False) as c:
                    frame = {k: c[k].copy() for k in c.files}
                frame["metadata"] = json.loads(meta_path.read_text())
                frame["stem"] = stem
                x = torch.from_numpy(frame["features"])
                frame["base_utility"] = models["uniform"](x).numpy()
                if not np.array_equal(select_actions(frame["base_utility"], frame["valid"], "absolute"),
                                      frame["uniform"]):
                    raise ValueError("Frozen uniform utility replay differs.")
                initial = np.repeat(frame["base_utility"][:, :, None], 4, axis=2)
                for composed in (False, True):
                    replay, _ = choose_coordinates(frame["candidates"], frame["boxes"], frame["valid"],
                                                    frame["uniform"], initial, composed)
                    if not np.array_equal(replay, frame["baseline"]):
                        raise ValueError("Initial whole/side readout differs from frozen uniform.")
                frame["regret"] = select_actions(models["regret"](x).numpy(), frame["valid"], "absolute")
                gt = np.asarray([o["box_xyxy"] for o in frame["metadata"]["valid_objects"]], float).reshape(-1, 4)
                owned = frame["owner"] >= 0
                target = np.zeros((len(frame["candidates"]), 4))
                target[owned] = gt[frame["owner"][owned]]
                width_height = frame["candidates"][:, 2:4] - frame["candidates"][:, :2]
                scale = np.tile(width_height, (1, 2))
                frame["errors"] = (np.abs(frame["boxes"] - target[:, None]) /
                                   np.maximum(scale[:, None], 1e-6)).astype(np.float32)
                result[split].append(frame)
    return result, plan, provenance


def fit(seed, frames, output, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    collected = {k: [] for k in ("features", "base", "errors", "valid", "owned", "reference")}
    for frame in frames:
        active = (frame["candidates"][:, 4] >= .05) & (frame["valid"].sum(1) >= 2)
        values = {
            "features": frame["features"], "base": frame["base_utility"], "errors": frame["errors"],
            "valid": frame["valid"], "owned": frame["owner"] >= 0, "reference": frame["uniform"],
        }
        for k, v in values.items():
            collected[k].append(v[active])
    tensors = {k: torch.from_numpy(np.concatenate(v)).to(gpu) for k, v in collected.items()}
    initial = SideResidual().to(gpu)
    models = {a: deepcopy(initial) for a in ("gain", "rank")}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.0002, weight_decay=.0001)
                  for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(tensors["owned"]),
            generator=torch.Generator().manual_seed(seed * 100 + epoch)).to(gpu)
        sums = dict.fromkeys(models, 0.)
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            for mode, model in models.items():
                utility = tensors["base"][rows, :, None] + model(tensors["features"][rows])
                loss = objective(utility, tensors["base"][rows], tensors["errors"][rows],
                                 tensors["valid"][rows], tensors["owned"][rows], tensors["reference"][rows], mode)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fixed-recipe side loss.")
                optimizers[mode].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[mode].step()
                sums[mode] += float(loss.detach()) * len(rows)
        history.append({"epoch": epoch+1, **{k: v / len(order) for k, v in sums.items()}})
    checkpoint = output / f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                           for a, m in models.items()}, "seed": seed, "epochs": 8}, checkpoint)
    save_json(output / f"losses-seed{seed}.json", history)
    return models, {"parameters_each": sum(p.numel() for p in initial.parameters()),
                     "active_training_rows": len(tensors["owned"]),
                     "checkpoint": checkpoint.name, "sha256": file_hash(checkpoint)}


def evaluate(models, frames, output, seed, gpu):
    arms = ("uniform", "regret", *ARMS)
    totals = {a: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                  "fp50": 0, "moves": 0, "invalid_fallback": 0,
                  "new_compositions_at_score025": 0} for a in arms}
    artifacts = {}
    predictions_path = output / f"predictions-seed{seed}"
    predictions_path.mkdir()
    with torch.inference_mode():
        for frame in frames:
            active = (frame["candidates"][:, 4] >= .05) & (frame["valid"].sum(1) >= 2)
            x = torch.from_numpy(frame["features"][active]).to(gpu)
            predictions = {
                "uniform": (frame["baseline"], {}),
                "regret": (apply_actions(frame["candidates"], frame["boxes"], frame["regret"]), {}),
            }
            for mode, model in models.items():
                utilities = np.repeat(frame["base_utility"][:, :, None], 4, axis=2)
                utilities[active] += model(x).cpu().numpy()
                for composed in (False, True):
                    arm = mode + ("_composed" if composed else "_whole")
                    predictions[arm] = choose_coordinates(frame["candidates"], frame["boxes"],
                        frame["valid"], frame["uniform"], utilities, composed)
            objects = frame["metadata"]["valid_objects"]
            gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
            classes = np.asarray([o["class_id"] for o in objects], int)
            areas = box_areas(gt) * (640 / max(frame["metadata"]["original_size"])) ** 2
            for arm, (prediction, counts) in predictions.items():
                scored = score_image(prediction, np.zeros(len(prediction), bool), gt, classes, areas)
                for g, row in scored["groups"].items():
                    for k, v in row.items():
                        totals[arm]["groups"][g][k] += v
                totals[arm]["fp50"] += scored["fp50"]
                totals[arm]["moves"] += int((prediction[:, :4] != frame["baseline"][:, :4]).any(1).sum())
                for k, v in counts.items():
                    totals[arm][k] += v
            path = predictions_path / f"{frame['stem']}.npz"
            np.savez_compressed(path, **{a: p for a, (p, _) in predictions.items()})
            artifacts[frame["stem"]] = {"file": path.relative_to(output).as_posix(), "sha256": file_hash(path)}
    save_json(output / f"frames-seed{seed}.json", artifacts)
    return {a: {
        **{g + "_ten_threshold_recall_percent": c["tp_ten"] / max(c["gt"] * 10, 1) * 100
           for g, c in counts["groups"].items()},
        "all_fppi50": counts["fp50"] / len(frames), "counts": counts,
    } for a, counts in totals.items()}


def decide(metrics, policy):
    local, key = metrics["rank_composed"], "small_all_ten_threshold_recall_percent"
    comparisons = {a: local[key] - metrics[a][key] for a in ("uniform", "regret", "rank_whole", "gain_composed")}
    tiny = local["tiny_ten_threshold_recall_percent"] - metrics["uniform"]["tiny_ten_threshold_recall_percent"]
    fp = local["all_fppi50"] - metrics["uniform"]["all_fppi50"]
    checks = {
        "tiny_gain": tiny >= policy["tiny_ten_threshold_vs_uniform_min_pp"],
        "small_gain": comparisons["uniform"] >= policy["small_all_ten_threshold_vs_uniform_min_pp"],
        **{a + "_gain": comparisons[a] > policy["small_all_ten_threshold_vs_" + a + "_min_exclusive_pp"]
           for a in ("regret", "rank_whole", "gain_composed")},
        "fp_budget": fp <= policy["all_fppi50_vs_uniform_max_increase"],
        "new_compositions": local["counts"]["new_compositions_at_score025"] >=
        policy["new_compositions_at_score025_min"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"small_all_gains_pp": comparisons, "tiny_vs_uniform_pp": tiny,
                       "fppi50_increase": fp, "new_compositions_at_score025":
                       local["counts"]["new_compositions_at_score025"]}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (REGISTRATION, "sdcl/postprocess/side_composition.py", "sdcl/training/side_composition_pilot.py",
             "scripts/training/run_side_composition_pilot.py", "sdcl/postprocess/localization.py",
             "sdcl/evaluation/censored_retrieval.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, plan, heads = load_frames()
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "heads": heads,
        "source_archive_sha256": file_hash(output / "source.zip"), "image_selection": plan["image_selection"],
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")),
    })
    gpu = torch.device("cuda:" + str(device))
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, saved = fit(seed, frames["train"], output, gpu)
        values = evaluate(models, frames["holdout"], output, seed, gpu)
        metrics[str(seed)], training[str(seed)] = values, saved
        decisions[str(seed)] = decide(values, registration["policy"])
        print(f"Side composition seed{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    mean_gain = float(np.mean([v["values"]["small_all_gains_pp"]["uniform"] for v in decisions.values()]))
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"] and
              mean_gain >= registration["policy"]["small_all_ten_threshold_vs_uniform_min_pp"] and
              all(v["checks"]["fp_budget"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_SIDE_COMPOSITION_AUDIT_ALLOWED" if passed else
               "STOP_THIS_GEOMETRY_SIDE_COMPOSITION_RECIPE",
               "per_seed": decisions, "seeds_executed": len(decisions), "positive_seeds": positive,
               "scope": "Reused train-head-holdout detection proxy, not AP or novelty proof."}
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "training.json", training)
    save_json(output / "verification.json", {
        "fit_images": len(frames["train"]), "head_holdout_images": len(frames["holdout"]),
        "detector_forwards": 0, "image_decodes": 0, "detector_training_steps": 0,
        "uniform_utility_replay_exact": True, "same_head_shared_between_whole_and_side_readout": True,
        "no_GT_in_prediction_features_or_selection": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
