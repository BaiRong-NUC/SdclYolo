"""Fixed cached matched local-scale residual pilot, expanding only after gates."""

from copy import deepcopy
from itertools import combinations
import json
import os
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.censored_retrieval import score_image
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, file_hash
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.neighbor_scale_residual import (
    ScaleResidual, context_channels, derange_blocks, objective,
)
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source
from sdcl.training.side_composition_pilot import load_frames


REGISTRATION = "docs/experiments/candidate33_neighbor_scale_residual_registration_20261014.json"
OUTPUT = "output/analysis/neighbor-scale-residual-fixed-20261014"
PRIOR = "output/analysis/neighbor-scale-signal-fixed-20261014"
ARMS = ("local_context", "global_context", "shuffled_context", "geometry_only")
GROUPS = ("all", "tiny", "small_all")
PAIRS = np.array(list(combinations(range(6), 2)))


def prepare():
    prior = project_path(PRIOR)
    record, plan = verify_record(prior, "docs/experiments/candidate32_neighbor_scale_signal_result_20261014.json")
    if record["decision"]["decision"] != "MATCHED_NEIGHBOR_SCALE_RESIDUAL_PILOT_ALLOWED":
        raise ValueError("Neighbor-scale information screen did not pass.")
    frames, menu, heads = load_frames()
    if menu["image_selection"] != plan["image_selection"] or heads != plan["heads"]:
        raise ValueError("Context/action frame scope changed.")
    manifest = json.loads((prior/"frames.json").read_text())
    models, _, _, _ = frozen_heads()
    with torch.inference_mode():
        for records in frames.values():
            for f in records:
                item = manifest[f["stem"]]
                path = prior/item["file"]
                if file_hash(path) != item["sha256"]:
                    raise ValueError("Pinned scale context changed.")
                with np.load(path, allow_pickle=False) as c:
                    context = {k: c[k].copy() for k in c.files}
                f["local_channels"] = context_channels(context, f["regret"])
                f["global_channels"] = context_channels(context, f["regret"], True)
                f["active"] = ((f["candidates"][:, 4] >= .05) & (f["valid"].sum(1) >= 2) &
                               context["supported"])
                f["regret_utility"] = models["regret"](torch.from_numpy(f["features"])).numpy()
                if not np.array_equal(select_actions(f["regret_utility"], f["valid"], "absolute"), f["regret"]):
                    raise ValueError("Frozen regret utility replay changed.")
    stats = {}
    for name in ("local_channels", "global_channels"):
        values = np.concatenate([f[name][f["active"]][f["valid"][f["active"]]] for f in frames["train"]])
        stats[name] = {"mean": values.mean(0), "std": values.std(0).clip(1e-3)}
    permutation = {}
    for split, records in frames.items():
        joined = np.concatenate([f["local_channels"] for f in records])
        shuffled, partner = derange_blocks(joined, 1331 if split == "train" else 1332)
        permutation[split] = partner.tolist()
        offset = 0
        for f in records:
            channels = {}
            for arm, name, values in (
                ("local_context", "local_channels", f["local_channels"]),
                ("global_context", "global_channels", f["global_channels"]),
                ("shuffled_context", "local_channels", shuffled[offset:offset+len(f["features"])])):
                stat = stats[name]
                channels[arm] = np.clip((values-stat["mean"])/stat["std"], -6, 6).astype(np.float32)
            channels["geometry_only"] = np.zeros_like(channels["local_context"])
            f["inputs"] = {a: np.concatenate((f["features"], c), -1) for a, c in channels.items()}
            offset += len(f["features"])
    return frames, menu, heads, stats, permutation


def fit(seed, frames, output, device, stats):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tensors = {a: torch.from_numpy(np.concatenate([f["inputs"][a][f["active"]] for f in frames])).to(device)
               for a in ARMS}
    values = {k: torch.from_numpy(np.concatenate([v(f)[f["active"]] for f in frames])).to(device)
              for k, v in {
                  "base": lambda f: f["regret_utility"], "ious": lambda f: f["ious"][:, 0].astype(np.float32),
                  "valid": lambda f: f["valid"], "owned": lambda f: f["owner"] >= 0,
                  "reference": lambda f: f["regret"],
              }.items()}
    initial = ScaleResidual().to(device)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.0002, weight_decay=.0001) for a, m in models.items()}
    pairs = torch.as_tensor(PAIRS, device=device)
    history = []
    for epoch in range(8):
        order = torch.randperm(len(values["owned"]), generator=torch.Generator().manual_seed(seed*100+epoch)).to(device)
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 512):
            rows = order[start:start+512]
            for arm, model in models.items():
                utility = values["base"][rows]+model(tensors[arm][rows])
                loss = objective(utility, values["ious"][rows], values["valid"][rows],
                                 values["owned"][rows], values["reference"][rows], pairs)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite scale-residual objective.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(rows)/len(order)
        history.append({"epoch": epoch+1, **losses})
    checkpoint = output/f"heads-seed{seed}.pt"
    torch.save({"models": {a: {k: v.detach().cpu() for k, v in m.state_dict().items()} for a, m in models.items()},
                "context_normalization": stats, "seed": seed, "epochs": 8}, checkpoint)
    save_json(output/f"losses-seed{seed}.json", history)
    return models, {"checkpoint": checkpoint.name, "sha256": file_hash(checkpoint),
                    "parameters_each": sum(p.numel() for p in initial.parameters()),
                    "active_training_rows": len(values["owned"])}


def evaluate(models, frames, output, seed, device):
    arms = ("uniform", "regret", *ARMS)
    totals = {a: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                  "fp50": 0, "moves_vs_regret": 0} for a in arms}
    path = output/f"predictions-seed{seed}"
    path.mkdir()
    manifest = {}
    with torch.inference_mode():
        for f in frames:
            reference = apply_actions(f["candidates"], f["boxes"], f["regret"])
            predictions = {"uniform": f["baseline"], "regret": reference}
            for arm, model in models.items():
                utility = f["regret_utility"].copy()
                utility[f["active"]] += model(torch.from_numpy(f["inputs"][arm][f["active"]]).to(device)).cpu().numpy()
                selected = select_actions(utility, f["valid"], "absolute")
                predictions[arm] = apply_actions(f["candidates"], f["boxes"], selected)
            objects = f["metadata"]["valid_objects"]
            gt = np.asarray([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
            classes = np.asarray([o["class_id"] for o in objects], int)
            areas = box_areas(gt)*(640/max(f["metadata"]["original_size"]))**2
            for arm, prediction in predictions.items():
                values = score_image(prediction, np.zeros(len(prediction), bool), gt, classes, areas)
                for g, row in values["groups"].items():
                    for k, v in row.items():
                        totals[arm]["groups"][g][k] += v
                totals[arm]["fp50"] += values["fp50"]
                totals[arm]["moves_vs_regret"] += int((prediction[:, :4] != reference[:, :4]).any(1).sum())
            target = path/(f["stem"]+".npz")
            np.savez_compressed(target, **predictions)
            manifest[f["stem"]] = {"file": target.relative_to(output).as_posix(), "sha256": file_hash(target)}
    save_json(output/f"frames-seed{seed}.json", manifest)
    return {a: {**{g+"_ten_threshold_recall_percent": c["tp_ten"]/max(c["gt"]*10, 1)*100
                    for g, c in values["groups"].items()},
                "all_fppi50": values["fp50"]/len(frames), "counts": values} for a, values in totals.items()}


def decide(metrics, policy):
    local = metrics["local_context"]
    key = "small_all_ten_threshold_recall_percent"
    gains = {a: local[key]-metrics[a][key] for a in ("regret", "geometry_only", "global_context", "shuffled_context")}
    tiny = local["tiny_ten_threshold_recall_percent"]-metrics["regret"]["tiny_ten_threshold_recall_percent"]
    fp = local["all_fppi50"]-metrics["regret"]["all_fppi50"]
    checks = {
        "tiny_gain": tiny >= policy["tiny_ten_threshold_vs_regret_min_pp"],
        "small_gain": gains["regret"] >= policy["small_all_ten_threshold_vs_regret_min_pp"],
        **{a: gains[a] > policy["small_all_ten_threshold_vs_"+name+"_min_exclusive_pp"]
           for a, name in (("geometry_only", "geometry"), ("global_context", "global"), ("shuffled_context", "shuffled"))},
        "fp_budget": fp <= policy["all_fppi50_vs_regret_max_increase"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "values": {"small_all_gains_pp": gains, "tiny_vs_regret_pp": tiny, "fppi50_increase": fp}}


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("Available CUDA index required.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(project_path(REGISTRATION).read_text())
    paths = (REGISTRATION, "sdcl/postprocess/neighbor_scale_residual.py", "sdcl/training/neighbor_scale_residual.py",
             "scripts/training/run_neighbor_scale_residual.py", "sdcl/training/side_composition_pilot.py",
             "sdcl/postprocess/localization.py", "sdcl/evaluation/censored_retrieval.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    frames, prior, heads, stats, permutation = prepare()
    save_json(output/"plan.json", {
        "registration": registration, "source_hashes": hashes, "image_selection": prior["image_selection"],
        "source_archive_sha256": file_hash(output/"source.zip"), "heads": heads,
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate32_neighbor_scale_signal_result_20261014.json")),
        "context_normalization": {k: {n: v.tolist() for n, v in row.items()} for k, row in stats.items()}})
    save_json(output/"shuffle-partners.json", permutation)
    initial = ScaleResidual().eval()
    with torch.inference_mode():
        for records in frames.values():
            for f in records:
                for arm in ARMS:
                    residual = initial(torch.from_numpy(f["inputs"][arm])).numpy()
                    if np.any(residual != 0) or not np.array_equal(
                            select_actions(f["regret_utility"]+residual, f["valid"], "absolute"), f["regret"]):
                        raise ValueError("Initial residual does not exactly replay frozen regret.")
    gpu = torch.device("cuda:"+str(device))
    metrics, decisions, training = {}, {}, {}
    for seed in (0, 1, 2):
        models, saved = fit(seed, frames["train"], output, gpu, stats)
        metrics[str(seed)] = evaluate(models, frames["holdout"], output, seed, gpu)
        training[str(seed)] = saved
        decisions[str(seed)] = decide(metrics[str(seed)], registration["policy"])
        print(f"Neighbor residual seed{seed}: {json.dumps(decisions[str(seed)])}, "
              f"{perf_counter()-started:.1f}s", flush=True)
        if seed == 0 and not decisions["0"]["passed"]:
            break
    positive = sum(v["passed"] for v in decisions.values())
    gain = float(np.mean([v["values"]["small_all_gains_pp"]["regret"] for v in decisions.values()]))
    passed = (len(decisions) == 3 and positive >= registration["policy"]["repeat_min_positive_seeds"] and
              gain >= registration["policy"]["small_all_ten_threshold_vs_regret_min_pp"] and
              all(v["checks"]["fp_budget"] for v in decisions.values()))
    verdict = {"decision": "BOUNDED_NEIGHBOR_SCALE_OFFICIAL_AUDIT_ALLOWED" if passed else
               "STOP_THIS_LOCAL_SCALE_MEDIAN_RESIDUAL", "per_seed": decisions,
               "seeds_executed": len(decisions), "positive_seeds": positive,
               "scope": "Reused train-head-holdout detection proxy, not AP or innovation."}
    save_json(output/"metrics.json", metrics)
    save_json(output/"decision.json", verdict)
    save_json(output/"training.json", training)
    save_json(output/"verification.json", {
        "fit_images": 128, "head_holdout_images": 64, "detector_forwards": 0, "image_decodes": 0,
        "detector_training_steps": 0, "initial_all_arms_exact_regret_replay": True,
        "same_training_rows_batches_labels_and_initial_weights": True,
        "context_GT_excluded_and_normalization_fit_only": True,
        "no_val_no_testdev_no_threshold_or_hyperparameter_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter()-started})
    return {"output": str(output), "decision": verdict}
