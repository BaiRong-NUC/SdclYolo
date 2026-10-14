"""Cached diagnostic of the independently registered native-coordinate action."""

import json
from time import perf_counter

import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.censored_retrieval import score_image
from sdcl.evaluation.localization_official_audit import frozen_heads
from sdcl.evaluation.size_analysis import box_areas, file_hash
from sdcl.postprocess.censored_coordinates import exchange
from sdcl.postprocess.localization import apply_actions, select_actions
from sdcl.postprocess.rescue import save_json
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate23_censored_coordinates_registration_20261014.json"
OUTPUT = "output/analysis/censored-coordinates-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
NATIVE = "output/analysis/native-censoring-fixed-20261014"
MENU = "output/analysis/action-order-sensitivity-fixed-20261014"
ARMS = ("uniform", "regret", "contrast_coordinates", "native_coordinates",
        "ordinary_agreement_coordinates", "kept_agreement_coordinates")
GROUPS = ("all", "tiny", "small_all")


def decide(metrics, policy):
    local = metrics["contrast_coordinates"]
    key = "small_all_ten_threshold_recall_percent"
    values = {a + "_gain_pp": local[key] - metrics[a][key] for a in ARMS
              if a != "contrast_coordinates"}
    values["tiny_vs_uniform_pp"] = (local["tiny_ten_threshold_recall_percent"] -
                                    metrics["uniform"]["tiny_ten_threshold_recall_percent"])
    values["fppi50_vs_uniform"] = local["all_fppi50"] - metrics["uniform"]["all_fppi50"]
    checks = {
        "tiny_gain": values["tiny_vs_uniform_pp"] >= policy["tiny_ten_threshold_vs_uniform_min_pp"],
        "small_gain": values["uniform_gain_pp"] >= policy["small_all_ten_threshold_vs_uniform_min_pp"],
        **{name + "_gain": values[arm + "_gain_pp"] >
           policy["small_all_ten_threshold_vs_" + name + "_min_exclusive_pp"]
           for name, arm in (("regret", "regret"), ("native", "native_coordinates"),
                             ("agreement", "ordinary_agreement_coordinates"),
                             ("kept", "kept_agreement_coordinates"))},
        "fp_budget": values["fppi50_vs_uniform"] <= policy["all_fppi50_vs_uniform_max_increase"],
    }
    return {"decision": "BOUNDED_CENSORED_LOCALIZATION_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_CENSORED_COORDINATE_EXCHANGE", "checks": checks, "values": values,
            "scope": "Reused64 train head-held-out diagnostics, no AP or novelty claim."}


def evaluate(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    base, native, menu = (project_path(p) for p in (BASE, NATIVE, MENU))
    _, base_plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    _, native_plan = verify_record(native, "docs/experiments/candidate21_native_censoring_result_20261014.json")
    _, menu_plan = verify_record(menu, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    stems = native_plan["images"]
    if stems != menu_plan["image_selection"]["holdout"] or len(stems) != 64:
        raise ValueError("Fixed64 image order differs.")
    models, _, _, heads = frozen_heads()
    if heads != base_plan["heads"]:
        raise ValueError("Frozen head provenance changed.")
    manifests = {name: json.loads((path / "frames.json").read_text())
                 for name, path in (("base", base), ("native", native), ("menu", menu))}
    snapshots = {r["stem"]: r for r in base_plan["records"]["holdout"]}
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/censored_coordinates.py",
             "sdcl/evaluation/censored_coordinates.py", "scripts/evaluation/diagnose_censored_coordinates.py",
             "sdcl/postprocess/censored_retrieval.py", "sdcl/evaluation/censored_retrieval.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/localization.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "images": stems,
        "source_archive_sha256": file_hash(output / "source.zip"), "heads": heads,
        "prior_results_sha256": {f"candidate{i}": file_hash(project_path(path)) for i, path in (
            (12, "docs/experiments/candidate12_class_choice_result_20261014.json"),
            (18, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json"),
            (21, "docs/experiments/candidate21_native_censoring_result_20261014.json"),
            (22, "docs/experiments/candidate22_censored_retrieval_result_20261014.json"))},
    })
    totals = {arm: {"groups": {g: dict.fromkeys(("gt", "tp50", "tp_ten"), 0) for g in GROUPS},
                    "fp50": 0, "moves": 0} for arm in ARMS}
    frames = {}
    for index, stem in enumerate(stems, 1):
        loaded = {}
        for name, path in (("base", base), ("native", native), ("menu", menu)):
            frame = manifests[name][stem]
            p = path / frame["file"]
            if file_hash(p) != frame["sha256"]:
                raise ValueError("Frozen frame changed.")
            with np.load(p, allow_pickle=False) as c:
                loaded[name] = {k: c[k].copy() for k in c.files}
        b, p, c = loaded["base"]["baseline"], loaded["native"], loaded["menu"]
        if not np.array_equal(apply_actions(c["candidates"], c["boxes"], c["uniform"]), b):
            raise ValueError("Frozen uniform replay differs.")
        arms, inventory = exchange(b, p["dense"], p["dense_view"], p["uncapped"], p["uncapped_view"])
        arms["uniform"] = b
        with torch.inference_mode():
            selected = select_actions(models["regret"](torch.from_numpy(c["features"])).numpy(),
                                      c["valid"], "absolute")
        arms["regret"] = apply_actions(c["candidates"], c["boxes"], selected)
        if any(v.shape != b.shape or not np.array_equal(v[:, 4:], b[:, 4:]) for v in arms.values()):
            raise ValueError("Coordinate action changed identity, scores or classes.")
        # All prediction choices precede reading diagnostic object labels.
        meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        if file_hash(meta_path) != snapshots[stem]["metadata_sha256"]:
            raise ValueError("Frozen diagnostic annotations changed.")
        meta = json.loads(meta_path.read_text())
        gt = np.asarray([o["box_xyxy"] for o in meta["valid_objects"]], float).reshape(-1, 4)
        classes = np.asarray([o["class_id"] for o in meta["valid_objects"]], int)
        areas = box_areas(gt) * (640 / max(meta["original_size"])) ** 2
        for arm, predictions in arms.items():
            scored = score_image(predictions, np.zeros(len(predictions), bool), gt, classes, areas)
            for g, counts in scored["groups"].items():
                for key, value in counts.items():
                    totals[arm]["groups"][g][key] += value
            totals[arm]["fp50"] += scored["fp50"]
            totals[arm]["moves"] += int((predictions[:, :4] != b[:, :4]).any(1).sum())
        target = output / "predictions" / f"{stem}.npz"
        np.savez_compressed(target, **arms)
        frames[stem] = {"file": target.relative_to(output).as_posix(),
                        "sha256": file_hash(target), "inventory": inventory}
        if index == 1 or index % 16 == 0:
            print(f"Censored coordinates: {index}/64, {perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for arm, counts in totals.items():
        metrics[arm] = {
            **{g + "_recall_at50_percent": c["tp50"] / max(c["gt"], 1) * 100
               for g, c in counts["groups"].items()},
            **{g + "_ten_threshold_recall_percent": c["tp_ten"] / max(c["gt"] * 10, 1) * 100
               for g, c in counts["groups"].items()},
            "all_fppi50": counts["fp50"] / len(stems), "counts": counts,
        }
    verdict = decide(metrics, registration["policy"])
    save_json(output / "frames.json", frames)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": len(stems), "zero_detector_forwards_zero_fitting": True,
        "uniform_reconstruction_exact": True, "all_scores_classes_counts_order_preserved": True,
        "no_val_no_testdev_no_threshold_search": True,
        "cache_has_diagnostic_ignore_filter": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
