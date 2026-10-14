"""Cached deployment and label audit; shared TAL/COCO/NMS engines are explicit."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.training.crossview_quality_rank import (
    ARMS, OUTPUT, PRIOR, Projection, assembled, decide, precision, precision_audit, target_rows,
)
from sdcl.training.transport_quality import diagnostic_filter, load_cache


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def independent_pairs(d):
    objects, representatives = {}, []
    for i, (image, owner, view) in enumerate(zip(d["image"], d["owner"], d["view"])):
        if owner >= 0:
            objects.setdefault((int(image), int(owner)), {}).setdefault(int(view), []).append(i)
    pending = []
    for identity, views in sorted(objects.items()):
        chosen = []
        for view, rows in sorted(views.items()):
            qs = sorted(rows, key=lambda i: (-d["quality"][i], i))
            ps = sorted(rows, key=lambda i: (-d["true_score"][i], i))
            chosen.extend(dict.fromkeys([qs[0], ps[0], qs[-1], ps[-1]]))
        representatives.extend(chosen)
        w, c = [], []
        for a, b in combinations(chosen, 2):
            if abs(float(d["quality"][a])-float(d["quality"][b])) >= .1:
                pair = (a, b) if d["quality"][a] > d["quality"][b] else (b, a)
                (w if d["view"][a] == d["view"][b] else c).append(pair)
        key = lambda pp: hashlib.sha256(f"candidate66-pair-v1:{identity}:{pp}".encode()).hexdigest()
        pending.extend(zip(sorted(w, key=key)[:8], sorted(c, key=key)[:8]))
    pools = {}
    for j in representatives:
        pools.setdefault(tuple(int(d[k][j]) for k in ("classes", "view", "levels")), []).append(j)
    result = dict(within=[], cross=[], unlinked=[])
    for w, (hi, lo) in pending:
        donors = []
        for j in pools[tuple(int(d[k][lo]) for k in ("classes", "view", "levels"))]:
            if (d["image"][j], d["owner"][j]) == (d["image"][lo], d["owner"][lo]):
                continue
            error = abs(float(d["quality"][j])-float(d["quality"][lo]))
            if error <= .025 and float(d["quality"][hi])-float(d["quality"][j]) >= .1:
                donors.append((error, j))
        if donors:
            donor = min(donors)[1]
            result["within"].append(w)
            result["cross"].append((hi, lo))
            result["unlinked"].append((hi, donor))
    return {a: np.asarray(v, np.int64).reshape(-1, 2) for a, v in result.items()}


def independent_deployment(frame, features, weight, bias, shape, device):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    scores = []
    for lev, (offset, side) in enumerate(zip((0, 14400, 18000), (120, 60, 30))):
        x = torch.from_numpy(features[offset:offset+side*side].T.copy()).to(device).reshape(1, 128, side, side)
        z = F.conv2d(x, weight[lev, :, :, None, None].to(x), bias[lev].to(x))
        scores.append(z.flatten(2))
    logits = torch.cat(scores, 2)
    decoded = torch.tensor(frame["decoded"][None], device=device)
    before = decoded[:, :4].clone()
    decoded[:, 4:14] = logits.sigmoid()
    assert torch.equal(decoded[:, :4], before)
    out = non_max_suppression(decoded, .001, .7, max_det=500)[0]
    out[:, :4] = scale_boxes((960, 960), out[:, :4], shape)
    return out.float().cpu().numpy()


def verify(seed=0, device="0"):
    root, prior = project_path(OUTPUT), project_path(PRIOR)
    plan, result = read(root/"plan.json"), read(root/"result.json")
    frames, old_frames = read(root/"frames.json"), read(prior/"frames.json")
    gpu = "cuda:"+str(device)
    precision(False)
    for p, h in result["artifacts_sha256"].items():
        assert file_hash(root/p) == h, p
    for p, h in plan["source_hashes"].items():
        assert file_hash(project_path(p)) == h, p
    for p, h in plan["installed_source_hashes"].items():
        assert file_hash(p) == h, p
    with zipfile.ZipFile(root/"source.zip") as z:
        assert set(z.namelist()) == set(plan["source_hashes"])
        for p, h in plan["source_hashes"].items():
            assert hashlib.sha256(z.read(p)).hexdigest() == h
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_sha256"]
    assert read(root/"detector_state.json")["before"] == read(root/"detector_state.json")["after"]
    control = frozen_control()
    calibrator = load_calibrator(control)
    states = torch.load(root/f"projections-seed{seed}.pt", map_location="cpu", weights_only=True)
    with np.load(root/"initial.npz") as z:
        states = dict(strong=dict(weight=torch.tensor(z["weight"]), bias=torch.tensor(z["bias"])), **states)
    selection = plan["image_selection"]
    counts = {}
    for split, stems in selection.items():
        for n, stem in enumerate(stems):
            e = old_frames[stem]
            metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(metadata_path) == e["metadata_sha256"]
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == e["image_sha256"]
            meta = read(metadata_path)
            for view, snapshot in enumerate(frames[stem]["views"]):
                f = load_cache(root, snapshot)
                old = load_cache(prior, e["views"][view])
                expected = target_rows(old, meta, e["windows"][view], e["size"], stem, view)
                for k, v in expected.items():
                    np.testing.assert_array_equal(v, f[k], err_msg=f"{stem}/{view}/{k}")
                np.testing.assert_array_equal(f["features"], f["features_full"][f["native"]])
            if n == 0 or (n+1) % 16 == 0:
                print(f"Verify66 cache/targets {split}: {n+1}/{len(stems)}", flush=True)
        data, main_pairs = assembled(root, frames, stems)
        alternate = independent_pairs(data)
        for arm in alternate:
            np.testing.assert_array_equal(alternate[arm], main_pairs[arm])
        counts[split] = {a: len(p) for a, p in alternate.items()}
        np.savez_compressed(root/f"independent-pairs-{split}.npz", **alternate)
        if split == "fit":
            with np.load(root/"fit-sampled-rows.npz") as stored:
                for k in data:
                    np.testing.assert_array_equal(stored[k], data[k])
            with np.load(root/"fit-pairs.npz") as stored:
                for k in alternate:
                    np.testing.assert_array_equal(stored[k], alternate[k])
            precision(True)
            audit = precision_audit(Projection(**states["strong"]).to(gpu), data, gpu)
            assert audit["passed"] and audit == read(root/"precision_audit.json")
            precision(False)
    metrics, snapshots = read(root/f"metrics-seed{seed}.json"), read(root/f"predictions-seed{seed}.json")
    for split in ("screen", "confirmation"):
        records, maps = [], {a: {} for a in states}
        for index, stem in enumerate(selection[split]):
            e = old_frames[stem]
            meta = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
            records.append(dict(id=index+1, stem=stem, size=e["size"], objects=meta["valid_objects"]))
            saved = load_cache(root, snapshots[stem])
            cached = [load_cache(prior, f) for f in e["views"]]
            feats = [load_cache(root, f)["features_full"] for f in frames[stem]["views"]]
            for arm, state in states.items():
                views = []
                with torch.inference_mode():
                    for view, window in enumerate(e["windows"]):
                        local = independent_deployment(cached[view], feats[view], state["weight"], state["bias"],
                                                       (window[3]-window[1], window[2]-window[0]), gpu)
                        if arm == "strong":
                            np.testing.assert_array_equal(local, cached[view]["original_local"])
                        if view:
                            # Original mapper adds offset and clips; keep shared mapper explicit.
                            from sdcl.evaluation.slicing_screen import map_tile_predictions
                            local = map_tile_predictions(local, window)
                        views.append(local)
                    output = calibrated_candidates(views, e["windows"], e["size"], [], calibrator)[0]
                np.testing.assert_array_equal(output, saved[arm])
                if arm == "strong":
                    ref = prior/e["reference"]["file"]
                    assert file_hash(ref) == e["reference"]["sha256"]
                    np.testing.assert_array_equal(output, np.load(ref))
                maps[arm][stem] = diagnostic_filter(output, meta)
                np.testing.assert_array_equal(maps[arm][stem], saved["diagnostic_"+arm])
            if index == 0 or (index+1) % 8 == 0:
                print(f"Verify66 deployment {split}: {index+1}/{len(selection[split])}", flush=True)
        ap = {a: coco_size_metrics(records, m, dict(enumerate(NAMES)), reference_size=640) for a, m in maps.items()}
        assert ap == metrics[split]["AP"]
        for arm in ARMS:
            fp = recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[arm]),
                                      budgets=(10,), bootstrap_samples=0)[0]
            assert fp == metrics[split]["fixed_fp"][arm]
        with np.load(root/f"mechanism-seed{seed}-{split}.npz") as stored:
            prefixes = np.array([str(s)[:7] for s in stored["pair_stems"]])
            unique = sorted(set(prefixes))
            count = np.array([(prefixes == s).sum() for s in unique])
            for arm in ARMS:
                np.testing.assert_allclose(float(stored[arm].mean()), metrics[split]["mechanism"]["NLL"][arm], rtol=0, atol=1e-14)
            for arm in ("within", "unlinked"):
                delta = stored[arm]-stored["cross"]
                sums = np.array([delta[prefixes == s].sum() for s in unique])
                draws = np.random.default_rng(0).multinomial(len(unique), [1/len(unique)]*len(unique), size=400)
                ci = np.percentile((draws@sums)/(draws@count), [2.5, 97.5]).tolist()
                assert ci == metrics[split]["mechanism"]["comparisons"][arm]["sequence95ci"]
    decision = decide(metrics, plan["registration"]["policy_per_half"])
    assert decision == read(root/f"decision-seed{seed}.json")
    # Repeat the registered arithmetic without reusing main decision's booleans.
    for split, r in metrics.items():
        p, c, ap = plan["registration"]["policy_per_half"], decision["checks"][split], r["AP"]
        assert c["tiny_GT"] == (ap["strong"]["tiny"]["gt"] >= p["tiny_GT_min"])
        for g, a in [("tiny", a) for a in states if a != "cross"]+[("small_all", "strong"), ("all", "strong")]:
            key = f"{g}_AP_vs_{a}"
            assert c[key] == ((ap["cross"][g]["ap50_95"]-ap[a][g]["ap50_95"])*100 >= p[key+"_min_pp"])
        assert c["fixed_FP"] == (r["fixed_fp"]["cross"][0]["delta_recall_pp"]["small_all"] >= p["small_all_recall_FPPI10_vs_strong_min_pp"])
        assert c["processing"] == (np.mean(r["timing_ms"]["cross"])-np.mean(r["timing_ms"]["strong"]) <= p["cached_added_processing_mean_ms_max"])
        assert c["mechanism"] == all(v["mean_gain"] >= p["crosspair_NLL_gain_vs_each_rankcontrol_min"]
                                    for v in r["mechanism"]["comparisons"].values())
        assert c["sequence_interval"] == all(v["sequence95ci"][0] > p["crosspair_NLL_gain_sequence95_lower_min_exclusive"]
                                            for v in r["mechanism"]["comparisons"].values())
    report = dict(passed=True, source_cache_metadata_hashes=True, independent_pair_counts=counts,
                  cached_full_deployment_exact=True, shared_TAL_target_replay=True,
                  precision_audit=audit, shared_COCO_and_FP_metrics_exact=True,
                  saved_mechanism_loss_bootstrap_rebuilt=True, fixed_gates_exact=True,
                  decision=decision, added_detector_forwards=0,
                  limits="Shared TAL,PyTorch convolution,NMS,mapper,calibrator,COCO and FP engines. "
                         "Pair construction independently rewritten. Does not independently extract detector features, "
                         "retrain optimizer steps, time end-to-end, or prove novelty/generalization.")
    (root/"independent_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    verify(**vars(parser.parse_args()))
