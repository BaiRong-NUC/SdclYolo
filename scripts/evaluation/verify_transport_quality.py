"""Independent scalar-SciPy feature and complete cached deployment audit for62."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from scipy.special import softmax
from scipy.stats import wasserstein_distance
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.transport_quality import QualityHead


ARMS = ("geometry", "dgqp", "centered", "transport")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def independent_boxes(decoded, window):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy
    width, height = window[2]-window[0], window[3]-window[1]
    boxes = scale_boxes((960, 960), xywh2xyxy(torch.tensor(
        decoded[:4].T, device="cuda")).float(), (height, width)).cpu().numpy()
    return boxes+np.array([window[0], window[1], window[0], window[1]], np.float32)


def reconstruct_features(frame, window, size):
    """Eight fixed query ranks/view; W1 comes from scipy.stats, not main code."""
    decoded, dfl, raw = frame["decoded"], frame["raw_dfl"], frame["raw_class"]
    score = decoded[4:14].max(0)
    candidates = np.flatnonzero(score >= .001)
    chosen = sorted(candidates, key=lambda j: (-float(score[j]), int(j)))[:4096]
    np.testing.assert_array_equal(chosen, frame["indices"])
    boxes = independent_boxes(decoded, window)
    np.testing.assert_array_equal(boxes, frame["boxes"])
    p = softmax(dfl.astype(np.float32), axis=-1).astype(float)
    width, height = window[2]-window[0], window[3]-window[1]
    gain = 960/max(width, height)
    pad = [round((960-round(width*gain))/2-.1), round((960-round(height*gain))/2-.1)]
    stride, anchors = frame["strides"], frame["anchors"]
    sample = np.unique(np.linspace(0, len(chosen)-1, min(8, len(chosen)), dtype=int))
    def support(index, side):
        axis = side % 2
        values = (float(anchors[index, axis])*float(stride[index])+
                  (-1 if side < 2 else 1)*np.arange(16)*float(stride[index])-pad[axis])/gain
        return np.clip(values, 0, [width, height][axis])+window[axis]
    def stats(index):
        top = np.sort(p[index], axis=-1)[:, -4:][:, ::-1]
        return np.c_[top, top.mean(1)].reshape(20)
    mean_drift = 0.
    for rank in sample:
        index = chosen[rank]
        lev, base, grid = (0, 0, 120) if index < 14400 else (
            (1, 14400, 60) if index < 18000 else (2, 18000, 30))
        x, y = (index-base) % grid, (index-base)//grid
        cls = decoded[4:14, index].argmax()
        own = boxes[index]
        dimension = np.maximum(own[2:]-own[:2], 1.)
        slots, quality, physical, centered = [], [], [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                xx, yy = x+dx, y+dy
                neighbor = base+yy*grid+xx if 0 <= xx < grid and 0 <= yy < grid else -1
                valid = neighbor >= 0 and np.isfinite(boxes[neighbor]).all() and (
                    boxes[neighbor, 2:] > boxes[neighbor, :2]).all()
                if not valid:
                    slots.extend([0.]*8)
                    physical.extend([0.]*4)
                    centered.extend([0.]*4)
                    continue
                other = boxes[neighbor]
                inter = np.maximum(np.minimum(own[2:], other[2:])-np.maximum(own[:2], other[:2]), 0).prod()
                union = np.maximum(own[2:]-own[:2], 0).prod()+np.maximum(other[2:]-other[:2], 0).prod()-inter
                slots.extend([*np.clip((other-own)/dimension[[0, 1, 0, 1]], -4, 4),
                              decoded[4+cls, neighbor], decoded[4:14, neighbor].argmax() == cls,
                              inter/max(union, 1e-12), 1.])
                quality.append(stats(neighbor))
                for side in range(4):
                    a, b = support(index, side), support(neighbor, side)
                    pa, pb = p[index, side], p[neighbor, side]
                    physical.append(min(wasserstein_distance(a, b, pa, pb)/dimension[side % 2], 4.))
                    # Frozen main uses FP32-softmax masses without a second
                    # normalization for centering. Audit its rounding explicitly.
                    ca, cb = np.dot(a, pa), np.dot(b, pb)
                    mean_drift = max(mean_drift, abs(ca-ca/pa.sum())/dimension[side % 2],
                                     abs(cb-cb/pb.sum())/dimension[side % 2])
                    centered.append(min(wasserstein_distance(a-ca, b-cb, pa, pb)/dimension[side % 2], 4.))
        base_values = np.r_[np.clip(raw[index], -20, 20), np.eye(10)[cls], np.eye(3)[lev],
                            own/np.array([*size, *size]), score[index], slots]
        q = np.r_[stats(index), np.mean(quality, 0) if quality else np.zeros(20)]
        for arm in ARMS:
            extra = physical if arm == "transport" else centered if arm == "centered" else np.zeros(32)
            expected = np.r_[base_values, q if arm != "geometry" else np.zeros(40), extra].astype(np.float32)
            np.testing.assert_allclose(frame["input_"+arm][rank], expected, atol=2e-6, rtol=2e-6)
    return len(sample), len(sample)*32, float(mean_drift)


def manual_filter(values, metadata):
    keep = np.ones(len(values), bool)
    for j, dt in enumerate(values):
        area = max(float((dt[2]-dt[0])*(dt[3]-dt[1])), 1e-12)
        for box in metadata["ignore_boxes_xyxy"]:
            inter = max(0, min(dt[2], box[2])-max(dt[0], box[0]))*max(0, min(dt[3], box[3])-max(dt[1], box[1]))
            if inter/area > .5:
                keep[j] = False
                break
    return values[keep]


def map_local(values, window, view):
    values = values.copy()
    if view:
        values[:, 0] = np.clip(values[:, 0], 0, window[2]-window[0])+window[0]
        values[:, 2] = np.clip(values[:, 2], 0, window[2]-window[0])+window[0]
        values[:, 1] = np.clip(values[:, 1], 0, window[3]-window[1])+window[1]
        values[:, 3] = np.clip(values[:, 3], 0, window[3]-window[1])+window[1]
    return values


def replay(decoded, indices, shift, window, device):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    tensor = torch.tensor(decoded[None], device=device)
    original_coordinates = tensor[:, :4].clone()
    if shift is not None:
        scores = tensor[:, 4:14, indices].float().clamp(1e-7, 1-1e-7)
        tensor[:, 4:14, indices] = torch.sigmoid(torch.logit(scores)+torch.tensor(
            shift[None, None], device=device)).to(tensor.dtype)
    assert torch.equal(tensor[:, :4], original_coordinates)
    kept = non_max_suppression(tensor, .001, .7, max_det=500)[0]
    kept[:, :4] = scale_boxes((960, 960), kept[:, :4], (window[3]-window[1], window[2]-window[0]))
    return kept.float().cpu().numpy()


def verify(device="0"):
    torch.set_num_threads(1)
    torch.cuda.set_device(int(device))
    root = project_path("output/analysis/transport-quality-fixed-20261015")
    audit_path = root/"independent_verification.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    result, plan = read(root/"result.json"), read(root/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
    for name, digest in plan["installed_source_hashes"].items():
        assert file_hash(name) == digest
    assert file_hash(project_path("docs/experiments/candidate61_peak_gap_survival_result_20261015.json")) == plan[
        "prior61_closeout_sha256"]
    frozen_control()
    selection, manifest = plan["image_selection"], read(root/"frames.json")
    assert list(map(len, selection.values())) == [64, 32, 17]
    groups = [{s[:7] for s in stems} for stems in selection.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(groups[2]) == 5 and len(manifest) == 113
    excluded = set()
    for name, digest in plan["exclusion_plan_hashes"].items():
        assert file_hash(project_path(name)) == digest
        excluded.update(re.findall(r"\d{7}_\d{5}_[a-z]_\d{7}", project_path(name).read_text(encoding="utf-8")))
    assert not ({s for stems in selection.values() for s in stems} & excluded)
    assert not groups[2] & {s[:7] for s in excluded}
    fit_inputs = {a: [] for a in ARMS}
    checked_queries, checked_W1, max_centering_drift = 0, 0, 0.
    for number, (stem, entry) in enumerate(manifest.items(), 1):
        assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == entry["image_sha256"]
        assert file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) == entry["metadata_sha256"]
        assert file_hash(root/entry["reference"]["file"]) == entry["reference"]["sha256"]
        metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
        gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
        gc = np.array([o["class_id"] for o in metadata["valid_objects"]])
        for view, snap in enumerate(entry["views"]):
            assert file_hash(root/snap["file"]) == snap["sha256"]
            with np.load(root/snap["file"], allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            q, w, drift = reconstruct_features(f, entry["windows"][view], entry["size"])
            checked_queries += q
            checked_W1 += w
            max_centering_drift = max(max_centering_drift, drift)
            boxes = f["boxes"][f["indices"]].astype(float)
            cls = f["decoded"][4:14, f["indices"]].argmax(0)
            maximum = np.zeros(len(boxes))
            for box, category in zip(gt, gc):
                inter = np.maximum(np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2]), 0).prod(1)
                union = ((box[2]-box[0])*(box[3]-box[1])+
                         np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(1)-inter)
                maximum = np.maximum(maximum, np.where(cls == category, inter/np.maximum(union, 1e-12), 0))
            np.testing.assert_array_equal(f["target"], np.where(maximum >= .5, maximum, 0).astype(np.float32))
            native = np.c_[boxes, np.zeros((len(boxes), 2))]
            # Rebuild availability by applying the scalar filter to indexed rows.
            native[:, 4] = np.arange(len(boxes))
            expected_eligible = np.zeros(len(boxes), bool)
            expected_eligible[manual_filter(native, metadata)[:, 4].astype(int)] = True
            np.testing.assert_array_equal(f["eligible"], expected_eligible)
            if entry["split"] == "fit":
                seed = int(hashlib.sha256(f"candidate62-sample-v1:{stem}:{view}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                rows, weights = [], []
                for positive in (False, True):
                    pool = np.flatnonzero(expected_eligible & ((f["target"] > 0) == positive))
                    n = min(128, len(pool))
                    if n:
                        rows.extend(sorted(rng.choice(pool, n, replace=False)))
                        weights.extend([len(pool)/n]*n)
                np.testing.assert_array_equal(f["sample_rows"], rows)
                np.testing.assert_array_equal(f["sample_weights"], np.array(weights, np.float32))
                for arm in ARMS:
                    fit_inputs[arm].append(f["input_"+arm][rows])
        if number % 16 == 0:
            print(f"Independent transport features and labels: {number}/113", flush=True)
    normal = read(root/"normalization.json")
    for arm in ARMS:
        values = np.concatenate(fit_inputs[arm])
        np.testing.assert_array_equal(values.mean(0), normal[arm]["mean"])
        np.testing.assert_array_equal(np.maximum(values.std(0), 1e-6), normal[arm]["std"])
    del fit_inputs
    calibrator = load_calibrator(frozen_control())
    decisions, scalar_reports = {}, {}
    output_replays, model_queries = 0, 0
    for seed in result["seeds"]:
        checkpoint = torch.load(root/f"heads-seed{seed}.pt", map_location="cpu", weights_only=False)
        models = {}
        for arm in ARMS:
            model = QualityHead().to("cuda:"+device).eval()
            model.load_state_dict(checkpoint["models"][arm])
            assert sum(p.numel() for p in model.parameters()) == 5313
            models[arm] = model
        traces = read(root/f"predictions-seed{seed}.json")
        reports = read(root/f"metrics-seed{seed}.json")
        rebuilt = {}
        for split in ("screen", "confirmation"):
            records, maps, losses = [], {a: {} for a in ("strong", *ARMS)}, []
            for number, stem in enumerate(selection[split], 1):
                entry, snap = manifest[stem], traces[stem]
                assert file_hash(root/snap["file"]) == snap["sha256"]
                with np.load(root/snap["file"], allow_pickle=False) as z:
                    outputs = {k: z[k].copy() for k in z.files}
                metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
                records.append(dict(id=number, stem=stem, size=entry["size"], objects=metadata["valid_objects"]))
                views, sums, mass = {a: [] for a in maps}, dict.fromkeys(ARMS, 0.), 0
                for view, cache in enumerate(entry["views"]):
                    with np.load(root/cache["file"], allow_pickle=False) as z:
                        f = {k: z[k].copy() for k in z.files}
                    window = entry["windows"][view]
                    for arm in maps:
                        shift = outputs[f"shifts_v{view}_{arm}"] if arm != "strong" else None
                        if arm != "strong":
                            x = np.clip((f["input_"+arm]-normal[arm]["mean"])/normal[arm]["std"], -8, 8).astype(np.float32)
                            with torch.inference_mode():
                                own_shift = models[arm](torch.tensor(x, device="cuda:"+device)).cpu().numpy()
                            np.testing.assert_allclose(shift, own_shift, atol=1e-5, rtol=1e-5)
                            assert np.isfinite(shift).all() and np.abs(shift).max(initial=0) <= 2.000001
                            model_queries += len(shift)
                            z = f["offset"].astype(float)+shift
                            sums[arm] += float((np.logaddexp(0, z)-f["target"]*z)[f["eligible"]].sum())
                        local = replay(f["decoded"], f["indices"], shift, window, "cuda:"+device)
                        if arm == "strong":
                            np.testing.assert_array_equal(local, f["original_local"])
                        views[arm].append(map_local(local, window, view))
                    mass += int(f["eligible"].sum())
                for arm in maps:
                    actual = calibrated_candidates(views[arm], entry["windows"], entry["size"], [], calibrator)[0]
                    np.testing.assert_array_equal(actual, outputs[arm])
                    maps[arm][stem] = manual_filter(actual, metadata)
                    np.testing.assert_array_equal(maps[arm][stem], outputs["diagnostic_"+arm])
                    output_replays += 1
                losses.append(dict(stem=stem, rows=mass, loss_sums=sums))
            assert losses == reports[split]["per_image_losses"]
            ap = {a: coco_size_metrics(records, mapping, dict(enumerate(NAMES)), reference_size=640)
                  for a, mapping in maps.items()}
            assert ap == reports[split]["AP"]
            fixed = {}
            for arm in ARMS:
                budgets, _ = recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[arm]),
                                                 budgets=(10,), bootstrap_samples=0)
                fixed[arm] = budgets
            assert fixed == reports[split]["fixed_fp"]
            prefixes = sorted({r["stem"][:7] for r in losses})
            count = np.array([sum(r["rows"] for r in losses if r["stem"][:7] == s) for s in prefixes])
            delta = np.array([sum(r["loss_sums"]["centered"]-r["loss_sums"]["transport"]
                                 for r in losses if r["stem"][:7] == s) for s in prefixes])
            draws = np.random.default_rng(0).multinomial(len(prefixes), np.full(len(prefixes), 1/len(prefixes)), size=400)
            ci = np.percentile((draws@delta)/(draws@count), [2.5, 97.5]).tolist() if ((draws@count) > 0).all() else None
            expected_ci = reports[split]["mechanism"]["centered_minus_transport_NLL_sequence_bootstrap95ci"]
            assert ci == expected_ci
            m = reports[split]["mechanism"]
            nll = {a: sum(r["loss_sums"][a] for r in losses)/sum(r["rows"] for r in losses) for a in ARMS}
            assert m["NLL"] == nll and m["sequence_units"] == len(prefixes)
            policy = plan["registration"]["policy_per_half"]
            c = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"])
            for group, control in (("tiny", "strong"), ("small_all", "strong"), ("all", "strong"),
                                   ("tiny", "geometry"), ("tiny", "dgqp"), ("tiny", "centered")):
                x, y = ap["transport"][group]["ap50_95"], ap[control][group]["ap50_95"]
                name = f"{group}_ap_vs_{control}"
                c[name] = x is not None and y is not None and (x-y)*100 >= policy[name+"_min_pp"]
            fp = fixed["transport"][0]["delta_recall_pp"]["small_all"]
            c["fixed_FP"] = fp is not None and fp >= policy["small_all_recall_fppi10_vs_strong_min_pp"]
            c["NLL"] = nll["centered"]-nll["transport"] >= policy["NLL_gain_vs_centered_min"]
            c["sequence_interval"] = ci is not None and ci[0] > policy[
                "NLL_gain_vs_centered_sequence_bootstrap_lower_min_exclusive"]
            rebuilt[split] = c
            print(f"Independent transport deployment and metrics: seed{seed} {split}", flush=True)
        decision = dict(checks=rebuilt, passed=all(all(c.values()) for c in rebuilt.values()))
        assert decision == result["seeds"][seed] == read(root/f"decision-seed{seed}.json")
        decisions[seed] = decision
    full_pass = sum(d["passed"] for d in decisions.values()) >= 2 and all(
        c["all_ap_vs_strong"] and c["fixed_FP"] for d in decisions.values() for c in d["checks"].values())
    expected = ("REPLICATED_DEPLOYABLE_TRANSPORT_QUALITY_PILOT_SUPPORTS_OFFICIAL_AND_EXTERNAL_AUDIT" if full_pass else
                "STOP_THIS_COMMON_COORDINATE_W1_PRENMS_QUALITY_RECIPE")
    assert result["decision"] == expected
    audit = dict(all_recorded_source_archive_installed_prior_cache_image_metadata_prediction_hashes_verified=True,
        fit_screen_confirmation_sequence_disjointness_and_plan_unlisted_five_confirmation_sequences_verified=True,
        all565view_labels_sampling_weights_and_fit_normalization_rebuilt=True,
        scalar_Scipy_feature_queries_checked=checked_queries, scalar_Scipy_W1_distances_checked=checked_W1,
        sampled_max_centering_mean_drift_from_FP32_softmax_mass_normalization=max_centering_drift,
        cached_final_outputs_replayed_exactly=output_replays, learned_head_query_outputs_checked=model_queries,
        all_AP_fixedFP_NLL_sequence_bootstrap_and_gates_recomputed=True, added_detector_forwards=0,
        added_JPEG_decodes=0, added_fits=0, verifier_sha256=file_hash(Path(__file__)),
        limits="Sampled8queryranks/view for independent feature construction,not everyfeature. Complete prediction outputs replayed using savedshifts; checkpointshift replay tolerance1e-5. Reuses originalNMS/frozencalibrator/COCO/FPbudgetengines,notindependentimplementations. NativeDFLneednotbe calibratedposteriors. Fiveconfirmationsequenceunits only; detector-training data; notofficialAP/novelty/externalgeneralization.")
    audit_path.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    verify(**vars(parser.parse_args()))
