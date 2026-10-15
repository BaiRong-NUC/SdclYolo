"""Independent masks/moments/folding and shared postprocessing replay for69."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.instance_moments import ARMS, OUTPUT, PRIOR, metrics, read
from sdcl.evaluation.size_analysis import file_hash
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.crossview_quality_rank import precision, state_hash
from sdcl.training.transport_quality import native_nms


def independently_rebuild_maps(metadata, window, stem, view):
    w, h = window[2]-window[0], window[3]-window[1]
    ratio = 960/max(w, h)
    px = round((960-round(w*ratio))/2-.1)
    py = round((960-round(h*ratio))/2-.1)
    gy, gx = np.mgrid[:120, :120]
    def raster(rect):
        left = (rect[0]-window[0])*ratio+px
        top = (rect[1]-window[1])*ratio+py
        right = (rect[2]-window[0])*ratio+px
        bottom = (rect[3]-window[1])*ratio+py
        return (np.clip(np.minimum(right, gx*8+8)-np.maximum(left, gx*8), 0, 8) *
                np.clip(np.minimum(bottom, gy*8+8)-np.maximum(top, gy*8), 0, 8))/64
    spatial = raster(window)
    for ignored in metadata["ignore_boxes_xyxy"]:
        spatial[raster(ignored) > 0] = 0
    area_weight, equal_weight = np.zeros((120, 120)), np.zeros((120, 120))
    count = 0
    for obj in metadata["valid_objects"]:
        left, top, right, bottom = obj["box_xyxy"]
        area = max(right-left, 0)*max(bottom-top, 0)
        if area <= 0 or area*(640/max(metadata["original_size"]))**2 >= 256:
            continue
        rect = [max(left, window[0]), max(top, window[1]),
                min(right, window[2]), min(bottom, window[3])]
        if max(rect[2]-rect[0], 0)*max(rect[3]-rect[1], 0)/area < .9:
            continue
        mask = raster(rect)*spatial
        mass = np.sum(mask)
        if mass:
            area_weight += mask
            equal_weight += mask/mass
            count += 1
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f"candidate69-shuffle-v1:{stem}:{view}".encode()).digest()[:8], "little"))
    supported = spatial > 0
    shuffled = np.zeros_like(spatial)
    shuffled[supported] = rng.permutation(equal_weight[supported])
    return dict(global_=spatial, tiny_pixel=area_weight, instance=equal_weight,
                shuffle=shuffled), count


def verify(device="0"):
    precision(False)
    root, prior = project_path(OUTPUT), project_path(PRIOR)
    plan = read(root/"plan.json")
    assert file_hash(root/"source.zip") == plan["source_archive_sha256"]
    with zipfile.ZipFile(root/"source.zip") as archive:
        for path, expected in plan["source_hashes"].items():
            assert file_hash(project_path(path)) == expected
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
    for path, expected in plan["installed_source_hashes"].items():
        assert file_hash(path) == expected
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_sha256"]
    frames = read(prior/"frames.json")
    for stem, entry in frames.items():
        assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == entry["image_sha256"]
        assert file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) == entry["metadata_sha256"]
    accum = dict(mass=np.zeros(4), sums=np.zeros((4, 128)), squares=np.zeros((4, 128)))
    for index, item in enumerate(read(root/"fit_manifest.json").values()):
        path = root/item["file"]
        assert file_hash(path) == item["sha256"]
        stem, view = item["stem"], item["view"]
        metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
        maps, count = independently_rebuild_maps(metadata, frames[stem]["windows"][view], stem, view)
        with np.load(path, allow_pickle=False) as z:
            assert int(z["tiny_count"]) == count
            for key in maps:
                np.testing.assert_array_equal(z[key], maps[key])
            independently = {k: [] for k in accum}
            for arm in ARMS:
                weights = maps["global_" if arm == "global" else arm].reshape(-1)
                flat = z["z"].astype(np.float64).reshape(128, -1)
                independently["mass"].append(float(np.sum(weights)))
                independently["sums"].append(np.einsum("cn,n->c", flat, weights, optimize=False))
                independently["squares"].append(np.einsum("cn,cn,n->c", flat, flat, weights, optimize=False))
            for key in accum:
                value = np.asarray(independently[key])
                np.testing.assert_allclose(value, z[key], rtol=1e-10, atol=1e-7)
                accum[key] += value
        if index == 0 or (index+1) % 64 == 0:
            print(f"Independent instance moment fit audit: {index+1}/320", flush=True)
    mean = accum["sums"]/accum["mass"][:, None]
    variance = np.maximum(accum["squares"]/accum["mass"][:, None]-mean**2, 0)
    with np.load(root/"moments.npz") as z:
        for key, value in dict(**accum, mean=mean, variance=variance).items():
            np.testing.assert_allclose(value, z[key], rtol=1e-10, atol=1e-7)
    with np.load(root/"bn.npz") as z:
        gamma, beta, old_var, eps = z["gamma"], z["beta"], z["old_var"], float(z["eps"])
    with np.load(root/"folded-strong.npz") as z:
        old_weight, old_bias = z["weight"], z["bias"]
    for i, arm in enumerate(ARMS):
        active = np.abs(gamma) >= 1e-6
        # Reconstruct pre-BN fitted moments, then compute the new running stats directly.
        fit_mean_shift = np.zeros(128)
        fit_variance = old_var.copy()
        fit_mean_shift[active] = (mean[i, active]-beta[active])*np.sqrt(old_var[active]+eps)/gamma[active]
        fit_variance[active] = variance[i, active]*(old_var[active]+eps)/gamma[active]**2
        new_var = .97*old_var+.03*fit_variance
        a = np.sqrt((old_var+eps)/(new_var+eps))
        b = beta*(1-a)-gamma*.03*fit_mean_shift/np.sqrt(new_var+eps)
        a[~active], b[~active] = 1, 0
        with np.load(root/f"folded-{arm}.npz") as z:
            np.testing.assert_allclose(a, z["a"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(b, z["b"], rtol=1e-10, atol=1e-12)
            np.testing.assert_array_equal(z["weight"], (old_weight.astype(float)*a[:, None, None, None]).astype(np.float16))
            np.testing.assert_array_equal(z["bias"], (old_bias.astype(float)*a+b).astype(np.float16))
    control = frozen_control()
    calibrator = load_calibrator(control)
    eval_manifest, predictions = read(root/"eval_manifest.json"), read(root/"predictions.json")
    for stem in plan["selection"]["screen"]+plan["selection"]["confirmation"]:
        entry = frames[stem]
        views = {a: [] for a in ("strong", *ARMS)}
        for view, window in enumerate(entry["windows"]):
            item = eval_manifest[f"{stem}:{view}"]
            assert file_hash(root/item["file"]) == item["sha256"]
            with np.load(root/item["file"], allow_pickle=False) as z:
                for arm in views:
                    values = torch.as_tensor(z[arm+"_decoded"][None], device="cuda:"+device)
                    if arm != "strong":
                        logits = torch.as_tensor(z[arm+"_raw_class"], device=values.device)
                        torch.testing.assert_close(values[0, 4:14], logits.sigmoid(), atol=0, rtol=0)
                        np.testing.assert_array_equal(z[arm+"_decoded"][:4], z["strong_decoded"][:4])
                    local = native_nms(values, (window[3]-window[1], window[2]-window[0]))
                    np.testing.assert_array_equal(local, z[arm+"_local"])
                    views[arm].append(map_tile_predictions(local, window) if view else local)
        item = predictions[stem]
        assert file_hash(root/item["file"]) == item["sha256"]
        with np.load(root/item["file"], allow_pickle=False) as z:
            for arm, arm_views in views.items():
                output = calibrated_candidates(arm_views, entry["windows"], entry["size"], [], calibrator)[0]
                np.testing.assert_array_equal(output, z[arm])
    report = metrics(root, plan["selection"], predictions, frames)
    assert report == read(root/"metrics.json")
    checks, differences = {}, {}
    p = plan["registration"]["policy_per_half"]
    for split, r in report.items():
        ap = r["AP"]
        c = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= p["tiny_GT_min"])
        d = {}
        for group, arm in [("tiny", a) for a in ("strong", *ARMS[:-1])] + [
                ("small_all", "strong"), ("all", "strong")]:
            key = f"{group}_AP_vs_{arm}"
            d[key] = (ap["instance"][group]["ap50_95"]-ap[arm][group]["ap50_95"])*100
            c[key] = d[key] >= p[key+"_min_pp"]
        d["recall"] = r["fixed_fp"]["instance"][0]["delta_recall_pp"]["small_all"]
        c["fixed_FP"] = d["recall"] >= p["small_all_recall_FPPI10_vs_strong_min_pp"]
        checks[split], differences[split] = c, d
    assert read(root/"decision.json") == dict(passed=all(all(c.values()) for c in checks.values()),
                                             checks=checks, differences=differences)
    stem = plan["selection"]["screen"][0]
    window = frames[stem]["windows"][0]
    image = cv2.imread(str(project_path(f"data/visdrone/images/train/{stem}.jpg")))
    extractor = FrozenViews(device)
    conv = extractor.head.cv3[0][1][1].conv
    original_hash = state_hash(extractor.model.predictor.model.model)
    try:
        with torch.inference_mode(), np.load(root/eval_manifest[f"{stem}:0"]["file"]) as z:
            for arm in ("strong", *ARMS):
                with np.load(root/f"folded-{arm}.npz") as weights:
                    conv.weight.copy_(torch.as_tensor(weights["weight"], device=conv.weight.device))
                    conv.bias.copy_(torch.as_tensor(weights["bias"], device=conv.bias.device))
                extractor.predict(image)
                np.testing.assert_array_equal(extractor.decoded[0].cpu().numpy(), z[arm+"_decoded"])
            conv.weight.copy_(torch.as_tensor(old_weight, device=conv.weight.device))
            conv.bias.copy_(torch.as_tensor(old_bias, device=conv.bias.device))
            assert state_hash(extractor.model.predictor.model.model) == original_hash
    finally:
        extractor.close()
    state = read(root/"state.json")
    assert state["before"] == state["after"] == original_hash
    result = dict(passed=True, independently_reconstructed_fit_views=320,
        cached_evaluation_views=245, cached_arms_each=5, actual_audit_forwards=5,
        explicit_audit_initialization_predictions=1, additional_fits=0,
        boundary="Independent masks, float64 moments and direct-BN folding; shared NMS/calibrator/COCO evaluator replay. Actual detector replay on ONE held-out image, not every modified forward. No official AP or training efficacy claim.")
    save_json(root/"verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
