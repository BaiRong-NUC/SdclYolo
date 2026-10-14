"""Actual last-class-projection quality-gap gradients, without parameter updates."""

import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import base, dataset as dataset_module, augment
from ultralytics.nn.modules import head as head_module
from ultralytics.utils import loss, tal
from ultralytics.utils.tal import make_anchors
import yaml

from sdcl.config import SDCLConfig, project_path
from sdcl.data.dataset import IgnoreDataset
from sdcl.data.geometry import anchor_ignore_mask
from sdcl.evaluation.scale_gradient_conflict import model_digest
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash
from sdcl.training.criterion import SDCLLoss
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate65_positive_rank_gradient_registration_20261015.json"
AUDIT = "docs/literature/positive_rank_gradient_prior_audit_20261015.json"
PRIOR50 = "output/analysis/assignment-reachability-fixed-20261014"
PRIOR63 = "output/analysis/assignment-quality-conflict-fixed-20261015"
OUTPUT = "output/analysis/positive-rank-gradient-fixed-20261015"
ARMS = ("original", "transport", "ordinary")
GROUPS = dict(all=float("inf"), tiny=256., small_all=1024.)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def pair_and_targets(frame, ignored):
    n = len(frame["gt"])
    pairs = np.full((n, 2), -1, int)
    pair_iou, pair_probability = np.zeros((n, 2)), np.zeros((n, 2))
    transport, ordinary = frame["target_mass"].astype(float).copy(), frame["target_mass"].astype(float).copy()
    for g, gt in enumerate(frame["gt"]):
        pool = np.flatnonzero(frame["foreground"] & (frame["owner"] == g) & (frame["target_mass"] > 0))
        if not len(pool):
            continue
        iou = box_iou_matrix([gt], frame["decoded_boxes"][pool])[0]
        if np.any(iou <= 0):
            raise ValueError("Productive positive has zero IoU; preserve fixed quality-control failure.")
        p = frame["probabilities"][pool, int(frame["classes"][g])]
        best, peak = np.lexsort((pool, -p, -iou))[0], np.lexsort((pool, -iou, -p))[0]
        pairs[g] = pool[[best, peak]]
        pair_iou[g], pair_probability[g] = iou[[best, peak]], p[[best, peak]]
        rank = np.lexsort((pool, frame["target_mass"][pool], iou))
        mass = frame["target_mass"][pool].astype(float)
        transport[pool[rank]] = np.sort(mass, kind="stable")
        total = mass.sum()
        lo, hi = 0., 1/min(iou)
        for _ in range(60):
            middle = (lo+hi)/2
            if np.minimum(1., middle*iou).sum() < total:
                lo = middle
            else:
                hi = middle
        ordinary[pool] = np.minimum(1., (lo+hi)/2*iou)
        assert abs(ordinary[pool].sum()-total) <= 1e-10
        np.testing.assert_array_equal(np.sort(transport[pool]), np.sort(mass))
    eligible = ((pairs >= 0).all(1) & (pair_iou[:, 0] >= .75) &
                (pair_iou[:, 0]-pair_iou[:, 1] >= .10) &
                (pair_probability[:, 1]-pair_probability[:, 0] >= .05))
    safe = np.maximum(pairs, 0)
    eligible &= ~ignored[safe].any(1)
    areas = np.maximum(frame["gt"][:, 2:]-frame["gt"][:, :2], 0).prod(1)*(640/960)**2
    return dict(pairs=pairs, pair_iou=pair_iou, pair_probability=pair_probability, eligible=eligible,
                areas=areas, mass_original=frame["target_mass"].copy(), mass_transport=transport.astype(np.float32),
                mass_ordinary=ordinary.astype(np.float32))


def gap_directions(features, levels, pairs, classes, gradients):
    result = {a: np.zeros(len(pairs)) for a in ARMS}
    hnorm = np.zeros(len(pairs))
    for g, (best, peak) in enumerate(pairs):
        if min(best, peak) < 0:
            continue
        cls, lb, lp = int(classes[g]), int(levels[best]), int(levels[peak])
        b, p = features[best].astype(float), features[peak].astype(float)
        hn = np.sqrt(np.dot(b-p, b-p)) if lb == lp else np.sqrt(np.dot(b, b)+np.dot(p, p)+2.)
        hnorm[g] = hn
        for arm, (weight, bias) in gradients.items():
            weight, bias = weight.astype(float), bias.astype(float)
            dot = np.dot(weight[lb, cls], b)-np.dot(weight[lp, cls], p)+bias[lb, cls]-bias[lp, cls]
            gn = np.sqrt((weight**2).sum()+(bias**2).sum())
            result[arm][g] = -dot/(gn*hn) if gn > 0 and hn > 0 else 0.
    return result, hnorm


def frame_statistics(values, directions):
    eligible = values["eligible"]
    worsening = eligible & (directions["original"] < -.01)
    repaired = {a: worsening & (directions[a] > .01) for a in ("transport", "ordinary")}
    groups = {}
    for group, limit in GROUPS.items():
        keep = values["areas"] < limit
        groups[group] = dict(gt=int(keep.sum()), eligible=int((keep & eligible).sum()),
            worsening=int((keep & worsening).sum()), worsening_images=int((keep & worsening).any()),
            repaired_transport=int((keep & repaired["transport"]).sum()),
            repaired_ordinary=int((keep & repaired["ordinary"]).sum()))
    return groups


def aggregate(traces, policy):
    metrics, checks = {}, {}
    for split, rows in traces.items():
        total = {g: {k: sum(r["groups"][g][k] for r in rows) for k in rows[0]["groups"][g]} for g in GROUPS}
        for m in total.values():
            m["worsening_fraction_allGT"] = m["worsening"]/m["gt"] if m["gt"] else 0.
            m["repair_gain_fraction_allGT"] = (m["repaired_transport"]-m["repaired_ordinary"])/m["gt"] if m["gt"] else 0.
        prefixes = sorted({r["stem"][:7] for r in rows})
        gt = np.array([sum(r["groups"]["tiny"]["gt"] for r in rows if r["stem"][:7] == s) for s in prefixes])
        delta = np.array([sum(r["groups"]["tiny"]["repaired_transport"]-r["groups"]["tiny"]["repaired_ordinary"]
                             for r in rows if r["stem"][:7] == s) for s in prefixes])
        draws = np.random.default_rng(0).multinomial(len(prefixes), [1/len(prefixes)]*len(prefixes), size=400)
        denominator = draws@gt
        ci = np.percentile((draws@delta)/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        total.update(sequence_units=len(prefixes), repair_gain_sequence_bootstrap95ci=ci)
        m = total["tiny"]
        checks[split] = dict(GT=m["gt"] >= policy["tiny_GT_min"],
            worsening=m["worsening"] >= policy["tiny_worsening_GT_min"],
            images=m["worsening_images"] >= policy["tiny_worsening_images_min"],
            fraction=m["worsening_fraction_allGT"] >= policy["tiny_worsening_fraction_allGT_min"],
            repair=m["repaired_transport"] >= policy["tiny_transport_repaired_GT_min"],
            vs_ordinary=m["repair_gain_fraction_allGT"] >= policy["tiny_repair_gain_vs_ordinary_fraction_allGT_min"],
            interval=ci is not None and ci[0] > policy["tiny_repair_gain_sequence_interval_lower_min_exclusive"])
        metrics[split] = total
    passed = all(all(v.values()) for v in checks.values())
    return metrics, dict(checks=checks, decision=(
        "REPLICATED_LAST_CLASS_PROJECTION_RANK_HARM_SUPPORTS_SEPARATE_BOUNDED_UPDATE_PILOT" if passed else
        "STOP_THIS_WITHINGT_QUALITY_HISTOGRAM_TRANSPORT_CLASS_PROJECTION_ROUTE"),
        scope="Actual last-classprojection first-order directions only; no optimizerupdate,AP,deployability,generalization ornovelty claim.")


def run(device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    check_version()
    if device != "0" or not torch.cuda.is_available():
        raise ValueError("Fixed CUDA0 FP32 probe.")
    root, prior50, prior63 = project_path(OUTPUT), project_path(PRIOR50), project_path(PRIOR63)
    if root.exists():
        raise FileExistsError(root)
    closeouts = {n: project_path(f"docs/experiments/candidate{n}_"+name+"_result_20261015.json") for n, name in
                 ((63, "assignment_quality_conflict"),)}
    closeouts[50] = project_path("docs/experiments/candidate50_assignment_reachability_result_20261014.json")
    for number, prior in ((50, prior50), (63, prior63)):
        for name, digest in read(closeouts[number])["artifacts_sha256"].items():
            assert file_hash(prior/name) == digest
        plan = read(prior/"plan.json")
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == digest
        for item in plan.get("installed_sources", {}).values():
            assert file_hash(item["path"]) == item["sha256"]
        for name, digest in plan.get("installed_source_hashes", {}).items():
            assert file_hash(name) == digest
    old, previous = read(prior50/"plan.json"), read(prior50/"frames.json")
    assignment_plan, assignment_frames = read(prior63/"plan.json"), read(prior63/"frames.json")
    selection = assignment_plan["image_selection"]
    assert sum(map(len, selection.values())) == 64
    audit = read(project_path(AUDIT))
    for paper in audit["papers"]:
        for key in ("pdf", "text", "preview"):
            if key in paper:
                assert file_hash(project_path(paper[key])) == paper[key+"_sha256"]
    registration = read(project_path(REGISTRATION))
    paths = (REGISTRATION, AUDIT, "sdcl/evaluation/positive_rank_gradient.py", "scripts/evaluation/diagnose_positive_rank_gradient.py",
        "tests/test_positive_rank_gradient.py", "sdcl/data/dataset.py", "sdcl/data/geometry.py",
        "sdcl/training/criterion.py", "sdcl/training/trainer.py", "sdcl/config.py",
        "sdcl/evaluation/scale_gradient_conflict.py", "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = {inspect.getfile(m): file_hash(inspect.getfile(m)) for m in (base, dataset_module, augment, head_module, loss, tal)}
    label_cache = project_path("data/visdrone/labels/train.cache")
    cache_hash = file_hash(label_cache)
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            archive.write(project_path(name), name)
        for path in installed:
            archive.write(path, "installed/"+Path(path).name)
    plan = dict(registration=registration, image_selection=selection, source_hashes=hashes,
        installed_source_hashes=installed, source_archive_sha256=file_hash(root/"source.zip"),
        prior_closeout_sha256={str(n): file_hash(p) for n, p in closeouts.items()},
        prior_plan_sha256={"50": file_hash(prior50/"plan.json"), "63": file_hash(prior63/"plan.json")},
        checkpoint=old["checkpoint"], checkpoint_sha256=old["checkpoint_sha256"], label_cache_sha256=cache_hash)
    save(root/"plan.json", plan)
    args = get_cfg(overrides=dict(imgsz=960, batch=1, workers=0, rect=False, plots=False, device=device))
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = IgnoreDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
        augment=False, hyp=args, rect=False, cache=False, single_cls=False, stride=32, pad=0.,
        prefix="positive-rank: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    checkpoint = project_path(old["checkpoint"])
    assert file_hash(checkpoint) == old["checkpoint_sha256"]
    model = YOLO(str(checkpoint)).model.float().cuda().eval()
    model.args, model.sdcl_settings = args, SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions, model.coverage_settings = True, {"enabled": False}
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]
    parameters, names, captured = [], [], {}
    for level, module in enumerate(head.cv3):
        for name in ("weight", "bias"):
            p = getattr(module[-1], name)
            p.requires_grad_(True)
            parameters.append(p)
            names.append(next(n for n, v in model.named_parameters() if v is p))
    assert sum(p.numel() for p in parameters) == 3870
    handles = []
    for level, module in enumerate(head.cv3):
        handles.append(module[-1].register_forward_pre_hook(
            lambda m, inputs, level=level: captured.update({level: inputs[0].detach()})))
    criterion = SDCLLoss(model)
    assert {k: getattr(criterion.assigner, k) for k in old["actual_assigner"]} == old["actual_assigner"]
    before = model_digest(model)
    plan.update(parameter_names=names, parameter_count=3870, model_state_before=before,
                cls_gain=float(args.cls), actual_assigner=old["actual_assigner"])
    save(root/"plan.json", plan)
    traces, manifest, count = {s: [] for s in selection}, {}, 0
    quantization = 0.
    try:
        for split, stems in selection.items():
            for stem in stems:
                entry = previous[stem]
                assert file_hash(prior50/entry["file"]) == entry["sha256"]
                for kind, folder, suffix in (("image", "images", ".jpg"), ("metadata", "metadata", ".json"), ("label", "labels", ".txt")):
                    assert file_hash(project_path(f"data/visdrone/{folder}/train/{stem}{suffix}")) == entry[kind+"_sha256"]
                with np.load(prior50/entry["file"], allow_pickle=False) as archive:
                    frame = {k: archive[k].copy() for k in archive.files}
                aentry = assignment_frames[stem]
                assert file_hash(prior63/aentry["file"]) == aentry["sha256"]
                with np.load(prior63/aentry["file"], allow_pickle=False) as archive:
                    old_ignored = archive["ignored"].copy()
                batch = dataset.collate_fn([dataset[indices[stem]]])
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch["img"] = batch["img"].float()/255
                raw = model(batch["img"])[1]
                logits = raw["scores"].permute(0, 2, 1).contiguous()
                anchors, stride = make_anchors(raw["feats"], criterion.stride, .5)
                decoded = criterion.bbox_decode(anchors, raw["boxes"].permute(0, 2, 1).contiguous())*stride
                tensor = lambda a: torch.tensor(a, device="cuda")
                labels, gt = tensor(frame["classes"][None, :, None]), tensor(frame["gt"][None])
                targets_input = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1)
                rebuilt = criterion.preprocess(targets_input, 1, tensor([960]*4))
                np.testing.assert_array_equal(rebuilt[0, :, 1:].cpu().numpy(), frame["gt"])
                np.testing.assert_array_equal(rebuilt[0, :, 0].cpu().numpy(), frame["classes"])
                np.testing.assert_array_equal(logits.detach().sigmoid()[0].cpu().numpy(), frame["probabilities"])
                np.testing.assert_array_equal(decoded.detach()[0].cpu().numpy(), frame["decoded_boxes"])
                with torch.no_grad():
                    assigned = criterion.assigner(logits.sigmoid(), decoded, anchors*stride, labels, gt, gt.sum(2, keepdim=True) > 0)
                for value, name in ((assigned[1], "assigned_boxes"), (assigned[3], "foreground"), (assigned[4], "owner")):
                    np.testing.assert_array_equal(value[0].cpu().numpy(), frame[name])
                np.testing.assert_array_equal(assigned[2][0].sum(-1).cpu().numpy(), frame["target_mass"])
                ignored = anchor_ignore_mask(anchors*stride, batch["ignore_bboxes"], batch["ignore_batch_idx"], 1, (960, 960))[0]
                np.testing.assert_array_equal(ignored.cpu().numpy(), old_ignored)
                values = pair_and_targets(frame, old_ignored)
                targets = {"original": assigned[2].detach()}
                positive = assigned[3][0]
                classes_at_row = labels[0, assigned[4][0], 0].long()
                for arm in ("transport", "ordinary"):
                    targets[arm] = assigned[2].detach().clone()
                    targets[arm][0, positive, classes_at_row[positive]] = tensor(values["mass_"+arm])[positive]
                for g in range(len(frame["gt"])):
                    pool = frame["foreground"] & (frame["owner"] == g) & (frame["target_mass"] > 0)
                    quantization = max(quantization, abs(values["mass_ordinary"][pool].astype(float).sum()-frame["target_mass"][pool].astype(float).sum()))
                gradients, losses = {}, {}
                denominator = float(assigned[2].sum().clamp_min(1))
                for arm in ARMS:
                    bce = F.binary_cross_entropy_with_logits(logits, targets[arm], reduction="none")
                    bce = bce.masked_fill((ignored[None] & ~assigned[3])[..., None], 0)
                    objective = bce.sum()/denominator*args.cls
                    computed = torch.autograd.grad(objective, parameters, retain_graph=arm != ARMS[-1])
                    w = np.stack([computed[2*i].detach().cpu().numpy()[:, :, 0, 0] for i in range(3)])
                    b = np.stack([computed[2*i+1].detach().cpu().numpy() for i in range(3)])
                    gradients[arm], losses[arm] = (w, b), float(objective.detach())
                features = torch.cat([captured[i].flatten(2).transpose(1, 2) for i in range(3)], 1)[0].cpu().numpy()
                levels = np.concatenate([np.full(captured[i].shape[-2]*captured[i].shape[-1], i, np.int8) for i in range(3)])
                directions, norms = gap_directions(features, levels, values["pairs"], frame["classes"], gradients)
                groups = frame_statistics(values, directions)
                path = root/"frames"/(stem+".npz")
                np.savez_compressed(path, **values, features=features, levels=levels,
                    logits=logits.detach()[0].cpu().numpy(), probabilities=frame["probabilities"],
                    gt=frame["gt"], classes=frame["classes"], foreground=frame["foreground"], owner=frame["owner"],
                    ignored=old_ignored, hnorm=norms, denominator=np.array(denominator),
                    **{"cosine_"+a: v for a, v in directions.items()},
                    **{"gradient_weight_"+a: v[0] for a, v in gradients.items()},
                    **{"gradient_bias_"+a: v[1] for a, v in gradients.items()})
                manifest[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path), split=split,
                    prior50_frame_sha256=entry["sha256"], prior63_frame_sha256=aentry["sha256"])
                traces[split].append(dict(stem=stem, groups=groups, losses=losses))
                count += 1
                if count % 16 == 0:
                    print(f"Positive-rank projection gradients: {count}/64", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    after = model_digest(model)
    assert before == after
    assert file_hash(label_cache) == cache_hash
    assert all(file_hash(project_path(p)) == d for p, d in hashes.items())
    assert all(file_hash(p) == d for p, d in installed.items())
    metrics, decision = aggregate(traces, registration["policy_per_half"])
    for name, value in (("frames.json", manifest), ("per_image.json", traces), ("metrics.json", metrics), ("decision.json", decision)):
        save(root/name, value)
    result = dict(experiment=registration["experiment"], metrics=metrics, decision=decision,
        model_state_before=before, model_state_after=after, ordinary_FP32_target_quantization_max_GTmass_drift=quantization,
        costs=dict(JPEG_samples=64, detector_forwards=64, actual_projection_autograd_calls=192,
            optimizer_steps=0, parameter_updates=0, AP_evaluations=0, official_AP_evaluations=0,
            elapsed_seconds=perf_counter()-started),
        artifacts_sha256={p.name: file_hash(p) for p in root.iterdir() if p.is_file()})
    save(root/"result.json", result)
    print(json.dumps(decision, indent=2), flush=True)
    return result
