"""Cache frozen visual features and train two matched 3870-parameter classifiers."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash, write_csv
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.class_margin_probe import (
    ARMS, CACHE, RUN, CONTROL_SHA, Projection, archive_source, check_sources, checked_control,
    objective, recipe, select_records, sources, verify_archive,
)
from sdcl.postprocess.rescue import save_json
from sdcl.runtime.files import atomic_replace
from sdcl.training.rescue import save_checkpoint
from sdcl.training.trainer import check_version


class FrozenViews:
    """Extract penultimate class features; hooks never alter the detector forward."""
    def __init__(self, device="0"):
        from ultralytics import YOLO
        check_version()
        self.control = checked_control()
        self.model = YOLO(self.control["detector"]["path"])
        self.kwargs = dict(imgsz=960, conf=.001, iou=.7, max_det=500, quantize=16,
                           rect=False, batch=1, device=device, verbose=False,
                           save=False, augment=False, agnostic_nms=False)
        self.model.predict(np.zeros((960, 960, 3), np.uint8), **self.kwargs)
        if not self.model.predictor.model.fp16:
            raise ValueError("Frozen visual probe requires FP16 square960 inference.")
        self.head = self.model.predictor.model.model.model[-1]
        self.weight = torch.stack([c[-1].weight[:, :, 0, 0].float().detach().cpu() for c in self.head.cv3])
        self.bias = torch.stack([c[-1].bias.float().detach().cpu() for c in self.head.cv3])
        if self.weight.shape != (3, 10, 128) or self.bias.shape != (3, 10):
            raise ValueError("Unexpected frozen head structure.")
        self.features = {}
        self.handles = [c[-1].register_forward_pre_hook(self.feature_hook(i))
                        for i, c in enumerate(self.head.cv3)]
        self.handles.append(self.head.register_forward_hook(self.output_hook))

    def feature_hook(self, level):
        def hook(module, inputs):
            self.features[level] = inputs[0].detach()
        return hook

    def output_hook(self, module, inputs, output):
        if not isinstance(output, tuple):
            raise ValueError("Expected decoded and raw detector outputs.")
        self.decoded = output[0].detach().clone()
        self.raw = output[1]

    def predict(self, image, projections=None):
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import scale_boxes
        original = self.model.predict(image, **self.kwargs)[0]
        base = torch.cat([original.boxes.xyxy, original.boxes.conf[:, None],
                          original.boxes.cls[:, None]], 1).float().cpu().numpy()
        outputs, timing = {"control": base}, {}
        self.projection_scores = {}
        input_shape = tuple(int(x * self.head.stride[0]) for x in self.raw["feats"][0].shape[2:])
        for arm, projection in (projections or {}).items():
            torch.cuda.synchronize()
            tick = perf_counter()
            scores = [F.conv2d(self.features[i], projection.weight[i, :, :, None, None].to(self.features[i]),
                               projection.bias[i].to(self.features[i])).view(1, 10, -1)
                      for i in range(3)]
            values = self.decoded.clone()
            self.projection_scores[arm] = torch.cat(scores, -1)
            values[:, 4:14] = self.projection_scores[arm].sigmoid()
            if not torch.equal(values[:, :4], self.decoded[:, :4]):
                raise AssertionError("Classification probe changed decoded regression boxes.")
            predicted = non_max_suppression(values, .001, .7, max_det=500)[0]
            predicted[:, :4] = scale_boxes(input_shape, predicted[:, :4], image.shape[:2])
            torch.cuda.synchronize()
            timing[arm] = (perf_counter() - tick) * 1000
            outputs[arm] = predicted.float().cpu().numpy()
        return outputs, timing

    def close(self):
        for handle in self.handles:
            handle.remove()


def targets_for_view(extractor, record, window, ignores):
    from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
    from ultralytics.utils.loss import v8DetectionLoss
    x1, y1, x2, y2 = window
    width, height = x2 - x1, y2 - y1
    device = extractor.raw["scores"].device
    scale = min(960 / width, 960 / height)
    padding = np.array([round((960 - round(width * scale)) / 2 - .1),
                        round((960 - round(height * scale)) / 2 - .1)])
    boxes, labels, tiny = [], [], []
    for obj in record["objects"]:
        box = np.asarray(obj["box_xyxy"], dtype=float)
        clipped = np.r_[np.maximum(box[:2], [x1, y1]), np.minimum(box[2:], [x2, y2])]
        area = np.prod(np.maximum(box[2:] - box[:2], 0))
        overlap = np.prod(np.maximum(clipped[2:] - clipped[:2], 0))
        if area <= 0 or overlap / area < .9:
            continue
        boxes.append((clipped - [x1, y1, x1, y1]) * scale + np.tile(padding, 2))
        labels.append(obj["class_id"])
        tiny.append(np.sqrt(area) * 640 / max(record["size"]) < recipe().tiny_reference_side)
    features = torch.cat([extractor.features[i].flatten(2).permute(0, 2, 1) for i in range(3)], 1)[0]
    levels = torch.cat([torch.full((extractor.features[i].shape[2] * extractor.features[i].shape[3],),
                                   i, device=device, dtype=torch.long) for i in range(3)])
    logits = extractor.raw["scores"].permute(0, 2, 1).float()
    anchors, stride = make_anchors(extractor.raw["feats"], extractor.head.stride, .5)
    if boxes:
        gt = torch.tensor(np.array(boxes), device=device, dtype=torch.float32)[None]
        cls = torch.tensor(labels, device=device, dtype=torch.float32)[None, :, None]
        decoder = v8DetectionLoss(extractor.model.predictor.model.model)
        distances = extractor.raw["boxes"].permute(0, 2, 1).float()
        decoded = decoder.bbox_decode(anchors.float(), distances)
        assigner = TaskAlignedAssigner(topk=10, num_classes=10, alpha=.5, beta=6.,
                                       stride=extractor.head.stride.tolist(), topk2=None)
        _, _, targets, fg, indices = assigner(logits.sigmoid(), decoded * stride,
                                             anchors * stride, cls, gt, torch.ones_like(cls, dtype=torch.bool))
        is_tiny = torch.tensor(tiny, device=device)[indices[0]] & fg[0]
    else:
        targets = logits.new_zeros(logits.shape)
        fg = torch.zeros(logits.shape[:2], device=device, dtype=torch.bool)
        is_tiny = fg[0].clone()
    points = anchors * stride
    ignored = torch.zeros(len(points), device=device, dtype=torch.bool)
    for box in ignores:
        box = np.asarray(box, dtype=float)
        lo = np.maximum(box[:2], [x1, y1])
        hi = np.minimum(box[2:], [x2, y2])
        if np.all(hi > lo):
            lo = torch.tensor((lo - [x1, y1]) * scale + padding, device=device)
            hi = torch.tensor((hi - [x1, y1]) * scale + padding, device=device)
            ignored |= ((points >= lo) & (points <= hi)).all(-1)
    # Same sample order for both objectives; background excludes train-only ignores.
    selected = []
    confidence = logits[0].sigmoid().amax(-1)
    for level in range(3):
        pos = torch.where((levels == level) & fg[0])[0]
        if len(pos):
            seed = int.from_bytes(hashlib.sha256(
                f"{record['stem']}:{window}:{level}".encode()).digest()[:8], 'little')
            order = np.random.default_rng(seed).permutation(len(pos))[:recipe().positive_per_level]
            selected.append(pos[torch.tensor(order, device=device)])
        neg = torch.where((levels == level) & ~fg[0] & ~ignored)[0]
        neg = neg[torch.argsort(confidence[neg], descending=True, stable=True)[:recipe().negative_per_level]]
        selected.append(neg)
    chosen = torch.cat(selected)
    return (features[chosen].cpu().numpy(), levels[chosen].cpu().numpy(),
            targets[0, chosen].cpu().numpy(), is_tiny[chosen].cpu().numpy())


def checked_records():
    control = checked_control()
    dataset = project_path("data/visdrone")
    records = load_records(dataset, "train")
    val = load_records(dataset, "val")
    from sdcl.postprocess.localization import frozen_control
    if len(records) != 6471 or len(val) != 548 or record_hash(val) != frozen_control()["frozen_validation"]["records_sha256"]:
        raise ValueError("Frozen train/validation data mismatch.")
    if {r['stem'] for r in records} & {r['stem'] for r in val}:
        raise ValueError("Training/validation overlap.")
    return dataset, records


def prepare(output=CACHE, device="0", limit=None):
    if not str(device).isdigit() or (limit is not None and not 1 <= limit <= 128):
        raise ValueError("Use a CUDA index; optional limit 1..128 is smoke only.")
    dataset, records = checked_records()
    chosen = select_records(records, limit)
    output = project_path(output)
    expected = dict(recipe=asdict(recipe()), source_hashes=sources(), control_sha256=CONTROL_SHA,
                    train_records_sha256=record_hash(records), selected_records_sha256=record_hash([r for r, _ in chosen]),
                    full_fixed_subset=limit is None, images=len(chosen), split="train", limit=limit)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != expected:
            raise ValueError("Cache source/data/recipe mismatch; refuse reuse.")
        verify_archive(output, expected["source_hashes"], manifest["source_archive_sha256"])
        for item in manifest["chunks"].values():
            if file_hash(output / item['file']) != item['sha256']:
                raise ValueError("Corrupt feature cache chunk.")
        if manifest["complete"]:
            return {"output": str(output), "complete": True, "images": len(chosen)}
    else:
        if output.exists():
            raise FileExistsError("Refuse an unregistered cache directory.")
        (output / "chunks").mkdir(parents=True)
        archive_source(output, expected["source_hashes"])
        manifest = dict(settings=expected, source_archive_sha256=file_hash(output / "source.zip"),
                        complete=False, chunks={})
        save_json(path, manifest)
    extractor = FrozenViews(device)
    with (output / "initial_projection.npz").open("wb") as stream:
        np.savez(stream, weight=extractor.weight.numpy(), bias=extractor.bias.numpy())
    manifest['initial_projection_sha256'] = file_hash(output / "initial_projection.npz")
    started = perf_counter()
    try:
        for index, (record, monitor) in enumerate(chosen, 1):
            if record['stem'] in manifest['chunks']:
                continue
            image = cv2.imread(record['image'])
            if image is None or list(image.shape[1::-1]) != record['size']:
                raise ValueError("Invalid training image.")
            ignores = json.loads((dataset / "metadata/train" / f"{record['stem']}.json").read_text())['ignore_boxes_xyxy']
            rows = []
            windows = [[0, 0, *record['size']], *tile_windows(*record['size'])]
            with torch.inference_mode():
                for window in windows:
                    x1, y1, x2, y2 = window
                    extractor.predict(image[y1:y2, x1:x2].copy())
                    rows.append(targets_for_view(extractor, record, window, ignores))
            arrays = [np.concatenate([r[i] for r in rows]) for i in range(4)]
            target = output / "chunks" / f"{record['stem']}.npz"
            partial = target.with_suffix('.partial')
            with partial.open('wb') as stream:
                np.savez_compressed(stream, features=arrays[0], levels=arrays[1].astype(np.int8),
                                    targets=arrays[2], tiny=arrays[3])
            atomic_replace(partial, target)
            manifest['chunks'][record['stem']] = dict(file=target.relative_to(output).as_posix(),
                sha256=file_hash(target), monitor=monitor, rows=len(arrays[0]), tiny_positive=int(arrays[3].sum()))
            save_json(path, manifest)
            if index == 1 or index % 25 == 0 or index == len(chosen):
                print(f"Visual feature cache: {index}/{len(chosen)}; {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    check_sources(expected['source_hashes'])
    if record_hash(load_records(dataset,'train')) != expected['train_records_sha256']:
        raise ValueError("Training inputs changed during caching.")
    manifest['complete'] = True
    save_json(path, manifest)
    return dict(output=str(output), complete=True, images=len(chosen), full_fixed_subset=limit is None)


def read_cache(path, *, source_check=check_sources):
    path = project_path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    source_check(manifest['settings']['source_hashes'])
    verify_archive(path, manifest['settings']['source_hashes'], manifest['source_archive_sha256'])
    settings = manifest['settings']
    if (not manifest['complete'] or settings['recipe'] != asdict(recipe())
            or settings['control_sha256'] != CONTROL_SHA or settings['split'] != 'train'
            or settings['full_fixed_subset'] != (settings['limit'] is None)):
        raise ValueError("Incomplete/changed visual cache.")
    if file_hash(path / 'initial_projection.npz') != manifest['initial_projection_sha256']:
        raise ValueError("Original classifier weights changed.")
    _, records = checked_records()
    chosen = select_records(records, manifest['settings']['limit'])
    if record_hash(records) != manifest['settings']['train_records_sha256'] or record_hash([r for r,_ in chosen]) != manifest['settings']['selected_records_sha256']:
        raise ValueError("Probe train split changed.")
    if settings['images'] != len(chosen) or set(manifest['chunks']) != {r['stem'] for r,_ in chosen}:
        raise ValueError("Wrong feature chunk set.")
    groups = {False: [], True: []}
    for record, monitor in chosen:
        item = manifest['chunks'][record['stem']]
        if item['monitor'] != monitor or file_hash(path / item['file']) != item['sha256']:
            raise ValueError("Cache chunk content/split changed.")
        with np.load(path / item['file']) as chunk:
            groups[monitor].append(tuple(chunk[k] for k in ('features','levels','targets','tiny')))
    arrays = {key: tuple(np.concatenate([chunk[i] for chunk in values]) for i in range(4))
              for key, values in groups.items()}
    original = np.load(path / 'initial_projection.npz')
    if original['weight'].shape != (3,10,128) or original['bias'].shape != (3,10):
        raise ValueError('Unexpected initial projection layout.')
    return manifest, arrays, original


def train(cache=CACHE, output=RUN, device="0", resume=False):
    fixed = recipe()
    cache, output = project_path(cache), project_path(output)
    manifest, arrays, original = read_cache(cache)
    digest = file_hash(cache / 'manifest.json')
    path = output / 'last.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False) if resume else None
    if saved is not None:
        if (saved['cache_sha256'] != digest or saved['recipe'] != asdict(fixed)
                or saved['full_fixed_subset'] != manifest['settings']['full_fixed_subset']
                or not 0 <= saved['epoch'] <= fixed.epochs):
            raise ValueError("Resume cache/recipe mismatch.")
        check_sources(saved['source_hashes'])
        verify_archive(output, saved['source_hashes'], saved['source_archive_sha256'])
        if saved['epoch'] == fixed.epochs:
            return dict(output=str(output), epoch=fixed.epochs, already_complete=True)
    else:
        if output.exists():
            raise FileExistsError("Refusing to overwrite/increment this run.")
        output.mkdir(parents=True)
        archive_source(output, sources())
    gpu = torch.device(f'cuda:{device}')
    torch.manual_seed(fixed.seed)
    torch.use_deterministic_algorithms(True)
    models = {arm: Projection(torch.from_numpy(original['weight']), torch.from_numpy(original['bias'])).to(gpu)
              for arm in ARMS}
    base = Projection(torch.from_numpy(original['weight']), torch.from_numpy(original['bias'])).to(gpu).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=fixed.learning_rate,
                    weight_decay=fixed.weight_decay) for arm, model in models.items()}
    if saved is not None:
        for arm in ARMS:
            models[arm].load_state_dict(saved['models'][arm])
            optimizers[arm].load_state_dict(saved['optimizers'][arm])
    data = {group: tuple(torch.from_numpy(a).to(gpu, dtype=torch.float32 if i in (0,2) else (torch.long if i==1 else torch.bool))
                        for i,a in enumerate(items)) for group,items in arrays.items()}
    plan = dict(recipe=asdict(fixed), source_hashes=sources(), source_archive_sha256=file_hash(output/'source.zip'),
                cache_path=str(cache), cache_sha256=digest, control_sha256=CONTROL_SHA,
                full_fixed_subset=manifest['settings']['full_fixed_subset'], parameters=3870,
                fit_rows=len(data[False][0]), monitor_rows=len(data[True][0]),
                training_scope='Only final class projections. Detector/backbone/box branch/calibrator weights remain unchanged.',
                epoch_selection='Last epoch10 only; monitor loss never selects epochs or hyperparameters.')
    save_json(output/'plan.json', plan)
    logs = saved['logs'] if saved else []
    x, levels, targets, tiny = data[False]
    for epoch in range(saved['epoch'] if saved else 0, fixed.epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(epoch+fixed.seed)).to(gpu)
        losses = dict.fromkeys(ARMS,0.)
        for start in range(0,len(order),fixed.batch):
            idx = order[start:start+fixed.batch]
            with torch.no_grad():
                reference = base(x[idx],levels[idx])
            for arm in ARMS:
                optimizers[arm].zero_grad(set_to_none=True)
                loss = objective(models[arm](x[idx],levels[idx]),targets[idx],tiny[idx],reference,arm)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite classification loss.")
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(idx)
        row = dict(epoch=epoch+1, **{f'{arm}_fit_loss':v/len(x) for arm,v in losses.items()})
        hx, hl, hy, ht = data[True]
        with torch.no_grad():
            for arm in ARMS:
                total = 0.
                for start in range(0,len(hx),fixed.batch):
                    stop=start+fixed.batch
                    total += float(objective(models[arm](hx[start:stop],hl[start:stop]),hy[start:stop],ht[start:stop],
                                              base(hx[start:stop],hl[start:stop]),arm))*len(hx[start:stop])
                row[f'{arm}_monitor_loss']=total/len(hx)
        logs.append(row)
        save_checkpoint(path,{**plan,'epoch':epoch+1,'models':{arm:{k:v.detach().cpu() for k,v in model.state_dict().items()}
            for arm,model in models.items()},'optimizers':{arm:o.state_dict() for arm,o in optimizers.items()},'logs':logs})
        write_csv(output/'losses.csv',logs)
        print(json.dumps(row),flush=True)
    check_sources(plan['source_hashes'])
    return dict(output=str(output), epoch=fixed.epochs, full_fixed_subset=plan['full_fixed_subset'])
