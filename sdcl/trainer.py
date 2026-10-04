"""Local trainer/model extension; no site-packages monkey-patching."""

from copy import copy
import json

import ultralytics
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import unwrap_model

from .config import SDCLConfig, SUPPORTED_ULTRALYTICS
from .criterion import SDCLLoss
from .dataset import IgnoreDataset
from .geometry import prediction_ignore_mask


def check_version():
    if ultralytics.__version__ != SUPPORTED_ULTRALYTICS:
        raise RuntimeError(
            f"Adapter supports ultralytics=={SUPPORTED_ULTRALYTICS}, "
            f"found {ultralytics.__version__}. Install the pinned requirements."
        )


def make_dataset(args, img_path, data, mode, batch, stride, ignore):
    dataset = IgnoreDataset if ignore else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=args.rect or mode != "train",
        cache=args.cache or None,
        single_cls=False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task="detect",
        classes=None,
        data=data,
        fraction=args.fraction,
    )


class SDCLDetectionModel(DetectionModel):
    def init_criterion(self):
        if getattr(self.model[-1], "one2one_cv2", None) is not None:
            raise ValueError("End-to-end heads are not supported yet. Use YOLO11 n/s.")
        return SDCLLoss(self)


class SDCLValidator(DetectionValidator):
    """Diagnostic IoF filtering; metrics are not official VisDrone AP."""

    def __init__(self, *args, ignore_regions=True, **kwargs):
        self.ignore_regions = ignore_regions
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        return make_dataset(self.args, img_path, self.data, mode, batch, self.stride, self.ignore_regions)

    def update_metrics(self, preds, batch):
        if self.ignore_regions:
            filtered = []
            for index, pred in enumerate(preds):
                boxes = batch["ignore_bboxes"][batch["ignore_batch_idx"] == index]
                scale = boxes.new_tensor([batch["img"].shape[-1], batch["img"].shape[-2]] * 2)
                ignored = xywh2xyxy(boxes) * scale
                drop = prediction_ignore_mask(pred["bboxes"], ignored)
                filtered.append({key: value[~drop] for key, value in pred.items()})
            preds = filtered
        return super().update_metrics(preds, batch)


class SDCLTrainer(DetectionTrainer):
    def __init__(self, *args, sdcl=None, ignore_regions=True, **kwargs):
        check_version()
        overrides = kwargs.get("overrides") or {}
        if isinstance(overrides.get("device"), (list, tuple)) or "," in str(overrides.get("device", "0")):
            raise ValueError("Only single-device training is supported in this framework.")
        if overrides.get("single_cls") or overrides.get("classes") is not None:
            raise ValueError("Class filtering is not supported by the ignore adapter.")
        if overrides.get("task", "detect") != "detect":
            raise ValueError("Only horizontal-box detection is supported.")
        self.sdcl = sdcl or SDCLConfig()
        self.ignore_regions = ignore_regions
        self._last_logged_epoch = -1
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._sync_epoch)
        self.add_callback("on_fit_epoch_end", self._log_diagnostics)
        self.add_callback("on_train_start", self._save_manifest)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            SDCLDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        model.sdcl_settings = self.sdcl.to_dict()
        model.sdcl_ignore_regions = self.ignore_regions
        model.sdcl_epoch = 0
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return make_dataset(self.args, img_path, self.data, mode, batch, stride, self.ignore_regions)

    def get_validator(self):
        return SDCLValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks, ignore_regions=self.ignore_regions,
        )

    def _sync_epoch(self, trainer):
        model = unwrap_model(trainer.model)
        model.sdcl_epoch = trainer.epoch
        if getattr(model, "criterion", None) is not None:
            model.criterion.epoch = trainer.epoch
        if getattr(trainer, "ema", None) is not None:
            trainer.ema.ema.sdcl_epoch = trainer.epoch
            if getattr(trainer.ema.ema, "criterion", None) is not None:
                trainer.ema.ema.criterion.epoch = trainer.epoch

    def _log_diagnostics(self, trainer):
        if trainer.epoch == self._last_logged_epoch or trainer.epoch >= trainer.epochs:
            return
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        stats = getattr(criterion, "diagnostics", {})
        if stats:
            with (trainer.save_dir / "sdcl_statistics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": trainer.epoch, **stats}, allow_nan=False) + "\n")
            self._last_logged_epoch = trainer.epoch

    def _save_manifest(self, trainer):
        from .environment import environment_info
        from .data import sha256
        from pathlib import Path
        manifest = environment_info()
        data_yaml = Path(trainer.args.data)
        manifest_files = list((data_yaml.parent / "manifests").glob("*.json"))
        manifest.update(
            sdcl=self.sdcl.to_dict(),
            ignore_regions=self.ignore_regions,
            train=vars(trainer.args),
            metric_protocol="ignore-aware Ultralytics (diagnostic), NOT official VisDrone",
            dataset_hashes={str(path): sha256(path) for path in [data_yaml, *manifest_files] if path.is_file()},
        )
        (trainer.save_dir / "experiment.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

