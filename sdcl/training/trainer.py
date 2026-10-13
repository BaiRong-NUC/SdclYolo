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

from sdcl.config import SDCLConfig, SUPPORTED_ULTRALYTICS
from sdcl.training.criterion import SDCLLoss
from sdcl.training.coverage import CoverageConfig
from sdcl.data.dataset import IgnoreDataset
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.training.phase import PhaseConfig, phase_model_loss


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
    def loss(self, batch, preds=None):
        return phase_model_loss(self, batch, preds)

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
    def __init__(self, *args, sdcl=None, ignore_regions=True, phase=None, coverage=None, **kwargs):
        check_version()
        overrides = kwargs.get("overrides") or {}
        if isinstance(overrides.get("device"), (list, tuple)) or "," in str(overrides.get("device", "0")):
            raise ValueError("Only single-device training is supported in this framework.")
        if overrides.get("single_cls") or overrides.get("classes") is not None:
            raise ValueError("Class filtering is not supported by the ignore adapter.")
        if overrides.get("task", "detect") != "detect":
            raise ValueError("Only horizontal-box detection is supported.")
        self.sdcl = sdcl or SDCLConfig()
        self.phase = phase or PhaseConfig()
        self.coverage = coverage or CoverageConfig()
        if self.coverage.enabled and (self.phase.enabled or self.sdcl.enabled or not ignore_regions):
            raise ValueError("Coverage screen requires phase/SDCL off and ignore regions on.")
        if self.coverage.enabled and overrides.get("resume"):
            raise ValueError("The frozen coverage screen requires an uninterrupted run.")
        if self.phase.enabled and self.sdcl.enabled:
            raise ValueError("Phase screen must disable SDCL to isolate the new mechanism.")
        if self.phase.enabled and overrides.get("resume"):
            raise ValueError("Phase screen does not support resume.")
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
        model.phase_settings = self.phase.to_dict()
        model.coverage_settings = self.coverage.to_dict()
        model.phase_step = 0
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return make_dataset(self.args, img_path, self.data, mode, batch, stride, self.ignore_regions)

    def get_validator(self):
        if self.phase.enabled:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "phase_loss")
        return SDCLValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks, ignore_regions=self.ignore_regions,
        )

    def _sync_epoch(self, trainer):
        model = unwrap_model(trainer.model)
        model.sdcl_epoch = trainer.epoch
        if self.phase.enabled:
            model.phase_statistics = {}
            model.phase_diagnostics = {}
        if getattr(model, "criterion", None) is not None:
            model.criterion.epoch = trainer.epoch
            if self.coverage.enabled:
                model.criterion.assigner.statistics = {}
        if getattr(trainer, "ema", None) is not None:
            trainer.ema.ema.sdcl_epoch = trainer.epoch
            if getattr(trainer.ema.ema, "criterion", None) is not None:
                trainer.ema.ema.criterion.epoch = trainer.epoch

    def _log_diagnostics(self, trainer):
        if trainer.epoch == self._last_logged_epoch or trainer.epoch >= trainer.epochs:
            return
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        stats = getattr(criterion, "diagnostics", {})
        phase_stats = getattr(unwrap_model(trainer.model), "phase_statistics", {})
        if self.coverage.enabled:
            coverage_stats = getattr(getattr(criterion, "assigner", None), "statistics", {})
            with (trainer.save_dir / "coverage_statistics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": trainer.epoch, **coverage_stats}, allow_nan=False) + "\n")
            self._last_logged_epoch = trainer.epoch
        if phase_stats:
            phase_stats = dict(phase_stats)
            phase_stats["mean_distance_active"] = phase_stats.get("distance_sum", 0) / max(
                1, phase_stats.get("active_steps", 0))
            phase_stats["mean_weighted_loss_all_batches"] = phase_stats.get("weighted_loss_sum", 0) / max(
                1, phase_stats.get("batches", 0))
            with (trainer.save_dir / "phase_statistics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": trainer.epoch, **phase_stats}, allow_nan=False) + "\n")
        if stats:
            with (trainer.save_dir / "sdcl_statistics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": trainer.epoch, **stats}, allow_nan=False) + "\n")
            self._last_logged_epoch = trainer.epoch

    def _save_manifest(self, trainer):
        from sdcl.runtime.environment import environment_info
        from sdcl.data.prepare import sha256
        from pathlib import Path
        manifest = environment_info()
        data_yaml = Path(trainer.args.data)
        manifest_files = list((data_yaml.parent / "manifests").glob("*.json"))
        manifest.update(
            sdcl=self.sdcl.to_dict(),
            phase=self.phase.to_dict(),
            coverage=self.coverage.to_dict(),
            ignore_regions=self.ignore_regions,
            train=vars(trainer.args),
            metric_protocol="ignore-aware Ultralytics (diagnostic), NOT official VisDrone",
            dataset_hashes={str(path): sha256(path) for path in [data_yaml, *manifest_files] if path.is_file()},
        )
        (trainer.save_dir / "experiment.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

