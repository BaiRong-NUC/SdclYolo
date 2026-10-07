"""Command-line entry points for the first reproducible experiment loop."""

import argparse
from datetime import datetime
from functools import partial
import json

from .config import SDCLConfig, load_experiment, project_path


def train_experiment(config_file, **overrides):
    from .trainer import SDCLTrainer

    arguments, settings, ignore = load_experiment(config_file)
    for key, value in overrides.items():
        if value is not None:
            arguments[key] = str(project_path(value)) if key in {"data", "resume"} else value
    if arguments.get("resume"):
        from pathlib import Path
        checkpoint = Path(arguments["resume"])
        manifest_path = checkpoint.parent.parent / "experiment.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing resume manifest: {manifest_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_settings = saved.get("sdcl")
        if (
            not isinstance(saved_settings, dict)
            or SDCLConfig.from_dict(saved_settings) != settings
            or saved.get("ignore_regions") != ignore
        ):
            raise ValueError("Resume requires the same SDCL settings and ignore protocol as the saved run.")
    trainer = SDCLTrainer(overrides=arguments, sdcl=settings, ignore_regions=ignore)
    trainer.train()
    return {"run": str(trainer.save_dir), "metric_protocol": "ignore-aware Ultralytics, not official VisDrone"}


def smoke_test(output=None, device="0"):
    from .data import audit_dataset, make_toy_visdrone, prepare_visdrone
    from .trainer import SDCLTrainer, SDCLValidator
    from ultralytics import YOLO

    root = project_path(output) if output else project_path(f"output/smoke/{datetime.now():%Y%m%d-%H%M%S-%f}")
    if root.exists():
        raise FileExistsError(f"Smoke output already exists: {root}")
    raw = make_toy_visdrone(root / "raw")
    data = prepare_visdrone(raw, root / "dataset")
    audit = audit_dataset(data.parent)
    if not audit["ok"]:
        raise RuntimeError(audit)
    settings = SDCLConfig(warmup_epochs=0, ramp_epochs=0, log_interval=1)
    completed = {}
    for name, enabled in (("baseline", False), ("sdcl", True)):
        config = SDCLConfig(**{**settings.to_dict(), "enabled": enabled})
        trainer = SDCLTrainer(
            overrides={
                "model": "yolo11n.yaml", "data": str(data), "device": device,
                "epochs": 1, "batch": 2, "imgsz": 128, "workers": 0,
                "amp": False, "plots": False, "pretrained": False,
                "project": str(root / "runs"), "name": name, "exist_ok": False,
                "optimizer": "SGD", "lr0": 0.001, "warmup_epochs": 0,
                "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
                "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                "fliplr": 0.0, "flipud": 0.0, "close_mosaic": 0,
                "max_det": 500, "patience": 0, "deterministic": True,
            },
            sdcl=config, ignore_regions=True,
        )
        trainer.train()
        checkpoint = trainer.save_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Checkpoint not saved: {checkpoint}")
        metrics = YOLO(str(checkpoint)).val(
            data=str(data), validator=partial(SDCLValidator, ignore_regions=True),
            device=device, imgsz=128, batch=2, workers=0, plots=False,
            project=str(root / "validation"), name=name, max_det=500,
        )
        completed[name] = {
            "checkpoint": str(checkpoint), "validation": metrics.results_dict,
            "statistics": str(trainer.save_dir / "sdcl_statistics.jsonl"),
        }
    report = {"synthetic_only": True, "output": str(root), "dataset_audit": audit, "runs": completed}
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="SDCL-YOLO experiment framework")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="Report versions and GPU")
    prepare = commands.add_parser("prepare", help="Convert extracted VisDrone; no automatic download")
    prepare.add_argument("--raw-root", default="data/raw")
    prepare.add_argument("--output", default="data/visdrone")
    audit = commands.add_parser("audit", help="Check converted data")
    audit.add_argument("--dataset", default="data/visdrone")
    train = commands.add_parser("train", help="Train baseline or SDCL")
    train.add_argument("--config", default="configs/experiments/baseline.yaml")
    for flag, kind in (("data", str), ("model", str), ("device", str), ("epochs", int),
                       ("batch", int), ("imgsz", int), ("workers", int), ("resume", str)):
        train.add_argument(f"--{flag}", type=kind)
    validate = commands.add_parser("validate", help="Diagnostic validation, not official VisDrone AP")
    validate.add_argument("--model", required=True)
    validate.add_argument("--data", default="data/visdrone/dataset.yaml")
    validate.add_argument("--device", default="0")
    validate.add_argument("--imgsz", type=int, default=640)
    validate.add_argument("--no-ignore", action="store_true")
    export = commands.add_parser("export", help="Write official-format VisDrone detections")
    export.add_argument("--model", required=True)
    export.add_argument("--images", default="data/visdrone/images/val")
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="0")
    export.add_argument("--imgsz", type=int, default=640)
    smoke = commands.add_parser("smoke", help="Local synthetic baseline + SDCL training check")
    smoke.add_argument("--output")
    smoke.add_argument("--device", default="0")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "env":
        from .environment import environment_info
        result = environment_info()
    elif args.command == "prepare":
        from .data import prepare_visdrone
        result = {"dataset_yaml": str(prepare_visdrone(project_path(args.raw_root), project_path(args.output)))}
    elif args.command == "audit":
        from .data import audit_dataset
        result = audit_dataset(project_path(args.dataset))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    elif args.command == "train":
        flags = {key: value for key, value in vars(args).items() if key not in {"command", "config"}}
        result = train_experiment(args.config, **flags)
    elif args.command == "validate":
        from ultralytics import YOLO
        from .trainer import SDCLValidator, check_version
        check_version()
        metrics = YOLO(str(project_path(args.model))).val(
            data=str(project_path(args.data)), device=args.device, imgsz=args.imgsz,
            validator=partial(SDCLValidator, ignore_regions=not args.no_ignore),
            plots=False, max_det=500, project=str(project_path("output/validation")),
        )
        result = {"metric_protocol": "Ultralytics diagnostic, NOT official VisDrone", **metrics.results_dict}
    elif args.command == "export":
        from .export import export_visdrone
        result = export_visdrone(
            project_path(args.model), project_path(args.images), project_path(args.output),
            args.device, args.imgsz,
        )
    else:
        result = smoke_test(args.output, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

