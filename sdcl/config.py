"""Validated experiment configuration and project-relative paths."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ULTRALYTICS = "8.4.172"


@dataclass(frozen=True)
class SDCLConfig:
    enabled: bool = True
    signal: Literal["joint", "scale", "difficulty", "additive"] = "joint"
    scale_reference: float = 640.0
    scale_s0: float = 32.0
    difficulty_alpha: float = 0.5
    lambda_max: float = 0.5
    warmup_epochs: int = 3
    ramp_epochs: int = 10
    log_interval: int = 50

    def __post_init__(self):
        if self.signal not in {"joint", "scale", "difficulty", "additive"}:
            raise ValueError(f"Unknown signal: {self.signal}")
        if self.scale_reference <= 0 or self.scale_s0 <= 0:
            raise ValueError("Scale values must be positive.")
        if not 0 <= self.difficulty_alpha <= 1:
            raise ValueError("difficulty_alpha must be in [0, 1].")
        if not 0 <= self.lambda_max < 1:
            raise ValueError("lambda_max must be in [0, 1).")
        if self.warmup_epochs < 0 or self.ramp_epochs < 0 or self.log_interval < 1:
            raise ValueError("Invalid schedule or log_interval.")

    @classmethod
    def from_dict(cls, values):
        values = values or {}
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown SDCL settings: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def strength(self, epoch: int) -> float:
        if not self.enabled:
            return 0.0
        if self.ramp_epochs == 0:
            return self.lambda_max if epoch >= self.warmup_epochs else 0.0
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        return self.lambda_max * min(1.0, max(0.0, progress))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_experiment(path: str | Path):
    file = project_path(path)
    with file.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {file}")
    unknown = set(config) - {"train", "sdcl", "ignore_regions"}
    if unknown:
        raise ValueError(f"Unknown experiment sections: {sorted(unknown)}")
    train = dict(config.get("train") or {})
    if "data" not in train:
        raise ValueError("train.data is required.")
    train["data"] = str(project_path(train["data"]))
    train["project"] = str(project_path(train.get("project", "output/runs")))
    model = train.get("model", "yolo11s.pt")
    if "/" in str(model) or "\\" in str(model):
        train["model"] = str(project_path(model))
    else:
        train["model"] = model
    if train.get("task", "detect") != "detect":
        raise ValueError("Only horizontal-box detection is supported.")
    if train.get("single_cls") or train.get("classes") is not None:
        raise ValueError("Class filtering is not supported by the ignore adapter.")
    device = str(train.get("device", "0"))
    if "," in device or isinstance(train.get("device"), (list, tuple)):
        raise ValueError("The first framework supports a single device only.")
    ignore = config.get("ignore_regions", {})
    if not isinstance(ignore, dict) or set(ignore) - {"enabled"}:
        raise ValueError("ignore_regions supports only 'enabled'.")
    settings = SDCLConfig.from_dict(config.get("sdcl"))
    return train, settings, bool(ignore.get("enabled", True))

