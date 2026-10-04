"""Reproducibility metadata without credentials or environment-variable dumps."""

import importlib.metadata
import hashlib
import platform
import subprocess
import sys

from .config import PROJECT_ROOT


def environment_info():
    import torch

    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sdcl_source_hashes": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted((PROJECT_ROOT / "sdcl").glob("*.py"))
        },
    }

