"""Evaluate one new resolution training run and report fixed continue/stop gates."""

import _bootstrap
import argparse
import json

from sdcl.resolution import run_resolution_screen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="output/runs/baseline_yolo11s_100ep_seed0/weights/best.pt")
    parser.add_argument("--candidate", default="output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt")
    parser.add_argument("--dataset", default="data/visdrone")
    parser.add_argument("--output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_resolution_screen(**vars(args)), indent=2, ensure_ascii=False))
