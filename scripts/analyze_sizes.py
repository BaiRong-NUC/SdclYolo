"""Compare checkpoints by object size without changing training."""

import _bootstrap
import argparse
import json

from sdcl.size_analysis import run_size_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="output/runs/baseline_yolo11s_seed0-2/weights/best.pt")
    parser.add_argument("--sdcl", "--candidate", dest="sdcl",
                        default="output/runs/sdcl_yolo11s_seed0/weights/best.pt",
                        help="Candidate checkpoint; it need not use SDCL.")
    parser.add_argument("--dataset", default="data/visdrone")
    parser.add_argument("--output")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline-imgsz", type=int, help="Override baseline inference size.")
    parser.add_argument("--sdcl-imgsz", "--candidate-imgsz", dest="sdcl_imgsz", type=int,
                        help="Override candidate inference size.")
    parser.add_argument("--reference-size", type=int,
                        help="Size-group reference long edge; defaults to --imgsz. "
                             "Use 640 for comparisons with the existing reports.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--prediction-cache", help="Reuse a previous analysis only if hashes/settings match.")
    parser.add_argument("--baseline-prediction-cache", help="Directory containing a baseline capture manifest.")
    parser.add_argument("--candidate-prediction-cache", help="Directory containing a candidate capture manifest.")
    args = parser.parse_args()
    print(json.dumps(run_size_analysis(**vars(args)), indent=2, ensure_ascii=False))
