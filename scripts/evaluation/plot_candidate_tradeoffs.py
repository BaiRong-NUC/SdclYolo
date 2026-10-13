"""Plot measured official AP/AR tradeoffs; seed SD is not an AP confidence interval."""

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


def plot():
    inputs = {
        "seeds": project_path("output/analysis/ranking-seeds-fixed-20261013/official_metrics.json"),
        "prior": project_path("output/analysis/prior-gap-fixed-20261013/official_metrics.json"),
        "threshold": project_path("output/analysis/threshold-gain-probe-fixed-20261013/official_metrics.json"),
        "consensus": project_path("output/analysis/coordinate-consensus-fixed-20261013/official_metrics.json"),
    }
    hashes = {str(p): file_hash(p) for p in [*inputs.values(), Path(__file__).resolve()]}
    data = {k: json.loads(p.read_text()) for k, p in inputs.items()}
    points = {}

    def point(name, values, marker, color, offset):
        ap, ar = [[v[k] for v in values] for k in ("AP", "AR500")]
        points[name] = {
            "AP": float(np.mean(ap)), "AR500": float(np.mean(ar)),
            "AP_head_seed_sd": float(np.std(ap, ddof=1)) if len(ap) > 1 else 0.,
            "AR500_head_seed_sd": float(np.std(ar, ddof=1)) if len(ar) > 1 else 0.,
            "count": len(values), "marker": marker, "color": color, "offset": offset,
        }

    point("Calibration", [data["seeds"]["0"]["calibration"]], "D", "#252525", (-90, -26))
    point("Uniform (3-seed mean)", [v["uniform"] for v in data["seeds"].values()],
          "o", "#126782", (12, 13))
    point("Regret (3-seed mean)", [v["regret"] for v in data["seeds"].values()],
          "o", "#c55a11", (-13, 18))
    point("Prior gap (3-seed mean)", list(data["prior"].values()),
          "o", "#5c6b73", (-78, 25))
    point("Threshold gain (seed0)", [data["threshold"]["threshold_gain"]],
          "^", "#6a7f47", (8, -25))
    point("Uniform consensus", [data["consensus"]["methods"]["uniform_consensus"]],
          "s", "#126782", (12, 15))
    point("Regret consensus", [data["consensus"]["methods"]["regret_consensus"]],
          "s", "#c55a11", (-88, -18))
    output = project_path("output/analysis/candidate-tradeoffs-20261013")
    output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, p in points.items():
        ax.errorbar(p["AR500"], p["AP"], xerr=p["AR500_head_seed_sd"],
                    yerr=p["AP_head_seed_sd"], fmt=p["marker"], color=p["color"],
                    markersize=8, capsize=3, linewidth=1.3, zorder=4)
        ax.annotate(name, (p["AR500"], p["AP"]), xytext=p["offset"],
                    textcoords="offset points", fontsize=9, color=p["color"])
    floor = points["Calibration"]["AP"] + .1
    ax.axhline(floor, color="#a1a1a1", linestyle="--", linewidth=1)
    ax.text(45.585, floor - .006, "Calibration AP + 0.10 pp budget line",
            fontsize=8, color="#707070", va="top")
    start, end = points["Regret consensus"], points["Uniform (3-seed mean)"]
    ax.annotate("", xy=(end["AR500"], end["AP"]),
                xytext=(start["AR500"], start["AP"]),
                arrowprops={"arrowstyle": "->", "color": "#126782", "lw": 1.5})
    ax.set(xlim=(45.58, 46.07), ylim=(30.84, 31.185),
           xlabel="Official AR500 (%)  -  higher is better",
           ylabel="Official AP (%)  -  higher is better",
           title="Frozen 960 coordinate candidates: AP and recall tradeoffs")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.11, .025,
             "Error bars: head-seed SD (n=3), not AP confidence intervals. Same fixed detector and test-dev.\n"
             "Arrow: ordinary uniform mean has higher AP and AR500 than regret consensus.",
             fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(output / "official_ap_ar_tradeoffs.png", dpi=180)
    fig.savefig(output / "official_ap_ar_tradeoffs.svg")
    plt.close(fig)
    if any(file_hash(p) != h for p, h in hashes.items()):
        raise ValueError("Plot inputs/source changed.")
    shutil.copyfile(__file__, output / "plot_source.py")
    (output / "data.json").write_text(json.dumps({
        "inputs_source_sha256": hashes, "points": points,
        "scope": "Measured official metrics. Head seed SD, not independent-data uncertainty or AP CI.",
    }, indent=2) + "\n")
    print(str(output.resolve()))


if __name__ == "__main__":
    plot()
