"""Replay saved pair losses from actual projections; no additional fitting."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.training.crossview_quality_rank import OUTPUT, assembled, precision


def run():
    root = project_path(OUTPUT)
    plan = json.loads((root/"plan.json").read_text())
    frames = json.loads((root/"frames.json").read_text())
    states = torch.load(root/"projections-seed0.pt", map_location="cpu", weights_only=True)
    precision(False)
    reports = {}
    for split in ("fit", "screen", "confirmation"):
        data, pairs = assembled(root, frames, plan["image_selection"][split])
        statistics = {}
        for arm, pp in pairs.items():
            gap = data["quality"][pp[:, 0]]-data["quality"][pp[:, 1]]
            statistics[arm] = dict(pairs=len(pp), quality_gap_mean=float(gap.mean()),
                                   quality_gap_min=float(gap.min()), quality_gap_max=float(gap.max()))
        reports[split] = dict(pair_quality_distributions=statistics)
        if split == "fit":
            audit = json.loads((root/"training-audit-seed0.json").read_text())
            assert not audit["cudnn_allow_tf32"] and not audit["matmul_allow_tf32"]
            assert audit["rows"] == len(data["features"]) and audit["pairs"] == len(pairs["cross"])
            assert audit["updates_per_arm"] == 8*((len(data["features"])+2047)//2048)
            for epoch in range(8):
                gen = torch.Generator().manual_seed(epoch)
                a = torch.randperm(len(data["features"]), generator=gen).numpy()
                b = torch.randperm(len(pairs["cross"]), generator=gen).numpy()
                saved = audit["orders"][epoch]
                assert hashlib.sha256(a.tobytes()).hexdigest() == saved["bce_order_sha256"]
                assert hashlib.sha256(b.tobytes()).hexdigest() == saved["rank_order_sha256"]
            with np.load(root/"initial.npz") as initial:
                parameter_changes = {}
                for arm, state in states.items():
                    delta = np.r_[state["weight"].numpy().ravel()-initial["weight"].ravel(),
                                  state["bias"].numpy().ravel()-initial["bias"].ravel()]
                    assert np.isfinite(delta).all() and np.linalg.norm(delta) > 0
                    parameter_changes[arm] = float(np.linalg.norm(delta))
            reports[split].update(initial_and_orders_checked=True, projection_delta_L2=parameter_changes)
            continue
        pp = pairs["cross"]
        loss_report = {}
        with np.load(root/f"mechanism-seed0-{split}.npz") as saved:
            for arm, state in states.items():
                rebuilt = []
                with torch.inference_mode():
                    for start in range(0, len(pp), 2048):
                        rows = pp[start:start+2048].reshape(-1)
                        x = torch.tensor(data["features"][rows], dtype=torch.float32, device="cuda:0")
                        levels = data["levels"][rows]
                        z = torch.empty((len(rows), 10), device="cuda:0")
                        for lev in range(3):
                            chosen = levels == lev
                            z[chosen] = F.linear(x[chosen], state["weight"][lev].to("cuda:0"),
                                                 state["bias"][lev].to("cuda:0"))
                        logits = z.double().cpu().numpy()[np.arange(len(rows)), data["classes"][rows]].reshape(-1, 2)
                        rebuilt.extend(np.logaddexp(0, logits[:, 1]-logits[:, 0]))
                np.testing.assert_array_equal(np.asarray(rebuilt), saved[arm])
                loss_report[arm] = dict(pairs=len(pp), maximum_saved_loss_error=0.)
        reports[split]["actual_projection_pair_loss_replay"] = loss_report
    source = Path(__file__).resolve()
    result = dict(passed=True, splits=reports, verifier_sha256=file_hash(source),
                  added_detector_forwards=0, added_fits=0,
                  scope="Replays pair-logit losses with shared PyTorch linear backend and actual saved projection "
                        "weights. Rebuilds recorded sample orders, validates finite nonzero parameter updates. "
                        "Does not reproduce every optimizer step or prove causal identity isolation.")
    (root/"mechanism_projection_replay.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
