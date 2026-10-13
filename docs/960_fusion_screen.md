# Fixed standard fusion control, 2026-10-13

This is one independent, training-free control following the failed fixed
four-tile screen. That screen's existing STOP decision is preserved.

Run from PowerShell:

```powershell
Set-Location D:\Python\YOLO
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/evaluation/diagnose_fusion.py --device 0
```

A wiring check uses `--limit 2` and always returns `SMOKE_ONLY`. Existing output
directories are never overwritten. There are no fusion-parameter CLI overrides.

## Frozen arms

The checkpoint and 548 validation images/labels/metadata are hash-pinned to the
existing 960 baseline and the previous fixed slicing screen.

All three arms share the exact same captured predictions:

1. Global 960 input, class-aware hard NMS at IoU 0.7.
2. Global plus four fixed tiles, class-aware hard NMS at IoU 0.7.
3. The same five views, class-aware Gaussian Soft-NMS with sigma 0.5.

Tiles use the existing 2x2 policy with approximately 20% crop-relative overlap.
Every view uses square 960 input, batch 1, FP16, confidence 0.001 and per-view
hard NMS at IoU 0.7. The final cap is 500 detections per image for all arms.
Size groups retain the fixed long-edge reference of 640.

The fusion candidate follows the Gaussian score decay in Soft-NMS
(arXiv:1704.04503v2): repeatedly select the current highest score, then multiply
remaining same-class scores by `exp(-IoU**2 / 0.5)`. Keep scores >= 0.001 and
cap at 500 after reranking. Coordinates stay unchanged. IoU uses continuous
YOLO xyxy coordinates, without the inclusive-pixel `+1` in the legacy Cython
reference implementation.

Fusion uses predicted boxes, scores and classes only. The existing IoF > 0.5
ignore filter is shared diagnostic preprocessing. Ground truth is used for
evaluation only; it does not determine tiles, score decay or duplicates.

## Predeclared decision

All three primary gates from the previous agreed proposal must pass:

| Candidate versus global baseline | Required delta, percentage points |
|---|---:|
| Small-all recall at FPPI <= 10 | >= +1.00 |
| Lower endpoint of paired-image 95% recall-delta interval | > 0 |
| Overall AP50-95 | >= -0.10 |

Tiny AP, small-all AP, FPPI <= 5 recall, comparison against hard NMS, and
latency are secondary observations. They do not change these primary gates.
The old slicing screen's five gates are not being relabeled or rerun as this
new three-gate control.

This is one fixed method and one full screen, without a parameter sweep,
confidence calibration fit, detector training or validation-based selection.
The bootstrap uses 2000 paired image resamples, seed 0, and fixed selected
thresholds. It does not cover threshold-selection or training-seed uncertainty.

If STOP, end investment in this fixed slicing/fusion route and retain the
960 baseline. Do not tune sigma or start training a rescue module.

If CONTINUE, the standard fusion control has passed the minimum investment
screen. It is not a new trained method or evidence of novelty; a proposed
learned method would still have to beat this standard control.

## Evidence and reuse

`plan.json` is written before prediction and stores the exact gates, method,
checkpoint/data/source hashes and environment. The run also saves:

- All five mapped views before global fusion in `raw_views.jsonl`.
- Predictions and manifests for all three arms.
- Component timing and the final detection caps.
- An exact evaluation-source ZIP.
- The candidate/global report and the candidate/hard-NMS FP-budget comparison.
- `decision.json`, `decision.md` and `verification.json`.

The historical hard-NMS metrics are reused only if both the newly captured
global and hard-NMS prediction files exactly match the historical content
hashes. Otherwise, hard-NMS metrics are recomputed from the current shared
capture. Historical artifacts are never changed.

Timing sums warmed, synchronized shared global/tile inference components and
each arm's own ignore/merge cost. It includes image decoding and crop copies,
but excludes model setup, metadata reads, artifact writes and metric
computation. This is not a deployment-throughput benchmark.

All metrics are COCO-style / IoF-ignore diagnostics, not official VisDrone AP.

## Completed run

The full run at `output/analysis/fusion-softnms-fixed-20261013` returned
**CONTINUE**, with all three primary gates passing:

| Primary metric | Measured delta (pp) | Required |
|---|---:|---:|
| Small-all recall at FPPI <= 10 | +3.6788 | >= +1.00 |
| Lower endpoint of its paired-image 95% interval | +3.2061 | > 0 |
| Overall AP50-95 | +4.3091 | >= -0.10 |

The complete FPPI-10 recall-delta interval is [3.2061, 4.1564] pp.

| Diagnostic metric (%) | Global | Tiles hard NMS | Tiles Gaussian Soft-NMS |
|---|---:|---:|---:|
| Overall AP50-95 | 28.5333 | 31.7150 | 32.8424 |
| Tiny AP50-95 | 19.4535 | 25.0078 | 25.8712 |
| Small-all AP50-95 | 25.8896 | 29.7934 | 30.8312 |
| Small-all recall, FPPI <= 5 | 45.9123 | 38.4570 | 46.5961 |
| Small-all recall, FPPI <= 10 | 54.6013 | 53.8253 | 58.2801 |

Soft-NMS versus the hard-NMS tiled control raises FPPI-10 small-all recall by
4.4547 pp, with a paired-image interval [4.1950, 4.7248] pp. Versus global
inference, the secondary FPPI-5 recall delta is +0.6838 pp with interval
[0.2216, 1.1548] pp.

Current measured mean component-sum latency is 19.253 / 75.249 / 108.810
ms/image for global / tiles hard NMS / tiles Soft-NMS. Mean fusion time is
0.832 / 2.178 / 35.739 ms/image. The unoptimized NumPy Soft-NMS implementation
has a material CPU cost; these are shared-capture diagnostic timings, not a
deployment benchmark.

The 43 targeted tests passed. New global and hard-NMS prediction hashes exactly
match the historical run. Source, checkpoint, dataset and prediction hashes,
final detection caps, and preservation of the old STOP decision all passed
post-run verification.

Retain this standard fusion as a required control for future work. A learned
fusion candidate must beat this control, preserve newly detected small
objects, and account for computation cost. These results do not demonstrate
novelty, independent-data generalization or training-seed robustness.
