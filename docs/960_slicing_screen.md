# Frozen local magnification screen

This screen does not train a detector. It tests whether the existing 960
baseline has useful local magnification headroom before another training run.
The older coverage-assignment candidate has already failed its frozen gates.

Run from PowerShell:

```powershell
Set-Location D:\Python\YOLO
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/evaluation/diagnose_slicing.py --device 0
```

For a wiring check only:

```powershell
& 'C:\Users\BaiRong-NUC\.conda\envs\yolo\python.exe' scripts/evaluation/diagnose_slicing.py --device 0 --limit 2
```

The smoke command always returns `SMOKE_ONLY`, regardless of metric values.
Output directories are unique by default. Existing outputs are never overwritten.

## Frozen comparison

- Checkpoint: `output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt`.
- All 548 validation images, with pinned checkpoint, image, label, metadata and
  dataset-YAML hashes.
- Baseline: whole image, square 960 predictor input.
- Candidate: same whole image plus four fixed overlapping crops.
- Crop dimensions: `ceil(width / 1.8)` by `ceil(height / 1.8)`, anchored to the
  four image corners. Overlap is approximately 20% of crop dimensions.
- Every view uses square 960 input, batch 1, FP16, confidence 0.001 and
  class-aware NMS at IoU 0.7.
- Map boxes to original coordinates, apply the existing IoF > 0.5 diagnostic
  ignore filter, class-aware global NMS at IoU 0.7, then retain at most 500
  detections per image for either arm.
- Tile placement uses image shape only. Ground truth and ignored regions never
  determine crop locations.
- Size groups use the original box area rescaled to a fixed long edge of 640.

The previous validator used rectangular inputs. Both arms are recomputed with
the same square predictor, so compare AP within this report. Do not compare its
absolute AP to previous rectangular-validator reports.

## Investment gates

All five must pass on the full frozen validation split:

| Metric | Required delta in percentage points |
|---|---:|
| Tiny AP50-95 | >= +1.00 |
| Small-all AP50-95 | >= +0.50 |
| Small-all recall at FPPI <= 10 | >= +1.00 |
| Lower endpoint of paired-image 95% recall-delta interval | > 0 |
| Overall AP50-95 | >= -0.10 |

These are predeclared project investment thresholds. The interval uses 2000
paired image resamples with seed 0 and fixed selected confidence thresholds;
it does not include threshold-selection or training-seed uncertainty.

`STOP` means stop investing in this fixed local-magnification candidate.
`CONTINUE` means local magnification has enough headroom to justify a bounded
follow-up under a fixed compute budget. It does not demonstrate a training
improvement or a novel method.

A follow-up would have to compare region selection against fixed tiling,
simple density/uncertainty selection, and larger whole-image inputs under
matched compute. Train any learned selection policy on the training split,
never this validation split. Do not start another detector training run merely
because five-view inference outperforms one-view inference.

## Artifacts and timing

Each run writes `plan.json` before prediction, an exact evaluation-source ZIP,
paired predictions and manifests, `timings.csv`, `report.json`,
`size_comparison.csv`, plots, example crops, and `decision.json` / `decision.md`.

Latency uses synchronized serial batch-1 inference after warmup. It includes
image decode, predictor preprocessing, forward and NMS, crop copies, ignore
filtering and merging. Candidate latency sums the shared global component and
the additional tile/merge component. It excludes model setup, metadata reads,
JSON writes and metric computation. This is not a deployment-throughput
benchmark.

All AP and recall values are COCO-style / IoF-ignore diagnostics, not official
VisDrone AP. Successful implementation or a passed screen is not evidence of
publication novelty.

## Completed run: 2026-10-12

Artifacts: `output/analysis/slicing-960-fixed-20261012`.

The full 548-image run returned **STOP**, with three of five gates passing:

| Metric | Global | Global + 4 tiles | Delta (pp) |
|---|---:|---:|---:|
| Overall AP50-95 (%) | 28.5333 | 31.7150 | +3.1817 |
| Tiny AP50-95 (%) | 19.4535 | 25.0078 | +5.5543 |
| Small-all AP50-95 (%) | 25.8896 | 29.7934 | +3.9038 |
| Small-all recall at FPPI <= 10 (%) | 54.6013 | 53.8253 | -0.7760 |

The paired-image 95% recall-delta interval is [-1.3023, -0.2917] pp.
At confidence 0.25, false positives increase from 10840 to 23848.
Mean measured latency increases from 12.7124 to 48.9731 ms/image (3.8524x).

This result supports an AP benefit from local magnification, but rejects this
fixed global-plus-four-tiles candidate under the predeclared error-budget
criteria. Retain the 960 baseline and do not launch additional detector
training for this candidate. It does not establish that all magnification or
region-selection methods must fail.

The source, checkpoint and prediction hashes were verified. The 32 relevant
tests passed, including predictor-policy parity and tile-coordinate mapping.
