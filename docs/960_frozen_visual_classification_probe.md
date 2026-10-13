# Frozen visual classification probe: one bounded investment decision

## Research question and boundaries

This is a low-cost learnability probe, not a new detector architecture or a
novelty claim. Test whether the existing 960 detector's frozen visual features
contain usable true-class versus wrong-class information for tiny objects.
Unlike the stopped geometry/action-ranking experiments, this probe trains
pre-NMS image-feature classification weights, not boxes, actions or a metadata
rescoring MLP. Previous STOP decisions remain unchanged.

Ordinary positive-margin learning overlaps existing classification methods.
A positive result is evidence for further investigation, not publishable novelty.
A negative result stops only this frozen-feature, final-linear-layer recipe;
it does not prove that a larger model, different data or representation learning
cannot work. Do not change seeds, subsets, epochs or margins after viewing val.

## Frozen control and matched experiment

- Detector: completed YOLO11s 960 seed0 best checkpoint, unchanged backbone,
  box branch, classification feature layers, batch normalization and decoder.
- Control: ordinary Soft-NMS calibration with validation ignores passed as an
  empty list during prediction/fusion/features. The official control AP is
  34.12335262742041 percent using the pinned upstream toolkit aggregation.
- Shared train subset: 512 fitting images and 128 monitor images selected by
  SHA256(seed0, image stem). Train-only scene grouping separates fit/monitor.
  Five frozen 960 views per image, as used by the strong control.
- Training cache: penultimate 128-dimensional features for each P3/P4/P5
  classifier. Frozen baseline TAL assigns soft class targets. At most 24
  positives and 16 strongest non-ignore negatives per level and view are kept.
  This is a deliberately bounded sample objective, not the full detector loss.
- Train annotations and ignore regions may define targets; val annotations
  never enter prediction, fusion, feature extraction or calibration.
- Tiles retain GT with at least 90 percent visible box area. Tiny labels use
  original GT area normalized to reference640, side less than 16 pixels.
- Both arms start from identical final class projections, with 3,870 trainable
  parameters each. Features, boxes, normalization and calibrator stay frozen.
- BCE control: soft-target BCE plus 0.01 mean-squared logit retention.
- Margin arm: identical BCE/retention plus quality-weighted
  `0.25 * softplus(0.2 + max_wrong_logit - true_class_logit)` on tiny foreground.
- Fixed AdamW, lr0.0002, weight decay0.0001, batch4096, seed0, ten epochs.
  Monitor losses are informational, never used for checkpoint/hyperparameter selection.
- Evaluate last epoch10 only. Ordinary BCE is a matched control, not a fallback
  method selected if the margin arm fails.

The original calibration head was fitted with train ignore-region filtering.
It remains frozen in all three evaluation arms; distribution-shift limitations
are not removed by this probe. Final box/score lists can change because changed
class logits affect NMS and fusion, even though dense regression boxes are frozen.

## Commands

Run the two commands once, sequentially, from the project root:

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python D:\Python\YOLO\scripts\training\train_class_margin_probe.py --device 0
python D:\Python\YOLO\scripts\evaluation\diagnose_class_margin_probe.py --device 0
```

The first command prepares the fixed feature subset and trains both arms. It
does not resume or retrain the 100-epoch detector. Cache preparation is resumable
by repeating the command before a training run exists. If training is interrupted:

```powershell
python D:\Python\YOLO\scripts\training\train_class_margin_probe.py --device 0 --resume
```

Default cache: `output/class_margin/train_visual_seed0`.
Default run: `output/runs/class_margin_probe_960_seed0`.
Evaluations create a new `output/analysis/class-margin-probe-<timestamp>` directory.
The evaluator refuses smoke-trained weights for a formal decision and requires
an exact raw detector replay and corrected strong-control prediction replay.
The local Octave path defaults to the previously validated portable installation.

No automatic full detector training, dataset download or test-dev method search.

## Ten fixed gates: every gate must pass

| Metric, margin arm | Requirement |
|---|---:|
| Official AP versus corrected calibration | >= +0.30 pp |
| Official AP versus matched BCE head | >= +0.20 pp |
| Project tiny AP50-95 versus corrected calibration | >= +0.50 pp |
| Project small_all AP50-95 versus corrected calibration | >= +0.30 pp |
| Project small_all recall at FPPI<=5 versus control | >= -0.10 pp |
| Project small_all recall at FPPI<=10 versus control | >= -0.10 pp |
| Official AR500 versus control | >= -0.10 pp |
| Conditional owned-tiny correct-class rate versus control | >= +1.00 pp |
| Conditional owned-tiny correct-class rate versus BCE | >= +0.50 pp |
| Mean added cached head/NMS/fusion/calibration processing | <= 10 ms/image |

Official headline AP and AR use the pinned VisDrone toolkit, including its
image-presence-weighted class aggregation. Size AP and FPPI are separately
labelled project COCO-style/IoF-ignore diagnostics, not official tiny metrics.
For the latter, all arms receive the same ignore mask only after annotation-free
predictions; do not reuse old pre-fusion-ignore diagnostic values as controls.

The mechanism diagnostic uses the same highest-IoU baseline anchors, selected
class-agnostically from the top500 scored pre-NMS anchors in each view. GT only
defines diagnostic ownership; the diagnostic is conditional, not detector AP,
not an oracle method, and not proof that all classification errors are recoverable.
Latency excludes the shared detector forward and official evaluation runtime;
it is not end-to-end deployment FPS. Initial kernels/postprocessing are warmed.

STOP if any gate fails. Do not rescue the experiment with another checkpoint,
training subset, margin, seed or additional epochs. A CONTINUE permits a frozen
independent test-dev verification before broader training. The repeatedly used
val split is not independent generalization evidence.

## Reproducibility and verification

Each cache/run/evaluation registers full source hashes and a source archive,
data/split/chunk/checkpoint hashes and the corrected control manifest hash.
Original detector/calibrator checkpoints, prediction captures and prior STOP
records are not overwritten. Only smoke cache/training/evaluation is run during
implementation validation; smoke metrics never influence the recipe or gates.

### Empty-view evaluation repair (2026-10-13)

A captured JSON `[]` must be restored as an empty `(0, 6)` detection array.
The earlier evaluator compared it with a `(0,)` array and stopped on identical
empty views. This repair retains exact value and row-order checks for nonempty
views. It does not relax replay tolerances or change any of the ten gates.

Completed checkpoints and their cache/source archives remain unchanged.
The evaluator records the old and repaired source hashes in
`plan.json/evaluation_source_patch`. Only the original registered evaluator and
a checked cache-validator injection may differ from the training snapshot;
other code changes still fail verification. Reuse the epoch10 checkpoint and
run the same evaluation command. No new training is required.

## Completed experiment: STOP, zero of ten gates passed

Full evaluation finished on 2026-10-13 with the registered epoch10 checkpoint.
The 548-image validation prediction replay and all seven official strong-control
metrics reproduced exactly. Detector, calibrator, trained projections, original
training source archive and feature-cache manifest hashes were unchanged.

| Arm | Official AP, percent | Official AR500, percent |
|---|---:|---:|
| Corrected calibration control | 34.1233526274 | 49.5993079061 |
| Matched BCE projection | 33.4591794179 | 48.5325618031 |
| Tiny-margin projection | 33.5108136426 | 48.6806890714 |

| Fixed gate | Measured | Requirement | Passed |
|---|---:|---:|---|
| Official AP versus control, pp | -0.612539 | >= +0.30 | No |
| Official AP versus BCE, pp | +0.051634 | >= +0.20 | No |
| Project tiny AP versus control, pp | -0.597433 | >= +0.50 | No |
| Project small_all AP versus control, pp | -0.546479 | >= +0.30 | No |
| small_all recall at FPPI5 versus control, pp | -0.862478 | >= -0.10 | No |
| small_all recall at FPPI10 versus control, pp | -1.169509 | >= -0.10 | No |
| Official AR500 versus control, pp | -0.918619 | >= -0.10 | No |
| Owned-tiny correct class versus control, pp | +0.004238 | >= +1.00 | No |
| Owned-tiny correct class versus BCE, pp | -0.059327 | >= +0.50 | No |
| Mean extra cached processing, ms/image | 47.330132 | <= 10 | No |

Stop this sampled frozen-feature projection/margin recipe. Do not add epochs,
change seeds, tune margins or select BCE as a fallback after seeing the results.
Both learned projections degrade official AP versus the corrected control.
This result does not establish a general limit on representation learning or
the entire small-object detection field.

Evidence:

- `output/analysis/class-margin-probe-20261013-182356/decision.json`
- `output/analysis/class-margin-probe-20261013-182356/official_metrics.json`
- `output/analysis/class-margin-probe-20261013-182356/verification.json`
- `output/verification/class_margin_replay_fix_20261013/repair_verification.json`

The empty-view repair and official evaluator regression suite passed 19 tests.
