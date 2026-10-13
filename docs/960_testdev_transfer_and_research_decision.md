# Final frozen test-dev transfer and research investment audit

Registered on 2026-10-13 before test-dev inference or metrics.

The current program has exhausted bounded scale weighting, phase, assignment,
localization, ranking and frozen classification probes. Preserve all prior STOP
decisions. This last no-training experiment checks whether the existing
engineering control transfers to the local, previously unevaluated test-dev.
It is not an innovation experiment and never authorizes restarting a stopped
recipe or selecting a new method on test-dev.

## Fixed experiment

- Local `test` is the supplied VisDrone2019-DET-test-dev: 1,610 images.
- Validate original annotations and images against the original manifest.
- No identical image-byte hashes, names or first filename prefixes versus
  train/val. Prefix nonoverlap is not proof of geographical independence.
- Same frozen 960 detector, FP16, square letterbox, confidence .001, NMS IoU .7,
  max500, whole image and four overlap-.20 tiles as the corrected strong control.
- Single-view baseline, five-view Gaussian Soft-NMS sigma .5, and the previously
  trained ordinary calibration head. No training, temperature fitting, class
  remapping, parameter sweep or annotation-based prediction filtering.
- Original eight-column annotations enter the official evaluator only.
- Use the pinned unmodified upstream toolkit and validated native helpers.
- Preserve official presence-weighted aggregation and all seven headline metrics.
- Three official processes may run concurrently; official runtime is not a
  deployment speed measurement.

All three gates must pass to call the frozen engineering control transferable:

| Calibration compared with | Metric | Requirement |
|---|---|---:|
| Five-view Soft-NMS | Official AP | >= +0.30 pp |
| Single-view 960 | Official AP | >= +1.00 pp |
| Five-view Soft-NMS | Official AR500 | >= -0.10 pp |

A pass retains the engineering control. A stop ends the tested calibration
transfer claim. Neither outcome proves publishable novelty. Test-dev is used
once here for the frozen controls; it must not become another tuning split.

## Registered command

```powershell
conda activate yolo
Set-Location 'D:\Python\YOLO'
python D:\Python\YOLO\scripts\evaluation\diagnose_testdev_transfer.py --device 0
```

Formal output: `output/analysis/testdev-transfer-fixed-20261013`.
Plan, source snapshot and gates are saved before inference. Predictions,
original input hashes, per-arm official logs, metrics and decision are retained.
Any `--limit` evaluation is SMOKE_ONLY, never transfer evidence.

## Prior-art boundaries

The following concrete examples prevent treating common ingredients as novelty:

- ESOD: Efficient Small Object Detection on High-Resolution Images.
  arXiv:2407.16424v2; author's repository citation lists IEEE TIP2025.
  Reusing backbone features for object seeking and patch slicing, sparse
  processing and speed/accuracy already form a documented research direction.
- SET: Spectral Enhancement for Tiny Object Detection. CVPR2025.
  Background smoothing and foreground feature enhancement already target
  weak tiny-object representations through a frequency-domain analysis.
- Uncertainty-Aware Gradient Stabilization for Small Object Detection.
  ICCV2025. The localization-regression/gradient-stability problem has already
  motivated uncertainty-aware optimization. A new generic loss is insufficient
  to establish novelty without closer comparison.

Primary sources inspected:

- https://arxiv.org/abs/2407.16424
- https://github.com/alibaba/esod
- https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf
- https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.pdf

These are examples of overlap, not an exhaustive novelty exclusion proof.
Do not label another generic crop, margin, weighting or context block innovative
without a narrower falsifiable gap and close matched prior-art controls.

## Program decision, fixed before official test-dev results

The investment recommendation is to stop the current broad "improve small-object
detection by extending YOLO on VisDrone" thesis program after this transfer
audit, regardless of whether the engineering control passes.

This is a resource-allocation recommendation, not a claim that all small-object
detection is solved, impossible, or unpublishable. A successful transfer retains
a useful engineering baseline but does not turn ordinary calibration/fusion
into a new scientific contribution. Failure only weakens the transfer claim.

Verified completed screens:

| Hypothesis / recipe | Passed fixed gates | Final decision |
|---|---:|---|
| Instance phase consistency | 2/7 | STOP |
| Coverage-preserving label assignment repair | 1/5 | STOP |
| View-conditioned rescue rescoring | 4/9 | STOP |
| Anchor-referenced localization gain selection | 7/11 | STOP |
| Regret-weighted action ranking | 6/10 | STOP |
| Frozen final-layer tiny classification margin | 0/10 | STOP |

The rows use different controls and diagnostics; gate counts are not effect
sizes and must not be pooled into a statistical rejection of the entire field.
Some individual gains were positive. They did not satisfy the predeclared
matched-control and mechanism requirements. Historical oracle geometry
headroom is label-assisted evidence, not proof of deployable learnability.

The 548-image val split has already been used repeatedly for design decisions.
Trying more modules on it would increase method-selection uncertainty.
Current probes do not provide a positive matched-method result that justifies
a new full-detector run under the requested low-cost, single-screen constraint.

Do not resume the stopped recipes or tune the new test-dev split. Do not download
new datasets, replace the architecture or start a longer training run as an
automatic continuation. Preserve the completed 960 weights, corrected control,
source snapshots and reports as a reusable engineering/reproduction project.

Reopening a small-object thesis later requires a new specific observable
failure, a genuinely separate development/confirmation split, close prior-art
controls and a resource budget adequate for the representation/data experiment.
It should be a new project decision, not another rescue of the failed losses.

Prior decisions were verified from the original result JSON files, including
their hashes, and recorded in
`output/verification/research_closeout_20261013/prior_decisions.json`.
Primary-source pages/README and retrieval hashes were saved in
`output/verification/research_closeout_20261013/primary_sources`.

## Completed transfer: TRANSFER_STOP, two of three gates passed

Full1610test-dev evaluation completed on 2026-10-13. No training or parameter
changes were made. All prepared input hashes, registered source hashes,
detector/calibrator hashes and original runtime references remained unchanged.

| Frozen arm | Official AP, percent | Official AR500, percent |
|---|---:|---:|
| Single-view 960 | 26.6267735095 | 37.6749098453 |
| Five-view Soft-NMS | 30.3595787507 | 46.2505428389 |
| Ordinary calibration | 30.8942687639 | 46.0181804755 |

| Gate | Measured, pp | Requirement, pp | Passed |
|---|---:|---:|---|
| Calibration AP versus Soft-NMS | +0.5346900132 | >= +0.30 | Yes |
| Calibration AP versus single-view | +4.2674952544 | >= +1.00 | Yes |
| Calibration AR500 versus Soft-NMS | -0.2323623634 | >= -0.10 | No |

The engineering AP improvement transfers to this split, but the registered
recall tolerance does not. This is not an all-gates-passing strong-control
transfer claim. Preserve the AP/recall tradeoff transparently; do not retune
scores on this test-dev or rescue it with another chosen checkpoint.

The program investment recommendation remains STOP as fixed before results.
Do not use the positive AP gain to infer novel small-object mechanisms or to
erase stopped matched learning experiments.

Evidence:

- `output/analysis/testdev-transfer-fixed-20261013/plan.json`
- `output/analysis/testdev-transfer-fixed-20261013/source.zip`
- `output/analysis/testdev-transfer-fixed-20261013/official_metrics.json`
- `output/analysis/testdev-transfer-fixed-20261013/decision.json`
- `output/analysis/testdev-transfer-fixed-20261013/verification.json`
- `docs/experiments/testdev_transfer_20261013.json`

Validation: 254 related tests passed. The two pre-existing incompatible queue
test files were excluded, as in prior runs; queue behavior was not changed.
Three-image end-to-end smoke passed before the frozen formal run, and its
numbers were never used to alter gates or inference settings.
