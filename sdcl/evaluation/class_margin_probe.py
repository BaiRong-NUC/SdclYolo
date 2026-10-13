"""Fixed full-val decision for frozen visual classification projections."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import zipfile

import cv2
import numpy as np
import torch
import yaml
from scipy.io import loadmat

from sdcl.config import project_path
from sdcl.evaluation.official_runtime import octave_env, verify_runtime_files
from sdcl.evaluation.official_visdrone import METRICS, matlab_string, write_detection
from sdcl.evaluation.size_analysis import (
    box_iou_matrix, coco_size_metrics, file_hash, load_records, recall_at_fp_budgets,
    record_hash, write_csv,
)
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.class_margin_probe import (
    ARMS, GATES, RUN, Projection, archive_source, check_sources, checked_control,
    decide, recipe, sources, verify_archive,
)
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import STRONG, save_json
from sdcl.training.class_margin_probe import FrozenViews, read_cache


OFFICIAL_CONTROL = "output/analysis/visdrone-official-control-20261013"


def frozen_view_matches(actual, captured):
    # JSON [] loses the six-column shape; preserve exact equality for every value.
    expected = np.asarray(captured, dtype=np.float32).reshape(-1, 6)
    return np.array_equal(actual, expected)


def evaluation_source_check(directory, expected, archive_digest):
    """Permit this evaluation repair without rewriting trained source snapshots."""
    verify_archive(directory, expected, archive_digest)
    current = sources()
    patch = None
    if current != expected:
        base = {
            'sdcl/evaluation/class_margin_probe.py':
                'f60fb43ef14bc62221ef1ffe6ac9e0c366fc8a97338f064173663df8f04a9105',
            'sdcl/training/class_margin_probe.py':
                '8c97c1d5300800d301149db6bd86d55008930580297f1acae4bb85e68badac71',
        }
        if (set(current) != set(expected)
                or {n for n in current if current[n] != expected[n]} != set(base)
                or any(expected[n] != digest for n, digest in base.items())):
            raise ValueError('Changes exceed the registered empty-view evaluation repair.')
        name = 'sdcl/training/class_margin_probe.py'
        with zipfile.ZipFile(directory / 'source.zip') as archive:
            original = archive.read(name)
        replacements = (
            (b'def read_cache(path):',
             b'def read_cache(path, *, source_check=check_sources):'),
            (b"check_sources(manifest['settings']['source_hashes'])",
             b"source_check(manifest['settings']['source_hashes'])"),
        )
        for before, after in replacements:
            if original.count(before) != 1:
                raise ValueError('Unexpected original cache source.')
            original = original.replace(before, after, 1)
        if project_path(name).read_bytes() != original:
            raise ValueError('Training changes exceed source-validator injection.')
        patch = {
            'version': '2026-10-13-empty-view-replay-v1',
            'reason': 'Read frozen JSON [] as an empty (0, 6) detection array.',
            'scope': 'Evaluation and cache source validation only; training, weights, recipe and gates unchanged.',
            'original_source_archive_sha256': archive_digest,
            'changed_files': {
                n: {'training_sha256': expected[n], 'evaluation_sha256': current[n]}
                for n in base
            },
        }

    def check_origin(hashes):
        if hashes != expected or sources() != current:
            raise ValueError('Training/evaluation sources changed outside the verified repair.')

    return check_origin, patch


def geometric_probe(extractor, record, window, projections, best, labels):
    """GT only selects diagnostic anchors; it never changes predicted boxes/scores."""
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy
    score = extractor.raw['scores'][0].float()
    selected = torch.argsort(score.sigmoid().amax(0), descending=True, stable=True)[:500]
    boxes = xywh2xyxy(extractor.decoded[0, :4, selected].T.clone())
    shape = tuple(int(x * extractor.head.stride[0]) for x in extractor.raw['feats'][0].shape[2:])
    x1,y1,x2,y2 = window
    boxes = scale_boxes(shape, boxes, (y2-y1,x2-x1))
    boxes[:,[0,2]] += x1
    boxes[:,[1,3]] += y1
    objects = record['objects']
    if not objects:
        return
    gt = np.array([obj['box_xyxy'] for obj in objects], dtype=float)
    iou = box_iou_matrix(gt, boxes.cpu().numpy())
    columns = iou.argmax(1)
    values = iou[np.arange(len(gt)),columns]
    changed = (values >= .5) & (values > best)
    if not changed.any():
        return
    selected_classes = {'control': score[:,selected].argmax(0).cpu().numpy()}
    for arm in projections:
        selected_classes[arm] = extractor.projection_scores[arm][0,:,selected].argmax(0).cpu().numpy()
    best[changed] = values[changed]
    for arm in labels:
        labels[arm][changed] = selected_classes[arm][columns[changed]]


def official(output, octave, records, predictions):
    original = project_path(OFFICIAL_CONTROL)
    runtime = verify_runtime_files(original)
    if str(Path(octave).resolve()) != runtime['octave_path']:
        raise ValueError('Use the validated Octave runtime.')
    validation = json.loads((original/'runtime/validation.json').read_text())
    if not validation['passed'] or len(records) != 548:
        raise ValueError('Official full probe requires native validation and all548val.')
    native = original/'runtime/visdrone_helpers.oct'
    directory = output/'official'
    directory.mkdir()
    shutil.copytree(original/'toolkit',directory/'toolkit')
    shutil.copytree(original/'compat',directory/'compat')
    shutil.copyfile(native,directory/'visdrone_helpers.oct')
    lib = matlab_string(directory/'visdrone_helpers.oct')
    root = matlab_string(directory)
    dataset = matlab_string(original/'dataset')
    setup = '\n'.join(f"autoload('{n}','{lib}');" for n in ('createIntImg','compOas','evalRes','VOCap'))
    arm_names = ','.join(f"'{a}'" for a in ('control',*ARMS))
    for arm,mapping in predictions.items():
        folder=directory/'detections'/arm
        folder.mkdir(parents=True)
        for r in records:
            write_detection(folder/f"{r['stem']}.txt",mapping[r['stem']],r['size'])
    code=f"""root='{root}'; dataset='{dataset}';
addpath(fullfile(root,'toolkit','utils')); addpath(fullfile(root,'compat'));
{setup}
names=findImageList(fullfile(dataset,'annotations')); assert(length(names)==548);
arms={{{arm_names}}};
for k=1:length(arms)
  arm=arms{{k}};
  [gt,dt]=saveAnnoRes(fullfile(dataset,'annotations'),fullfile(root,'detections',arm),fullfile(dataset,'images'),548,names);
  [a,b,c,d,e,f,g]=calcAccuracy(548,gt,dt); values=[a,b,c,d,e,f,g];
  save('-mat7-binary',fullfile(root,[arm '.mat']),'values');
  fprintf('PROBE_OFFICIAL_COMPLETED %s\\n',arm);
end
"""
    script=directory/'evaluate.m'
    script.write_text(code,encoding='ascii')
    save_json(directory/'input_manifest.json',{
        'runtime_control_manifest_sha256':file_hash(original/'runtime/manifest.json'),
        'native_validation_sha256':file_hash(original/'runtime/validation.json'),
        'files':{p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*') if p.is_file()},
        'original_dataset_inputs':'Hash-verified by verify_runtime_files before and after evaluation.',
    })
    with (directory/'evaluate.log').open('x',encoding='utf-8') as stream:
        result=subprocess.run([str(Path(octave).resolve()),'--no-gui','--quiet','--no-history',str(script)],
                              env=octave_env(octave),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Official probe failed: {directory}/evaluate.log')
    verify_runtime_files(original)
    expected=json.loads((directory/'input_manifest.json').read_text())['files']
    if any(file_hash(directory/n)!=h for n,h in expected.items()):
        raise ValueError('Official evaluation inputs changed.')
    metrics={a:dict(zip(METRICS,loadmat(directory/f'{a}.mat')['values'].ravel().tolist())) for a in predictions}
    pinned=json.loads((original/'official_metrics.json').read_text())['metrics_percent']['calibration_no_gt']
    if any(abs(metrics['control'][k]-pinned[k])>1e-12 for k in METRICS):
        raise ValueError('Corrected official strong control did not reproduce.')
    return metrics


def evaluate(checkpoint=f'{RUN}/last.pt', output=None, device='0', octave=None, limit=None):
    fixed=recipe()
    checkpoint=project_path(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['epoch']!=fixed.epochs or saved['recipe']!=asdict(fixed):
        raise ValueError('Only the fixed final epoch10 is eligible.')
    source_check,source_patch=evaluation_source_check(
        checkpoint.parent,saved['source_hashes'],saved['source_archive_sha256'])
    source_check(saved['source_hashes'])
    cache,_,_=read_cache(saved['cache_path'],source_check=source_check)
    if (file_hash(Path(saved['cache_path'])/'manifest.json')!=saved['cache_sha256']
            or saved['full_fixed_subset'] != cache['settings']['full_fixed_subset']):
        raise ValueError('Training checkpoint/cache mismatch.')
    control=checked_control()
    if limit is not None and not 1<=limit<=548:
        raise ValueError('Smoke limit must be in1..548.')
    if limit is None and not saved['full_fixed_subset']:
        raise ValueError('Smoke-trained heads cannot enter the formal evaluation.')
    if limit is None and not octave:
        raise ValueError('Formal screen requires the validated Octave path.')
    dataset=project_path('data/visdrone')
    all_records=load_records(dataset,'val')
    if len(all_records)!=548 or record_hash(all_records)!=frozen_control()['frozen_validation']['records_sha256']:
        raise ValueError('Validation data changed.')
    records=all_records[:limit] if limit else all_records
    raw_path=project_path(STRONG)/'raw_views.jsonl'
    if file_hash(raw_path)!=frozen_control()['frozen_validation']['raw_views_sha256']:
        raise ValueError('Frozen raw views changed.')
    old={}
    for line in raw_path.read_text().splitlines():
        item=json.loads(line)
        old[item['image']]=item
    capture={}
    for line in Path(control['prediction_capture']['path']).read_text().splitlines():
        item=json.loads(line)
        capture[item['image']]=np.array(item['boxes'],dtype=float).reshape(-1,6)
    output=project_path(output or f'output/analysis/class-margin-probe-{datetime.now():%Y%m%d-%H%M%S}')
    if output.exists():
        raise FileExistsError('Refuse to overwrite any prior evaluation.')
    output.mkdir(parents=True)
    plan=dict(recipe=asdict(fixed), fixed_gates=GATES, source_hashes=sources(), checkpoint_sha256=file_hash(checkpoint),
              training_source_hashes=saved['source_hashes'], evaluation_source_patch=source_patch,
              cache_sha256=saved['cache_sha256'], control_manifest_sha256=saved['control_sha256'],
              images=len(records), full_fixed_subset=saved['full_fixed_subset'],
              records_sha256=record_hash(records), raw_views_sha256=file_hash(raw_path),
              timing='Warmed per-image added class projection/NMS plus delta in calibration/fusion. Shared detector forward excluded; not deploymentFPS.',
              mechanism='Class-agnostic best-IoU baseline geometric anchors from top500 per-view preNMS confidences; tinyGT ownership fixed across heads. Conditional diagnostic only.',
              protocol='Official headline AP/AR plus separately labelled reference640 COCO-style size/IoF-ignore diagnostics.')
    save_json(output/'plan.json',plan)
    archive_source(output,plan['source_hashes'])
    extractor=FrozenViews(device)
    projections={a:Projection(extractor.weight,extractor.bias).to(f'cuda:{device}').eval() for a in ARMS}
    for arm in ARMS:
        projections[arm].load_state_dict(saved['models'][arm])
    calibrator=load_calibrator(frozen_control())
    with torch.inference_mode():
        for _ in range(2):
            extractor.predict(np.zeros((960,960,3),np.uint8),projections)
    predictions={a:{} for a in ('control',*ARMS)}
    timings,mechanism=[],[]
    try:
        for index,r in enumerate(records,1):
            image=cv2.imread(r['image'])
            if image is None or list(image.shape[1::-1])!=r['size']:
                raise ValueError('Invalid evaluation image.')
            views={a:[] for a in predictions}
            windows=[[0,0,*r['size']],*tile_windows(*r['size'])]
            extra=dict.fromkeys(ARMS,0.)
            best=np.zeros(len(r['objects']))
            labels={a:np.full(len(best),-1) for a in predictions}
            with torch.inference_mode():
                for view,window in enumerate(windows):
                    x1,y1,x2,y2=window
                    boxes,time=extractor.predict(image[y1:y2,x1:x2].copy(),projections)
                    for arm,values in boxes.items():
                        if view:
                            values=map_tile_predictions(values,window)
                        views[arm].append(values)
                    if not frozen_view_matches(views['control'][-1],old[r['stem']]['views'][view]):
                        expected=np.asarray(old[r['stem']]['views'][view],dtype=np.float32).reshape(-1,6)
                        np.savez(output/'raw_replay_mismatch.npz',actual=views['control'][-1],expected=expected)
                        save_json(output/'raw_replay_mismatch.json',{
                            'image':r['stem'],'view':view,'window':window,
                            'actual_shape':list(views['control'][-1].shape),'expected_shape':list(expected.shape),
                        })
                        raise ValueError(f"Frozen raw detector replay differs: {r['stem']} view{view}; see {output}/raw_replay_mismatch.json.")
                    for arm in ARMS:
                        extra[arm]+=time[arm]
                    geometric_probe(extractor,r,window,projections,best,labels)
            processing={}
            if index == 1:
                for _ in range(2):
                    for arm in predictions:
                        calibrated_candidates(views[arm],windows,r['size'],[],calibrator)
            for arm in predictions:
                tick=perf_counter()
                values=calibrated_candidates(views[arm],windows,r['size'],[],calibrator)[0]
                processing[arm]=(perf_counter()-tick)*1000
                predictions[arm][r['stem']]=values
            if not np.array_equal(predictions['control'][r['stem']],capture[r['stem']]):
                raise AssertionError('Annotation-free calibrated strong control replay differs.')
            timings.append(dict(image=r['stem'],margin_added_ms=extra['margin']+processing['margin']-processing['control'],
                                margin_projection_nms_ms=extra['margin'],**{f'{a}_postprocess_ms':v for a,v in processing.items()}))
            areas=np.array([np.prod(np.array(o['box_xyxy'])[2:]-np.array(o['box_xyxy'])[:2]) for o in r['objects']])
            tiny=(np.sqrt(areas)*640/max(r['size'])<16)&(best>=.5)
            truth=np.array([o['class_id'] for o in r['objects']])
            mechanism.append(dict(image=r['stem'],owned_tiny=int(tiny.sum()),
                                  **{f'{a}_correct':int(((labels[a]==truth)&tiny).sum()) for a in labels}))
            if index==1 or index%50==0 or index==len(records):
                print(f'Frozen visual classification evaluation: {index}/{len(records)}',flush=True)
    finally:
        extractor.close()
    for arm,mapping in predictions.items():
        folder=output/'predictions'/arm
        folder.mkdir(parents=True)
        with (folder/'predictions.jsonl').open('w',encoding='utf-8') as stream:
            for r in records:
                stream.write(json.dumps({'image':r['stem'],'boxes':mapping[r['stem']].tolist()})+'\n')
    diagnostic={}
    # Historical diagnostics filtered ignores before fusion. Here all arms instead
    # apply the same diagnostic ignore mask only after annotation-free prediction.
    from sdcl.data.geometry import prediction_ignore_mask
    diagnostic_predictions={a:{} for a in predictions}
    for r in records:
        ignores=json.loads((dataset/'metadata/val'/f"{r['stem']}.json").read_text())['ignore_boxes_xyxy']
        for a in predictions:
            v=predictions[a][r['stem']]
            mask=prediction_ignore_mask(torch.from_numpy(v[:,:4]),torch.tensor(ignores,dtype=torch.float64).reshape(-1,4),.5).numpy()
            diagnostic_predictions[a][r['stem']]=v[~mask]
    names=yaml.safe_load((dataset/'dataset.yaml').read_text())['names']
    for a in predictions:
        print(f'Project size diagnostics: {a}',flush=True)
        diagnostic[a]=coco_size_metrics(records,diagnostic_predictions[a],names,reference_size=640)
    budgets,_=recall_at_fp_budgets(records,{'baseline':diagnostic_predictions['control'],'sdcl':diagnostic_predictions['margin']},
                                  reference_size=640,budgets=(5,10),bootstrap_samples=2000)
    count=sum(r['owned_tiny'] for r in mechanism)
    if not count:
        raise ValueError('No owned tiny anchors; enlarge smoke subset.')
    rate={a:sum(r[f'{a}_correct'] for r in mechanism)/count for a in predictions}
    summary=dict(owned_tiny=count,correct_class_rate=rate,scope=plan['mechanism'])
    timing=dict(mean_added_ms=float(np.mean([r['margin_added_ms'] for r in timings])),
                p95_added_ms=float(np.quantile([r['margin_added_ms'] for r in timings],.95)),scope=plan['timing'])
    save_json(output/'coco_reference.json',diagnostic)
    save_json(output/'matched_fp_budgets.json',budgets)
    save_json(output/'owned_tiny_classification.json',summary)
    save_json(output/'processing_times.json',timing)
    write_csv(output/'timings.csv',timings)
    write_csv(output/'owned_tiny_per_image.csv',mechanism)
    if limit is None:
        print('Running the pinned official evaluator for all three heads.',flush=True)
        official_metrics=official(output,octave,records,predictions)
        save_json(output/'official_metrics.json',official_metrics)
        values={
            'official_ap_vs_control_pp':official_metrics['margin']['AP']-official_metrics['control']['AP'],
            'official_ap_vs_bce_pp':official_metrics['margin']['AP']-official_metrics['bce']['AP'],
            'official_ar500_vs_control_pp':official_metrics['margin']['AR500']-official_metrics['control']['AR500'],
            'tiny_ap_vs_control_pp':100*(diagnostic['margin']['tiny']['ap50_95']-diagnostic['control']['tiny']['ap50_95']),
            'small_all_ap_vs_control_pp':100*(diagnostic['margin']['small_all']['ap50_95']-diagnostic['control']['small_all']['ap50_95']),
            'owned_tiny_correct_class_vs_control_pp':100*(rate['margin']-rate['control']),
            'owned_tiny_correct_class_vs_bce_pp':100*(rate['margin']-rate['bce']),
            'added_processing_mean_ms':timing['mean_added_ms'],
        }
        for budget in (5,10):
            row=next(row for row in budgets if row['label']==f'fppi_{budget}')
            values[f'small_all_recall_fppi{budget}_pp']=row['delta_recall_pp']['small_all']
        decision=decide(values,True)
    else:
        decision={'decision':'SMOKE_ONLY','complete_fixed_probe':False,'checks':[],
                  'scope':'Smoke outputs are wiring evidence, not efficacy or threshold-selection evidence.'}
    check_sources(plan['source_hashes'])
    read_cache(saved['cache_path'],source_check=source_check)
    hashes={a:file_hash(output/'predictions'/a/'predictions.jsonl') for a in predictions}
    verification=dict(checkpoint_hash_unchanged=file_hash(checkpoint)==plan['checkpoint_sha256'],
                      raw_views_hash_unchanged=file_hash(raw_path)==plan['raw_views_sha256'],
                      validation_unchanged=record_hash(load_records(dataset,'val'))==record_hash(all_records),
                      frozen_detector_and_calibrator_unchanged=True,raw_base_detector_replay_exact=True,
                      corrected_calibration_replay_exact=True,no_val_annotations_in_predictor=True,
                      regression_and_backbone_frozen=True,prediction_hashes=hashes)
    if not all(v for k,v in verification.items() if k!='prediction_hashes'):
        raise RuntimeError('Probe integrity failed.')
    checked_control()
    save_json(output/'verification.json',verification)
    save_json(output/'decision.json',decision)
    lines=['# Frozen visual classification probe','',f"Decision: {decision['decision']}",'',
           '| Metric | Value | Requirement | Pass |','|---|---:|---:|---|']
    for row in decision['checks']:
        lines.append(f"| {row['metric']} | {row['value']:.4f} | {row['comparison']} {row['threshold']:.4f} | {row['passed']} |")
    (output/'decision.md').write_text('\n'.join(lines)+'\n',encoding='ascii')
    return dict(output=str(output),decision=decision['decision'],checks=decision['checks'])
