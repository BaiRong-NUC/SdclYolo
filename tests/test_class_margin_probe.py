from dataclasses import asdict

import numpy as np
import pytest
import torch

from sdcl.postprocess.class_margin_probe import (
    GATES, Projection, Recipe, decide, objective, recipe, select_records,
)
from sdcl.postprocess.rescue import scene_holdout
from sdcl.evaluation.class_margin_probe import evaluation_source_check, frozen_view_matches


def test_recipe_is_fixed_and_projection_has_3870_parameters():
    assert asdict(recipe())==asdict(Recipe())
    model=Projection(torch.zeros(3,10,128),torch.zeros(3,10))
    assert sum(p.numel() for p in model.parameters())==3870


def test_projection_matches_each_original_class_conv():
    torch.manual_seed(0)
    weight=torch.randn(3,10,128)
    bias=torch.randn(3,10)
    x=torch.randn(23,128)
    levels=torch.arange(23)%3
    model=Projection(weight,bias)
    actual=model(x,levels)
    expected=torch.stack([torch.nn.functional.linear(x[i],weight[levels[i]],bias[levels[i]]) for i in range(23)])
    torch.testing.assert_close(actual,expected)


def test_margin_only_changes_tiny_foreground_gradients():
    logits=torch.zeros(4,10,requires_grad=True)
    y=torch.zeros(4,10)
    y[0,1]=.7
    y[1,4]=.5
    tiny=torch.tensor([True,False,True,False])
    original=logits.detach().clone()
    control=torch.autograd.grad(objective(logits,y,tiny,original,'bce'),logits)[0]
    candidate=torch.autograd.grad(objective(logits,y,tiny,original,'margin'),logits)[0]
    assert candidate[0,1]<control[0,1]
    assert (candidate[0]-control[0]).abs().sum()>0
    assert torch.equal(candidate[1:],control[1:])


def test_tied_or_empty_tiny_targets_are_finite_and_do_not_add_margin():
    logits=torch.zeros(3,10,requires_grad=True)
    targets=torch.zeros_like(logits)
    for arm in ('bce','margin'):
        loss=objective(logits,targets,torch.ones(3,dtype=torch.bool),logits.detach(),arm)
        assert torch.isfinite(loss)
    assert objective(logits,targets,torch.ones(3,dtype=torch.bool),logits.detach(),'bce')==objective(logits,targets,torch.ones(3,dtype=torch.bool),logits.detach(),'margin')


def test_same_scene_never_crosses_fit_monitor_and_subset_is_repeatable():
    records=[dict(stem=f'{i:07d}_{j:05d}_d') for i in range(2000) for j in range(2)]
    chosen=select_records(records)
    assert len(chosen)==640
    assert chosen==select_records(list(reversed(records)))
    assert sum(h for _,h in chosen)==128
    assert all(scene_holdout(r['stem'])==h for r,h in chosen)
    assert not ({r['stem'].split('_')[0] for r,h in chosen if h} & {r['stem'].split('_')[0] for r,h in chosen if not h})


def test_one_failed_gate_means_stop_and_smoke_cannot_continue():
    values={name: threshold for name,(_,threshold) in GATES.items()}
    assert decide(values,True)['decision']=='CONTINUE'
    values['official_ap_vs_bce_pp']-=.001
    assert decide(values,True)['decision']=='STOP'
    assert decide(values,False)['decision']=='SMOKE_ONLY'
    with pytest.raises(ValueError):
        decide({**values,'tiny_ap_vs_control_pp':np.nan},True)
    with pytest.raises(ValueError):
        decide({},True)


def test_empty_frozen_json_view_matches_empty_six_column_predictions():
    assert frozen_view_matches(np.empty((0,6),dtype=np.float32),[])
    assert not frozen_view_matches(np.ones((1,6),dtype=np.float32),[])


def test_raw_replay_still_requires_exact_values_and_order():
    actual=np.array([[0,0,4,4,.5,1],[1,1,5,5,.5,2]],dtype=np.float32)
    assert frozen_view_matches(actual,actual.tolist())
    assert not frozen_view_matches(actual,actual[::-1].tolist())
    changed=actual.copy()
    changed[0,4]=np.nextafter(changed[0,4],np.float32(1))
    assert not frozen_view_matches(actual,changed.tolist())
    with pytest.raises(ValueError):
        frozen_view_matches(actual,[[0,0]])


def test_evaluation_repair_preserves_training_sources_and_rejects_other_changes(monkeypatch):
    import hashlib
    import zipfile
    import sdcl.evaluation.class_margin_probe as evaluation
    import sdcl.postprocess.class_margin_probe as postprocess
    from sdcl.config import project_path

    original_run=project_path('output/runs/class_margin_probe_960_seed0')
    if not (original_run/'source.zip').is_file():
        pytest.skip('Local completed source archive is unavailable.')
    with zipfile.ZipFile(original_run/'source.zip') as archive:
        hashes={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    digest=hashlib.sha256((original_run/'source.zip').read_bytes()).hexdigest()
    # Model the repair's exact snapshot scope; later independent modules are
    # outside this fixture, and must not weaken the production source checker.
    all_current=postprocess.sources()
    current={n:all_current[n] for n in hashes}
    monkeypatch.setattr(evaluation,'sources',lambda:current)
    check,patch=evaluation_source_check(original_run,hashes,digest)
    assert patch['version']=='2026-10-13-empty-view-replay-v1'
    assert len(patch['changed_files'])==2
    check(hashes)
    monkeypatch.setattr(evaluation,'sources',lambda:{**current,'sdcl/postprocess/class_margin_probe.py':'changed'})
    with pytest.raises(ValueError):
        check(hashes)
    with pytest.raises(ValueError):
        evaluation_source_check(original_run,hashes,digest)
