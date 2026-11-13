"""Verify coordinate conversion and that T never selects a saved rotation."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_candidate_rotation_gallery import freeze_grid_search,evaluate_frozen,sha
from export_saved_search_heatmaps import select_saved_frames


@pytest.fixture(autouse=True)
def isolated_output(tmp_path,monkeypatch):
    import export_candidate_rotation_gallery as gallery
    monkeypatch.setattr(gallery,'OUTPUT',tmp_path/'gallery')


def fixture(tmp_path):
    rng=np.random.default_rng(19);z=rng.normal(size=(20,2))
    source=tmp_path/'cases_source';stage=tmp_path/'stage1';folder=tmp_path/'wave'
    source.mkdir();stage.mkdir();folder.mkdir()
    (folder/'protocol.json').write_text('{}')
    np.save(stage/'Zstd.npy',z);counts=np.ones(20,dtype=int)
    for key,value in [('trial_counts',counts),('subject_index',np.arange(20)),('trial_index',np.zeros(20,dtype=int))]:
        np.save(source/(key+'.npy'),value)
    theta=.43;a=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    for name,q in [('baseline',a),('rotated',np.eye(2))]:
        path=source/'cases'/name;path.mkdir(parents=True);np.save(path/'W.npy',z@q)
    dataset=tmp_path/'dataset.npz'
    np.savez(dataset,parameters=z.copy(),trial_counts=counts,subject_index=np.arange(20),trial_index=np.zeros(20,dtype=int))
    (stage/'manifest.json').write_text(json.dumps(dict(source_dataset=str(dataset),source_dataset_sha256=sha(dataset))))
    info=dict(source=source,folder=folder,index=dict(source_stage1=str(stage),Z_dim=2),
        losses=dict(baseline=2.,rotated=1.),scores_sha256='frozen_scores',source_tag='TEST')
    return info,z,a,dataset


def test_grid_R_uses_original_Z_convention_and_posthoc_alignment(tmp_path):
    info,z,a,dataset=fixture(tmp_path);out=tmp_path/'result'
    result=freeze_grid_search(info,out,'metric','p_nn')
    np.testing.assert_allclose(z@np.load(out/'R.npy').T,np.load(out/'U.npy'),atol=1e-12)
    np.testing.assert_allclose(np.load(out/'U.npy'),z,atol=1e-12)
    correlation,_=evaluate_frozen(info,out)
    np.testing.assert_allclose(np.diag(correlation),1.,atol=1e-12)
    assert result['T_used_by_search'] is False
    # A subject ordering mismatch must fail evaluation, after selection froze.
    f=dict(np.load(dataset));f['subject_index']=f['subject_index'][::-1];np.savez(dataset,**f)
    with pytest.raises(ValueError,match='alignment'):evaluate_frozen(info,out)


def test_pool_selects_lower_off_grid_loss_without_T(tmp_path):
    info,z,a,dataset=fixture(tmp_path)
    (info['folder']/'reveal.json').write_text(json.dumps(dict(mapping={'case_000':'baseline','case_001':'rotated'})))
    # Anonymous coordinates are Z @ a. Third frame is deliberately off grid.
    q=np.array([[0.,-1.],[1.,0.]])
    frames=np.array([np.eye(2),a.T,q]);directions=np.concatenate([x.T for x in frames]);membership=np.arange(6).reshape(3,2)
    np.savez(info['folder']/'W_only_geometry.npz',directions=directions,membership=membership)
    with (info['folder']/'blind_scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['anonymous_case','metric','channel','loss']);writer.writeheader()
        writer.writerows(dict(anonymous_case=k,metric='metric',channel='p_nn',loss=v)
            for k,v in [('case_000',2.),('case_001',1.),('audit_frame_000',.5)])
    # Delete T completely: scoring/selection still succeeds.
    dataset.unlink()
    out=tmp_path/'result';result=select_saved_frames(info,out,'metric','p_nn')
    assert result['loss']==.5 and result['selected_kind']=='saved off-grid stress frame'
    np.testing.assert_allclose(np.load(out/'U.npy'),z@a@q,atol=1e-12)
    assert not result['global_optimality_proven']
    assert result['candidate_frames']==3
