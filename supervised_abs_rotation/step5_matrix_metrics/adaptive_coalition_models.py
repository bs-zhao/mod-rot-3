"""Choose a generic conditional emulator by GCV, never by rotation labels."""
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

import conditional_ux as conditional
import conditional_model_quality as quality
import shapley_function_profiles as shapley


def subspace_key(basis):
    projector=np.round(basis@basis.T,6);projector[projector==0.]=0.
    return basis.shape[1],hashlib.sha256(projector.tobytes()).hexdigest()


class AdaptiveCoalitionModels(shapley.CoalitionModels):
    def __init__(self,z,target,output,sources,device='cuda',prediction_cache_size=16):
        self.z,self.target,self.output,self.device=z,target,Path(output),device
        self.output.mkdir(parents=True,exist_ok=True);self.models={};self.audits=[]
        self.sources={};self.components={};self.predictions=OrderedDict();self.prediction_cache_size=prediction_cache_size
        reference=target.detach().cpu().numpy();reference_energy=float(np.sum(reference*reference))
        snapshots=[];manifest=None
        for tag,folder in sources.items():
            folder=Path(folder);protocol=json.loads((folder/'protocol.json').read_text())
            digest=hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
            if protocol['status']!='COMPLETE' or digest!=protocol['scores_sha256']:raise ValueError('A candidate emulator source changed')
            if protocol['response_transform']!='identity':raise ValueError('Adaptive candidates must use the same raw response functions')
            current=protocol['source_matrix_manifest_sha256']
            if manifest is None:manifest=current
            if current!=manifest:raise ValueError('Candidate emulators use different actual matrices')
            with np.load(folder/'shared_response_basis.npz',allow_pickle=False) as saved:old=saved['latent'].copy()
            if len(old)!=len(reference):raise ValueError('Candidate emulators have different observed subjects')
            if np.array_equal(old,reference):alignment=None;error=0.
            else:
                left,_,right=np.linalg.svd(old.T@reference,full_matrices=False);alignment=left@right
                error=float(np.sum((old@alignment-reference)**2)/max(reference_energy,1e-30))
            if error>1e-8:raise ValueError('Candidate observed-function spaces could not be aligned within the compression budget')
            target_variance=float(np.sum(old*old)/len(old))
            audit_file=folder/'conditional_model_audits.json'
            fitted={int(a['index']):a for a in json.loads(audit_file.read_text())};catalog={}
            for file in sorted((folder/'conditional_models').glob('model*.npz')):
                index=int(file.stem[5:]);audit=fitted[index]
                with np.load(file,allow_pickle=False) as saved:
                    basis=saved['basis'].copy();length=float(saved['lengthscale'])
                identifier=subspace_key(basis)
                entry=dict(file=file,basis=basis,lengthscale=length,
                    relative_GCV=float(audit['options'][audit['selected_index']]['GCV'])/max(target_variance,1e-30))
                if identifier in catalog:raise ValueError('Ambiguous repeated candidate subspace in frozen source')
                catalog[identifier]=entry
            self.sources[tag]=dict(folder=folder,protocol=protocol,catalog=catalog,
                alignment=None if alignment is None else torch.tensor(alignment,dtype=torch.float64,device=device))
            self.components[tag]=shapley.CoalitionModels(z,target,self.output/'new_candidates'/tag,
                degree=protocol['degree'],centers=protocol['radial_centers'],seed=protocol['seed'],device=device)
            snapshots.append(dict(family=tag,source=str(folder),source_scores_sha256=digest,
                source_response_basis_sha256=hashlib.sha256((folder/'shared_response_basis.npz').read_bytes()).hexdigest(),
                source_model_audits_sha256=hashlib.sha256(audit_file.read_bytes()).hexdigest(),
                relative_squared_response_alignment_error=error,degree=protocol['degree'],radial_centers=protocol['radial_centers'],
                frozen_conditional_models=len(catalog)))
        self.protocol_info=dict(adaptive_GCV=True,degree='selected_by_GCV',radial_centers='selected_by_GCV',
            conditional_model='lowest normalized GCV among the same generic polynomial/radial candidates in every input subspace',
            candidate_sources=snapshots,source_matrix_manifest_sha256=manifest,
            selection_uses_case_labels=False,selection_uses_reference_angles=False,hand_chosen_mixture_weights=False,
            GCV_definition='candidate all-subject GCV divided by its retained observed-target variance',
            response_alignment_relative_squared_budget=1e-8,
            adaptive_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @torch.no_grad()
    def predict(self,axes):
        basis=np.linalg.qr(np.asarray(axes))[0][:,:axes.shape[1]];identifier=subspace_key(basis)
        if identifier in self.predictions:
            self.predictions.move_to_end(identifier);return self.predictions[identifier]
        if identifier not in self.models:
            candidates=[]
            for tag,source in self.sources.items():
                entry=source['catalog'].get(identifier)
                if entry is not None:
                    if np.max(np.abs(entry['basis']@entry['basis'].T-basis@basis.T))>1e-6:
                        raise ValueError('Candidate subspace hash collision')
                    candidates.append(dict(family=tag,relative_GCV=entry['relative_GCV'],origin='frozen_source'))
                else:
                    component=self.components[tag];component.predict(axes)
                    model=component.models[identifier];option=model['options'][model['selected']]
                    variance=float(self.target.square().sum()/len(self.target))
                    candidates.append(dict(family=tag,relative_GCV=float(option['GCV'])/max(variance,1e-30),origin='new_fit'))
            chosen=min(candidates,key=lambda candidate:candidate['relative_GCV'])
            record=dict(index=len(self.models),input_dimension=basis.shape[1],projector_key=identifier[1],
                selected_family=chosen['family'],selected_relative_GCV=chosen['relative_GCV'],
                origin=chosen['origin'],candidate_scores=candidates)
            self.models[identifier]=record;self.audits.append(record)
        chosen=self.models[identifier];tag=chosen['selected_family'];source=self.sources[tag]
        if chosen['origin']=='new_fit':prediction=self.components[tag].predict(axes)
        else:
            entry=source['catalog'][identifier]
            with np.load(entry['file'],allow_pickle=False) as saved:coefficient=saved['coefficient'].copy()
            design,_=conditional.subspace_design(self.z,entry['basis'],source['protocol']['degree'],
                source['protocol']['radial_centers'],entry['lengthscale'],source['protocol']['seed'])
            coefficient=torch.tensor(coefficient,dtype=torch.float64,device=self.device)
            if source['alignment'] is not None:coefficient=coefficient@source['alignment']
            prediction=torch.tensor(design,dtype=torch.float64,device=self.device)@coefficient
            chosen['selected_model_file']=entry['file'].name
            chosen['selected_model_sha256']=hashlib.sha256(entry['file'].read_bytes()).hexdigest()
        self.predictions[identifier]=prediction
        while len(self.predictions)>self.prediction_cache_size:self.predictions.popitem(last=False)
        return prediction

    def finalize(self):
        for tag,component in self.components.items():
            (component.output/'candidate_fit_audits.json').write_text(json.dumps(component.audits,indent=2),encoding='utf-8')
        return dict(selected_family_counts=dict(Counter(r['selected_family'] for r in self.audits)),
            new_candidate_fit_counts={tag:len(component.models) for tag,component in self.components.items()},
            selected_subspaces=len(self.models))

    @torch.no_grad()
    def fixed_fit_PRESS(self,axes):
        """Exact linear-smoother PRESS, conditional on the selected frozen fit."""
        self.predict(axes)
        identifier=subspace_key(np.linalg.qr(np.asarray(axes))[0][:,:axes.shape[1]])
        record=self.models[identifier];tag=record['selected_family'];source=self.sources[tag]
        if record['origin']=='new_fit':
            component=self.components[tag];model=component.models[identifier]
            design,audit=component.design(model['basis'],model['lengthscale'])
            coefficient=torch.tensor(model['coefficient'],dtype=torch.float64,device=self.device)
            ridge=model['options'][model['selected']]['ridge']
        else:
            entry=source['catalog'][identifier]
            design,audit=conditional.subspace_design(self.z,entry['basis'],source['protocol']['degree'],
                source['protocol']['radial_centers'],entry['lengthscale'],source['protocol']['seed'])
            with np.load(entry['file'],allow_pickle=False) as saved:
                coefficient=torch.tensor(saved['coefficient'],dtype=torch.float64,device=self.device)
                ridge=float(saved['selected_ridge'])
            if source['alignment'] is not None:coefficient=coefficient@source['alignment']
        _,PRESS,numerical=quality.audit(torch.tensor(design,dtype=torch.float64,device=self.device),
            coefficient,self.target,audit['linear_count'],ridge)
        return PRESS,dict(selected_family=tag,**numerical)


class ResidualAwareAdaptiveModels(AdaptiveCoalitionModels):
    """Use an estimated full coalition; retain its observed residual explicitly."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fitted_full=self.predict(np.eye(self.z.shape[1]));self.unassigned=self.target-self.fitted_full
        identifier=subspace_key(np.eye(self.z.shape[1]));selected=self.models[identifier]['selected_family']
        component=self.components[selected];model=component.models[identifier]
        design,audit=component.design(model['basis'],model['lengthscale'])
        _,self.full_PRESS,self.full_quality=quality.audit(torch.tensor(design,dtype=torch.float64,device=self.device),
            torch.tensor(model['coefficient'],dtype=torch.float64,device=self.device),self.target,
            audit['linear_count'],model['options'][model['selected']]['ridge'])
        self.metric_prefix='resolved_with_unassigned'
        self.protocol_info.update(residual_separated=True,effect_group_count=self.z.shape[1]+1,
            value_function='estimated conditional response means; fitted all-U mean for full coalition; zero empty coalition',
            decomposition='Shapley shares of the estimated-full-coalition game plus a separate observed residual component',
            residual_interpretation='unassigned, not presumed noise and not discarded',
            full_model_fixed_across_coordinate_frames=True,full_model_selected_family=selected,
            unassigned_relative_squared_energy=float(self.unassigned.square().sum()/self.target.square().sum()),
            full_model_fixed_hyperparameter_PRESS_relative_MSE=float(self.full_PRESS.square().sum()/self.target.square().sum()))
        np.savez_compressed(self.output/'unassigned_response_latent.npz',residual=self.unassigned.cpu().numpy())
        (self.output/'full_model_numerical_audit.json').write_text(json.dumps(self.full_quality,indent=2),encoding='utf-8')

    @torch.no_grad()
    def frame_shares(self,q):
        observed_shares,_=super().frame_shares(q)
        assigned=observed_shares-self.unassigned[None]/q.shape[1]
        parts=torch.cat((assigned,self.unassigned[None]),dim=0)
        error=float((parts.sum(0)-self.target).square().sum()/self.target.square().sum().clamp_min(1e-30))
        if error>1e-18:raise ArithmeticError('Resolved plus unassigned profiles did not reconstruct the observed response')
        return parts,error

    @torch.no_grad()
    def response_audits(self,projections,tasks,channels):
        rows=[]
        for p,(c,j) in zip(projections,tasks):
            total=float((self.target@p).square().sum())
            rows.append(dict(channel=channels[c],X_column=j,
                unassigned_relative_squared_energy=float((self.unassigned@p).square().sum())/max(total,1e-30),
                full_model_fixed_hyperparameter_PRESS_relative_MSE=float((self.full_PRESS@p).square().sum())/max(total,1e-30)))
        return rows
