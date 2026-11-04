"""Training utilities for categorical point-response Neural Processes."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset

from ....generators.dataset import GeneratedDataset
from ....nn.np.common.losses import gaussian_kl_loss,latent_l1_loss
from ....nn.np.common.preprocessing import masked_mean_std
from ....nn.np.point.discrete import DiscreteNPConfig,DiscreteNeuralProcess
from ...continuous import (
    make_fixed_context_masks,
    sample_context_mask,
    split_subject_indices,
    stable_run_seed,
    training_condition_tag,
)


@dataclass(frozen=True, slots=True)
class DiscreteNPTrainingConfig:
    beta: float
    lambda_sparse: float
    learning_rate: float=5e-4
    weight_decay: float=0.0
    batch_size: int=64
    max_epochs: int=1000
    patience: int=50
    min_delta: float=0.0
    gradient_clip_norm: float|None=1.0
    train_fraction: float=0.70
    validation_fraction: float=0.15
    test_fraction: float=0.15
    context_min_fraction: float=0.25
    context_max_fraction: float=0.80
    context_sampling_policy: str="per_presentation"
    deterministic: bool=True
    seed: int=31
    num_workers: int=0

    def __post_init__(self):
        if self.beta<0 or self.lambda_sparse<0:
            raise ValueError("beta and lambda_sparse must be non-negative.")
        if self.learning_rate<=0 or self.batch_size<=0:
            raise ValueError("learning_rate and batch_size must be positive.")
        if self.max_epochs<=0 or self.patience<0:
            raise ValueError("max_epochs must be positive and patience non-negative.")
        if not 0<self.context_min_fraction<=self.context_max_fraction<=1:
            raise ValueError("Context fractions must satisfy 0 < min <= max <= 1.")
        if self.context_sampling_policy not in {"per_presentation","per_epoch"}:
            raise ValueError("Unknown context sampling policy.")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscreteData:
    x: Tensor
    choice: Tensor
    trial_mask: Tensor

    @property
    def n_subjects(self):
        return int(self.choice.shape[0])

    @property
    def n_trials(self):
        return int(self.choice.shape[1])

    @property
    def x_dim(self):
        return int(self.x.shape[-1])

    @classmethod
    def from_generated(cls,dataset:GeneratedDataset):
        n_subjects=len(dataset.trial_counts)
        max_trials=int(np.max(dataset.trial_counts))
        x_dim=int(dataset.x.shape[1])
        x=torch.zeros((n_subjects,max_trials,x_dim),dtype=torch.float32)
        choice=torch.zeros((n_subjects,max_trials),dtype=torch.long)
        mask=torch.zeros((n_subjects,max_trials),dtype=torch.bool)
        for sid in range(n_subjects):
            idx=np.flatnonzero(dataset.subject_index==sid)
            n=len(idx)
            x[sid,:n]=torch.as_tensor(dataset.x[idx],dtype=torch.float32)
            choice[sid,:n]=torch.as_tensor(
                np.asarray(dataset.y[idx]).reshape(-1),dtype=torch.long
            )
            mask[sid,:n]=True
        return cls(x=x,choice=choice,trial_mask=mask)


class DiscreteSubjectDataset(Dataset):
    def __init__(self,data:DiscreteData,subject_indices:Tensor):
        self.data=data
        self.subject_indices=torch.as_tensor(subject_indices,dtype=torch.long)

    def __len__(self):
        return len(self.subject_indices)

    def __getitem__(self,index):
        sid=self.subject_indices[index]
        return {
            "subject_index":sid,
            "x":self.data.x[sid],
            "choice":self.data.choice[sid],
            "trial_mask":self.data.trial_mask[sid],
        }


def discrete_beta_vae_loss(
    logits:Tensor,
    choice:Tensor,
    target_mask:Tensor,
    mean:Tensor,
    log_variance:Tensor,
    beta:float,
    lambda_sparse:float,
)->dict[str,Tensor]:
    if logits.shape[:2]!=choice.shape or logits.shape[:2]!=target_mask.shape:
        raise ValueError("logits, choice, and target_mask must agree on B and T.")
    valid=target_mask.bool()
    if not valid.any():
        raise ValueError("At least one valid target is required.")
    reconstruction=F.cross_entropy(logits[valid],choice[valid])
    kl=gaussian_kl_loss(mean,log_variance)
    sparse=latent_l1_loss(mean)
    total=reconstruction+float(beta)*kl+float(lambda_sparse)*sparse
    return {
        "loss":total,
        "reconstruction":reconstruction,
        "kl":kl,
        "sparse":sparse,
    }


def _seed_everything(seed:int,deterministic:bool):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic,warn_only=True)
    torch.backends.cudnn.deterministic=deterministic
    if deterministic:
        torch.backends.cudnn.benchmark=False


def _fit_encoder_preprocessing(data:DiscreteData,train_indices:Tensor):
    ids=torch.as_tensor(train_indices,dtype=torch.long)
    obs=torch.cat(
        [data.x[ids],data.choice[ids].float().unsqueeze(-1)],dim=-1
    ).numpy()
    mask=data.trial_mask[ids].numpy()
    return masked_mean_std(obs,mask)


def _normalize_encoder_batch(x,choice,mean,std):
    obs=torch.cat([x,choice.float().unsqueeze(-1)],dim=-1)
    norm=(obs-mean)/std
    return norm[...,:x.shape[-1]],norm[...,x.shape[-1]:]


class DiscreteNPTrainer:
    def __init__(
        self,
        model_config:DiscreteNPConfig,
        training_config:DiscreteNPTrainingConfig,
        device="auto",
    ):
        self.model_config=model_config
        self.training_config=training_config
        if str(device)=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=torch.device(device)

    def fit(
        self,
        data:DiscreteData,
        subject_splits,
        fixed_context_masks,
        epoch_callback=None,
    ):
        cfg=self.training_config
        _seed_everything(cfg.seed,cfg.deterministic)

        mean_np,std_np=_fit_encoder_preprocessing(
            data,subject_splits["train"]
        )
        encoder_mean=torch.as_tensor(
            mean_np,dtype=torch.float32,device=self.device
        )
        encoder_std=torch.as_tensor(
            std_np,dtype=torch.float32,device=self.device
        )

        loaders={}
        for offset,name in enumerate(("train","validation","test")):
            generator=torch.Generator().manual_seed(cfg.seed+offset)
            loaders[name]=DataLoader(
                DiscreteSubjectDataset(data,subject_splits[name]),
                batch_size=cfg.batch_size,
                shuffle=name=="train",
                num_workers=cfg.num_workers,
                generator=generator,
            )

        model=DiscreteNeuralProcess(self.model_config).to(self.device)
        optimizer=Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        context_generator=torch.Generator().manual_seed(cfg.seed+1000)
        history={
            f"{prefix}_{name}":[]
            for prefix in ("train","validation")
            for name in (
                "loss","reconstruction","kl","sparse",
                "accuracy","brier",
            )
        }
        best_state=copy.deepcopy(model.state_dict())
        best_val=float("inf")
        best_epoch=-1
        bad_epochs=0

        for epoch in range(cfg.max_epochs):
            epoch_context=None
            if cfg.context_sampling_policy=="per_epoch":
                epoch_context=torch.zeros_like(data.trial_mask)
                epoch_context[subject_splits["train"]]=sample_context_mask(
                    data.trial_mask[subject_splits["train"]],
                    cfg.context_min_fraction,
                    cfg.context_max_fraction,
                    context_generator,
                )
            train_metrics=self._run_training_epoch(
                model,loaders["train"],optimizer,context_generator,
                encoder_mean,encoder_std,epoch_context,
            )
            val_metrics=self._evaluate(
                model,loaders["validation"],
                fixed_context_masks["validation"],
                encoder_mean,encoder_std,
            )
            for name,value in train_metrics.items():
                history[f"train_{name}"].append(value)
            for name,value in val_metrics.items():
                history[f"validation_{name}"].append(value)
            if epoch_callback is not None:
                epoch_callback(epoch,train_metrics,val_metrics)

            if val_metrics["loss"]<best_val-cfg.min_delta:
                best_val=val_metrics["loss"]
                best_state=copy.deepcopy(model.state_dict())
                best_epoch=epoch
                bad_epochs=0
            else:
                bad_epochs+=1
                if bad_epochs>=cfg.patience:
                    break

        model.load_state_dict(best_state)
        validation_metrics=self._evaluate(
            model,loaders["validation"],
            fixed_context_masks["validation"],
            encoder_mean,encoder_std,
        )
        test_metrics=self._evaluate(
            model,loaders["test"],
            fixed_context_masks["test"],
            encoder_mean,encoder_std,
        )
        return {
            "model":model,
            "model_config":self.model_config,
            "history":history,
            "best_epoch":best_epoch,
            "validation_metrics":validation_metrics,
            "test_metrics":test_metrics,
            "preprocessing":{
                "encoder_mean":torch.from_numpy(mean_np),
                "encoder_std":torch.from_numpy(std_np),
            },
            "subject_splits":subject_splits,
            "fixed_context_masks":fixed_context_masks,
        }

    def _move(self,batch):
        return {k:v.to(self.device) for k,v in batch.items()}

    def _run_training_epoch(
        self,model,loader,optimizer,context_generator,
        encoder_mean,encoder_std,epoch_context,
    ):
        model.train()
        totals={k:0.0 for k in ("loss","reconstruction","kl","sparse")}
        correct=0; n_obs=0; brier_sum=0.0
        for batch in loader:
            if epoch_context is None:
                context_mask=sample_context_mask(
                    batch["trial_mask"],
                    self.training_config.context_min_fraction,
                    self.training_config.context_max_fraction,
                    context_generator,
                )
            else:
                context_mask=epoch_context[batch["subject_index"].long()]
            batch=self._move(batch)
            context_mask=context_mask.to(self.device)
            x_enc,c_enc=_normalize_encoder_batch(
                batch["x"],batch["choice"],encoder_mean,encoder_std
            )
            optimizer.zero_grad(set_to_none=True)
            out=model(
                x_enc,c_enc,context_mask,
                batch["x"],batch["trial_mask"],
                sample_latent=True,
            )
            terms=discrete_beta_vae_loss(
                out["logits"],batch["choice"],batch["trial_mask"],
                out["mean"],out["log_variance"],
                self.training_config.beta,
                self.training_config.lambda_sparse,
            )
            terms["loss"].backward()
            if self.training_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.gradient_clip_norm,
                )
            optimizer.step()
            for k in totals:
                totals[k]+=float(terms[k].item())
            c,n,b=self._classification_sums(
                out["logits"].detach(),batch["choice"],batch["trial_mask"]
            )
            correct+=c; n_obs+=n; brier_sum+=b
        return self._finish(totals,len(loader),correct,n_obs,brier_sum)

    @torch.no_grad()
    def _evaluate(
        self,model,loader,fixed_context_mask,encoder_mean,encoder_std
    ):
        model.eval()
        totals={k:0.0 for k in ("loss","reconstruction","kl","sparse")}
        correct=0; n_obs=0; brier_sum=0.0
        for batch in loader:
            global_ids=batch["subject_index"].long()
            context_mask=fixed_context_mask[global_ids].to(self.device)
            batch=self._move(batch)
            x_enc,c_enc=_normalize_encoder_batch(
                batch["x"],batch["choice"],encoder_mean,encoder_std
            )
            out=model(
                x_enc,c_enc,context_mask,
                batch["x"],batch["trial_mask"],
                sample_latent=False,
            )
            terms=discrete_beta_vae_loss(
                out["logits"],batch["choice"],batch["trial_mask"],
                out["mean"],out["log_variance"],
                self.training_config.beta,
                self.training_config.lambda_sparse,
            )
            for k in totals:
                totals[k]+=float(terms[k].item())
            c,n,b=self._classification_sums(
                out["logits"],batch["choice"],batch["trial_mask"]
            )
            correct+=c; n_obs+=n; brier_sum+=b
        return self._finish(totals,len(loader),correct,n_obs,brier_sum)

    def _classification_sums(self,logits,choice,mask):
        valid=mask.bool()
        logits_v=logits[valid]
        choice_v=choice[valid]
        pred=logits_v.argmax(dim=-1)
        correct=int((pred==choice_v).sum().item())
        n=int(choice_v.numel())
        if self.model_config.num_choices==2:
            p1=torch.softmax(logits_v,dim=-1)[:,1]
            brier=float(torch.sum((p1-choice_v.float())**2).item())
        else:
            probs=torch.softmax(logits_v,dim=-1)
            one_hot=F.one_hot(
                choice_v,num_classes=self.model_config.num_choices
            ).float()
            brier=float(torch.sum(torch.sum((probs-one_hot)**2,dim=-1)).item())
        return correct,n,brier

    @staticmethod
    def _finish(totals,n_batches,correct,n_obs,brier_sum):
        if n_batches==0 or n_obs==0:
            raise ValueError("Empty data loader.")
        metrics={k:v/n_batches for k,v in totals.items()}
        metrics["accuracy"]=float(correct/n_obs)
        metrics["brier"]=float(brier_sum/n_obs)
        return metrics


__all__=[
    "DiscreteData",
    "DiscreteNPConfig",
    "DiscreteNeuralProcess",
    "DiscreteNPTrainer",
    "DiscreteNPTrainingConfig",
    "discrete_beta_vae_loss",
    "make_fixed_context_masks",
    "split_subject_indices",
    "stable_run_seed",
    "training_condition_tag",
]
