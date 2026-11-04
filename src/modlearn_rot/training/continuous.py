"""Training utilities for continuous-response Neural Processes."""

from __future__ import annotations

import copy
import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ..generators.dataset import GeneratedDataset
from ..nn import ContinuousNPConfig, ContinuousNeuralProcess
from ..nn.np.common.context import sample_context_keep_mask_torch
from ..nn.np.common.losses import gaussian_kl_loss, latent_l1_loss
from ..nn.np.common.preprocessing import masked_mean_std


@dataclass(frozen=True, slots=True)
class ContinuousNPTrainingConfig:
    beta: float
    lambda_sparse: float
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 1000
    patience: int = 50
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    context_min_fraction: float = 0.25
    context_max_fraction: float = 0.80
    context_sampling_policy: str = "per_presentation"
    deterministic: bool = True
    seed: int = 31
    num_workers: int = 0

    def __post_init__(self):
        if self.beta < 0 or self.lambda_sparse < 0:
            raise ValueError("beta and lambda_sparse must be non-negative.")
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning_rate and batch_size must be positive.")
        if self.max_epochs <= 0 or self.patience < 0:
            raise ValueError("max_epochs must be positive and patience non-negative.")
        split_total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(split_total - 1.0) > 1e-8:
            raise ValueError("Subject split fractions must sum to one.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("Subject split fractions must be positive.")
        if not 0 < self.context_min_fraction <= self.context_max_fraction <= 1:
            raise ValueError("Context fractions must satisfy 0 < min <= max <= 1.")
        if self.context_sampling_policy not in {"per_presentation", "per_epoch"}:
            raise ValueError("context_sampling_policy must be per_presentation or per_epoch.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContinuousData:
    x: Tensor
    y: Tensor
    trial_mask: Tensor

    @property
    def n_subjects(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_trials(self) -> int:
        return int(self.y.shape[1])

    @property
    def x_dim(self) -> int:
        return int(self.x.shape[-1])

    @classmethod
    def from_generated(cls, dataset: GeneratedDataset) -> "ContinuousData":
        n_subjects = len(dataset.trial_counts)
        max_trials = int(np.max(dataset.trial_counts))
        x_dim = dataset.x.shape[1]
        x = torch.zeros((n_subjects, max_trials, x_dim), dtype=torch.float32)
        y = torch.zeros((n_subjects, max_trials, 1), dtype=torch.float32)
        mask = torch.zeros((n_subjects, max_trials), dtype=torch.bool)
        for sid in range(n_subjects):
            idx = np.flatnonzero(dataset.subject_index == sid)
            n = len(idx)
            x[sid, :n] = torch.as_tensor(dataset.x[idx], dtype=torch.float32)
            y[sid, :n, 0] = torch.as_tensor(dataset.y[idx], dtype=torch.float32)
            mask[sid, :n] = True
        return cls(x=x, y=y, trial_mask=mask)


class ContinuousSubjectDataset(Dataset):
    def __init__(self, data: ContinuousData, subject_indices: Tensor):
        self.data = data
        self.subject_indices = torch.as_tensor(subject_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.subject_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sid = self.subject_indices[index]
        return {
            "subject_index": sid,
            "x": self.data.x[sid],
            "y": self.data.y[sid],
            "trial_mask": self.data.trial_mask[sid],
        }


def split_subject_indices(
    n_subjects: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = torch.randperm(n_subjects, generator=generator)
    n_train = max(1, round(n_subjects * train_fraction))
    n_validation = max(1, round(n_subjects * validation_fraction))
    if n_train + n_validation >= n_subjects:
        n_validation = 1
        n_train = n_subjects - 2
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }


def sample_context_mask(
    valid_trial_mask: Tensor,
    minimum_fraction: float,
    maximum_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    if valid_trial_mask.ndim != 2:
        raise ValueError("valid_trial_mask must have shape (B,T).")
    return sample_context_keep_mask_torch(
        valid_trial_mask,
        minimum_fraction,
        maximum_fraction,
        generator=generator,
    )


def continuous_beta_vae_loss(
    prediction: Tensor,
    target: Tensor,
    target_mask: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    beta: float,
    lambda_sparse: float,
) -> dict[str, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    expanded_mask = target_mask.unsqueeze(-1).expand_as(prediction)
    if not expanded_mask.any():
        raise ValueError("At least one valid target is required.")
    reconstruction = torch.mean(
        (prediction[expanded_mask] - target[expanded_mask]) ** 2
    )
    kl = gaussian_kl_loss(mean, log_variance)
    sparse = latent_l1_loss(mean)
    total = reconstruction + float(beta) * kl + float(lambda_sparse) * sparse
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "kl": kl,
        "sparse": sparse,
    }


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def stable_run_seed(base_seed: int, model_id: str, beta: float, repeat: int) -> int:
    """Seed is deliberately independent of lambda at fixed beta/repeat."""
    key = f"{int(base_seed)}|{model_id}|beta={float(beta):.12g}|repeat={int(repeat)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little") % 1_000_000_000
    return int((int(base_seed) + offset) % 2_147_483_647)


def number_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p").replace("+", "")


def training_condition_tag(
    beta: float,
    lambda_sparse: float,
    context_min_fraction: float,
    context_max_fraction: float,
    latent_dim: int,
) -> str:
    return (
        f"beta{number_tag(beta)}_lambda{number_tag(lambda_sparse)}_"
        f"ctx{number_tag(context_min_fraction)}to{number_tag(context_max_fraction)}_"
        f"L{int(latent_dim)}"
    )


def _fit_encoder_preprocessing(
    data: ContinuousData,
    train_indices: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    ids = torch.as_tensor(train_indices, dtype=torch.long)
    observations = torch.cat([data.x[ids], data.y[ids]], dim=-1).numpy()
    mask = data.trial_mask[ids].numpy()
    return masked_mean_std(observations, mask)


def _normalize_encoder_batch(
    x: Tensor,
    y: Tensor,
    encoder_mean: Tensor,
    encoder_std: Tensor,
) -> tuple[Tensor, Tensor]:
    observations = torch.cat([x, y], dim=-1)
    normalized = (observations - encoder_mean) / encoder_std
    x_dim = x.shape[-1]
    return normalized[..., :x_dim], normalized[..., x_dim:]


class ContinuousNPTrainer:
    def __init__(
        self,
        model_config: ContinuousNPConfig,
        training_config: ContinuousNPTrainingConfig,
        device: str | torch.device | None = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        if device is None or str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(
        self,
        data: ContinuousData,
        subject_splits: dict[str, Tensor],
        fixed_context_masks: dict[str, Tensor],
        epoch_callback=None,
    ) -> dict:
        _seed_everything(
            self.training_config.seed,
            self.training_config.deterministic,
        )
        encoder_mean_np, encoder_std_np = _fit_encoder_preprocessing(
            data, subject_splits["train"]
        )
        encoder_mean = torch.as_tensor(
            encoder_mean_np, dtype=torch.float32, device=self.device
        )
        encoder_std = torch.as_tensor(
            encoder_std_np, dtype=torch.float32, device=self.device
        )

        loaders = self._make_loaders(data, subject_splits)
        model = ContinuousNeuralProcess(self.model_config).to(self.device)
        optimizer = Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        train_context_generator = torch.Generator().manual_seed(
            self.training_config.seed + 1000
        )
        history = {
            f"{prefix}_{name}": []
            for prefix in ("train", "validation")
            for name in (
                "loss", "reconstruction", "kl", "sparse", "r2", "rmse"
            )
        }
        best_state = copy.deepcopy(model.state_dict())
        best_validation_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0

        for epoch in range(self.training_config.max_epochs):
            epoch_context = None
            if self.training_config.context_sampling_policy == "per_epoch":
                epoch_context = torch.zeros_like(data.trial_mask)
                epoch_context[subject_splits["train"]] = sample_context_mask(
                    data.trial_mask[subject_splits["train"]],
                    self.training_config.context_min_fraction,
                    self.training_config.context_max_fraction,
                    train_context_generator,
                )
            train_metrics = self._run_training_epoch(
                model,
                loaders["train"],
                optimizer,
                train_context_generator,
                encoder_mean,
                encoder_std,
                epoch_context,
            )
            validation_metrics = self._evaluate(
                model,
                loaders["validation"],
                fixed_context_masks["validation"],
                encoder_mean,
                encoder_std,
            )
            for name, value in train_metrics.items():
                history[f"train_{name}"].append(value)
            for name, value in validation_metrics.items():
                history[f"validation_{name}"].append(value)
            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, validation_metrics)

            improved = (
                validation_metrics["loss"]
                < best_validation_loss - self.training_config.min_delta
            )
            if improved:
                best_validation_loss = validation_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.training_config.patience:
                    break

        model.load_state_dict(best_state)
        validation_metrics = self._evaluate(
            model,
            loaders["validation"],
            fixed_context_masks["validation"],
            encoder_mean,
            encoder_std,
        )
        test_metrics = self._evaluate(
            model,
            loaders["test"],
            fixed_context_masks["test"],
            encoder_mean,
            encoder_std,
        )
        return {
            "model": model,
            "model_config": self.model_config,
            "history": history,
            "best_epoch": best_epoch,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "preprocessing": {
                "encoder_mean": torch.from_numpy(encoder_mean_np),
                "encoder_std": torch.from_numpy(encoder_std_np),
            },
            "subject_splits": subject_splits,
            "fixed_context_masks": fixed_context_masks,
        }

    def _make_loaders(self, data, splits):
        out = {}
        for offset, name in enumerate(("train", "validation", "test")):
            generator = torch.Generator().manual_seed(
                self.training_config.seed + offset
            )
            out[name] = DataLoader(
                ContinuousSubjectDataset(data, splits[name]),
                batch_size=self.training_config.batch_size,
                shuffle=name == "train",
                num_workers=self.training_config.num_workers,
                generator=generator,
            )
        return out

    def _move(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _run_training_epoch(
        self,
        model,
        loader,
        optimizer,
        context_generator,
        encoder_mean,
        encoder_std,
        epoch_context,
    ):
        model.train()
        totals = {
            k: 0.0
            for k in ("loss", "reconstruction", "kl", "sparse")
        }
        pred_parts, target_parts = [], []
        for batch in loader:
            if epoch_context is None:
                context_mask = sample_context_mask(
                    batch["trial_mask"],
                    self.training_config.context_min_fraction,
                    self.training_config.context_max_fraction,
                    context_generator,
                )
            else:
                context_mask = epoch_context[batch["subject_index"].long()]
            batch = self._move(batch)
            context_mask = context_mask.to(self.device)
            x_enc, y_enc = _normalize_encoder_batch(
                batch["x"], batch["y"], encoder_mean, encoder_std
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                x_enc,
                y_enc,
                context_mask,
                batch["x"],
                batch["trial_mask"],
                sample_latent=True,
            )
            terms = continuous_beta_vae_loss(
                output["prediction"],
                batch["y"],
                batch["trial_mask"],
                output["mean"],
                output["log_variance"],
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
                totals[k] += float(terms[k].item())
            mask = batch["trial_mask"].unsqueeze(-1).expand_as(
                output["prediction"]
            )
            pred_parts.append(output["prediction"][mask].detach().cpu())
            target_parts.append(batch["y"][mask].detach().cpu())
        return self._finish_metrics(
            totals, len(loader), pred_parts, target_parts
        )

    @torch.no_grad()
    def _evaluate(
        self,
        model,
        loader,
        fixed_context_mask,
        encoder_mean,
        encoder_std,
    ):
        model.eval()
        totals = {
            k: 0.0
            for k in ("loss", "reconstruction", "kl", "sparse")
        }
        pred_parts, target_parts = [], []
        for batch in loader:
            global_ids = batch["subject_index"].long()
            context_mask = fixed_context_mask[global_ids].to(self.device)
            batch = self._move(batch)
            x_enc, y_enc = _normalize_encoder_batch(
                batch["x"], batch["y"], encoder_mean, encoder_std
            )
            output = model(
                x_enc,
                y_enc,
                context_mask,
                batch["x"],
                batch["trial_mask"],
                sample_latent=False,
            )
            terms = continuous_beta_vae_loss(
                output["prediction"],
                batch["y"],
                batch["trial_mask"],
                output["mean"],
                output["log_variance"],
                self.training_config.beta,
                self.training_config.lambda_sparse,
            )
            for k in totals:
                totals[k] += float(terms[k].item())
            mask = batch["trial_mask"].unsqueeze(-1).expand_as(
                output["prediction"]
            )
            pred_parts.append(output["prediction"][mask].cpu())
            target_parts.append(batch["y"][mask].cpu())
        return self._finish_metrics(
            totals, len(loader), pred_parts, target_parts
        )

    @staticmethod
    def _finish_metrics(totals, n_batches, pred_parts, target_parts):
        if n_batches == 0:
            raise ValueError("Empty data loader.")
        metrics = {k: v / n_batches for k, v in totals.items()}
        pred = torch.cat(pred_parts).numpy()
        target = torch.cat(target_parts).numpy()
        residual = float(np.sum((target - pred) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        metrics["r2"] = (
            float(1.0 - residual / total) if total > 0 else float("nan")
        )
        metrics["rmse"] = float(np.sqrt(np.mean((target - pred) ** 2)))
        return metrics


def make_fixed_context_masks(
    data: ContinuousData,
    subject_splits: dict[str, Tensor],
    minimum_fraction: float,
    maximum_fraction: float,
    seed: int,
) -> dict[str, Tensor]:
    out = {}
    for offset, name in enumerate(("validation", "test")):
        generator = torch.Generator().manual_seed(
            int(seed) + 10_000 + offset
        )
        sampled = sample_context_mask(
            data.trial_mask[subject_splits[name]],
            minimum_fraction,
            maximum_fraction,
            generator,
        )
        full = torch.zeros_like(data.trial_mask)
        full[subject_splits[name]] = sampled
        out[name] = full
    return out


def save_history_csv(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max((len(v) for v in history.values()), default=0)
    fields = ["epoch", *history.keys()]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            row = {"epoch": i + 1}
            for k, v in history.items():
                row[k] = v[i] if i < len(v) else ""
            w.writerow(row)


def _choice_rt_preprocessing_from_result(result):
    if "mean_x_enc" not in result or "std_x_enc" not in result:
        return None
    payload = {
        "encoder_mean": torch.as_tensor(result["mean_x_enc"]).cpu(),
        "encoder_std": torch.as_tensor(result["std_x_enc"]).cpu(),
    }
    if "mean_rt" in result:
        payload["rt_mean"] = torch.as_tensor(float(result["mean_rt"]))
    if "std_rt" in result:
        payload["rt_std"] = torch.as_tensor(float(result["std_rt"]))
    return payload


def save_checkpoint(
    path: str | Path,
    result: dict,
    model_config,
    training_config,
    subject_splits: dict[str, Tensor],
    fixed_context_masks: dict[str, Tensor],
    extra_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actual_model_config = result.get("model_config", model_config)
    actual_splits = result.get("subject_splits", subject_splits)
    actual_masks = result.get("fixed_context_masks", fixed_context_masks)
    preprocessing = result.get("preprocessing")
    if preprocessing is None:
        preprocessing = _choice_rt_preprocessing_from_result(result)

    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in result["model"].state_dict().items()
            },
            "model_config": actual_model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "history": result["history"],
            "best_epoch": result["best_epoch"],
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
            "subject_splits": {
                k: torch.as_tensor(v).cpu()
                for k, v in actual_splits.items()
            },
            "fixed_context_masks": {
                k: torch.as_tensor(v).cpu()
                for k, v in actual_masks.items()
            },
            "preprocessing": preprocessing,
            "extra_state": extra_state or {},
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")
