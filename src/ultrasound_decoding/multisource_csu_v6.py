"""Frozen Correlated Style Uncertainty training for multi-source LOSO v6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import TASK_CLASS_NAMES, cycle_text
from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)
from ultrasound_decoding.multisource_loso_v5 import (
    FROZEN_SUPERVISED_CONFIG,
    V5_SEEDS,
    PreparedCrossSessionData,
    binary_roc_auc,
    epoch_draw_indices,
    source_sessions_for_target,
)


V6_SEEDS = V5_SEEDS
V6_CONDITIONS = ("MULTI_SOURCE_ERM", "MULTI_SOURCE_CSU")
CSU_ALPHA = 0.3
CSU_PROBABILITY = 0.5
CSU_EPSILON = 1e-6
CSU_PROJECTED_EIGENVALUE_FLOOR = 1e-12
CSU_INSERTION_POINT = "after_smallcnn_block1"
CSU_OFFICIAL_REPOSITORY = "https://github.com/freshman97/CSU"
CSU_OFFICIAL_COMMIT = "17e948728cad633a218bfd9467f97e80521da1ce"
CSU_PAPER_URL = (
    "https://openaccess.thecvf.com/content/WACV2024/"
    "papers/Zhang_Domain_Generalization_With_Correlated_Style_Uncertainty_WACV_2024_paper.pdf"
)
CSU_SUPPLEMENT_URL = (
    "https://openaccess.thecvf.com/content/WACV2024/supplemental/"
    "Zhang_Domain_Generalization_With_WACV_2024_supplemental.pdf"
)
V5_RUN_NAME = "multisource_loso_smallcnn_9sessions_v5"

REQUIRED_FORMAL_OUTPUTS = (
    "audit/v5_baseline_reuse.csv",
    "audit/smallcnn_identity_check.md",
    "audit/csu_implementation_audit.md",
    "audit/csu_insertion_point.md",
    "audit/target_holdout_leakage.csv",
    "audit/csu_batch_domain_diversity.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "downstream/fold_metrics.csv",
    "downstream/target_predictions.csv",
    "summaries/target_level_csu_comparison.csv",
    "summaries/planned_statistical_tests.csv",
    "summaries/within_cross_gap.csv",
    "summaries/seed_stability.csv",
    "figures/binary_csu_vs_erm_by_target.png",
    "figures/stimulus_type_csu_vs_erm_by_target.png",
    "figures/binary_csu_delta.png",
    "figures/stimulus_type_csu_delta.png",
    "figures/within_cross_gap_csu.png",
    "figures/csu_seed_stability.png",
    "report/csu_domain_generalization_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        nan_count = int(torch.isnan(value).sum().detach().cpu())
        inf_count = int(torch.isinf(value).sum().detach().cpu())
        raise FloatingPointError(
            f"CSU numerical STOP at {name}: nan_count={nan_count}, inf_count={inf_count}"
        )


def resolve_v5_artifact_dir(candidate: Path) -> Path:
    """Resolve direct and downloaded/nested v5 output layouts."""
    root = Path(candidate).expanduser().resolve()
    direct = [root, root / V5_RUN_NAME]
    for value in direct:
        if (value / "downstream/fold_metrics.csv").is_file():
            return value
    matches = sorted({path.parents[1] for path in root.glob("**/downstream/fold_metrics.csv")}) if root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous v5 artifact roots below {root}: {matches}")
    raise FileNotFoundError(f"v5 fold metrics not found below {root}")


def _csv_false(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    return str(value).strip().lower() in {"false", "0", "no"}


def v5_baseline_compatibility(
    row: pd.Series | dict[str, Any], *, task: str, target: str, seed: int
) -> tuple[bool, str]:
    """Check every frozen field needed to rename v5 balanced ERM as v6 ERM."""
    values = dict(row)
    expected_sources = ",".join(source_sessions_for_target(target))
    required = {
        "task", "target_session", "source_sessions", "n_source_sessions", "seed",
        "condition", "source_balance_mode", "normalization_weighting", "best_epoch",
        "early_stopping", "fold", "run_status", "target_frames_used_for_training",
        "target_labels_used_for_training", "target_used_for_normalization",
        "target_used_for_validation", "target_used_for_model_selection",
    }
    missing = required - set(values)
    if missing:
        return False, f"missing frozen v5 fields: {sorted(missing)}"
    checks = [
        (str(values["task"]) == str(task), "task mismatch"),
        (str(values["target_session"]) == str(target), "target mismatch"),
        (int(values["seed"]) == int(seed), "seed mismatch"),
        (str(values["condition"]) == "MULTI_SOURCE_BALANCED", "condition mismatch"),
        (str(values["source_sessions"]) == expected_sources, "eight-source LOSO fold mismatch"),
        (int(values["n_source_sessions"]) == 8, "source-session count mismatch"),
        (str(values["source_balance_mode"]) == "session_balanced", "sampler mismatch"),
        (
            str(values["normalization_weighting"])
            == "sample_frequency_weighted_source_only",
            "source-only preprocessing mismatch",
        ),
        (int(values["best_epoch"]) == FROZEN_SUPERVISED_CONFIG.max_epochs, "epoch budget mismatch"),
        (_csv_false(values["early_stopping"]), "early-stopping mismatch"),
        (str(values["fold"]) == "LOSO_target_session", "fold definition mismatch"),
        (str(values["run_status"]) == "VALID", "invalid v5 run"),
        (int(values["target_frames_used_for_training"]) == 0, "target frame leakage"),
        (_csv_false(values["target_labels_used_for_training"]), "target label leakage"),
        (_csv_false(values["target_used_for_normalization"]), "target normalization leakage"),
        (_csv_false(values["target_used_for_validation"]), "target validation leakage"),
        (_csv_false(values["target_used_for_model_selection"]), "target model-selection leakage"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    return (not failures), ("exact frozen v5 balanced-ERM artifact" if not failures else "; ".join(failures))


class CorrelatedStyleUncertainty(nn.Module):
    """Author-aligned CSU feature-statistic augmentation.

    The implementation follows the WACV 2024 supplementary pseudocode and
    the authors' ``CorrelatedDistributionUncertainty`` implementation.  The
    eigendecomposition is used only to obtain directions (no gradient through
    eigenvectors); gradients still flow through the projected covariance
    intensity, as in the official code.
    """

    def __init__(
        self,
        *,
        p: float = CSU_PROBABILITY,
        alpha: float = CSU_ALPHA,
        eps: float = CSU_EPSILON,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError("CSU p must be in [0, 1]")
        if float(alpha) <= 0.0 or float(eps) <= 0.0:
            raise ValueError("CSU alpha and eps must be positive")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)
        self.last_applied = False
        self.application_count = 0

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}, eps={self.eps}"

    def _correlated_square_root(self, covariance: torch.Tensor) -> torch.Tensor:
        _require_finite(covariance, "covariance")
        channels = int(covariance.shape[0])
        if covariance.shape != (channels, channels):
            raise ValueError("CSU covariance must be a square channel matrix")
        identity = torch.eye(channels, dtype=covariance.dtype, device=covariance.device)
        stabilized = channels * covariance + self.eps * identity
        _require_finite(stabilized, "stabilized_covariance")
        # Official numerical scheme: do not backpropagate through eigenvectors.
        with torch.no_grad():
            try:
                _eigenvalues, eigenvectors = torch.linalg.eigh(stabilized)
            except RuntimeError as error:
                raise FloatingPointError("CSU eigendecomposition failed; formal run STOP") from error
            _require_finite(eigenvectors, "eigenvectors")
        projected = torch.diagonal(eigenvectors.T @ covariance @ eigenvectors)
        safe_scale = torch.sqrt(torch.clamp(projected, min=CSU_PROJECTED_EIGENVALUE_FLOOR))
        root = eigenvectors @ torch.diag(safe_scale) @ eigenvectors.T
        _require_finite(root, "correlated_covariance_square_root")
        return root

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CSU expects [B, C, H, W], got {tuple(x.shape)}")
        _require_finite(x, "input")
        self.last_applied = False
        # Match the authors' train/eval gate and single Bernoulli draw per call.
        if (not self.training) or np.random.random() > self.p:
            return x

        batch, channels = int(x.shape[0]), int(x.shape[1])
        if batch < 2:
            raise ValueError("CSU requires at least two feature maps when active")
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        _require_finite(mu, "channel_mean")
        _require_finite(sigma, "channel_std")
        normalized = (x - mu) / sigma

        factor = self.beta.sample((batch, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_flat = mu.reshape(batch, channels)
        sigma_flat = sigma.reshape(batch, channels)
        centered_mu = mu_flat - mu_flat.mean(dim=0, keepdim=True)
        centered_sigma = sigma_flat - sigma_flat.mean(dim=0, keepdim=True)
        covariance_mu = centered_mu.T @ centered_mu / batch
        covariance_sigma = centered_sigma.T @ centered_sigma / batch

        root_mu = self._correlated_square_root(covariance_mu)
        root_sigma = self._correlated_square_root(covariance_sigma)
        noise_mu = (torch.randn(batch, 1, channels, device=x.device, dtype=x.dtype) @ root_mu).reshape(
            batch, channels, 1, 1
        )
        noise_sigma = (
            torch.randn(batch, 1, channels, device=x.device, dtype=x.dtype) @ root_sigma
        ).reshape(batch, channels, 1, 1)
        output = normalized * (sigma + factor * noise_sigma) + (mu + factor * noise_mu)
        _require_finite(output, "output")
        self.last_applied = True
        self.application_count += 1
        return output


class CSUSmallCNNFrameEncoder(nn.Module):
    """The frozen SmallCNN frame encoder with one CSU after block 1."""

    feature_dim = SmallCNNFrameEncoder.feature_dim
    insertion_feature_shape = (8, 64, 125)

    def __init__(self) -> None:
        super().__init__()
        frozen = SmallCNNFrameEncoder()
        layers = list(frozen.layers.children())
        if len(layers) != 9:
            raise AssertionError("unexpected frozen SmallCNN layer count")
        self.block1 = nn.Sequential(*layers[:4])
        self.csu = CorrelatedStyleUncertainty()
        self.remaining = nn.Sequential(*layers[4:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.block1(x)
        expected = self.insertion_feature_shape
        if tuple(features.shape[1:]) != expected:
            raise AssertionError(
                f"CSU insertion shape changed: expected [B,{expected}], got {tuple(features.shape)}"
            )
        return self.remaining(self.csu(features))


class CSUCNN2DMeanPool(nn.Module):
    """Frozen four-frame shared SmallCNN mean-pool classifier with CSU."""

    def __init__(self, n_classes: int, dropout: float = 0.25, temporal_length: int = 4) -> None:
        super().__init__()
        if int(temporal_length) != 4:
            raise ValueError("v6 temporal length is frozen at four")
        self.temporal_length = 4
        self.encoder = CSUSmallCNNFrameEncoder()
        self.encoder_feature_dim = self.encoder.feature_dim
        self.n_classes = int(n_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)), nn.Linear(self.encoder_feature_dim, self.n_classes)
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or int(x.shape[1]) != 4 or int(x.shape[2]) != 1:
            raise ValueError(f"expected [B, 4, 1, H, W], got {tuple(x.shape)}")
        batch, time, channels, height, width = x.shape
        encoded = self.encoder(x.reshape(batch * time, channels, height, width))
        return encoded.reshape(batch, time, self.encoder_feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_sequence(x).mean(dim=1))


@dataclass
class CSUTrainingResult:
    model: CSUCNN2DMeanPool
    train_predictions: np.ndarray
    train_probabilities: np.ndarray
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    history: list[dict[str, Any]]
    batch_domain_diversity: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    metrics: dict[str, Any]
    device: str


def train_prepared_csu(
    prepared: PreparedCrossSessionData,
    *,
    seed: int,
    config: DeepTrainingConfig = FROZEN_SUPERVISED_CONFIG,
    device: str | None = "auto",
) -> CSUTrainingResult:
    if prepared.balance_mode != "session_balanced":
        raise ValueError("MULTI_SOURCE_CSU requires the v5 session-balanced sampler")
    if len(prepared.source_sessions) < 2:
        raise ValueError("MULTI_SOURCE_CSU requires multiple source sessions")
    if prepared.target_session in prepared.source_sessions:
        raise AssertionError("target session entered CSU training")
    if config != FROZEN_SUPERVISED_CONFIG and (
        config.max_epochs > FROZEN_SUPERVISED_CONFIG.max_epochs
        or any(
            getattr(config, field) != getattr(FROZEN_SUPERVISED_CONFIG, field)
            for field in ("optimizer", "lr", "weight_decay", "batch_size", "dropout", "loss")
        )
    ):
        raise ValueError("CSU training may shorten smoke epochs but may not change frozen supervision")

    set_reproducible_seed(seed)
    classes = np.asarray(sorted(TASK_CLASS_NAMES[prepared.task]), dtype=np.int64)
    y_train_i = labels_to_class_indices(prepared.y_train, classes)
    train_tensor = blocks_to_sequence_tensor(prepared.X_train)
    test_tensor = blocks_to_sequence_tensor(prepared.X_test)
    torch_device = resolve_device(device)
    model = CSUCNN2DMeanPool(
        n_classes=len(classes), dropout=config.dropout, temporal_length=4
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    diversity: list[dict[str, Any]] = []
    batch_size = max(1, min(int(config.batch_size), len(train_tensor)))
    for epoch in range(1, int(config.max_epochs) + 1):
        indices = epoch_draw_indices(
            prepared.train_session_labels,
            seed=seed,
            epoch=epoch,
            balance_mode="session_balanced",
        )
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch_index, start in enumerate(range(0, len(indices), batch_size)):
            batch_indices = indices[start : start + batch_size]
            batch_sessions = prepared.train_session_labels[batch_indices].astype(str)
            if prepared.target_session in set(batch_sessions):
                raise AssertionError("target appeared in a CSU minibatch")
            diversity.append({
                "target": prepared.target_session,
                "task": prepared.task,
                "seed": int(seed),
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "batch_size": int(len(batch_indices)),
                "n_unique_source_sessions": int(len(np.unique(batch_sessions))),
            })
            xb = train_tensor[batch_indices].to(torch_device)
            yb = torch.from_numpy(y_train_i[batch_indices]).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            _require_finite(logits, "training_logits")
            loss = criterion(logits, yb)
            _require_finite(loss, "training_loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    _require_finite(parameter.grad, f"gradient:{name}")
            optimizer.step()
            for name, parameter in model.named_parameters():
                _require_finite(parameter, f"parameter:{name}")
            n_items = int(len(batch_indices))
            total_loss += float(loss.detach().cpu()) * n_items
            total_correct += int((logits.argmax(1) == yb).sum().detach().cpu())
            total_seen += n_items
        history.append({
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(total_seen, 1)),
            "train_accuracy_minibatch": float(total_correct / max(total_seen, 1)),
            "n_draws": int(total_seen),
            "n_unique_source_blocks": int(len(np.unique(indices))),
            "source_balance_mode": "session_balanced",
            "csu_application_count_cumulative": int(model.encoder.csu.application_count),
        })

    train_probs = predict_probabilities(
        model, train_tensor, device=torch_device, batch_size=config.batch_size
    )
    test_probs = predict_probabilities(
        model, test_tensor, device=torch_device, batch_size=config.batch_size
    )
    if model.encoder.csu.last_applied:
        raise AssertionError("CSU remained active during evaluation")
    if not np.isfinite(train_probs).all() or not np.isfinite(test_probs).all():
        raise FloatingPointError("non-finite evaluation probabilities")
    train_pred = classes[train_probs.argmax(axis=1)]
    test_pred = classes[test_probs.argmax(axis=1)]
    train_metrics = classification_metrics(prepared.y_train, train_pred)
    test_metrics = classification_metrics(prepared.y_test, test_pred)
    metrics = {
        "task": prepared.task,
        "target_session": prepared.target_session,
        "source_sessions": ",".join(prepared.source_sessions),
        "n_source_sessions": int(len(prepared.source_sessions)),
        "seed": int(seed),
        "condition": "MULTI_SOURCE_CSU",
        "source_balance_mode": "session_balanced",
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_F1": float(test_metrics["macro_f1"]),
        "ROC_AUC": binary_roc_auc(prepared.y_test, test_probs[:, 1]),
        "best_epoch": int(config.max_epochs),
        "train_test_gap_BA": float(
            train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"]
        ),
        "n_train_blocks": int(len(prepared.y_train)),
        "n_test_blocks": int(len(prepared.y_test)),
        "n_train_frames": int(4 * len(prepared.y_train)),
        "n_test_frames": int(4 * len(prepared.y_test)),
        "source_cycle_counts": ";".join(
            f"{session}:{prepared.source_cycle_counts[session]}"
            for session in prepared.source_sessions
        ),
        "test_cycles": cycle_text(prepared.test_cycles),
        "target_labels_used_for_training": False,
        "target_frames_used_for_training": 0,
        "target_used_for_normalization": False,
        "target_used_for_validation": False,
        "target_used_for_model_selection": False,
        "target_unlabeled_adaptation": False,
        "early_stopping": False,
        "csu_alpha": CSU_ALPHA,
        "csu_probability": CSU_PROBABILITY,
        "csu_epsilon": CSU_EPSILON,
        "csu_insertion_point": CSU_INSERTION_POINT,
        "csu_application_count": int(model.encoder.csu.application_count),
        "run_status": "VALID",
    }
    return CSUTrainingResult(
        model=model.cpu(),
        train_predictions=train_pred,
        train_probabilities=train_probs,
        test_predictions=test_pred,
        test_probabilities=test_probs,
        history=history,
        batch_domain_diversity=diversity,
        normalization_audit=dict(prepared.normalization_audit),
        metrics=metrics,
        device=str(torch_device),
    )


def assert_formal_cuda(device: str) -> torch.device:
    if str(device) != "cuda":
        raise RuntimeError("formal v6 requires --device cuda; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("formal v6 requires an available CUDA device; STOP")
    return torch.device("cuda")


def missing_formal_outputs(output_dir: Path) -> list[str]:
    missing = []
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    curve_dir = output_dir / "downstream/training_curves"
    if not curve_dir.is_dir() or not any(curve_dir.glob("*.csv")):
        missing.append("downstream/training_curves/*.csv")
    return missing
