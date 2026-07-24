from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_PATH = PROJECT_DIR / "scripts" / "generalization" / "run_cross_session_generalization.py"
spec = importlib.util.spec_from_file_location("run_cross_session_generalization", SCRIPT_PATH)
cross_session = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = cross_session
spec.loader.exec_module(cross_session)

from ultrasound_decoding.deep import _grouped_validation_indices
from ultrasound_decoding.deep import _normalize_frames


def test_grouped_validation_indices_accept_composite_session_cycle_groups() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.asarray(
        [
            "708_cycle0",
            "708_cycle0",
            "709_cycle0",
            "709_cycle0",
            "710_cycle0",
            "710_cycle0",
            "807_cycle0",
            "807_cycle0",
        ],
        dtype=object,
    )

    train_idx, val_idx, val_groups = _grouped_validation_indices(y, groups, seed=0)

    assert train_idx is not None
    assert val_idx is not None
    assert val_groups
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert all(isinstance(group, str) for group in val_groups)


def test_split_audit_rejects_target_session_in_inner_validation() -> None:
    source_groups = np.asarray(["708_cycle0", "708_cycle1", "709_cycle0", "709_cycle1"], dtype=object)
    metadata = {"inner_validation": {"val_cycles": ["710_cycle0"]}}

    try:
        cross_session.split_audit_rows(["708", "709"], "710", "loso", source_groups, metadata)
    except AssertionError as exc:
        assert "target session leaked" in str(exc)
    else:
        raise AssertionError("expected target leakage assertion")


def test_session_equal_normalization_weights_source_sessions_equally() -> None:
    X_a = np.zeros((2, 2, 2), dtype=np.float32)
    X_b = np.full((6, 2, 2), np.sinh(2.0), dtype=np.float32)
    X_train = np.concatenate([X_a, X_b], axis=0)
    labels = np.asarray(["708"] * len(X_a) + ["709"] * len(X_b), dtype=object)

    X_norm, _, stats = _normalize_frames(
        X_train,
        X_train[:1],
        statistics_scope="unit_test",
        normalization_weighting="session_equal",
        train_session_labels=labels,
    )

    assert stats["normalization_weighting"] == "session_equal"
    assert stats["target_used_for_stats"] is False
    assert abs(stats["mean_mean"] - 1.0) < 1e-6
    assert np.isfinite(X_norm).all()


def test_balanced_cycle_selection_uses_whole_cycles() -> None:
    data_by_session = {}
    for session, n_cycles in {"708": 2, "709": 4}.items():
        groups = np.repeat(np.arange(n_cycles), 16)
        y = np.asarray(["no_stimulus", "stimulus"] * (len(groups) // 2), dtype=object)
        meta = []
        for index, cycle in enumerate(groups):
            meta.append(
                {
                    "index": index,
                    "cycle": int(cycle),
                    "block_name": "dot",
                    "session": session,
                }
            )
        data_by_session[session] = cross_session.SessionData(
            session=session,
            X=np.zeros((len(groups), 128, 501), dtype=np.float32),
            y=y,
            groups=groups,
            meta=cross_session.pd.DataFrame(meta),
            classes=["no_stimulus", "stimulus"],
        )

    selected, selection_df = cross_session.select_balanced_cycles(["708", "709"], data_by_session, seed=0)
    X, y, groups, meta = cross_session.combine_sources(["708", "709"], data_by_session, selected)

    assert selection_df["n_selected_cycles"].tolist() == [2, 2]
    assert selection_df["n_selected_samples"].tolist() == [32, 32]
    assert len(y) == 64
    assert all(meta.groupby(["session", "cycle"]).size() == 16)
    assert len(set(groups)) == 4
