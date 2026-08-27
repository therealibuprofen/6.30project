"""Provenance-pinned outer-result reuse for Adaptive Mean/Std Nested-CV v1.

This module is intentionally separate from inner training and selection.  The
formal runner imports it only after all selection artifacts have been locked.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .adaptive_mean_std_nestedcv import file_sha256, read_locked_selection
from .fcnn_temporal_statistics import (
    INPUT_VARIANTS,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
)


EXPECTED_FIXED_RUN_FINGERPRINT = (
    "34387397a0e9712caddb70f06f9199b272fd89fa94bedcf2782b30da771075ca"
)
EXPECTED_FIXED_CONFIG_FINGERPRINT = (
    "f5b7f4175baebdefec5250586db2811cee33f54dfd2e0fc06b05c12e43a67a74"
)
EXPECTED_FIXED_GIT_COMMIT = "96d26bfac562f934e74e49e059d9ea74e78617af"
EXPECTED_FIXED_FILE_SHA256 = {
    "RUN_COMPLETE.json": "24f6fed65066fafb924679f80349b3c24464eb3bedd371787daec6675089bf3d",
    "config.json": "f7fd9d608b5d59d2ea068abae32b053af5c416b5649de474f955104299eb21f3",
    "task_plan.csv": "192ad9d44da3138e2b9f9c4005db7759361e1a4963559ea99ad7a34e7e80a106",
    "git_state.json": "e2500d3b3152112d94f4b7346302656591f0b04a82bad0164e09e36ecf9b83bf",
}
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "b947042232e46a0c93da17737e03ba8f45019a6010cc6be360db20ee4c88a645"
)


def _load_formal_runner():
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "scripts/baselines/run_fcnn_mean_std_temporal_statistics.py"
    spec = importlib.util.spec_from_file_location(
        "_approved_fcnn_mean_std_runner", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import approved formal runner {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_fixed_run(
    fixed_results_dir: Path, *, validate_all_tasks: bool
) -> dict[str, Any]:
    """Validate the exact completed candidate run pinned by this protocol."""

    for name, expected_hash in EXPECTED_FIXED_FILE_SHA256.items():
        path = fixed_results_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned fixed artifact is missing: {path}")
        if file_sha256(path) != expected_hash:
            raise AssertionError(f"pinned fixed artifact hash mismatch: {name}")
    completion = json.loads(
        (fixed_results_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (fixed_results_dir / "config.json").read_text(encoding="utf-8")
    )
    git_state = json.loads(
        (fixed_results_dir / "git_state.json").read_text(encoding="utf-8")
    )
    plan = pd.read_csv(
        fixed_results_dir / "task_plan.csv", dtype={"session": str}
    )
    if (
        completion.get("status") != "complete"
        or int(completion.get("completed_tasks", -1)) != 492
        or int(completion.get("expected_tasks", -1)) != 492
        or int(completion.get("number_of_folds", -1)) != 82
        or int(completion.get("number_of_seeds", -1)) != 3
        or int(completion.get("number_of_variants", -1)) != 2
    ):
        raise AssertionError("fixed run is not formally complete 492/492")
    if completion.get("run_fingerprint") != EXPECTED_FIXED_RUN_FINGERPRINT:
        raise AssertionError("fixed run fingerprint mismatch")
    if completion.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        raise AssertionError("fixed model implementation version mismatch")
    if config.get("git_commit") != EXPECTED_FIXED_GIT_COMMIT or git_state.get(
        "commit"
    ) != EXPECTED_FIXED_GIT_COMMIT:
        raise AssertionError("fixed git provenance mismatch")
    source_hash = config.get("project_source_sha256", {}).get(
        "src/ultrasound_decoding/multiframe/fcnn_temporal_statistics.py"
    )
    if source_hash != EXPECTED_CANDIDATE_SOURCE_SHA256:
        raise AssertionError("fixed candidate source hash mismatch")
    if len(plan) != 492 or set(plan["variant"].astype(str)) != set(INPUT_VARIANTS):
        raise AssertionError("fixed task plan coverage mismatch")
    if set(plan["config_fingerprint"].astype(str)) != {
        EXPECTED_FIXED_CONFIG_FINGERPRINT
    }:
        raise AssertionError("fixed task config fingerprint mismatch")
    if validate_all_tasks:
        formal_runner = _load_formal_runner()
        for expected in plan.to_dict(orient="records"):
            path = formal_runner.task_dir(
                fixed_results_dir,
                str(expected["session"]),
                str(expected["variant"]),
                int(expected["seed"]),
                int(expected["fold"]),
            )
            formal_runner.validate_completed_task(
                path,
                expected,
                EXPECTED_FIXED_RUN_FINGERPRINT,
                raise_on_error=True,
            )
    return {
        "status": "validated",
        "run_fingerprint": EXPECTED_FIXED_RUN_FINGERPRINT,
        "config_fingerprint": EXPECTED_FIXED_CONFIG_FINGERPRINT,
        "git_commit": EXPECTED_FIXED_GIT_COMMIT,
        "candidate_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "model": MODEL_NAME,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "completed_tasks": 492,
        "task_plan": plan,
    }


class SelectedOuterResultReader:
    """Capability object restricted to one already-locked selected variant."""

    def __init__(
        self,
        fixed_results_dir: Path,
        fixed_plan: pd.DataFrame,
        selection_path: Path,
        *,
        expected_protocol_fingerprint: str,
    ) -> None:
        self.fixed_results_dir = fixed_results_dir
        self.fixed_plan = fixed_plan
        self.selection = read_locked_selection(
            selection_path,
            expected_protocol_fingerprint=expected_protocol_fingerprint,
        )
        self.selected_variant = str(self.selection["selected_variant"])
        if self.selected_variant not in INPUT_VARIANTS:
            raise AssertionError("locked selection contains an unknown variant")

    def read(self, requested_variant: str | None = None) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
        variant = self.selected_variant if requested_variant is None else requested_variant
        if variant != self.selected_variant:
            raise PermissionError("outer reader refuses the unselected candidate")
        session = str(self.selection["session"])
        fold = int(self.selection["outer_fold"])
        seed = int(self.selection["seed"])
        subset = self.fixed_plan[
            self.fixed_plan["session"].astype(str).eq(session)
            & self.fixed_plan["variant"].astype(str).eq(variant)
            & pd.to_numeric(self.fixed_plan["seed"]).eq(seed)
            & pd.to_numeric(self.fixed_plan["fold"]).eq(fold)
        ]
        if len(subset) != 1:
            raise AssertionError("selected fixed task is missing or duplicated")
        expected = subset.iloc[0].to_dict()
        expected_train = tuple(
            sorted(int(value) for value in str(expected["train_cycles"]).split(","))
        )
        expected_test = tuple(
            sorted(int(value) for value in str(expected["test_cycles"]).split(","))
        )
        if expected_train != tuple(self.selection["outer_train_cycle_ids"]):
            raise AssertionError("selected outer train membership mismatch")
        if expected_test != tuple(self.selection["outer_test_cycle_ids"]):
            raise AssertionError("selected outer test membership mismatch")
        formal_runner = _load_formal_runner()
        path = formal_runner.task_dir(
            self.fixed_results_dir, session, variant, seed, fold
        )
        formal_runner.validate_completed_task(
            path,
            expected,
            EXPECTED_FIXED_RUN_FINGERPRINT,
            raise_on_error=True,
        )
        # No unselected result or prediction path is constructed or opened.
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        if result.get("model") != MODEL_NAME or result.get("variant") != variant:
            raise AssertionError("selected outer model/variant provenance mismatch")
        if int(result.get("final_epoch", -1)) != 40:
            raise AssertionError("selected outer result is not fixed epoch 40")
        return result, predictions, {
            "fixed_task_dir": str(path),
            "fixed_task_fingerprint": str(result["task_fingerprint"]),
            "fixed_run_fingerprint": EXPECTED_FIXED_RUN_FINGERPRINT,
            "fixed_config_fingerprint": EXPECTED_FIXED_CONFIG_FINGERPRINT,
            "selected_variant_only": True,
        }
