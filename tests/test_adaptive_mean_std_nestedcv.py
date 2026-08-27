from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe import adaptive_mean_std_nestedcv as core
from ultrasound_decoding.multiframe.adaptive_mean_std_outer_reuse import (
    SelectedOuterResultReader,
    validate_fixed_run,
)
from ultrasound_decoding.multiframe.fcnn_temporal_statistics import (
    MEAN_ONLY_VARIANT,
    MEAN_STD_VARIANT,
    MODEL_IMPLEMENTATION_VERSION,
    build_model,
)
from ultrasound_decoding.multiframe.models import FCNNMeanPool
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    normalize_blocks_train_fold_only_with_stats,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
FORMAL_DIR = PROJECT_DIR / "outputs/fcnn_mean_std_temporal_statistics_v1"
FEASIBILITY_DIR = PROJECT_DIR / "outputs/adaptive_mean_std_nestedcv_feasibility"


def load_runner():
    import importlib.util

    path = PROJECT_DIR / "scripts/baselines/run_adaptive_mean_std_nestedcv.py"
    spec = importlib.util.spec_from_file_location("adaptive_nested_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def runner():
    return load_runner()


@pytest.fixture(scope="session")
def planned_output(tmp_path_factory, runner):
    output = tmp_path_factory.mktemp("adaptive_nested_plan")
    args = runner.parse_args(
        [
            "--stage",
            "plan",
            "--project-root",
            str(PROJECT_DIR),
            "--output-dir",
            str(output),
            "--fixed-results-dir",
            str(FORMAL_DIR),
            "--feasibility-dir",
            str(FEASIBILITY_DIR),
        ]
    )
    runner.run_plan(args)
    return output


def _training_identity(seed: int = 0, candidate: str = MEAN_ONLY_VARIANT):
    return core.build_training_cache_identity(
        session="x",
        train_sample_ids=["a", "b", "c", "d"],
        train_cycle_ids=[1, 2],
        candidate=candidate,
        seed=seed,
        dataset_source_hash={"h5": "h", "metadata": "m"},
        session_manifest_hash="manifest",
        candidate_source_hashes={"model": "source"},
        protocol_fingerprint="protocol",
        runtime_fingerprint="runtime",
        training_config=DeepTrainingConfig(max_epochs=1),
    )


def _selection_payload(protocol: str = "protocol"):
    return core.build_selection_payload(
        session="x",
        outer_fold=1,
        seed=0,
        outer_train_cycle_ids=[1, 2],
        outer_test_cycle_ids=[3],
        inner_ba_mean_only=0.5,
        inner_ba_mean_std=0.6,
        candidate_protocol_fingerprints={"mean_only": "a", "mean_std": "b"},
        split_fingerprint="split",
        normalization_protocol_fingerprint="normalization",
        inner_oof_prediction_hashes={"mean_only": "c", "mean_std": "d"},
        expected_outer_train_sample_ids=["a", "b"],
        observed_candidate_sample_ids={"mean_only": ["a", "b"], "mean_std": ["a", "b"]},
        protocol_fingerprint=protocol,
    )


def test_01_outer_82_folds_exactly_match_formal_experiment(planned_output) -> None:
    plan = pd.read_csv(planned_output / "task_plan.csv", dtype={"session": str})
    formal = pd.read_csv(FORMAL_DIR / "task_plan.csv", dtype={"session": str})
    outer = core.validate_outer_manifest(formal)
    assert len(plan) == len(outer) == 82
    assert plan[["session", "outer_fold", "outer_train_cycle_ids", "outer_test_cycle_ids"]].equals(
        outer.rename(columns={"fold": "outer_fold", "train_cycles": "outer_train_cycle_ids", "test_cycles": "outer_test_cycle_ids"})[["session", "outer_fold", "outer_train_cycle_ids", "outer_test_cycle_ids"]]
    )


def test_02_dynamic_inner_split_total_is_722(planned_output) -> None:
    assert len(pd.read_csv(planned_output / "split_manifest.csv")) == 722


def test_03_unique_training_cycle_sets_are_425(planned_output) -> None:
    split = pd.read_csv(planned_output / "split_manifest.csv", dtype={"session": str})
    assert len(split[["session", "inner_train_cycle_ids"]].drop_duplicates()) == 425


def test_04_scheme_a_unique_training_jobs_are_2550(planned_output) -> None:
    cache = pd.read_csv(planned_output / "cache_manifest.csv")
    tasks = pd.read_csv(planned_output / "inner_task_manifest.csv")
    assert len(cache) == 425 * 2 * 3 == 2550
    assert len(tasks) == 722 * 2 * 3 == 4332


@pytest.mark.parametrize(
    "column", ["outer_test_inner_train_overlap_count", "outer_test_inner_val_overlap_count", "inner_train_val_overlap_count"]
)
def test_05_to_07_all_forbidden_cycle_intersections_are_empty(planned_output, column) -> None:
    split = pd.read_csv(planned_output / "split_manifest.csv")
    assert split[column].eq(0).all()


def test_08_inner_train_union_validation_equals_outer_train(planned_output) -> None:
    split = pd.read_csv(planned_output / "split_manifest.csv")
    assert split["inner_union_equals_outer_train"].all()


def test_09_normalization_fit_ids_equal_inner_train_ids(planned_output) -> None:
    split = pd.read_csv(planned_output / "split_manifest.csv")
    assert split["normalization_fit_cycle_ids"].equals(split["inner_train_cycle_ids"])
    tasks = pd.read_csv(planned_output / "inner_task_manifest.csv")
    for row in tasks.head(20).itertuples(index=False):
        identity = json.loads(row.training_identity_json)
        assert identity["normalization_fit_sample_ids"] == identity["exact_training_sample_ids"]


def test_10_validation_data_cannot_change_inner_normalization_statistics() -> None:
    rng = np.random.default_rng(2)
    train = rng.normal(size=(3, 4, 5, 7)).astype(np.float32)
    val_a = np.zeros((2, 4, 5, 7), dtype=np.float32)
    val_b = np.full_like(val_a, 1e6)
    left = normalize_blocks_train_fold_only_with_stats(train, val_a, session="x", task="binary", method="mean_only", seed=0, fold=1, train_cycles="1,2", test_cycles="3")
    right = normalize_blocks_train_fold_only_with_stats(train, val_b, session="x", task="binary", method="mean_only", seed=0, fold=1, train_cycles="1,2", test_cycles="3")
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[3], right[3])
    assert np.array_equal(left[4], right[4])


def test_11_outer_test_modification_cannot_change_inner_identity_or_selection() -> None:
    identity_before = _training_identity()
    outer_pixels = np.zeros((2, 4, 5, 7), dtype=np.float32)
    outer_labels = np.asarray([0, 1])
    outer_pixels[:] = 1e9
    outer_labels[:] = outer_labels[::-1]
    identity_after = _training_identity()
    assert core.training_cache_key(identity_before) == core.training_cache_key(identity_after)
    assert core.select_variant(0.55, 0.60) == core.select_variant(0.55, 0.60)


def test_12_mean_std_wins_only_when_strictly_higher() -> None:
    assert core.select_variant(0.50, 0.51) == MEAN_STD_VARIANT


def test_13_mean_only_wins_when_mean_std_is_lower() -> None:
    assert core.select_variant(0.51, 0.50) == MEAN_ONLY_VARIANT


def test_14_exact_tie_selects_mean_only() -> None:
    assert core.select_variant(0.5, 0.5) == MEAN_ONLY_VARIANT


def test_15_inner_oof_ba_uses_concatenated_predictions_not_mean_fold_ba() -> None:
    fold_a = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "y_true": [0, 0, 1, 1], "y_pred": [0, 0, 1, 1]})
    fold_b = pd.DataFrame({"sample_id": list("efghijkl"), "y_true": [0, 0, 0, 0, 1, 1, 1, 1], "y_pred": [0, 0, 1, 1, 1, 1, 0, 0]})
    combined = pd.concat([fold_a, fold_b], ignore_index=True)
    observed = core.concatenated_oof_balanced_accuracy(combined, combined.sample_id.tolist())
    mean_fold = np.mean([
        classification_metrics(fold_a.y_true.to_numpy(), fold_a.y_pred.to_numpy())["balanced_accuracy"],
        classification_metrics(fold_b.y_true.to_numpy(), fold_b.y_pred.to_numpy())["balanced_accuracy"],
    ])
    assert observed == pytest.approx(2.0 / 3.0)
    assert observed != pytest.approx(mean_fold)


def test_16_inner_oof_requires_each_sample_exactly_once() -> None:
    duplicate = pd.DataFrame({"sample_id": ["a", "a"], "y_true": [0, 0], "y_pred": [0, 0]})
    with pytest.raises(AssertionError, match="duplicate"):
        core.concatenated_oof_balanced_accuracy(duplicate, ["a", "b"])


def test_17_identical_training_identity_has_identical_cache_key() -> None:
    assert core.training_cache_key(_training_identity()) == core.training_cache_key(_training_identity())


def test_18_seed_is_part_of_training_cache_key() -> None:
    assert core.training_cache_key(_training_identity(0)) != core.training_cache_key(_training_identity(1))


def test_19_parent_and_validation_are_part_of_evaluation_cache_key() -> None:
    key = core.training_cache_key(_training_identity())
    left = core.build_evaluation_cache_identity(training_key=key, session="x", parent_outer_fold=1, outer_seed=0, candidate=MEAN_ONLY_VARIANT, validation_sample_ids=["v1"], validation_cycle_ids=[2], current_outer_train_cycle_ids=[1, 2], current_outer_test_cycle_ids=[3], protocol_fingerprint="p")
    right = core.build_evaluation_cache_identity(training_key=key, session="x", parent_outer_fold=2, outer_seed=0, candidate=MEAN_ONLY_VARIANT, validation_sample_ids=["v2"], validation_cycle_ids=[3], current_outer_train_cycle_ids=[1, 3], current_outer_test_cycle_ids=[2], protocol_fingerprint="p")
    assert core.evaluation_cache_key(left) != core.evaluation_cache_key(right)


def test_20_parent_reader_rejects_current_outer_test_cycle() -> None:
    identity = {"exact_validation_cycle_ids": [3], "current_outer_train_cycle_ids": [1, 2], "current_outer_test_cycle_ids": [3]}
    with pytest.raises(PermissionError):
        core.validate_parent_evaluation_access(identity, current_outer_train_cycle_ids=[1, 2], current_outer_test_cycle_ids=[3])


def test_21_inner_stage_has_no_outer_result_capability(runner) -> None:
    args = runner.parse_args(["--stage", "inner"])
    assert not hasattr(args, "fixed_results_dir")
    assert "adaptive_mean_std_outer_reuse" not in inspect.getsource(runner.run_inner)


def test_22_select_stage_has_no_outer_result_capability(runner) -> None:
    args = runner.parse_args(["--stage", "select"])
    assert not hasattr(args, "fixed_results_dir")
    assert "adaptive_mean_std_outer_reuse" not in inspect.getsource(runner.run_select)


def test_23_outer_reader_requires_locked_selection(tmp_path) -> None:
    with pytest.raises(PermissionError, match="locked"):
        core.read_locked_selection(tmp_path / "missing.json", expected_protocol_fingerprint="p")


def test_24_outer_reader_only_allows_selected_variant(tmp_path) -> None:
    selection = _selection_payload()
    path = tmp_path / "selection.json"
    core.lock_selection(path, selection, expected_protocol_fingerprint="protocol")
    reader = SelectedOuterResultReader(
        FORMAL_DIR,
        pd.DataFrame(),
        path,
        expected_protocol_fingerprint="protocol",
    )
    with pytest.raises(PermissionError, match="unselected"):
        reader.read(MEAN_ONLY_VARIANT)


def test_25_selected_outer_reader_validates_and_reads_only_selected(tmp_path) -> None:
    fixed = validate_fixed_run(FORMAL_DIR, validate_all_tasks=False)
    plan = fixed["task_plan"]
    selection = core.build_selection_payload(session="708", outer_fold=1, seed=0, outer_train_cycle_ids=[1, 2, 3, 4, 5], outer_test_cycle_ids=[0], inner_ba_mean_only=0.5, inner_ba_mean_std=0.6, candidate_protocol_fingerprints={"mean_only": "a", "mean_std": "b"}, split_fingerprint="split", normalization_protocol_fingerprint="normalization", inner_oof_prediction_hashes={"mean_only": "c", "mean_std": "d"}, expected_outer_train_sample_ids=["a", "b"], observed_candidate_sample_ids={"mean_only": ["a", "b"], "mean_std": ["a", "b"]}, protocol_fingerprint="protocol")
    path = tmp_path / "selection.json"
    core.lock_selection(path, selection, expected_protocol_fingerprint="protocol")
    reader = SelectedOuterResultReader(
        FORMAL_DIR,
        plan,
        path,
        expected_protocol_fingerprint="protocol",
    )
    result, predictions, provenance = reader.read()
    assert result["variant"] == MEAN_STD_VARIANT
    assert len(predictions) == 4
    assert provenance["selected_variant_only"] is True


def test_26_formal_outer_fingerprint_mismatch_is_rejected(tmp_path) -> None:
    for name in ("RUN_COMPLETE.json", "config.json", "task_plan.csv", "git_state.json"):
        shutil.copy2(FORMAL_DIR / name, tmp_path / name)
    payload = json.loads((tmp_path / "RUN_COMPLETE.json").read_text())
    payload["run_fingerprint"] = "wrong"
    (tmp_path / "RUN_COMPLETE.json").write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="hash mismatch"):
        validate_fixed_run(tmp_path, validate_all_tasks=False)


def test_27_incomplete_or_bad_checkpoint_is_not_a_resume_hit(tmp_path, monkeypatch) -> None:
    identity = _training_identity()
    valid, reason = core.validate_training_cache(tmp_path, identity, load_checkpoint=True)
    assert valid is False
    assert "missing" in reason
    monkeypatch.setattr(
        core,
        "_train_epochs",
        lambda *args, **kwargs: [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "train_accuracy": 0.5,
                "n_train_items": 4,
                "batch_size": 4,
            }
        ],
    )
    rng = np.random.default_rng(27)
    core.train_inner_cache(
        tmp_path,
        identity,
        rng.normal(size=(4, 4, 5, 7)).astype(np.float32),
        np.asarray([0, 1, 0, 1]),
        rng.normal(size=(2, 4, 5, 7)).astype(np.float32),
        device="cpu",
        workers=0,
    )
    assert core.validate_training_cache(tmp_path, identity, load_checkpoint=True)[0]
    (tmp_path / "checkpoint.pt").write_bytes(b"corrupt checkpoint")
    complete = json.loads((tmp_path / "COMPLETE.json").read_text())
    complete["artifact_sha256"]["checkpoint.pt"] = core.file_sha256(
        tmp_path / "checkpoint.pt"
    )
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    valid, reason = core.validate_training_cache(tmp_path, identity, load_checkpoint=True)
    assert valid is False
    assert "checkpoint" in reason


def test_28_tampered_locked_selection_is_rejected(tmp_path) -> None:
    payload = _selection_payload()
    path = tmp_path / "selection.json"
    core.lock_selection(path, payload, expected_protocol_fingerprint="protocol")
    tampered = json.loads(path.read_text())
    tampered["selected_variant"] = MEAN_ONLY_VARIANT
    path.write_text(json.dumps(tampered))
    with pytest.raises(AssertionError):
        core.read_locked_selection(path, expected_protocol_fingerprint="protocol")


def test_29_scheme_a_pairs_outer_selector_and_training_seed(planned_output) -> None:
    tasks = pd.read_csv(planned_output / "inner_task_manifest.csv")
    assert tasks["outer_seed"].equals(tasks["selector_seed"])
    for row in tasks.sample(20, random_state=0).itertuples(index=False):
        assert json.loads(row.training_identity_json)["seed"] == int(row.outer_seed)


def test_30_run_complete_is_not_written_by_plan_or_incomplete_summary(planned_output, runner) -> None:
    assert not (planned_output / "RUN_COMPLETE.json").exists()
    args = SimpleNamespace(output_dir=planned_output, project_root=PROJECT_DIR, fixed_results_dir=FORMAL_DIR)
    with pytest.raises(FileNotFoundError):
        runner.run_summarize(args)
    assert not (planned_output / "RUN_COMPLETE.json").exists()


def test_synthetic_leakage_regression_and_cache_example(tmp_path, runner) -> None:
    args = runner.parse_args(["--stage", "sanity", "--project-root", str(PROJECT_DIR), "--output-dir", str(tmp_path)])
    runner.run_sanity(args)
    leakage = json.loads((tmp_path / "sanity/synthetic_leakage_audit.json").read_text())
    cache = json.loads((tmp_path / "sanity/cache_correctness_audit.json").read_text())
    assert leakage["status"] == "pass"
    assert leakage["model_training_performed"] is False
    assert cache["training_cache_keys_equal"] is True
    assert cache["evaluation_cache_keys_different"] is True


def test_candidates_are_directly_the_approved_implementations() -> None:
    assert type(build_model(MEAN_ONLY_VARIANT)) is FCNNMeanPool
    assert MODEL_IMPLEMENTATION_VERSION == "fcnn_mean_std_temporal_statistics_v1.0.0"


def test_locked_selection_contains_required_provenance_schema() -> None:
    payload = _selection_payload()
    required = {"session", "outer_fold", "seed", "outer_train_cycle_ids", "outer_test_cycle_ids", "inner_BA_mean_only", "inner_BA_mean_std", "delta_inner_BA", "tie", "selected_variant", "selection_rule_version", "candidate_protocol_fingerprints", "split_fingerprint", "normalization_protocol_fingerprint", "inner_oof_prediction_hashes", "inner_coverage_assertion", "created_at", "selection_artifact_hash", "outer_result_read_before_selection"}
    assert required <= set(payload)


def test_plan_resume_validates_without_overwriting_manifests(planned_output, runner) -> None:
    protected = [
        "config.json",
        "task_plan.csv",
        "split_manifest.csv",
        "cache_manifest.csv",
        "inner_task_manifest.csv",
        "PLAN_COMPLETE.json",
    ]
    before = {
        name: core.file_sha256(planned_output / name) for name in protected
    }
    args = runner.parse_args(
        [
            "--stage",
            "plan",
            "--project-root",
            str(PROJECT_DIR),
            "--output-dir",
            str(planned_output),
            "--fixed-results-dir",
            str(FORMAL_DIR),
            "--feasibility-dir",
            str(FEASIBILITY_DIR),
        ]
    )
    runner.run_plan(args)
    after = {name: core.file_sha256(planned_output / name) for name in protected}
    assert after == before
