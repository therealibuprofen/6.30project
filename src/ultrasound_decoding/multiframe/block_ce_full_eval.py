from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS


MODEL_NAME = "block_ce_fcnn"
MODEL_VERSION = "block_ce_fcnn_full_eval_v1.0.0"
REUSED_SEED = 0
NEW_SEEDS = (1, 2)
ALL_SEEDS = (0, 1, 2)
EXPECTED_FOLDS = 82
EXPECTED_NEW_TRAININGS = 164
EXPECTED_BLOCKS_PER_SEED = 456
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
STABILITY_THRESHOLDS = {
    "A_overall_delta_exclusive_minimum": 0.0,
    "B_session_wins_minimum_inclusive": 5,
    "C_seed_wins_minimum_inclusive": 2,
    "D_minimum_leave_one_session_out_delta_exclusive": 0.0,
}
IDENTITY_COLUMNS = ("session", "seed", "fold", "block_id")


def build_new_training_plan(reference_plan: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session",
        "seed",
        "fold",
        "n_train_samples",
        "n_test_samples",
        "train_cycles",
        "test_cycles",
    }
    if not required.issubset(reference_plan.columns):
        raise AssertionError("historical reference plan lacks required columns")
    plan = reference_plan.copy()
    plan["session"] = plan["session"].astype(str)
    plan["seed"] = plan["seed"].astype(int)
    plan = plan[plan["seed"].isin(NEW_SEEDS)].copy()
    plan = plan.sort_values(["seed", "session", "fold"]).reset_index(drop=True)
    plan = plan.rename(
        columns={
            "n_train_samples": "n_train_blocks",
            "n_test_samples": "n_test_blocks",
        }
    )
    plan["model"] = MODEL_NAME
    plan["training_action"] = "train_new"
    plan["task_key"] = plan.apply(
        lambda row: f"{row['session']}:{int(row['seed'])}:{int(row['fold'])}:{MODEL_NAME}",
        axis=1,
    )
    columns = [
        "session",
        "seed",
        "fold",
        "model",
        "training_action",
        "n_train_blocks",
        "n_test_blocks",
        "train_cycles",
        "test_cycles",
        "task_key",
    ]
    plan = plan[columns]
    counts = plan.groupby("seed").size().to_dict()
    if counts != {1: EXPECTED_FOLDS, 2: EXPECTED_FOLDS}:
        raise AssertionError("new training plan must contain exactly 82 folds for seeds 1 and 2")
    if len(plan) != EXPECTED_NEW_TRAININGS or REUSED_SEED in set(plan["seed"]):
        raise AssertionError("new training plan must be 164 tasks and exclude seed 0")
    if set(plan["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("new training plan does not cover the frozen nine sessions")
    return plan


def _canonical_identity(table: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(table.columns))
    if missing:
        raise AssertionError(f"prediction table lacks identity columns: {missing}")
    result = table.copy()
    result["session"] = result["session"].astype(str)
    result["seed"] = result["seed"].astype(int)
    result["fold"] = result["fold"].astype(int)
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise AssertionError("prediction table contains duplicate held-out block identities")
    return result.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)


def validate_prediction_identity_alignment(
    historical: pd.DataFrame,
    blockce: pd.DataFrame,
    *,
    seeds: Iterable[int] = ALL_SEEDS,
) -> dict[str, Any]:
    requested = {int(seed) for seed in seeds}
    left = _canonical_identity(historical)
    right = _canonical_identity(blockce)
    left = left[left["seed"].isin(requested)].reset_index(drop=True)
    right = right[right["seed"].isin(requested)].reset_index(drop=True)
    left_identity = left[list(IDENTITY_COLUMNS)]
    right_identity = right[list(IDENTITY_COLUMNS)]
    if not left_identity.equals(right_identity):
        merged = left_identity.merge(
            right_identity,
            on=list(IDENTITY_COLUMNS),
            how="outer",
            indicator=True,
        )
        mismatch_count = int((merged["_merge"] != "both").sum())
        raise AssertionError(
            f"historical and Block-CE held-out identities differ ({mismatch_count} mismatches)"
        )
    for column in ("truth", "cycle", "block_name"):
        if column in left.columns and column in right.columns:
            if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
                raise AssertionError(f"historical and Block-CE {column} values differ")
    expected_rows = EXPECTED_BLOCKS_PER_SEED * len(requested)
    if len(left) != expected_rows:
        raise AssertionError(
            f"aligned prediction coverage {len(left)} differs from expected {expected_rows}"
        )
    return {
        "status": "PASS",
        "seeds": sorted(requested),
        "aligned_rows": int(len(left)),
        "aligned_fold_tasks": int(
            left[["session", "seed", "fold"]].drop_duplicates().shape[0]
        ),
        "identity_columns": list(IDENTITY_COLUMNS),
        "truth_cycle_block_name_match": True,
    }


def session_seed_balanced_accuracy(
    predictions: pd.DataFrame, *, value_name: str
) -> pd.DataFrame:
    required = {"session", "seed", "truth", "pred", "block_id"}
    if not required.issubset(predictions.columns):
        raise AssertionError("prediction table lacks columns required for OOF aggregation")
    table = predictions.copy()
    table["session"] = table["session"].astype(str)
    table["seed"] = table["seed"].astype(int)
    rows = []
    for (session, seed), group in table.groupby(["session", "seed"], sort=True):
        if group["block_id"].duplicated().any():
            raise AssertionError("session/seed OOF contains a duplicate held-out block")
        ba = classification_metrics(
            group["truth"].to_numpy(dtype=np.int64),
            group["pred"].to_numpy(dtype=np.int64),
        )["balanced_accuracy"]
        rows.append(
            {
                "session": str(session),
                "seed": int(seed),
                value_name: float(ba),
                "n_oof_blocks": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def validate_seed0_reuse(
    seed0_predictions: pd.DataFrame,
    screening_session_summary: pd.DataFrame,
    historical_predictions: pd.DataFrame,
    *,
    source_path: str,
) -> dict[str, Any]:
    source = seed0_predictions.copy()
    if "model" in source.columns:
        source = source[source["model"].eq(MODEL_NAME)].copy()
    source["seed"] = source["seed"].astype(int)
    if set(source["seed"]) != {REUSED_SEED}:
        raise AssertionError("seed-0 reuse source must contain only seed 0 Block-CE predictions")
    source = _canonical_identity(source)
    folds = source[["session", "seed", "fold"]].drop_duplicates()
    if len(folds) != EXPECTED_FOLDS or len(source) != EXPECTED_BLOCKS_PER_SEED:
        raise AssertionError("seed-0 reuse source lacks complete 82-fold/456-block coverage")
    if set(source["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("seed-0 reuse source lacks one or more frozen sessions")
    alignment = validate_prediction_identity_alignment(
        historical_predictions, source, seeds=(REUSED_SEED,)
    )
    recomputed = session_seed_balanced_accuracy(
        source, value_name="recomputed_blockce_seed0_BA"
    )
    saved = screening_session_summary.copy()
    saved["session"] = saved["session"].astype(str)
    saved = saved[["session", "block_ce_fcnn_seed0_BA"]]
    compared = recomputed.merge(saved, on="session", validate="one_to_one")
    differences = np.abs(
        compared["recomputed_blockce_seed0_BA"].to_numpy(float)
        - compared["block_ce_fcnn_seed0_BA"].to_numpy(float)
    )
    maximum_difference = float(differences.max())
    if maximum_difference > 1e-12:
        raise AssertionError("recomputed seed-0 session BA differs from screening summary")
    return {
        "status": "PASS",
        "seed0_retrained": False,
        "training_action": "reuse_predictions_only",
        "source": str(source_path),
        "source_experiment": "crr_fcnn_screening_v1",
        "source_model": MODEL_NAME,
        "source_seed": REUSED_SEED,
        "folds_complete": True,
        "fold_count": EXPECTED_FOLDS,
        "heldout_blocks_complete": True,
        "heldout_block_count": EXPECTED_BLOCKS_PER_SEED,
        "heldout_identity_alignment_with_historical": alignment,
        "session_BA_recomputed_matches_screening": True,
        "maximum_absolute_session_BA_difference": maximum_difference,
        "recomputed_nine_session_mean_BA": float(
            recomputed["recomputed_blockce_seed0_BA"].mean()
        ),
    }


def build_evaluation_summaries(
    historical_predictions: pd.DataFrame,
    blockce_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_prediction_identity_alignment(
        historical_predictions, blockce_predictions, seeds=ALL_SEEDS
    )
    historical = session_seed_balanced_accuracy(
        historical_predictions, value_name="historical_BA"
    ).drop(columns="n_oof_blocks")
    blockce = session_seed_balanced_accuracy(
        blockce_predictions, value_name="blockce_BA"
    ).drop(columns="n_oof_blocks")
    per_seed_session = historical.merge(
        blockce, on=["session", "seed"], validate="one_to_one"
    )
    per_seed_session["delta_blockce_vs_historical"] = (
        per_seed_session["blockce_BA"] - per_seed_session["historical_BA"]
    )
    if len(per_seed_session) != len(EXPECTED_SESSIONS) * len(ALL_SEEDS):
        raise AssertionError("per-seed/session summary must contain exactly 27 rows")

    rows = []
    for session in EXPECTED_SESSIONS:
        group = per_seed_session[per_seed_session["session"].eq(session)].set_index("seed")
        if set(group.index.astype(int)) != set(ALL_SEEDS):
            raise AssertionError(f"session {session} lacks one or more seeds")
        row: dict[str, Any] = {"session": session}
        for seed in ALL_SEEDS:
            row[f"historical_seed{seed}_BA"] = float(group.loc[seed, "historical_BA"])
            row[f"blockce_seed{seed}_BA"] = float(group.loc[seed, "blockce_BA"])
        row["historical_3seed_mean_BA"] = float(
            np.mean([row[f"historical_seed{seed}_BA"] for seed in ALL_SEEDS])
        )
        row["blockce_3seed_mean_BA"] = float(
            np.mean([row[f"blockce_seed{seed}_BA"] for seed in ALL_SEEDS])
        )
        row["delta_blockce_vs_historical"] = (
            row["blockce_3seed_mean_BA"] - row["historical_3seed_mean_BA"]
        )
        rows.append(row)
    per_session = pd.DataFrame(rows)

    seed_rows = []
    for seed in ALL_SEEDS:
        group = per_seed_session[per_seed_session["seed"].eq(seed)]
        historical_mean = float(group["historical_BA"].mean())
        blockce_mean = float(group["blockce_BA"].mean())
        seed_rows.append(
            {
                "seed": int(seed),
                "historical_9session_mean_BA": historical_mean,
                "blockce_9session_mean_BA": blockce_mean,
                "delta_blockce_vs_historical": blockce_mean - historical_mean,
                "blockce_better": bool(blockce_mean > historical_mean),
            }
        )
    seed_level = pd.DataFrame(seed_rows)
    return per_seed_session, per_session, seed_level


def evaluate_stability(
    per_session_summary: pd.DataFrame,
    seed_level_summary: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "session",
        "historical_3seed_mean_BA",
        "blockce_3seed_mean_BA",
        "delta_blockce_vs_historical",
    }
    if not required.issubset(per_session_summary.columns):
        raise AssertionError("per-session summary lacks stability inputs")
    table = per_session_summary.copy()
    table["session"] = table["session"].astype(str)
    if len(table) != 9 or set(table["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("stability assessment requires the frozen nine sessions")
    if set(seed_level_summary["seed"].astype(int)) != set(ALL_SEEDS):
        raise AssertionError("stability assessment requires all three seed summaries")
    deltas = table["delta_blockce_vs_historical"].to_numpy(float)
    historical_overall = float(table["historical_3seed_mean_BA"].mean())
    blockce_overall = float(table["blockce_3seed_mean_BA"].mean())
    overall_delta = blockce_overall - historical_overall
    session_wins = int((deltas > 0).sum())
    seed_wins = int(seed_level_summary["blockce_better"].astype(bool).sum())
    loso = {
        str(table.iloc[index]["session"]): float(np.delete(deltas, index).mean())
        for index in range(len(table))
    }
    minimum_loso = float(min(loso.values()))
    passes = {
        "A": bool(overall_delta > STABILITY_THRESHOLDS["A_overall_delta_exclusive_minimum"]),
        "B": bool(session_wins >= STABILITY_THRESHOLDS["B_session_wins_minimum_inclusive"]),
        "C": bool(seed_wins >= STABILITY_THRESHOLDS["C_seed_wins_minimum_inclusive"]),
        "D": bool(
            minimum_loso
            > STABILITY_THRESHOLDS[
                "D_minimum_leave_one_session_out_delta_exclusive"
            ]
        ),
    }
    all_pass = all(passes.values())
    return {
        "thresholds": dict(STABILITY_THRESHOLDS),
        "observed": {
            "overall_historical_3seed_mean_BA": historical_overall,
            "overall_blockce_3seed_mean_BA": blockce_overall,
            "overall_delta_blockce_vs_historical": overall_delta,
            "sessions_blockce_better": session_wins,
            "seeds_blockce_9session_mean_better": seed_wins,
            "leave_one_session_out_mean_deltas": loso,
            "minimum_leave_one_session_out_mean_delta": minimum_loso,
            "strong3_historical_mean_BA": float(
                table[table["session"].isin(STRONG_SESSIONS)][
                    "historical_3seed_mean_BA"
                ].mean()
            ),
            "strong3_blockce_mean_BA": float(
                table[table["session"].isin(STRONG_SESSIONS)][
                    "blockce_3seed_mean_BA"
                ].mean()
            ),
            "weak6_historical_mean_BA": float(
                table[table["session"].isin(WEAK_SESSIONS)][
                    "historical_3seed_mean_BA"
                ].mean()
            ),
            "weak6_blockce_mean_BA": float(
                table[table["session"].isin(WEAK_SESSIONS)][
                    "blockce_3seed_mean_BA"
                ].mean()
            ),
        },
        "criteria": {name: {"passed": passed} for name, passed in passes.items()},
        "controlling_criteria": ["A", "B", "C", "D"],
        "all_four_criteria_passed": all_pass,
        "decision": (
            "supports_block_level_training_as_new_baseline"
            if all_pass
            else "does_not_support_block_level_training_as_new_baseline"
        ),
        "confirmatory_claim": False,
        "p_value_controls_decision": False,
    }
