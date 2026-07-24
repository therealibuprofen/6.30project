#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.interpretability.aggregation import (
    cross_method_agreement_row,
    pairwise_stability_rows,
    signed_region_summary,
    write_completeness_report,
)
from ultrasound_decoding.interpretability.common import (
    IMAGE_SHAPE,
    aggregate_patch_values,
    display_model_name,
    load_session_data,
    make_patch_specs,
    mean_arcsinh_background,
    patch_dataframe,
    resolve_model_name,
    validate_request,
)
from ultrasound_decoding.interpretability.gradients import aggregate_gradient_maps, run_gradients_for_fold
from ultrasound_decoding.interpretability.occlusion import aggregate_occlusion, run_occlusion_for_fold
from ultrasound_decoding.interpretability.plotting import (
    plot_nn_maps,
    plot_session_searchlight,
    save_main_figure,
)
from ultrasound_decoding.interpretability.searchlight import run_pca_lda_searchlight


DEFAULT_RUN_NAME = "spatial_interpretability_binary_v1"
DEFAULT_BENCHMARK_ROOT = PROJECT_DIR / "results" / "model_batch_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible spatial interpretability for binary fUS decoders.")
    parser.add_argument("--sessions", nargs="+", default=["708", "709", "710"])
    parser.add_argument("--task", default="binary")
    parser.add_argument("--methods", nargs="+", default=["pca_lda", "cnn", "fcnn_berthon2023"])
    parser.add_argument(
        "--interpretation-methods",
        nargs="+",
        default=["searchlight", "occlusion", "input_gradient", "gradient_x_input", "integrated_gradients"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patch-height", type=int, default=32)
    parser.add_argument("--patch-width", type=int, default=64)
    parser.add_argument("--stride-height", type=int, default=16)
    parser.add_argument("--stride-width", type=int, default=32)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=str(PROJECT_DIR / "results" / "runs" / "interpretability"))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-compatible-results", action="store_true")
    parser.add_argument("--stage", choices=["audit", "smoke", "session", "full"], default="full")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional diagnostic limit; smoke overrides to 3.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional diagnostic limit; smoke overrides to 2.")
    parser.add_argument("--only-folds", nargs="+", type=int, default=None)
    return parser.parse_args()


def output_run_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.run_name


def read_baseline_summary(benchmark_root: Path, sessions: list[str]) -> pd.DataFrame:
    path = benchmark_root / "benchmark" / "summary.csv"
    if not path.exists():
        path = benchmark_root / "summary.csv"
    df = pd.read_csv(path)
    return df[(df["task"].astype(str) == "binary") & (df["session"].astype(str).isin(sessions))].copy()


def audit_payload(args: argparse.Namespace, patches_n: int, baseline: pd.DataFrame) -> dict[str, object]:
    return {
        "benchmark_path": str(Path(args.benchmark_root)),
        "task": args.task,
        "sessions": args.sessions,
        "methods": args.methods,
        "interpretation_methods": args.interpretation_methods,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patches_per_session": patches_n,
        "expected_searchlight_fits": {
            "708": patches_n * 6,
            "709": patches_n * 10,
            "710": patches_n * 10,
        },
        "baseline_balanced_accuracy": baseline[
            baseline["method"].isin(["pca_lda", "cnn", "fcnn"])
        ][["session", "method", "seed", "balanced_accuracy"]].to_dict("records"),
        "fold_source": "results/model_batch_test/benchmark/fold_metrics_all.csv when present, otherwise regenerated grouped cycle CV",
        "linear_preprocessing": "arcsinh flatten; standardize=False for window_size=1; PCA variance=0.95; LDA reg=1e-3",
        "torch_normalization": "arcsinh_then_train_pixel_zscore with outer train fold statistics only",
        "checkpoint_source": "results/model_batch_test/models compatible fold/seed checkpoints",
    }


def save_session_common_outputs(session_dir: Path, data, patches) -> np.ndarray:
    (session_dir / "splits").mkdir(parents=True, exist_ok=True)
    data.split_manifest.to_csv(session_dir / "splits" / "split_manifest.csv", index=False)
    background = mean_arcsinh_background(data.X)
    (session_dir / "background").mkdir(parents=True, exist_ok=True)
    np.save(session_dir / "background" / "mean_fus_background.npy", background)
    patch_dataframe(patches).to_csv(session_dir / "splits" / "patch_definitions.csv", index=False)
    return background


def selected_fold_items(splits: list[tuple[np.ndarray, np.ndarray]], only_folds: list[int] | None):
    for fold, split in enumerate(splits, start=1):
        if only_folds is not None and fold not in set(only_folds):
            continue
        yield fold, split


def run_session(args: argparse.Namespace, session: str, patches, aggregate_rows: dict[str, list[dict[str, object]]]) -> None:
    run_root = output_run_root(args)
    benchmark_root = Path(args.benchmark_root)
    session_dir = run_root / f"session_{session}"
    stability_start = len(aggregate_rows["stability"])
    master_start = len(aggregate_rows["master"])
    data = load_session_data(PROJECT_DIR, benchmark_root, session, args.task)
    background = save_session_common_outputs(session_dir, data, patches)
    completeness = [
        {"session": session, "item": "input_shape", "status": tuple(data.X.shape[1:]) == IMAGE_SHAPE, "detail": str(data.X.shape)},
        {"session": session, "item": "split_manifest", "status": True, "detail": str(session_dir / "splits" / "split_manifest.csv")},
        {"session": session, "item": "mean_fus_background", "status": True, "detail": str(session_dir / "background" / "mean_fus_background.npy")},
    ]

    max_patches = args.max_patches
    max_test_samples = args.max_test_samples
    ig_steps = args.ig_steps
    only_folds = args.only_folds
    seeds = args.seeds
    if args.stage == "smoke":
        max_patches = 3
        max_test_samples = 2
        ig_steps = min(args.ig_steps, 8)
        only_folds = [1]
        seeds = [0]

    if "pca_lda" in args.methods and "searchlight" in args.interpretation_methods:
        searchlight_dir = session_dir / "pca_lda" / "searchlight"
        searchlight_dir.mkdir(parents=True, exist_ok=True)
        data.split_manifest.to_csv(searchlight_dir / "split_manifest.csv", index=False)
        result = run_pca_lda_searchlight(
            session=session,
            X=data.X,
            y=data.y,
            groups=data.groups,
            splits=[split for _, split in selected_fold_items(data.splits, only_folds)],
            patches=patches,
            output_dir=searchlight_dir,
            pca_variance=0.95,
            max_patches=max_patches,
        )
        completeness.append({"session": session, "item": "pca_lda_searchlight", "status": True, "detail": str(searchlight_dir)})
        plot_session_searchlight(session, background, searchlight_dir, session_dir / "figures")
        fold_maps = {}
        metrics = result["metrics"]
        used_patches = patches[:max_patches] if max_patches is not None else patches
        for fold, fold_df in metrics.groupby("fold"):
            vals = fold_df.set_index("patch_id").reindex([p.patch_id for p in used_patches])["balanced_accuracy"].to_numpy()
            fold_maps[f"fold{int(fold)}"] = aggregate_patch_values(used_patches, vals)[0]
        aggregate_rows["stability"].extend(
            pairwise_stability_rows(
                session=session,
                model="pca_lda",
                method="searchlight_ba",
                maps=fold_maps,
                comparison_type="fold",
            )
        )

    nn_methods = [method for method in args.methods if resolve_model_name(method) in {"cnn", "fcnn"}]
    gradient_methods = set(args.interpretation_methods) & {"input_gradient", "gradient_x_input", "integrated_gradients"}
    for requested_model in nn_methods:
        model_name = resolve_model_name(requested_model)
        model_dir = session_dir / display_model_name(model_name)
        seed_dirs = []
        for seed in seeds:
            seed_dir = model_dir / f"seed{seed}"
            seed_dirs.append(seed_dir)
            for fold, (train_idx, test_idx) in selected_fold_items(data.splits, only_folds):
                fold_dir = seed_dir / f"fold{fold}"
                if "occlusion" in args.interpretation_methods:
                    run_occlusion_for_fold(
                        project_dir=PROJECT_DIR,
                        benchmark_root=benchmark_root,
                        session=session,
                        task=args.task,
                        model_name=model_name,
                        seed=seed,
                        fold=fold,
                        X=data.X,
                        y=data.y,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        patches=patches,
                        output_dir=fold_dir,
                        device=args.device,
                        max_patches=max_patches,
                    )
                if gradient_methods:
                    run_gradients_for_fold(
                        project_dir=PROJECT_DIR,
                        benchmark_root=benchmark_root,
                        session=session,
                        task=args.task,
                        model_name=model_name,
                        seed=seed,
                        fold=fold,
                        X=data.X,
                        y=data.y,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        output_dir=fold_dir,
                        interpretation_methods=gradient_methods,
                        ig_steps=ig_steps,
                        device=args.device,
                        max_test_samples=max_test_samples,
                    )
        aggregate_dir = model_dir / "aggregate"
        if "occlusion" in args.interpretation_methods:
            aggregate_occlusion(seed_dirs, aggregate_dir)
            completeness.append({"session": session, "item": f"{display_model_name(model_name)}_occlusion", "status": True, "detail": str(aggregate_dir)})
        if gradient_methods:
            aggregate_gradient_maps(seed_dirs, aggregate_dir)
            completeness.append({"session": session, "item": f"{display_model_name(model_name)}_gradients", "status": True, "detail": str(aggregate_dir)})
        plot_nn_maps(session, display_model_name(model_name), background, aggregate_dir, session_dir / "figures")
        add_nn_stability_rows(session, display_model_name(model_name), model_dir, seeds, data.splits, aggregate_rows)

    write_completeness_report(session_dir / "interpretability_completeness_report.csv", completeness)
    pd.DataFrame(aggregate_rows["stability"][stability_start:]).to_csv(
        session_dir / "interpretability_stability.csv",
        index=False,
    )
    pd.DataFrame(aggregate_rows["master"][master_start:]).to_csv(
        session_dir / "interpretability_master_summary.csv",
        index=False,
    )


def load_mean(paths: list[Path]) -> np.ndarray | None:
    arrays = [np.load(path) for path in paths if path.exists()]
    return np.nanmean(np.stack(arrays, axis=0), axis=0) if arrays else None


def add_nn_stability_rows(session: str, model: str, model_dir: Path, seeds: list[int], splits, aggregate_rows) -> None:
    method_specs = [
        ("integrated_gradients_absolute", "integrated_gradients_absolute_mean.npy"),
        ("occlusion_probability_drop", "occlusion_probability_drop.npy"),
        ("occlusion_ba_drop", "occlusion_ba_drop.npy"),
    ]
    for method, filename in method_specs:
        seed_maps = {}
        for seed in seeds:
            seed_maps[f"seed{seed}"] = load_mean(sorted((model_dir / f"seed{seed}").glob(f"fold*/{filename}")))
        seed_maps = {key: value for key, value in seed_maps.items() if value is not None}
        aggregate_rows["stability"].extend(
            pairwise_stability_rows(
                session=session,
                model=model,
                method=method,
                maps=seed_maps,
                comparison_type="seed",
            )
        )
        fold_maps = {}
        for fold in range(1, len(splits) + 1):
            fold_maps[f"fold{fold}"] = load_mean(sorted(model_dir.glob(f"seed*/fold{fold}/{filename}")))
        fold_maps = {key: value for key, value in fold_maps.items() if value is not None}
        aggregate_rows["stability"].extend(
            pairwise_stability_rows(
                session=session,
                model=model,
                method=method,
                maps=fold_maps,
                comparison_type="fold",
            )
        )
    for path in [
        model_dir / "aggregate" / "integrated_gradients_stimulus_signed_mean_mean.npy",
        model_dir / "aggregate" / "integrated_gradients_no_stimulus_signed_mean_mean.npy",
    ]:
        if path.exists():
            summary = signed_region_summary(np.load(path))
            aggregate_rows["master"].append(
                {
                    "session": session,
                    "model": model,
                    "method": path.stem,
                    **summary,
                }
            )


def build_cross_method_agreement(run_root: Path, sessions: list[str]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        sdir = run_root / f"session_{session}"
        maps = {
            "pca_lda_searchlight_ba": sdir / "pca_lda" / "searchlight" / "searchlight_ba_mean.npy",
            "cnn_integrated_gradients_absolute": sdir / "cnn" / "aggregate" / "integrated_gradients_absolute_mean_mean.npy",
            "cnn_occlusion_ba_drop": sdir / "cnn" / "aggregate" / "occlusion_ba_drop_mean.npy",
            "fcnn_integrated_gradients_absolute": sdir / "fcnn_berthon2023" / "aggregate" / "integrated_gradients_absolute_mean_mean.npy",
        }
        loaded = {key: np.load(path) for key, path in maps.items() if path.exists()}
        pairs = [
            ("pca_lda_searchlight_ba", "cnn_integrated_gradients_absolute"),
            ("pca_lda_searchlight_ba", "cnn_occlusion_ba_drop"),
            ("cnn_integrated_gradients_absolute", "cnn_occlusion_ba_drop"),
            ("cnn_integrated_gradients_absolute", "fcnn_integrated_gradients_absolute"),
        ]
        for a, b in pairs:
            if a in loaded and b in loaded:
                rows.append(cross_method_agreement_row(session=session, method_a=a, method_b=b, map_a=loaded[a], map_b=loaded[b]))
    return pd.DataFrame(rows)


def collect_session_tables(run_root: Path, sessions: list[str], new_rows: dict[str, list[dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_frames = []
    master_frames = []
    for session in sessions:
        sdir = run_root / f"session_{session}"
        stability_path = sdir / "interpretability_stability.csv"
        master_path = sdir / "interpretability_master_summary.csv"
        if stability_path.exists():
            stability_frames.append(pd.read_csv(stability_path))
        if master_path.exists():
            master_frames.append(pd.read_csv(master_path))
    if new_rows["stability"]:
        stability_frames.append(pd.DataFrame(new_rows["stability"]))
    if new_rows["master"]:
        master_frames.append(pd.DataFrame(new_rows["master"]))
    stability = pd.concat(stability_frames, ignore_index=True).drop_duplicates() if stability_frames else pd.DataFrame()
    master = pd.concat(master_frames, ignore_index=True).drop_duplicates() if master_frames else pd.DataFrame()
    return stability, master


def write_findings(run_root: Path, sessions: list[str], baseline: pd.DataFrame, stability: pd.DataFrame, agreement: pd.DataFrame) -> None:
    lines = [
        "# Spatial Interpretability Findings",
        "",
        "This report describes image regions and pixel locations only. No atlas or cross-session registration was used, so hotspots are not named as anatomical regions.",
        "",
        "## Baseline decoding",
    ]
    for session in sessions:
        sub = baseline[(baseline["session"].astype(str) == session) & (baseline["method"].isin(["pca_lda", "cnn", "fcnn"]))]
        values = "; ".join(
            f"{row.method}/seed{int(row.seed)} BA={float(row.balanced_accuracy):.3f}"
            for row in sub.itertuples(index=False)
        )
        lines.append(f"- session {session}: {values}")
    lines.extend(["", "## Stability audit"])
    if stability.empty:
        lines.append("- Stability rows were not produced, likely because the current stage was a smoke test with too few folds/seeds.")
    else:
        summary = stability.groupby(["session", "model", "method", "comparison_type"])["spearman_r"].mean().reset_index()
        for row in summary.itertuples(index=False):
            val = "NA" if pd.isna(row.spearman_r) else f"{float(row.spearman_r):.3f}"
            lines.append(f"- session {row.session}, {row.model}, {row.method}, {row.comparison_type}: mean Spearman r={val}")
    lines.extend(["", "## Cross-method spatial agreement"])
    if agreement.empty:
        lines.append("- Cross-method agreement rows were not produced in this stage.")
    else:
        for row in agreement.itertuples(index=False):
            val = "NA" if pd.isna(row.spearman_r) else f"{float(row.spearman_r):.3f}"
            lines.append(f"- session {row.session}: {row.method_a} vs {row.method_b}, Spearman r={val}, top10 overlap={float(row.top10_overlap):.3f}")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "- Searchlight BA is a local decoding description, not an independently validated best-region performance estimate.",
            "- Integrated Gradients and occlusion measure different sensitivities; agreement is described as spatial-pattern consistency only.",
            "- Negative occlusion drops and signed attributions were retained.",
            "- If fold or seed stability is low, local hotspots should not be interpreted as stable image regions.",
            "- Session maps are not registered, so session differences may reflect imaging position differences.",
        ]
    )
    (run_root / "aggregate" / "interpretability_findings.md").write_text("\n".join(lines), encoding="utf-8")


def write_session_findings(run_root: Path, session: str, baseline: pd.DataFrame, stability: pd.DataFrame, agreement: pd.DataFrame) -> None:
    sdir = run_root / f"session_{session}"
    lines = [
        f"# Session {session} Spatial Interpretability Findings",
        "",
        "This session report describes image regions and pixel coordinates only. The session is not atlas-registered.",
        "",
        "## Baseline decoding",
    ]
    sub = baseline[(baseline["session"].astype(str) == session) & (baseline["method"].isin(["pca_lda", "cnn", "fcnn"]))]
    for row in sub.itertuples(index=False):
        model = "fcnn_berthon2023" if row.method == "fcnn" else row.method
        lines.append(f"- {model}/seed{int(row.seed)}: Balanced Accuracy={float(row.balanced_accuracy):.3f}")
    summary_path = sdir / "pca_lda" / "searchlight" / "searchlight_summary.csv"
    if summary_path.exists():
        top = pd.read_csv(summary_path).head(3)
        lines.extend(["", "## PCA+LDA Searchlight"])
        for row in top.itertuples(index=False):
            lines.append(
                f"- patch {int(row.patch_id)} centered at row {float(row.center_row):.1f}, col {float(row.center_col):.1f}: "
                f"mean BA={float(row.balanced_accuracy_mean):.3f}, fold SD={float(row.balanced_accuracy_std):.3f}. "
                "This is descriptive, not an independently validated best region."
            )
    session_stability = stability[stability["session"].astype(str) == session] if not stability.empty else pd.DataFrame()
    lines.extend(["", "## Stability"])
    if session_stability.empty:
        lines.append("- Stability comparisons are unavailable for this stage.")
    else:
        means = session_stability.groupby(["model", "method", "comparison_type"])["spearman_r"].mean().reset_index()
        for row in means.itertuples(index=False):
            val = "NA" if pd.isna(row.spearman_r) else f"{float(row.spearman_r):.3f}"
            lines.append(f"- {row.model}, {row.method}, {row.comparison_type}: mean Spearman r={val}")
    session_agreement = agreement[agreement["session"].astype(str) == session] if not agreement.empty else pd.DataFrame()
    lines.extend(["", "## Cross-method agreement"])
    if session_agreement.empty:
        lines.append("- Cross-method agreement is unavailable for this stage.")
    else:
        for row in session_agreement.itertuples(index=False):
            val = "NA" if pd.isna(row.spearman_r) else f"{float(row.spearman_r):.3f}"
            lines.append(f"- {row.method_a} vs {row.method_b}: Spearman r={val}, top10 overlap={float(row.top10_overlap):.3f}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Do not call hotspots anatomical brain regions without atlas/registration.",
            "- Do not interpret gradient magnitude or occlusion drops as causal evidence.",
            "- Main interpretation should use cross-fold and cross-seed average maps, not individual samples.",
            "- Negative occlusion drops and signed attributions are retained as informative audit outputs.",
        ]
    )
    (sdir / f"session_{session}_interpretability_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.sessions = validate_request(args.task, args.sessions)
    args.methods = [display_model_name(resolve_model_name(method)) if method != "pca_lda" else method for method in args.methods]
    if args.max_epochs != 40:
        raise ValueError("This workflow must reuse max_epochs=40 benchmark configuration")
    patches = make_patch_specs(
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        stride_height=args.stride_height,
        stride_width=args.stride_width,
    )
    baseline = read_baseline_summary(Path(args.benchmark_root), args.sessions)
    if args.stage == "smoke":
        args.sessions = ["710"]
    if args.stage == "audit" or args.dry_run:
        print(json.dumps(audit_payload(args, len(patches), baseline), indent=2, ensure_ascii=False))
        return

    run_root = output_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "aggregate").mkdir(parents=True, exist_ok=True)
    aggregate_rows: dict[str, list[dict[str, object]]] = {"stability": [], "master": []}
    for session in args.sessions:
        session_dir = run_root / f"session_{session}"
        if (
            args.reuse_compatible_results
            and args.stage != "smoke"
            and (session_dir / "interpretability_completeness_report.csv").exists()
        ):
            print(f"Reusing existing interpretability outputs for session {session}")
            continue
        run_session(args, session, patches, aggregate_rows)

    stability, session_master = collect_session_tables(run_root, args.sessions, aggregate_rows)
    stability.to_csv(run_root / "aggregate" / "interpretability_stability.csv", index=False)
    agreement = build_cross_method_agreement(run_root, args.sessions)
    agreement.to_csv(run_root / "aggregate" / "cross_method_spatial_agreement.csv", index=False)
    master = session_master
    if not baseline.empty:
        baseline_rows = baseline[baseline["method"].isin(["pca_lda", "cnn", "fcnn"])].copy()
        baseline_rows["model"] = baseline_rows["method"].replace({"fcnn": "fcnn_berthon2023"})
        baseline_rows["method"] = "benchmark_balanced_accuracy"
        baseline_rows = baseline_rows[["session", "model", "method", "seed", "balanced_accuracy"]]
        master = pd.concat([baseline_rows, master], ignore_index=True)
    master.to_csv(run_root / "aggregate" / "interpretability_master_summary.csv", index=False)
    completeness_rows = []
    for session in args.sessions:
        path = run_root / f"session_{session}" / "interpretability_completeness_report.csv"
        if path.exists():
            completeness_rows.extend(pd.read_csv(path).to_dict("records"))
    completeness_rows.extend(
        [
            {"session": "aggregate", "item": "interpretability_stability", "status": True, "detail": str(run_root / "aggregate" / "interpretability_stability.csv")},
            {"session": "aggregate", "item": "cross_method_spatial_agreement", "status": True, "detail": str(run_root / "aggregate" / "cross_method_spatial_agreement.csv")},
        ]
    )
    write_completeness_report(run_root / "aggregate" / "interpretability_completeness_report.csv", completeness_rows)
    write_findings(run_root, args.sessions, baseline, stability, agreement)
    for session in args.sessions:
        write_session_findings(run_root, session, baseline, stability, agreement)
    if args.stage == "full":
        save_main_figure(run_root, args.sessions, run_root / "aggregate" / "interpretability_main_figure")
    elif args.stage == "session":
        save_main_figure(run_root, args.sessions, run_root / "aggregate" / "interpretability_main_figure")
    print(f"Saved interpretability outputs under {run_root}")


if __name__ == "__main__":
    main()
