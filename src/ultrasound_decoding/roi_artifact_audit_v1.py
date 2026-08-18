"""Read-only ROI artifact and provenance audit.

The audit deliberately does not create, modify, resize, register, or decode an ROI.
Every write is confined to the caller supplied audit output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


RUN_NAME = "roi_artifact_audit_v1"
EXPECTED_SESSIONS = ("626", "628", "708", "709", "710", "807", "813", "817", "822")
EXPECTED_SHAPE = (128, 501)
FIXED_ORIENTATIONS = {session: "identity" for session in EXPECTED_SESSIONS}
FIXED_ORIENTATIONS["807"] = "flip_vertical"

CLASS_EXPERT = "EXPERT_SESSION_SPECIFIC"
CLASS_RECONSTRUCTED = "EXPERT_GUIDED_RECONSTRUCTED"
CLASS_RECONSTRUCTED_SEARCHLIGHT = "EXPERT_GUIDED_RECONSTRUCTED_FROM_SEARCHLIGHT_DISPLAY"
CLASS_CIRCULAR = "LABEL_DERIVED_CIRCULAR"
CLASS_TRANSFER = "CROSS_SESSION_TRANSFER_UNRELIABLE"
CLASS_UNKNOWN = "UNKNOWN_PROVENANCE"
CLASS_MISSING = "MISSING"

ROI_KEYWORDS = (
    "roi", "mask", "candidate_roi", "manual_roi", "expert_roi", "region",
    "polygon", "poly", "draw_roi", "fillpoly", "searchlight", "expert",
    "annotation", "anatomical", "crop", "zero", "mean_roi", "roi_mean",
    "roi_crop", "roi_zero",
)
CODE_KEYWORDS = (
    "roi_mask", "mask_path", "roi_path", "candidate_roi", "manual_mask",
    "expert_mask", "cv2.fillpoly", "polygon", "roi_crop", "roi_zero", "roi_mean",
)
AUDITED_EXTENSIONS = {
    ".npy", ".npz", ".mat", ".pkl", ".pt", ".csv", ".json", ".yaml",
    ".yml", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
MASK_EXTENSIONS = {".npy", ".npz", ".mat", ".png", ".tif", ".tiff"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"}
LABEL_TOKENS = ("accuracy", "activation", "attribution", "prediction", "glm", "fdr", "beta", "t_map", "t-map")
TRANSFER_TOKENS = ("warp", "warped", "registration", "registered", "transferred", "transfer")

INVENTORY_FIELDS = (
    "artifact_id", "artifact_path", "file_type", "artifact_kind", "session",
    "session_inferred_or_explicit", "mask_shape", "expected_shape", "shape_valid",
    "mask_area_pixels", "mask_fraction", "mask_sha256", "file_sha256",
    "source_image_path", "source_image_type", "annotation_type", "annotator",
    "creation_script", "creation_config", "creation_date_if_known",
    "derived_from_searchlight", "derived_from_glm", "derived_from_attribution",
    "derived_from_decoding_labels", "label_information_used", "session_specific",
    "transferred_from_session", "registration_used", "orientation_state",
    "provenance_known", "classification", "usable_for_primary_roi_decoding",
    "usable_for_exploratory_roi_decoding", "exclusion_reason", "notes",
    "overlay_path",
)


@dataclass(frozen=True)
class DiscoveredArtifact:
    path: Path
    kind: str


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _excluded(path: Path, output_dir: Path | None = None) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    if output_dir is not None:
        try:
            path.resolve().relative_to(output_dir.resolve())
            return True
        except ValueError:
            pass
    return False


def iter_project_files(project_root: Path, output_dir: Path | None = None) -> Iterable[Path]:
    root = Path(project_root).resolve()
    for path in root.rglob("*"):
        if path.is_file() and not _excluded(path, output_dir):
            yield path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_array_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(mask).astype(np.bool_, copy=False))
    header = f"{canonical.shape}|bool|C".encode("ascii")
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result[name.lower()] = item
            result.update(_flatten(item, name))
    return result


def _metadata_for(path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    candidates: list[Path] = []
    if path.suffix.lower() == ".json":
        candidates.append(path)
    candidates.extend(sorted(path.parent.glob("*metadata*.json")))
    candidates.extend(sorted(path.parent.glob("*provenance*.json")))
    candidates.extend(sorted(path.parent.glob("*manifest*.json")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        data = _safe_json(candidate)
        if data:
            merged.update(data)
            merged.setdefault("_metadata_paths", []).append(str(candidate.resolve()))
    return merged


def _text_contains_roi(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > 5_000_000:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return any(keyword in text for keyword in ROI_KEYWORDS + CODE_KEYWORDS)


def artifact_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    lower_path = path.as_posix().lower()
    if suffix not in AUDITED_EXTENSIONS:
        return None
    if any(token in lower_name for token in ("summary", "metric", "score", "result")) and "mask" not in lower_name:
        return None
    if suffix in MASK_EXTENSIONS and "mask" in lower_name:
        return "MASK"
    if suffix in {".npy", ".npz", ".mat", ".pkl", ".pt"} and any(
        token in lower_name for token in ("roi", "manual", "expert")
    ):
        return "SERIALIZED_ROI"
    if ("polygon" in lower_name or "draw_roi" in lower_name or "poly" in lower_name) and suffix in {".json", ".csv", ".yaml", ".yml"}:
        return "POLYGON_OR_COORDINATES"
    if suffix == ".json" and any(token in lower_name for token in ("roi", "annotation", "metadata", "manifest")) and "roi" in lower_path:
        return "PROVENANCE_SIDECAR"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        direct_expert = bool(re.search(r"(?:^|[-_])(?:roi|expert)(?:[-_.]|$)", lower_name))
        if direct_expert and not any(token in lower_name for token in ("overlay", "summary", "overview", "background", "mean")):
            return "MANUAL_ANNOTATION_IMAGE"
    if any(token in lower_path for token in ("manual_roi", "expert_roi", "candidate_roi")) and _text_contains_roi(path):
        return "ROI_DEFINITION_OR_RECORD"
    return None


def discover_roi_artifacts(project_root: Path, output_dir: Path | None = None) -> list[DiscoveredArtifact]:
    artifacts: list[DiscoveredArtifact] = []
    for path in iter_project_files(project_root, output_dir):
        kind = artifact_kind(path)
        if kind:
            artifacts.append(DiscoveredArtifact(path.resolve(), kind))
    return sorted(artifacts, key=lambda item: item.path.as_posix())


def recursive_search_evidence(project_root: Path, output_dir: Path | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in iter_project_files(project_root, output_dir):
        lower_path = path.as_posix().lower()
        filename_hits = sorted({key for key in ROI_KEYWORDS if key in lower_path})
        content_hits: list[str] = []
        if path.suffix.lower() in TEXT_EXTENSIONS and path.stat().st_size <= 5_000_000:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                content_hits = sorted({key for key in ROI_KEYWORDS + CODE_KEYWORDS if key in text})
            except OSError:
                pass
        if filename_hits or content_hits:
            rows.append({
                "path": _relative(path, project_root),
                "file_type": path.suffix.lower(),
                "filename_keyword_hits": ";".join(filename_hits),
                "content_keyword_hits": ";".join(content_hits),
            })
    return rows


def infer_session(path: Path, metadata: Mapping[str, Any]) -> tuple[str, str]:
    flat = _flatten(metadata)
    explicit: list[str] = []
    for key, value in flat.items():
        if key.split(".")[-1] in {"session", "session_id"}:
            match = re.fullmatch(r"(?:session[_-]?)?(\d{3})", str(value).strip(), flags=re.IGNORECASE)
            if match and match.group(1) in EXPECTED_SESSIONS:
                explicit.append(match.group(1))
    explicit = sorted(set(explicit))
    if len(explicit) == 1:
        return explicit[0], "explicit"
    inferred: list[str] = []
    for match in re.finditer(r"(?<!\d)(?:session[_-]?)?(626|628|708|709|710|807|813|817|822)(?!\d)", path.as_posix(), re.I):
        inferred.append(match.group(1))
    inferred = sorted(set(inferred))
    if len(inferred) == 1:
        return inferred[0], "inferred"
    return "UNKNOWN", "unknown_or_ambiguous"


def _load_image(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            arr = mpimg.imread(path)
        else:
            return None
    except Exception:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr if arr.ndim == 2 else None


def load_mask_array(path: Path, kind: str | None = None) -> tuple[np.ndarray | None, str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            raw = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = [key for key in archive.files if "mask" in key.lower() or "roi" in key.lower()]
                if not keys and len(archive.files) == 1:
                    keys = list(archive.files)
                if len(keys) != 1:
                    return None, "NPZ_MASK_KEY_AMBIGUOUS"
                raw = archive[keys[0]]
        elif suffix in {".png", ".tif", ".tiff"} and (kind == "MASK" or "mask" in path.name.lower()):
            image = _load_image(path)
            if image is None:
                return None, "IMAGE_MASK_UNREADABLE"
            finite = image[np.isfinite(image)]
            if finite.size == 0 or np.unique(finite).size > 4:
                return None, "IMAGE_NOT_UNAMBIGUOUSLY_BINARY"
            raw = image > (float(finite.min()) + float(finite.max())) / 2.0
        elif suffix == ".mat":
            try:
                import h5py
                with h5py.File(path, "r") as handle:
                    names: list[str] = []
                    handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
                    names = [name for name in names if "mask" in name.lower() or "roi" in name.lower()]
                    if len(names) != 1:
                        return None, "MAT_MASK_DATASET_AMBIGUOUS"
                    raw = np.asarray(handle[names[0]])
            except Exception:
                return None, "MAT_MASK_INSPECTION_UNAVAILABLE"
        else:
            return None, "SAFE_MASK_LOADER_NOT_AVAILABLE"
    except Exception as exc:
        return None, f"MASK_LOAD_FAILED:{type(exc).__name__}"
    raw = np.asarray(raw)
    if raw.ndim != 2:
        return None, f"MASK_NOT_2D:{raw.shape}"
    if raw.dtype == np.bool_:
        return raw.copy(), "OK"
    finite = raw[np.isfinite(raw)] if np.issubdtype(raw.dtype, np.number) else np.asarray([])
    if finite.size and set(np.unique(finite).tolist()).issubset({0, 1, 0.0, 1.0}):
        return raw.astype(bool), "OK"
    return None, "MASK_VALUES_NOT_BINARY"


def _lookup(metadata: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    flat = _flatten(metadata)
    wanted = {name.lower() for name in names}
    for key, value in flat.items():
        if key.split(".")[-1] in wanted:
            return value
    return default


def _truth(value: Any) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "YES"
    if text in {"false", "no", "0"}:
        return "NO"
    return "UNKNOWN"


def audit_orientation(session: str, metadata: Mapping[str, Any]) -> str:
    expected = FIXED_ORIENTATIONS.get(session, "unknown")
    value = _lookup(
        metadata, "fixed_orientation_normalization", "fixed_orientation", "orientation",
        "orientation_state", "moving_orientation", default="",
    )
    normalized = _truth(_lookup(metadata, "orientation_normalized", "normalization_applied", default=""))
    text = str(value).lower().replace("-", "_").replace(" ", "_")
    if session == "807":
        if "flip_vertical" in text or (normalized == "YES" and expected == "flip_vertical"):
            return "NORMALIZED_FLIP_VERTICAL"
        return "ORIENTATION_MISMATCH_OR_UNCERTAIN"
    if text and any(token in text for token in ("flip", "rot", "transpose")) and "identity" not in text:
        return "ORIENTATION_MISMATCH_OR_UNCERTAIN"
    return "IDENTITY_EXPECTED"


def classify_provenance(path: Path, metadata: Mapping[str, Any]) -> dict[str, str]:
    lower = path.as_posix().lower()
    blob = (lower + " " + json.dumps(metadata, ensure_ascii=True, default=str).lower())
    provenance = str(_lookup(metadata, "roi_provenance", "provenance", default="")).lower()
    annotation = str(_lookup(metadata, "annotation_type", "annotation_method", default="")).lower()
    annotation_reason = str(_lookup(metadata, "annotation_reason", "reason", default="")).lower()
    source_value = str(_lookup(metadata, "source_image_path", "source_image_type", "annotation_image", default="")).lower()
    source_evidence = " ".join((lower, provenance, annotation, annotation_reason, source_value))
    label = _truth(_lookup(metadata, "label_information_used", "used_labels", default=""))
    direct_searchlight_threshold = (
        "searchlight" in source_evidence
        and any(token in source_evidence for token in ("threshold", "top 10", "top_10", "peak", "hotspot"))
        and "analyst_reconstructed" not in source_evidence and "expert_guided" not in source_evidence
    )
    derived_searchlight = _truth(_lookup(
        metadata, "derived_from_searchlight", "used_searchlight_or_activation_map",
        "used_searchlight_accuracy_or_hotspot", default="",
    ))
    if derived_searchlight == "UNKNOWN" and "searchlight" in blob:
        derived_searchlight = "YES"
    derived_glm = _truth(_lookup(metadata, "derived_from_glm", "used_glm", default=""))
    if derived_glm == "UNKNOWN" and any(token in blob for token in ("glm", "fdr", "beta map", "t-map", "t_map")):
        derived_glm = "YES"
    derived_attr = _truth(_lookup(metadata, "derived_from_attribution", default=""))
    if derived_attr == "UNKNOWN" and "attribution" in blob:
        derived_attr = "YES"
    derived_labels = _truth(_lookup(metadata, "derived_from_decoding_labels", default=""))
    if derived_labels == "UNKNOWN" and any(token in blob for token in ("decoding accuracy", "classification weight", "oof prediction")):
        derived_labels = "YES"

    transferred = str(_lookup(metadata, "transferred_from_session", default="")).strip()
    registration = _truth(_lookup(metadata, "registration_used", "registration", default=""))
    transfer_evidence = " ".join((lower, provenance, annotation, annotation_reason, transferred.lower()))
    explicit_transfer = bool(transferred) or registration == "YES" or any(token in transfer_evidence for token in TRANSFER_TOKENS)
    reconstructed = any(token in provenance + " " + annotation + " " + blob for token in (
        "analyst_reconstructed_from_expert", "expert_guided_reconstructed", "expert indicated",
    ))
    searchlight_display = reconstructed and "searchlight" in source_evidence
    direct_expert = (
        any(token in provenance + " " + annotation for token in ("expert_session_specific", "direct_expert", "expert_direct"))
        and not reconstructed
    )
    direct_label_derived = (
        label == "YES" or direct_searchlight_threshold or derived_glm == "YES"
        or derived_attr == "YES" or derived_labels == "YES"
        or (derived_searchlight == "YES" and not reconstructed)
    )

    if direct_label_derived:
        classification = CLASS_CIRCULAR
        primary, exploratory = "NO", "NO"
        exclusion = "CIRCULAR_ANALYSIS_RISK"
    elif explicit_transfer:
        classification = CLASS_TRANSFER
        primary, exploratory = "NO", "NO"
        exclusion = "UNVALIDATED_CROSS_SESSION_TRANSFER"
    elif reconstructed:
        classification = CLASS_RECONSTRUCTED_SEARCHLIGHT if searchlight_display else CLASS_RECONSTRUCTED
        primary, exploratory = "REVIEW_REQUIRED", "YES"
        exclusion = "POTENTIAL_LABEL_DEPENDENCE_REQUIRES_REVIEW" if searchlight_display else "EXPERT_GUIDED_RECONSTRUCTION_REQUIRES_REVIEW"
    elif direct_expert and "searchlight" in source_evidence:
        classification = CLASS_RECONSTRUCTED_SEARCHLIGHT
        primary, exploratory = "REVIEW_REQUIRED", "YES"
        exclusion = "POTENTIAL_LABEL_DEPENDENCE_REQUIRES_REVIEW"
    elif direct_expert:
        classification = CLASS_EXPERT
        primary, exploratory = "YES", "YES"
        exclusion = ""
    else:
        classification = CLASS_UNKNOWN
        primary, exploratory = "NO", "NO"
        exclusion = "PROVENANCE_UNKNOWN"
    return {
        "classification": classification,
        "usable_for_primary_roi_decoding": primary,
        "usable_for_exploratory_roi_decoding": exploratory,
        "exclusion_reason": exclusion,
        "derived_from_searchlight": derived_searchlight,
        "derived_from_glm": derived_glm,
        "derived_from_attribution": derived_attr,
        "derived_from_decoding_labels": derived_labels,
        "label_information_used": label,
        "transferred_from_session": transferred or "NONE",
        "registration_used": registration,
        "provenance_known": "YES" if classification != CLASS_UNKNOWN else "NO",
    }


def _source_info(path: Path, metadata: Mapping[str, Any], project_root: Path) -> tuple[str, str]:
    value = _lookup(metadata, "source_image_path", "mean_image_source", "annotation_image", default="")
    source = str(value).strip()
    if source and not Path(source).is_absolute():
        candidates = (path.parent / source, project_root / source)
        for candidate in candidates:
            if candidate.exists():
                source = str(candidate.resolve())
                break
    source_type = str(_lookup(metadata, "source_image_type", default="")).strip()
    blob = (source + " " + source_type).lower()
    positive_label_source = any(
        _truth(_lookup(metadata, key, default="")) == "YES"
        for key in (
            "derived_from_searchlight", "used_searchlight_or_activation_map",
            "used_searchlight_accuracy_or_hotspot", "derived_from_glm", "used_glm",
            "derived_from_attribution", "derived_from_decoding_labels", "label_information_used",
        )
    )
    if not source_type:
        if positive_label_source or any(token in blob for token in LABEL_TOKENS) or "searchlight visualization" in blob:
            source_type = "LABEL_DEPENDENT_VISUALIZATION"
        elif source:
            source_type = "STRUCTURAL_OR_LABEL_INDEPENDENT"
        else:
            source_type = "UNKNOWN"
    return source or "UNKNOWN", source_type


def inspect_artifact(item: DiscoveredArtifact, project_root: Path, artifact_id: str) -> tuple[dict[str, Any], np.ndarray | None]:
    path = item.path
    metadata = _metadata_for(path)
    session, session_basis = infer_session(path, metadata)
    mask, load_note = load_mask_array(path, item.kind)
    provenance = classify_provenance(path, metadata)
    source_path, source_type = _source_info(path, metadata, project_root)
    if mask is None:
        shape = "UNKNOWN"
        shape_valid = "NOT_A_VALID_MASK"
        area: int | str = ""
        fraction: float | str = ""
        mask_hash = ""
    else:
        shape = "x".join(map(str, mask.shape))
        shape_valid = "YES" if mask.shape == EXPECTED_SHAPE else "NO"
        area = int(mask.sum())
        fraction = float(area / mask.size)
        mask_hash = mask_array_sha256(mask)
    orientation = audit_orientation(session, metadata) if session in EXPECTED_SESSIONS else "UNKNOWN_SESSION"
    if mask is not None and mask.shape != EXPECTED_SHAPE:
        provenance["usable_for_primary_roi_decoding"] = "NO"
        provenance["usable_for_exploratory_roi_decoding"] = "NO"
        provenance["exclusion_reason"] = "SHAPE_MISMATCH"
    elif mask is not None and (not mask.any() or mask.all()):
        provenance["usable_for_primary_roi_decoding"] = "NO"
        provenance["usable_for_exploratory_roi_decoding"] = "NO"
        provenance["exclusion_reason"] = "EMPTY_OR_FULL_MASK"
    if orientation == "ORIENTATION_MISMATCH_OR_UNCERTAIN":
        provenance["usable_for_primary_roi_decoding"] = "NO"
        if provenance["classification"] == CLASS_EXPERT:
            provenance["usable_for_exploratory_roi_decoding"] = "NO"
        provenance["exclusion_reason"] = "ORIENTATION_MISMATCH_OR_UNCERTAIN"
    session_specific = _truth(_lookup(metadata, "session_specific", default=""))
    if session_specific == "UNKNOWN":
        session_specific = "YES" if session in EXPECTED_SESSIONS and provenance["transferred_from_session"] == "NONE" else "UNKNOWN"
    row: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_path": str(path.resolve()),
        "file_type": path.suffix.lower(),
        "artifact_kind": item.kind,
        "session": session,
        "session_inferred_or_explicit": session_basis,
        "mask_shape": shape,
        "expected_shape": "128x501",
        "shape_valid": shape_valid,
        "mask_area_pixels": area,
        "mask_fraction": fraction,
        "mask_sha256": mask_hash,
        "file_sha256": file_sha256(path),
        "source_image_path": source_path,
        "source_image_type": source_type,
        "annotation_type": _lookup(metadata, "annotation_type", "annotation_version", default="UNKNOWN"),
        "annotator": _lookup(metadata, "annotator", "analyst", default="UNKNOWN"),
        "creation_script": _lookup(
            metadata, "creation_script", "script",
            default="src/ultrasound_decoding/roi_annotation.py" if "candidate_roi" in path.as_posix().lower() else "UNKNOWN",
        ),
        "creation_config": _lookup(metadata, "creation_config", "config", default="UNKNOWN"),
        "creation_date_if_known": _lookup(
            metadata, "creation_date", "created_at", "annotation_timestamp_utc",
            default=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        ),
        "session_specific": session_specific,
        "orientation_state": orientation,
        "notes": load_note,
        "overlay_path": "",
        **provenance,
    }
    return row, mask


def detect_duplicate_masks(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> list[dict[str, str]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        digest = str(row.get("mask_sha256", ""))
        if digest:
            groups.setdefault(digest, []).append(row)
    duplicates: list[dict[str, str]] = []
    for digest, members in groups.items():
        sessions = sorted({str(row["session"]) for row in members if str(row["session"]) != "UNKNOWN"})
        if len(sessions) < 2:
            continue
        first = arrays[str(members[0]["artifact_id"])]
        same_array = all(np.array_equal(first, arrays[str(row["artifact_id"])]) for row in members[1:])
        file_hashes = {str(row.get("file_sha256", "")) for row in members}
        duplicates.append({
            "finding": "IDENTICAL_MASK_ACROSS_SESSIONS",
            "artifact_ids": ";".join(str(row["artifact_id"]) for row in members),
            "sessions": ";".join(sessions),
            "paths": ";".join(str(row["artifact_path"]) for row in members),
            "same_hash": "YES" if len(file_hashes) == 1 else "NO",
            "same_array": "YES" if same_array else "NO",
            "mask_array_sha256": digest,
            "notes": "Manual review required; equality alone does not establish an error.",
        })
    return duplicates


def _safe_background_path(path: Path) -> bool:
    lower = path.as_posix().lower()
    prohibited = ("mask", "overlay", "accuracy", "hotspot", "fdr", "glm", "beta", "attribution", "prediction")
    return not any(token in lower for token in prohibited)


def find_background(row: Mapping[str, Any], project_root: Path) -> tuple[np.ndarray | None, str]:
    candidates: list[tuple[Path, str]] = []
    source = str(row.get("source_image_path", ""))
    if source not in {"", "UNKNOWN"}:
        candidates.append((Path(source), "ORIGINAL_ANNOTATION_IMAGE"))
    artifact_path = Path(str(row["artifact_path"]))
    for name in ("mean_image.npy", "mean_fus_background.npy", "mean_image.png", "mean_image_enhanced.png"):
        candidates.append((artifact_path.parent / name, "LOCAL_STRUCTURAL_BACKGROUND"))
    session = str(row.get("session", ""))
    if session in EXPECTED_SESSIONS:
        patterns = (
            f"**/session_{session}/mean_image.npy", f"**/session_{session}/mean_fus_background.npy",
            f"**/session_{session}/mean_image.png", f"**/session_{session}*background*.png",
        )
        for pattern in patterns:
            for candidate in sorted(project_root.glob(pattern)):
                candidates.append((candidate, "BACKGROUND_NOT_ORIGINAL_ANNOTATION_IMAGE"))
    seen: set[Path] = set()
    for candidate, status in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file() or not _safe_background_path(resolved):
            continue
        seen.add(resolved)
        image = _load_image(resolved)
        if image is not None and image.shape == EXPECTED_SHAPE:
            return image.astype(float), f"{status}:{resolved}"
    return None, "BACKGROUND_NOT_FOUND"


def _scaled(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=float)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - low) / (high - low), 0, 1)


def make_overlay(mask: np.ndarray, background: np.ndarray | None, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg = np.zeros(mask.shape, dtype=float) if background is None else _scaled(background)
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto")
    ax.contour(mask.astype(float), levels=[0.5], colors=["#ff3b30"], linewidths=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _rank(row: Mapping[str, Any]) -> tuple[int, int, str]:
    order = {
        CLASS_EXPERT: 0,
        CLASS_RECONSTRUCTED: 1,
        CLASS_RECONSTRUCTED_SEARCHLIGHT: 2,
        CLASS_CIRCULAR: 3,
        CLASS_TRANSFER: 4,
        CLASS_UNKNOWN: 5,
    }
    valid = 0 if row.get("shape_valid") == "YES" else 1
    return order.get(str(row.get("classification")), 99), valid, str(row.get("artifact_path"))


def build_session_status(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        members = [row for row in rows if str(row.get("session")) == session]
        valid_masks = [row for row in members if row.get("mask_sha256")]
        best_pool = valid_masks or members
        best = sorted(best_pool, key=_rank)[0] if best_pool else None
        has_valid_mask = bool(best and best.get("mask_sha256") and best.get("shape_valid") == "YES")
        primary = bool(has_valid_mask and best and best.get("usable_for_primary_roi_decoding") == "YES")
        exploratory = bool(has_valid_mask and best and best.get("usable_for_exploratory_roi_decoding") == "YES")
        result.append({
            "session": session,
            "n_roi_artifacts": len(members),
            "best_available_roi_path": best["artifact_path"] if best else "MISSING",
            "best_available_classification": best["classification"] if best else CLASS_MISSING,
            "provenance_status": best["provenance_known"] if best else "MISSING",
            "primary_roi_available": "YES" if primary else "NO",
            "exploratory_roi_available": "YES" if exploratory else "NO",
            "needs_expert_annotation": "NO" if primary else "YES",
            "reason": "PRIMARY_USABLE" if primary else (str(best.get("exclusion_reason")) if best else "MISSING"),
        })
    return result


def make_overview(
    rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], project_root: Path, output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
    for ax, session in zip(axes.flat, EXPECTED_SESSIONS):
        members = [row for row in rows if str(row.get("session")) == session and str(row.get("artifact_id")) in arrays]
        if not members:
            ax.text(0.5, 0.5, "NO ROI FOUND", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(session)
            ax.set_axis_off()
            continue
        best = sorted(members, key=_rank)[0]
        mask = arrays[str(best["artifact_id"])]
        if mask.shape != EXPECTED_SHAPE:
            ax.text(0.5, 0.5, f"NO VALID MASK\nshape={mask.shape}", ha="center", va="center", transform=ax.transAxes)
        else:
            background, _ = find_background(best, project_root)
            bg = np.zeros(EXPECTED_SHAPE) if background is None else _scaled(background)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto")
            ax.contour(mask.astype(float), levels=[0.5], colors=["#ff3b30"], linewidths=1.0)
        ax.set_title(f"{session}\n{best['classification']}", fontsize=7)
        ax.set_axis_off()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _decision(status: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    missing_primary = [str(row["session"]) for row in status if row["primary_roi_available"] != "YES"]
    if not missing_primary:
        return "Decision R-A: READY_FOR_9SESSION_PRIMARY_ROI_DECODING", []
    credible = [row for row in status if row["best_available_classification"] in {
        CLASS_EXPERT, CLASS_RECONSTRUCTED, CLASS_RECONSTRUCTED_SEARCHLIGHT,
    }]
    if credible:
        return "Decision R-B: PARTIAL_ROI_ONLY_NEED_EXPERT_ANNOTATION", missing_primary
    return "Decision R-C: CURRENT_ROIS_NOT_PRIMARY_USABLE", missing_primary


def make_report(rows: Sequence[Mapping[str, Any]], status: Sequence[Mapping[str, Any]], duplicates: Sequence[Mapping[str, Any]]) -> str:
    primary = sum(row["primary_roi_available"] == "YES" for row in status)
    exploratory = sum(row["primary_roi_available"] != "YES" and row["exploratory_roi_available"] == "YES" for row in status)
    unusable = sum(row["best_available_classification"] not in {CLASS_MISSING, CLASS_EXPERT, CLASS_RECONSTRUCTED, CLASS_RECONSTRUCTED_SEARCHLIGHT} for row in status)
    missing = sum(row["best_available_classification"] == CLASS_MISSING for row in status)
    decision, need = _decision(status)
    lines = [
        "# ROI artifact and provenance audit v1", "",
        "This is a read-only provenance audit. It did not create, modify, resize, register, or decode any ROI.", "",
        "## Nine-session summary", "",
        f"- Primary-usable ROI: {primary}/9",
        f"- Exploratory-only ROI: {exploratory}/9",
        f"- Unusable/circular/unknown ROI: {unusable}/9",
        f"- Missing ROI: {missing}/9", "",
    ]
    for stat in status:
        members = [row for row in rows if str(row.get("session")) == str(stat["session"])]
        best = sorted([row for row in members if row.get("mask_sha256")] or members, key=_rank)[0] if members else None
        lines.extend([
            f"## Session {stat['session']}", "",
            f"- artifacts found: {len(members)}",
            f"- best ROI: {stat['best_available_roi_path']}",
            f"- provenance: {stat['best_available_classification']}",
            f"- label independence: {best['label_information_used'] if best else 'MISSING'}",
            f"- session-specific: {best['session_specific'] if best else 'MISSING'}",
            f"- orientation: {best['orientation_state'] if best else ('flip_vertical required; ROI missing' if stat['session'] == '807' else 'ROI missing')}",
            f"- usable for primary decoding: {stat['primary_roi_available']}",
            f"- reason: {stat['reason']}", "",
        ])
    lines.extend([
        "## Duplicate-mask audit", "",
        f"Identical cross-session mask groups: {len(duplicates)}.", "",
        "## Final decision", "", decision, "",
    ])
    if need:
        lines.append("Sessions requiring new independent expert annotation: " + ", ".join(need) + ".")
    return "\n".join(lines) + "\n"


def filesystem_signature(project_root: Path, output_dir: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in iter_project_files(project_root, output_dir)
    }


def expected_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / "roi_artifact_inventory.csv",
        output_dir / "session_roi_status.csv",
        output_dir / "audit/duplicate_mask_audit.csv",
        output_dir / "audit/shape_orientation_audit.csv",
        output_dir / "audit/recursive_search_evidence.csv",
        output_dir / "figures/roi_overlay_9sessions.png",
        output_dir / "roi_provenance_report.md",
    ]


def run_audit(project_root: Path, output_dir: Path, device: str = "cpu") -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    if device != "cpu":
        raise ValueError("ROI artifact audit is CPU-only")
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("output_dir must be inside project_root") from exc
    before = filesystem_signature(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(exist_ok=True)
    (output_dir / "figures/overlays").mkdir(parents=True, exist_ok=True)

    discovered = discover_roi_artifacts(project_root, output_dir)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, item in enumerate(discovered, start=1):
        artifact_id = f"ROI-{index:05d}"
        row, mask = inspect_artifact(item, project_root, artifact_id)
        if mask is not None:
            arrays[artifact_id] = mask
            if mask.shape == EXPECTED_SHAPE:
                overlay = output_dir / "figures/overlays" / f"{artifact_id}.png"
                background, background_status = find_background(row, project_root)
                make_overlay(mask, background, f"{row['session']} | {row['classification']} | {artifact_id}", overlay)
                row["overlay_path"] = str(overlay)
                row["notes"] = f"{row['notes']};{background_status}"
        rows.append(row)

    duplicates = detect_duplicate_masks(rows, arrays)
    status = build_session_status(rows)
    evidence = recursive_search_evidence(project_root, output_dir)
    _write_csv(output_dir / "roi_artifact_inventory.csv", rows, INVENTORY_FIELDS)
    _write_csv(output_dir / "session_roi_status.csv", status, (
        "session", "n_roi_artifacts", "best_available_roi_path", "best_available_classification",
        "provenance_status", "primary_roi_available", "exploratory_roi_available",
        "needs_expert_annotation", "reason",
    ))
    _write_csv(output_dir / "audit/duplicate_mask_audit.csv", duplicates, (
        "finding", "artifact_ids", "sessions", "paths", "same_hash", "same_array", "mask_array_sha256", "notes",
    ))
    shape_rows = [{key: row[key] for key in (
        "artifact_id", "artifact_path", "session", "mask_shape", "expected_shape", "shape_valid", "orientation_state", "exclusion_reason",
    )} for row in rows]
    _write_csv(output_dir / "audit/shape_orientation_audit.csv", shape_rows, (
        "artifact_id", "artifact_path", "session", "mask_shape", "expected_shape", "shape_valid", "orientation_state", "exclusion_reason",
    ))
    _write_csv(output_dir / "audit/recursive_search_evidence.csv", evidence, (
        "path", "file_type", "filename_keyword_hits", "content_keyword_hits",
    ))
    make_overview(rows, arrays, project_root, output_dir / "figures/roi_overlay_9sessions.png")
    (output_dir / "roi_provenance_report.md").write_text(make_report(rows, status, duplicates), encoding="utf-8")

    missing = [str(path) for path in expected_outputs(output_dir) if not path.is_file()]
    if missing:
        raise AssertionError(f"output completeness failure: {missing}")
    after = filesystem_signature(project_root, output_dir)
    if before != after:
        changed = sorted(set(before) ^ set(after))
        modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
        raise RuntimeError(f"read-only guard failed; changed={changed}; modified={modified}")
    decision, need = _decision(status)
    return {
        "n_artifacts": len(rows),
        "n_valid_masks": len(arrays),
        "n_duplicate_groups": len(duplicates),
        "decision": decision,
        "sessions_requiring_annotation": need,
        "outputs": [str(path) for path in expected_outputs(output_dir)],
    }
