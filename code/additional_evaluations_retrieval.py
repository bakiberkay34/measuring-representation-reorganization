from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:
    display = None

EXTRACT_DIR = Path(os.environ.get("ACTIVATION_CACHE_DIR", "results_vlm_feature_geometry_h100/activations")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("ADDITIONAL_EVALUATION_OUTPUT_DIR", "additional_evaluation_outputs")).expanduser().resolve()
AUDIT_DIR = OUTPUT_ROOT / "audit"
RESULTS_DIR = OUTPUT_ROOT / "results"
INTERMEDIATE_DIR = OUTPUT_ROOT / "intermediate"

for directory in (AUDIT_DIR, RESULTS_DIR, INTERMEDIATE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not EXTRACT_DIR.exists():
    raise FileNotFoundError(f"Activation cache directory not found: {EXTRACT_DIR}")

CANONICAL_CACHE_READY = False


@dataclass(frozen=True)
class ExperimentContract:
    schema_version: str = "activation_cache_contract_v1"

    primary_pooling: str = "final_quarter"

    expected_families: tuple[str, ...] = (
        "pali",
        "qwen",
        "smol",
    )

    expected_datasets: tuple[str, ...] = (
        "coco",
        "flickr30k",
        "pixelprose",
    )

    expected_poolings: tuple[str, ...] = (
        "all_valid",
        "final_quarter",
        "last_token",
    )

    expected_depth_count: int = 8
    expected_train_examples: int = 1024
    expected_eval_examples: int = 512

    required_array_keys: tuple[str, ...] = (
        "llm_train",
        "vlm_train",
        "vlm_eval_matched",
        "vlm_eval_text_swapped",
        "vlm_eval_image_swapped",
        "vlm_eval_image_only",
        "depth_fractions",
        "layers",
        "llm_layers",
        "vlm_layers",
        "train_examples",
        "eval_examples",
        "family",
        "dataset",
        "pooling",
    )

    activation_keys: tuple[str, ...] = (
        "llm_train",
        "vlm_train",
        "vlm_eval_matched",
        "vlm_eval_text_swapped",
        "vlm_eval_image_swapped",
        "vlm_eval_image_only",
    )

    finite_check_primary_only: bool = True
    finite_check_example_chunk: int = 128

    random_seed: int = 20260720


CONTRACT = ExperimentContract()


FILE_PATTERN = re.compile(
    r"^exp1_act_"
    r"(?P<family>pali|qwen|smol)_"
    r"(?P<dataset>coco_karpathy|flickr30k|pixelprose)_"
    r"(?P<pooling>all_valid|final_quarter|last_token)"
    r"\.npz$",
    flags=re.IGNORECASE,
)


def canonical_dataset(raw_value: str) -> str:
    raw_value = str(raw_value).lower()

    if raw_value == "coco_karpathy":
        return "coco"

    return raw_value


def parse_cache_filename(path: Path) -> dict[str, Any]:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        raise ValueError(
            f"Unexpected cache filename: {path.name}"
        )

    values = match.groupdict()

    return {
        "family": values["family"].lower(),
        "dataset": canonical_dataset(
            values["dataset"]
        ),
        "pooling": values["pooling"].lower(),
    }


all_npz_paths = sorted(
    path for path in EXTRACT_DIR.rglob("*.npz")
    if path.is_file()
)

if len(all_npz_paths) != 27:
    raise RuntimeError(
        "The canonical contract requires exactly 27 NPZ files, "
        f"but found {len(all_npz_paths)}."
    )

index_rows: list[dict[str, Any]] = []

for path in all_npz_paths:
    parsed = parse_cache_filename(path)

    index_rows.append(
        {
            **parsed,
            "path": str(path.resolve()),
            "relative_path": str(
                path.relative_to(EXTRACT_DIR)
            ),
            "size_bytes": int(path.stat().st_size),
        }
    )

CACHE_INDEX = pd.DataFrame(index_rows).sort_values(
    ["family", "dataset", "pooling"],
    ignore_index=True,
)

duplicate_count = int(
    CACHE_INDEX.duplicated(
        ["family", "dataset", "pooling"],
        keep=False,
    ).sum()
)

if duplicate_count:
    raise RuntimeError(
        "Duplicate family–dataset–pooling cache cells detected."
    )

expected_cells = {
    (family, dataset, pooling)
    for family in CONTRACT.expected_families
    for dataset in CONTRACT.expected_datasets
    for pooling in CONTRACT.expected_poolings
}

observed_cells = set(
    zip(
        CACHE_INDEX["family"],
        CACHE_INDEX["dataset"],
        CACHE_INDEX["pooling"],
    )
)

if observed_cells != expected_cells:
    raise RuntimeError(
        "Cache coverage differs from the frozen 3×3×3 contract.\n"
        f"Missing: {sorted(expected_cells - observed_cells)}\n"
        f"Unexpected: {sorted(observed_cells - expected_cells)}"
    )

PRIMARY_CACHE_INDEX = CACHE_INDEX[
    CACHE_INDEX["pooling"] == CONTRACT.primary_pooling
].copy().reset_index(drop=True)

if len(PRIMARY_CACHE_INDEX) != 9:
    raise RuntimeError(
        "Exactly nine final-quarter primary files are required."
    )


def scalar_string(array: np.ndarray) -> str:
    values = np.asarray(array).reshape(-1)

    if values.size != 1:
        raise ValueError(
            f"Expected one scalar string, got shape {array.shape}."
        )

    return str(values[0])


def scalar_integer(array: np.ndarray) -> int:
    values = np.asarray(array).reshape(-1)

    if values.size != 1:
        raise ValueError(
            f"Expected one scalar integer, got shape {array.shape}."
        )

    return int(values[0])


def array_is_finite_in_chunks(
    array: np.ndarray,
    example_chunk: int,
) -> tuple[bool, int]:
    if array.ndim != 3:
        raise ValueError(
            f"Expected a rank-3 activation array, got {array.shape}."
        )

    nonfinite_count = 0

    for start in range(0, array.shape[0], example_chunk):
        end = min(
            start + example_chunk,
            array.shape[0],
        )

        finite_mask = np.isfinite(array[start:end])

        if not bool(finite_mask.all()):
            nonfinite_count += int(
                finite_mask.size
                - np.count_nonzero(finite_mask)
            )

        del finite_mask

    return nonfinite_count == 0, nonfinite_count


audit_rows: list[dict[str, Any]] = []
reference_depths_by_cell: dict[
    tuple[str, str],
    np.ndarray
] = {}

print("=" * 72)
print("PRIMARY CACHE CONTRACT AUDIT")
print("=" * 72)
print(f"Primary pooling: {CONTRACT.primary_pooling}")
print(f"Primary files:   {len(PRIMARY_CACHE_INDEX)}")
print()

for file_number, row in PRIMARY_CACHE_INDEX.iterrows():
    path = Path(row["path"])

    print(
        f"[{file_number + 1:02d}/09] "
        f"{row['family']} / {row['dataset']} / "
        f"{row['pooling']}"
    )

    with np.load(path, allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        required_keys = set(CONTRACT.required_array_keys)

        missing_keys = sorted(
            required_keys - actual_keys
        )

        unexpected_keys = sorted(
            actual_keys - required_keys
        )

        if missing_keys:
            raise RuntimeError(
                f"{path.name} is missing keys: {missing_keys}"
            )

        if unexpected_keys:
            raise RuntimeError(
                f"{path.name} contains unexpected keys: "
                f"{unexpected_keys}"
            )

        metadata_family = scalar_string(
            archive["family"]
        ).lower()

        metadata_dataset = canonical_dataset(
            scalar_string(archive["dataset"])
        )

        metadata_pooling = scalar_string(
            archive["pooling"]
        ).lower()

        if metadata_family != row["family"]:
            raise RuntimeError(
                f"Family metadata mismatch in {path.name}: "
                f"{metadata_family} != {row['family']}"
            )

        if metadata_dataset != row["dataset"]:
            raise RuntimeError(
                f"Dataset metadata mismatch in {path.name}: "
                f"{metadata_dataset} != {row['dataset']}"
            )

        if metadata_pooling != row["pooling"]:
            raise RuntimeError(
                f"Pooling metadata mismatch in {path.name}: "
                f"{metadata_pooling} != {row['pooling']}"
            )

        train_examples = scalar_integer(
            archive["train_examples"]
        )

        eval_examples = scalar_integer(
            archive["eval_examples"]
        )

        if train_examples != CONTRACT.expected_train_examples:
            raise RuntimeError(
                f"Unexpected train count in {path.name}: "
                f"{train_examples}"
            )

        if eval_examples != CONTRACT.expected_eval_examples:
            raise RuntimeError(
                f"Unexpected eval count in {path.name}: "
                f"{eval_examples}"
            )

        depth_fractions = np.asarray(
            archive["depth_fractions"],
            dtype=np.float64,
        )

        llm_layers = np.asarray(
            archive["llm_layers"],
            dtype=np.int64,
        )

        vlm_layers = np.asarray(
            archive["vlm_layers"],
            dtype=np.int64,
        )

        generic_layers = np.asarray(
            archive["layers"],
            dtype=np.int64,
        )

        expected_meta_shape = (
            CONTRACT.expected_depth_count,
        )

        for name, values in {
            "depth_fractions": depth_fractions,
            "llm_layers": llm_layers,
            "vlm_layers": vlm_layers,
            "layers": generic_layers,
        }.items():
            if values.shape != expected_meta_shape:
                raise RuntimeError(
                    f"{name} in {path.name} has shape "
                    f"{values.shape}; expected {expected_meta_shape}."
                )

        if not bool(np.isfinite(depth_fractions).all()):
            raise RuntimeError(
                f"Non-finite depth fractions in {path.name}."
            )

        if not bool(
            np.all(np.diff(depth_fractions) > 0)
        ):
            raise RuntimeError(
                f"Depth fractions are not strictly increasing "
                f"in {path.name}."
            )

        if (
            float(depth_fractions.min()) < 0.0
            or float(depth_fractions.max()) > 1.0
        ):
            raise RuntimeError(
                f"Depth fractions fall outside [0,1] "
                f"in {path.name}."
            )

        cell_key = (
            row["family"],
            row["dataset"],
        )

        reference_depths_by_cell[
            cell_key
        ] = depth_fractions.copy()

        activation_shapes: dict[str, tuple[int, ...]] = {}
        hidden_dimensions: dict[str, int] = {}
        nonfinite_counts: dict[str, int] = {}

        for key in CONTRACT.activation_keys:
            activation = np.asarray(
                archive[key]
            )

            if activation.dtype != np.float32:
                raise RuntimeError(
                    f"{key} in {path.name} uses dtype "
                    f"{activation.dtype}; expected float32."
                )

            expected_examples = (
                CONTRACT.expected_train_examples
                if key in {"llm_train", "vlm_train"}
                else CONTRACT.expected_eval_examples
            )

            if activation.ndim != 3:
                raise RuntimeError(
                    f"{key} in {path.name} has rank "
                    f"{activation.ndim}; expected rank 3."
                )

            if activation.shape[0] != expected_examples:
                raise RuntimeError(
                    f"{key} in {path.name} has "
                    f"{activation.shape[0]} examples; expected "
                    f"{expected_examples}."
                )

            if (
                activation.shape[1]
                != CONTRACT.expected_depth_count
            ):
                raise RuntimeError(
                    f"{key} in {path.name} has "
                    f"{activation.shape[1]} depth positions; "
                    f"expected {CONTRACT.expected_depth_count}."
                )

            if activation.shape[2] <= 0:
                raise RuntimeError(
                    f"{key} in {path.name} has invalid "
                    f"hidden dimension {activation.shape[2]}."
                )

            finite_pass, nonfinite_count = (
                array_is_finite_in_chunks(
                    activation,
                    CONTRACT.finite_check_example_chunk,
                )
            )

            if not finite_pass:
                raise RuntimeError(
                    f"{key} in {path.name} contains "
                    f"{nonfinite_count} NaN/Inf values."
                )

            activation_shapes[key] = tuple(
                int(value)
                for value in activation.shape
            )

            hidden_dimensions[key] = int(
                activation.shape[2]
            )

            nonfinite_counts[key] = int(
                nonfinite_count
            )

            del activation
            gc.collect()

        unique_hidden_dimensions = set(
            hidden_dimensions.values()
        )

        if len(unique_hidden_dimensions) != 1:
            raise RuntimeError(
                f"Hidden dimensions differ across arrays "
                f"in {path.name}: {hidden_dimensions}"
            )

        hidden_dimension = next(
            iter(unique_hidden_dimensions)
        )

        audit_rows.append(
            {
                "family": row["family"],
                "dataset": row["dataset"],
                "pooling": row["pooling"],
                "path": str(path),
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "depth_count": len(depth_fractions),
                "depth_fractions": depth_fractions.tolist(),
                "llm_layers": llm_layers.tolist(),
                "vlm_layers": vlm_layers.tolist(),
                "layers": generic_layers.tolist(),
                "hidden_dimension": hidden_dimension,
                "all_required_keys_present": True,
                "no_unexpected_keys": True,
                "all_finite": True,
                "nonfinite_counts": nonfinite_counts,
                "activation_shapes": activation_shapes,
            }
        )

    gc.collect()


PRIMARY_AUDIT = pd.DataFrame(audit_rows)

if len(PRIMARY_AUDIT) != 9:
    raise RuntimeError(
        "Primary audit did not produce nine valid records."
    )


pooling_consistency_rows: list[dict[str, Any]] = []

for (
    family,
    dataset,
), group in CACHE_INDEX.groupby(
    ["family", "dataset"],
    sort=True,
):
    records: list[dict[str, Any]] = []

    for _, file_row in group.sort_values(
        "pooling"
    ).iterrows():
        path = Path(file_row["path"])

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            records.append(
                {
                    "pooling": file_row["pooling"],
                    "depth_fractions": np.asarray(
                        archive["depth_fractions"],
                        dtype=np.float64,
                    ),
                    "llm_layers": np.asarray(
                        archive["llm_layers"],
                        dtype=np.int64,
                    ),
                    "vlm_layers": np.asarray(
                        archive["vlm_layers"],
                        dtype=np.int64,
                    ),
                    "train_examples": scalar_integer(
                        archive["train_examples"]
                    ),
                    "eval_examples": scalar_integer(
                        archive["eval_examples"]
                    ),
                }
            )

    reference = records[0]

    metadata_identical = all(
        np.array_equal(
            record["depth_fractions"],
            reference["depth_fractions"],
        )
        and np.array_equal(
            record["llm_layers"],
            reference["llm_layers"],
        )
        and np.array_equal(
            record["vlm_layers"],
            reference["vlm_layers"],
        )
        and record["train_examples"]
        == reference["train_examples"]
        and record["eval_examples"]
        == reference["eval_examples"]
        for record in records[1:]
    )

    pooling_consistency_rows.append(
        {
            "family": family,
            "dataset": dataset,
            "poolings_seen": sorted(
                record["pooling"]
                for record in records
            ),
            "metadata_identical_across_poolings": (
                metadata_identical
            ),
        }
    )


POOLING_METADATA_AUDIT = pd.DataFrame(
    pooling_consistency_rows
)

if not bool(
    POOLING_METADATA_AUDIT[
        "metadata_identical_across_poolings"
    ].all()
):
    raise RuntimeError(
        "Layer/depth/example metadata differ across pooling "
        "variants for at least one family–dataset cell."
    )


cache_index_path = (
    AUDIT_DIR / "canonical_cache_index.csv"
)

primary_audit_path = (
    AUDIT_DIR / "primary_numerical_audit.json"
)

pooling_audit_path = (
    AUDIT_DIR / "pooling_metadata_audit.csv"
)



contract_path = (
    AUDIT_DIR / "experiment_contract.json"
)

CACHE_INDEX.to_csv(
    cache_index_path,
    index=False,
)

POOLING_METADATA_AUDIT.to_csv(
    pooling_audit_path,
    index=False,
)


primary_audit_path.write_text(
    json.dumps(
        audit_rows,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


contract_path.write_text(
    json.dumps(
        asdict(CONTRACT),
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


CANONICAL_CACHE_READY = bool(
    len(CACHE_INDEX) == 27
    and len(PRIMARY_CACHE_INDEX) == 9
    and bool(PRIMARY_AUDIT["all_finite"].all())
    and bool(
        POOLING_METADATA_AUDIT[
            "metadata_identical_across_poolings"
        ].all()
    )
)

print("\n" + "=" * 72)
print("CANONICAL CACHE CONTRACT SUMMARY")
print("=" * 72)
print(f"Full cache cells:          {len(CACHE_INDEX)} / 27")
print(f"Primary cache cells:       {len(PRIMARY_CACHE_INDEX)} / 9")
print(
    "Primary numerical audit: "
    f"{'PASS' if bool(PRIMARY_AUDIT['all_finite'].all()) else 'FAIL'}"
)
print(
    "Pooling metadata audit:  "
    f"{'PASS' if bool(POOLING_METADATA_AUDIT['metadata_identical_across_poolings'].all()) else 'FAIL'}"
)
print(f"Canonical cache ready:     {CANONICAL_CACHE_READY}")

print("\nPrimary activation cells:")
if display is not None:
    display(
        PRIMARY_AUDIT[
            [
                "family",
                "dataset",
                "pooling",
                "train_examples",
                "eval_examples",
                "depth_count",
                "hidden_dimension",
                "all_finite",
            ]
        ].sort_values(
            ["family", "dataset"]
        )
    )
else:
    print(
        PRIMARY_AUDIT[
            [
                "family",
                "dataset",
                "pooling",
                "train_examples",
                "eval_examples",
                "depth_count",
                "hidden_dimension",
                "all_finite",
            ]
        ].sort_values(
            ["family", "dataset"]
        ).to_string(index=False)
    )

print("\nGenerated audit files:")
print(f"1. {contract_path}")
print(f"2. {cache_index_path}")
print(f"3. {primary_audit_path}")
print(f"4. {pooling_audit_path}")

if not CANONICAL_CACHE_READY:
    raise RuntimeError(
        "Canonical cache contract did not pass. "
        "Do not run downstream experiments."
    )

print("\nActivation cache audit completed successfully.")

PRIMARY_VALIDATION_READY = False

if globals().get("CANONICAL_CACHE_READY") is not True:
    raise RuntimeError(
        "The activation cache has not passed the canonical validation gate."
    )

_REQUIRED_OBJECTS = (
    "PRIMARY_CACHE_INDEX",
    "CONTRACT",
    "AUDIT_DIR",
    "RESULTS_DIR",
    "INTERMEDIATE_DIR",
)

_missing_objects = [
    name
    for name in _REQUIRED_OBJECTS
    if name not in globals()
]

if _missing_objects:
    raise RuntimeError(
        "Required cache-validation objects are unavailable: "
        + ", ".join(_missing_objects)
    )

AUDIT_DIR = Path(AUDIT_DIR).expanduser().resolve()
RESULTS_DIR = Path(RESULTS_DIR).expanduser().resolve()
INTERMEDIATE_DIR = Path(
    INTERMEDIATE_DIR
).expanduser().resolve()

for _directory in (
    AUDIT_DIR,
    RESULTS_DIR,
    INTERMEDIATE_DIR,
):
    _directory.mkdir(
        parents=True,
        exist_ok=True,
    )


ANALYSIS_VERSION = "primary_validation_v2"

PRIMARY_POOLING = "final_quarter"
GLOBAL_SEED = 20260720

ALIGNMENT_PCA_RANK = 64

RETRIEVAL_FOLDS = 4
RETRIEVAL_CANDIDATE_POOL = 256
RETRIEVAL_K_VALUES = (
    10,
    20,
    50,
)

CLASSIFICATION_FOLDS = 5
CLASSIFICATION_PCA_RANK = 64
CLASSIFICATION_C = 1.0
CLASSIFICATION_MAX_ITER = 3000
CLASSIFICATION_RETRY_MAX_ITER = 10000

CONDITION_NAMES = (
    "matched",
    "text_swapped",
    "image_swapped",
    "image_only",
)

CONDITION_KEYS = (
    "vlm_eval_matched",
    "vlm_eval_text_swapped",
    "vlm_eval_image_swapped",
    "vlm_eval_image_only",
)

NULL_CONTROL_TARGET_DEPTH = 0.75

BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 10000

EPSILON = 1e-8
TIE_TOLERANCE = 1e-7

RUN_CROSS_DATASET_TRANSFER = True


EVALUATION_ROOT = (
    RESULTS_DIR
    / "primary_validation"
)

RETRIEVAL_CHECKPOINT_DIR = (
    EVALUATION_ROOT
    / "checkpoints"
    / "paired_retrieval"
)

CLASSIFICATION_CHECKPOINT_DIR = (
    EVALUATION_ROOT
    / "checkpoints"
    / "condition_classification"
)

MISMATCH_CHECKPOINT_DIR = (
    EVALUATION_ROOT
    / "checkpoints"
    / "dense_mismatch"
)

CROSS_DATASET_CHECKPOINT_DIR = (
    EVALUATION_ROOT
    / "checkpoints"
    / "cross_dataset_retrieval"
)

for _directory in (
    EVALUATION_ROOT,
    RETRIEVAL_CHECKPOINT_DIR,
    CLASSIFICATION_CHECKPOINT_DIR,
    MISMATCH_CHECKPOINT_DIR,
    CROSS_DATASET_CHECKPOINT_DIR,
):
    _directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def stable_seed(*parts: Any) -> int:
    payload = "||".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_builtin(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def atomic_json_write(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            to_builtin(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def atomic_csv_write(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def read_completed_json(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    if payload.get("completed") is not True:
        return None

    if (
        payload.get("analysis_version")
        != ANALYSIS_VERSION
    ):
        return None

    return payload


def unit_filename(
    *parts: Any,
) -> str:
    safe_parts = []

    for part in parts:
        value = str(part)

        value = (
            value
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(".", "p")
        )

        safe_parts.append(value)

    return (
        "__".join(safe_parts)
        + ".json"
    )


def require_finite(
    name: str,
    array: np.ndarray,
) -> None:
    finite = np.isfinite(array)

    if not bool(finite.all()):
        nonfinite_count = int(
            finite.size
            - np.count_nonzero(finite)
        )

        raise FloatingPointError(
            f"{name} contains "
            f"{nonfinite_count} "
            "NaN or infinite values."
        )


def row_l2_normalize(
    array: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    values = np.asarray(
        array,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    normalized = (
        values
        / np.maximum(
            norms,
            epsilon,
        )
    )

    require_finite(
        "row-normalized array",
        normalized,
    )

    return normalized.astype(
        np.float32,
        copy=False,
    )


def harmonic_number(
    number: int,
) -> float:
    return float(
        sum(
            1.0 / index
            for index in range(
                1,
                number + 1,
            )
        )
    )


def make_nearly_balanced_fold_assignment(
    item_count: int,
    fold_count: int,
    seed: int,
) -> np.ndarray:

    if item_count < fold_count:
        raise ValueError(
            f"Cannot divide {item_count} "
            f"items into {fold_count} "
            "nonempty folds."
        )

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(
        item_count
    )

    assignment = np.empty(
        item_count,
        dtype=np.int64,
    )

    assignment[permutation] = (
        np.arange(
            item_count,
            dtype=np.int64,
        )
        % fold_count
    )

    fold_sizes = np.bincount(
        assignment,
        minlength=fold_count,
    )

    if int(fold_sizes.min()) <= 0:
        raise RuntimeError(
            "At least one fold is empty: "
            f"{fold_sizes.tolist()}"
        )

    if int(
        fold_sizes.max()
        - fold_sizes.min()
    ) > 1:
        raise RuntimeError(
            "Fold sizes differ by more than one: "
            f"{fold_sizes.tolist()}"
        )

    return assignment


_fold_test_classification = np.bincount(
    make_nearly_balanced_fold_assignment(
        item_count=512,
        fold_count=5,
        seed=GLOBAL_SEED,
    ),
    minlength=5,
)

if sorted(
    _fold_test_classification.tolist()
) != [
    102,
    102,
    102,
    103,
    103,
]:
    raise RuntimeError(
        "Five-fold classification "
        "split self-test failed: "
        f"{_fold_test_classification.tolist()}"
    )

_fold_test_retrieval = np.bincount(
    make_nearly_balanced_fold_assignment(
        item_count=1024,
        fold_count=4,
        seed=GLOBAL_SEED,
    ),
    minlength=4,
)

if not bool(
    np.all(
        _fold_test_retrieval
        == 256
    )
):
    raise RuntimeError(
        "Four-fold retrieval "
        "split self-test failed: "
        f"{_fold_test_retrieval.tolist()}"
    )

del (
    _fold_test_classification,
    _fold_test_retrieval,
)


def fit_projection_pair(
    vlm_train: np.ndarray,
    llm_train: np.ndarray,
    vlm_test: np.ndarray,
    llm_test: np.ndarray,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    if (
        vlm_train.shape
        != llm_train.shape
    ):
        raise ValueError(
            "Paired training arrays "
            "must have equal shape."
        )

    if (
        vlm_test.shape
        != llm_test.shape
    ):
        raise ValueError(
            "Paired test arrays "
            "must have equal shape."
        )

    if (
        vlm_train.shape[1]
        != vlm_test.shape[1]
    ):
        raise ValueError(
            "Train and test hidden "
            "dimensions differ."
        )

    vlm_train_norm = (
        row_l2_normalize(
            vlm_train
        )
    )

    llm_train_norm = (
        row_l2_normalize(
            llm_train
        )
    )

    vlm_test_norm = (
        row_l2_normalize(
            vlm_test
        )
    )

    llm_test_norm = (
        row_l2_normalize(
            llm_test
        )
    )

    effective_rank = int(
        min(
            rank,
            vlm_train_norm.shape[0] - 1,
            vlm_train_norm.shape[1],
        )
    )

    if effective_rank < 2:
        raise ValueError(
            "Invalid effective PCA rank: "
            f"{effective_rank}"
        )

    vlm_pca = PCA(
        n_components=effective_rank,
        whiten=True,
        svd_solver="randomized",
        random_state=stable_seed(
            seed,
            "vlm_pca",
        ),
    )

    llm_pca = PCA(
        n_components=effective_rank,
        whiten=True,
        svd_solver="randomized",
        random_state=stable_seed(
            seed,
            "llm_pca",
        ),
    )

    vlm_train_projected = (
        vlm_pca.fit_transform(
            vlm_train_norm
        )
    )

    llm_train_projected = (
        llm_pca.fit_transform(
            llm_train_norm
        )
    )

    vlm_test_projected = (
        vlm_pca.transform(
            vlm_test_norm
        )
    )

    llm_test_projected = (
        llm_pca.transform(
            llm_test_norm
        )
    )

    for name, array in (
        (
            "vlm_train_projected",
            vlm_train_projected,
        ),
        (
            "llm_train_projected",
            llm_train_projected,
        ),
        (
            "vlm_test_projected",
            vlm_test_projected,
        ),
        (
            "llm_test_projected",
            llm_test_projected,
        ),
    ):
        require_finite(
            name,
            array,
        )

    return {
        "vlm_train_projected": (
            vlm_train_projected
        ),
        "llm_train_projected": (
            llm_train_projected
        ),
        "vlm_test_projected": (
            vlm_test_projected
        ),
        "llm_test_projected": (
            llm_test_projected
        ),
        "effective_rank": (
            effective_rank
        ),
        "vlm_pca_explained_variance_sum": float(
            np.sum(
                vlm_pca
                .explained_variance_ratio_
            )
        ),
        "llm_pca_explained_variance_sum": float(
            np.sum(
                llm_pca
                .explained_variance_ratio_
            )
        ),
    }


def orthogonal_rotation(
    source_train: np.ndarray,
    target_train: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    cross_covariance = (
        source_train.T
        @ target_train
    )

    (
        left_vectors,
        singular_values,
        right_vectors_t,
    ) = np.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )

    rotation = (
        left_vectors
        @ right_vectors_t
    )

    require_finite(
        "orthogonal rotation",
        rotation,
    )

    return (
        rotation,
        singular_values,
    )


def mapped_test_representations(
    projection: dict[str, Any],
    seed: int,
    shuffle_training_pairs: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    target_train = projection[
        "llm_train_projected"
    ]

    if shuffle_training_pairs:
        rng = np.random.default_rng(
            stable_seed(
                seed,
                "shuffle_pairs",
            )
        )

        target_train = target_train[
            rng.permutation(
                target_train.shape[0]
            )
        ]

    (
        rotation,
        singular_values,
    ) = orthogonal_rotation(
        projection[
            "vlm_train_projected"
        ],
        target_train,
    )

    mapped_vlm_test = (
        row_l2_normalize(
            projection[
                "vlm_test_projected"
            ]
            @ rotation
        )
    )

    normalized_llm_test = (
        row_l2_normalize(
            projection[
                "llm_test_projected"
            ]
        )
    )

    diagnostics = {
        "effective_rank": int(
            projection[
                "effective_rank"
            ]
        ),
        "cross_covariance_singular_min": float(
            np.min(
                singular_values
            )
        ),
        "cross_covariance_singular_max": float(
            np.max(
                singular_values
            )
        ),
        "vlm_pca_explained_variance_sum": float(
            projection[
                "vlm_pca_explained_variance_sum"
            ]
        ),
        "llm_pca_explained_variance_sum": float(
            projection[
                "llm_pca_explained_variance_sum"
            ]
        ),
    }

    return (
        mapped_vlm_test,
        normalized_llm_test,
        diagnostics,
    )


def no_alignment_test_representations(
    vlm_train: np.ndarray,
    llm_train: np.ndarray,
    vlm_test: np.ndarray,
    llm_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    vlm_mean = np.mean(
        vlm_train,
        axis=0,
        keepdims=True,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    llm_mean = np.mean(
        llm_train,
        axis=0,
        keepdims=True,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    return (
        row_l2_normalize(
            np.asarray(
                vlm_test,
                dtype=np.float32,
            )
            - vlm_mean
        ),
        row_l2_normalize(
            np.asarray(
                llm_test,
                dtype=np.float32,
            )
            - llm_mean
        ),
    )


def paired_ranks_from_similarity(
    similarity: np.ndarray,
    tolerance: float = TIE_TOLERANCE,
) -> np.ndarray:
    if (
        similarity.ndim != 2
        or similarity.shape[0]
        != similarity.shape[1]
    ):
        raise ValueError(
            "Paired retrieval requires "
            "a square query-by-candidate "
            "similarity matrix."
        )

    item_count = similarity.shape[0]

    true_scores = similarity[
        np.arange(item_count),
        np.arange(item_count),
    ]

    greater_counts = np.sum(
        similarity
        > (
            true_scores[:, None]
            + tolerance
        ),
        axis=1,
    )

    tied_counts = (
        np.sum(
            np.abs(
                similarity
                - true_scores[:, None]
            )
            <= tolerance,
            axis=1,
        )
        - 1
    )

    ranks = (
        1.0
        + greater_counts.astype(
            np.float64
        )
        + 0.5
        * tied_counts.astype(
            np.float64
        )
    )

    require_finite(
        "retrieval ranks",
        ranks,
    )

    return ranks


def retrieval_metrics_from_ranks(
    ranks: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        ranks,
        dtype=np.float64,
    )

    return {
        "mrr": float(
            np.mean(
                1.0 / values
            )
        ),
        "recall_at_1": float(
            np.mean(
                values <= 1.0
            )
        ),
        "recall_at_5": float(
            np.mean(
                values <= 5.0
            )
        ),
        "recall_at_10": float(
            np.mean(
                values <= 10.0
            )
        ),
        "median_rank": float(
            np.median(values)
        ),
        "mean_rank": float(
            np.mean(values)
        ),
    }


def bootstrap_retrieval_intervals(
    rank_vectors: dict[
        str,
        np.ndarray,
    ],
    replicates: int,
    seed: int,
) -> dict[
    str,
    dict[
        str,
        list[float],
    ],
]:
    baseline_names = sorted(
        rank_vectors
    )

    observation_count = len(
        rank_vectors[
            baseline_names[0]
        ]
    )

    for name in baseline_names:
        if len(
            rank_vectors[name]
        ) != observation_count:
            raise ValueError(
                "All retrieval rank "
                "vectors must have equal length."
            )

    rng = np.random.default_rng(
        seed
    )

    sample_indices = rng.integers(
        0,
        observation_count,
        size=(
            replicates,
            observation_count,
        ),
        dtype=np.int32,
    )

    intervals: dict[
        str,
        dict[
            str,
            list[float],
        ],
    ] = {}

    for baseline_name in baseline_names:
        sampled_ranks = (
            rank_vectors[
                baseline_name
            ][
                sample_indices
            ]
        )

        intervals[
            baseline_name
        ] = {
            "mrr_ci95": np.quantile(
                np.mean(
                    1.0
                    / sampled_ranks,
                    axis=1,
                ),
                [
                    0.025,
                    0.975,
                ],
            ).tolist(),
            "recall_at_1_ci95": np.quantile(
                np.mean(
                    sampled_ranks
                    <= 1.0,
                    axis=1,
                ),
                [
                    0.025,
                    0.975,
                ],
            ).tolist(),
            "recall_at_5_ci95": np.quantile(
                np.mean(
                    sampled_ranks
                    <= 5.0,
                    axis=1,
                ),
                [
                    0.025,
                    0.975,
                ],
            ).tolist(),
            "recall_at_10_ci95": np.quantile(
                np.mean(
                    sampled_ranks
                    <= 10.0,
                    axis=1,
                ),
                [
                    0.025,
                    0.975,
                ],
            ).tolist(),
        }

    del sample_indices

    gc.collect()

    return intervals


def top_k_indices(
    similarity: np.ndarray,
    k: int,
) -> np.ndarray:
    if (
        similarity.shape[0]
        != similarity.shape[1]
    ):
        raise ValueError(
            "Neighborhood similarity "
            "matrix must be square."
        )

    if k >= similarity.shape[1]:
        raise ValueError(
            f"k={k} must be smaller than "
            f"{similarity.shape[1]}."
        )

    working = np.array(
        similarity,
        dtype=np.float32,
        copy=True,
    )

    np.fill_diagonal(
        working,
        -np.inf,
    )

    return np.argpartition(
        -working,
        kth=k - 1,
        axis=1,
    )[:, :k]


def neighborhood_overlap_metrics(
    representation_a: np.ndarray,
    representation_b: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, float]:
    a = row_l2_normalize(
        representation_a
    )

    b = row_l2_normalize(
        representation_b
    )

    similarity_a = (
        a
        @ a.T
    )

    similarity_b = (
        b
        @ b.T
    )

    metrics: dict[
        str,
        float,
    ] = {}

    for k in k_values:
        if k >= a.shape[0]:
            continue

        neighbors_a = top_k_indices(
            similarity_a,
            k,
        )

        neighbors_b = top_k_indices(
            similarity_b,
            k,
        )

        overlap_counts = np.empty(
            a.shape[0],
            dtype=np.float64,
        )

        for row_index in range(
            a.shape[0]
        ):
            overlap_counts[
                row_index
            ] = len(
                np.intersect1d(
                    neighbors_a[
                        row_index
                    ],
                    neighbors_b[
                        row_index
                    ],
                    assume_unique=False,
                )
            )

        overlap_fraction = (
            overlap_counts
            / float(k)
        )

        union = (
            2.0 * k
            - overlap_counts
        )

        metrics[
            f"knn_overlap_at_{k}"
        ] = float(
            np.mean(
                overlap_fraction
            )
        )

        metrics[
            f"knn_jaccard_at_{k}"
        ] = float(
            np.mean(
                overlap_counts
                / np.maximum(
                    union,
                    1.0,
                )
            )
        )

        metrics[
            f"knn_overlap_chance_at_{k}"
        ] = float(
            k
            / (
                a.shape[0]
                - 1
            )
        )

    return metrics


def run_paired_retrieval_unit(
    llm_all: np.ndarray,
    vlm_all: np.ndarray,
    family: str,
    dataset: str,
    depth_index: int,
    depth_fraction: float,
) -> dict[str, Any]:
    if (
        llm_all.shape
        != vlm_all.shape
    ):
        raise ValueError(
            "LLM and VLM paired "
            "caches have different shape."
        )

    example_count = (
        llm_all.shape[0]
    )

    if example_count != 1024:
        raise ValueError(
            "Expected 1024 paired examples; "
            f"found {example_count}."
        )

    fold_assignment = (
        make_nearly_balanced_fold_assignment(
            item_count=example_count,
            fold_count=RETRIEVAL_FOLDS,
            seed=stable_seed(
                GLOBAL_SEED,
                family,
                dataset,
                depth_index,
                "retrieval_folds",
            ),
        )
    )

    fold_sizes = np.bincount(
        fold_assignment,
        minlength=RETRIEVAL_FOLDS,
    )

    if not bool(
        np.all(
            fold_sizes
            == RETRIEVAL_CANDIDATE_POOL
        )
    ):
        raise RuntimeError(
            "Retrieval fold sizes "
            "are incorrect: "
            f"{fold_sizes.tolist()}"
        )

    rank_lists: dict[
        str,
        list[np.ndarray],
    ] = {
        "aligned": [],
        "shuffled_alignment": [],
        "no_alignment": [],
    }

    fold_rows: list[
        dict[str, Any]
    ] = []

    neighborhood_rows: list[
        dict[str, float]
    ] = []

    for fold_index in range(
        RETRIEVAL_FOLDS
    ):
        test_indices = np.flatnonzero(
            fold_assignment
            == fold_index
        )

        train_indices = np.flatnonzero(
            fold_assignment
            != fold_index
        )

        if np.intersect1d(
            train_indices,
            test_indices,
        ).size:
            raise RuntimeError(
                "Retrieval train/test "
                "overlap detected."
            )

        llm_train = np.asarray(
            llm_all[
                train_indices
            ],
            dtype=np.float32,
        )

        vlm_train = np.asarray(
            vlm_all[
                train_indices
            ],
            dtype=np.float32,
        )

        llm_test = np.asarray(
            llm_all[
                test_indices
            ],
            dtype=np.float32,
        )

        vlm_test = np.asarray(
            vlm_all[
                test_indices
            ],
            dtype=np.float32,
        )

        fold_seed = stable_seed(
            GLOBAL_SEED,
            family,
            dataset,
            depth_index,
            fold_index,
            "paired_retrieval",
        )

        projection = (
            fit_projection_pair(
                vlm_train=vlm_train,
                llm_train=llm_train,
                vlm_test=vlm_test,
                llm_test=llm_test,
                rank=ALIGNMENT_PCA_RANK,
                seed=fold_seed,
            )
        )

        (
            aligned_vlm,
            aligned_llm,
            _,
        ) = mapped_test_representations(
            projection=projection,
            seed=fold_seed,
            shuffle_training_pairs=False,
        )

        (
            shuffled_vlm,
            shuffled_llm,
            _,
        ) = mapped_test_representations(
            projection=projection,
            seed=fold_seed,
            shuffle_training_pairs=True,
        )

        (
            no_alignment_vlm,
            no_alignment_llm,
        ) = no_alignment_test_representations(
            vlm_train=vlm_train,
            llm_train=llm_train,
            vlm_test=vlm_test,
            llm_test=llm_test,
        )

        rank_vectors = {
            "aligned": (
                paired_ranks_from_similarity(
                    aligned_vlm
                    @ aligned_llm.T
                )
            ),
            "shuffled_alignment": (
                paired_ranks_from_similarity(
                    shuffled_vlm
                    @ shuffled_llm.T
                )
            ),
            "no_alignment": (
                paired_ranks_from_similarity(
                    no_alignment_vlm
                    @ no_alignment_llm.T
                )
            ),
        }

        for (
            baseline_name,
            ranks,
        ) in rank_vectors.items():
            rank_lists[
                baseline_name
            ].append(ranks)

            fold_rows.append(
                {
                    "fold": fold_index,
                    "baseline": (
                        baseline_name
                    ),
                    **retrieval_metrics_from_ranks(
                        ranks
                    ),
                }
            )

        neighborhood_rows.append(
            neighborhood_overlap_metrics(
                representation_a=(
                    aligned_vlm
                ),
                representation_b=(
                    aligned_llm
                ),
                k_values=(
                    RETRIEVAL_K_VALUES
                ),
            )
        )

        del (
            llm_train,
            vlm_train,
            llm_test,
            vlm_test,
            projection,
            aligned_vlm,
            aligned_llm,
            shuffled_vlm,
            shuffled_llm,
            no_alignment_vlm,
            no_alignment_llm,
        )

        gc.collect()

    concatenated_ranks = {
        baseline: np.concatenate(
            vectors,
            axis=0,
        )
        for baseline, vectors
        in rank_lists.items()
    }

    for (
        baseline,
        ranks,
    ) in concatenated_ranks.items():
        if len(ranks) != example_count:
            raise RuntimeError(
                f"OOF rank count for "
                f"{baseline} is {len(ranks)}, "
                f"expected {example_count}."
            )

    intervals = (
        bootstrap_retrieval_intervals(
            rank_vectors=(
                concatenated_ranks
            ),
            replicates=(
                BOOTSTRAP_REPLICATES
            ),
            seed=stable_seed(
                GLOBAL_SEED,
                family,
                dataset,
                depth_index,
                "retrieval_bootstrap",
            ),
        )
    )

    summary_rows = [
        {
            "baseline": baseline,
            **retrieval_metrics_from_ranks(
                ranks
            ),
            **intervals[
                baseline
            ],
        }
        for baseline, ranks
        in concatenated_ranks.items()
    ]

    mean_neighborhood_metrics = {
        key: float(
            np.mean(
                [
                    row[key]
                    for row
                    in neighborhood_rows
                ]
            )
        )
        for key
        in neighborhood_rows[0]
    }

    candidate_count = (
        RETRIEVAL_CANDIDATE_POOL
    )

    return {
        "completed": True,
        "analysis_version": (
            ANALYSIS_VERSION
        ),
        "experiment": (
            "held_out_paired_"
            "representation_retrieval"
        ),
        "family": family,
        "dataset": dataset,
        "pooling": PRIMARY_POOLING,
        "depth_index": int(
            depth_index
        ),
        "depth_fraction": float(
            depth_fraction
        ),
        "pairing_basis": (
            "shared cache row index; "
            "explicit sample identifiers "
            "were not stored"
        ),
        "pca_rank": (
            ALIGNMENT_PCA_RANK
        ),
        "fold_count": (
            RETRIEVAL_FOLDS
        ),
        "candidate_pool": (
            candidate_count
        ),
        "summary_rows": (
            summary_rows
        ),
        "fold_rows": (
            fold_rows
        ),
        "neighborhood_metrics": (
            mean_neighborhood_metrics
        ),
        "chance_metrics": {
            "candidate_pool": (
                candidate_count
            ),
            "chance_recall_at_1": float(
                1.0
                / candidate_count
            ),
            "chance_recall_at_5": float(
                5.0
                / candidate_count
            ),
            "chance_recall_at_10": float(
                10.0
                / candidate_count
            ),
            "chance_mrr": float(
                harmonic_number(
                    candidate_count
                )
                / candidate_count
            ),
        },
        "completed_at_unix": (
            time.time()
        ),
    }


def confusion_metrics(
    matrix: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    total = float(
        np.sum(matrix)
    )

    if total <= 0:
        raise ValueError(
            "Confusion matrix "
            "contains no observations."
        )

    true_positive = np.diag(
        matrix
    )

    predicted_total = np.sum(
        matrix,
        axis=0,
    )

    actual_total = np.sum(
        matrix,
        axis=1,
    )

    precision = np.divide(
        true_positive,
        predicted_total,
        out=np.zeros_like(
            true_positive
        ),
        where=(
            predicted_total > 0
        ),
    )

    recall = np.divide(
        true_positive,
        actual_total,
        out=np.zeros_like(
            true_positive
        ),
        where=(
            actual_total > 0
        ),
    )

    class_f1 = np.divide(
        2.0
        * precision
        * recall,
        precision + recall,
        out=np.zeros_like(
            precision
        ),
        where=(
            precision
            + recall
        ) > 0,
    )

    return {
        "accuracy": float(
            np.sum(
                true_positive
            )
            / total
        ),
        "balanced_accuracy": float(
            np.mean(recall)
        ),
        "macro_f1": float(
            np.mean(class_f1)
        ),
    }


def group_bootstrap_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[
    str,
    list[float],
]:
    unique_groups = np.unique(
        groups
    )

    class_count = len(
        CONDITION_NAMES
    )

    group_confusions = np.zeros(
        (
            len(unique_groups),
            class_count,
            class_count,
        ),
        dtype=np.int64,
    )

    for (
        group_position,
        group_value,
    ) in enumerate(
        unique_groups
    ):
        mask = (
            groups
            == group_value
        )

        group_confusions[
            group_position
        ] = confusion_matrix(
            y_true[mask],
            y_pred[mask],
            labels=np.arange(
                class_count
            ),
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_weights = rng.multinomial(
        n=len(unique_groups),
        pvals=np.full(
            len(unique_groups),
            1.0
            / len(unique_groups),
        ),
        size=replicates,
    )

    bootstrap_confusions = np.einsum(
        "bg,gij->bij",
        bootstrap_weights,
        group_confusions,
        optimize=True,
    )

    true_positive = np.diagonal(
        bootstrap_confusions,
        axis1=1,
        axis2=2,
    )

    predicted_total = np.sum(
        bootstrap_confusions,
        axis=1,
    )

    actual_total = np.sum(
        bootstrap_confusions,
        axis=2,
    )

    precision = np.divide(
        true_positive,
        predicted_total,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=(
            predicted_total > 0
        ),
    )

    recall = np.divide(
        true_positive,
        actual_total,
        out=np.zeros_like(
            true_positive,
            dtype=np.float64,
        ),
        where=(
            actual_total > 0
        ),
    )

    class_f1 = np.divide(
        2.0
        * precision
        * recall,
        precision + recall,
        out=np.zeros_like(
            precision,
            dtype=np.float64,
        ),
        where=(
            precision
            + recall
        ) > 0,
    )

    bootstrap_macro_f1 = (
        np.mean(
            class_f1,
            axis=1,
        )
    )

    bootstrap_balanced_accuracy = (
        np.mean(
            recall,
            axis=1,
        )
    )

    bootstrap_accuracy = np.divide(
        np.sum(
            true_positive,
            axis=1,
        ),
        np.sum(
            bootstrap_confusions,
            axis=(1, 2),
        ),
    )

    result = {
        "macro_f1_ci95": np.quantile(
            bootstrap_macro_f1,
            [
                0.025,
                0.975,
            ],
        ).tolist(),
        "balanced_accuracy_ci95": np.quantile(
            bootstrap_balanced_accuracy,
            [
                0.025,
                0.975,
            ],
        ).tolist(),
        "accuracy_ci95": np.quantile(
            bootstrap_accuracy,
            [
                0.025,
                0.975,
            ],
        ).tolist(),
    }

    del (
        bootstrap_weights,
        bootstrap_confusions,
    )

    gc.collect()

    return result


def make_within_group_null_labels(
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    null_labels = np.array(
        labels,
        dtype=np.int64,
        copy=True,
    )

    rng = np.random.default_rng(
        seed
    )

    for group_value in np.unique(
        groups
    ):
        positions = np.flatnonzero(
            groups
            == group_value
        )

        if len(
            positions
        ) != len(
            CONDITION_NAMES
        ):
            raise RuntimeError(
                "Each classification group "
                "must contain one observation "
                "from every condition."
            )

        null_labels[
            positions
        ] = rng.permutation(
            null_labels[
                positions
            ]
        )

    if not np.array_equal(
        np.bincount(
            labels,
            minlength=len(
                CONDITION_NAMES
            ),
        ),
        np.bincount(
            null_labels,
            minlength=len(
                CONDITION_NAMES
            ),
        ),
    ):
        raise RuntimeError(
            "Within-group label permutation "
            "changed class balance."
        )

    return null_labels


def fit_logistic_with_retry(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    def fit_once(
        iteration_limit: int,
    ) -> tuple[
        LogisticRegression,
        bool,
        list[str],
    ]:
        classifier = LogisticRegression(
            C=CLASSIFICATION_C,
            solver="lbfgs",
            max_iter=(
                iteration_limit
            ),
            tol=1e-6,
            random_state=seed,
        )

        with warnings.catch_warnings(
            record=True
        ) as captured:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            classifier.fit(
                train_features,
                train_labels,
            )

        convergence_warning = any(
            issubclass(
                record.category,
                ConvergenceWarning,
            )
            for record in captured
        )

        messages = [
            str(record.message)
            for record in captured
        ]

        return (
            classifier,
            convergence_warning,
            messages,
        )

    (
        classifier,
        convergence_warning,
        warning_messages,
    ) = fit_once(
        CLASSIFICATION_MAX_ITER
    )

    retried = False

    if convergence_warning:
        retried = True

        (
            classifier,
            convergence_warning,
            retry_messages,
        ) = fit_once(
            CLASSIFICATION_RETRY_MAX_ITER
        )

        warning_messages.extend(
            retry_messages
        )

    if convergence_warning:
        raise RuntimeError(
            "Logistic regression did not "
            "converge after the extended retry."
        )

    predictions = classifier.predict(
        test_features
    )

    return (
        predictions,
        {
            "iterations": int(
                np.max(
                    classifier.n_iter_
                )
            ),
            "retried": retried,
            "warnings": (
                warning_messages
            ),
        },
    )


def run_condition_classification_unit(
    condition_arrays: dict[
        str,
        np.ndarray,
    ],
    family: str,
    dataset: str,
    depth_index: int,
    depth_fraction: float,
    run_null_control: bool,
) -> dict[str, Any]:
    example_counts = {
        key: value.shape[0]
        for key, value
        in condition_arrays.items()
    }

    if set(
        example_counts.values()
    ) != {512}:
        raise ValueError(
            "Expected 512 examples "
            "per condition; found "
            f"{example_counts}."
        )

    hidden_dimensions = {
        value.shape[1]
        for value
        in condition_arrays.values()
    }

    if len(
        hidden_dimensions
    ) != 1:
        raise ValueError(
            "Condition arrays have "
            "different hidden dimensions."
        )

    features = np.concatenate(
        [
            np.asarray(
                condition_arrays[
                    name
                ],
                dtype=np.float32,
            )
            for name
            in CONDITION_NAMES
        ],
        axis=0,
    )

    features = row_l2_normalize(
        features
    )

    labels = np.concatenate(
        [
            np.full(
                512,
                condition_index,
                dtype=np.int64,
            )
            for condition_index
            in range(
                len(
                    CONDITION_NAMES
                )
            )
        ]
    )

    groups = np.concatenate(
        [
            np.arange(
                512,
                dtype=np.int64,
            )
            for _
            in CONDITION_NAMES
        ]
    )

    for group_value in range(
        512
    ):
        positions = np.flatnonzero(
            groups
            == group_value
        )

        if len(positions) != 4:
            raise RuntimeError(
                "Classification group "
                "construction failed."
            )

        if not np.array_equal(
            np.sort(
                labels[
                    positions
                ]
            ),
            np.arange(4),
        ):
            raise RuntimeError(
                "A classification group "
                "does not contain one item "
                "from every condition."
            )

    fold_assignment = (
        make_nearly_balanced_fold_assignment(
            item_count=512,
            fold_count=(
                CLASSIFICATION_FOLDS
            ),
            seed=stable_seed(
                GLOBAL_SEED,
                family,
                dataset,
                depth_index,
                "classification_group_folds",
            ),
        )
    )

    group_fold_sizes = np.bincount(
        fold_assignment,
        minlength=(
            CLASSIFICATION_FOLDS
        ),
    )

    if sorted(
        group_fold_sizes.tolist()
    ) != [
        102,
        102,
        102,
        103,
        103,
    ]:
        raise RuntimeError(
            "Unexpected five-fold "
            "group sizes: "
            f"{group_fold_sizes.tolist()}"
        )

    null_labels = None

    if run_null_control:
        null_labels = (
            make_within_group_null_labels(
                labels=labels,
                groups=groups,
                seed=stable_seed(
                    GLOBAL_SEED,
                    family,
                    dataset,
                    depth_index,
                    "within_group_null_labels",
                ),
            )
        )

    out_of_fold_predictions = np.full(
        len(labels),
        fill_value=-1,
        dtype=np.int64,
    )

    null_out_of_fold_predictions = (
        np.full(
            len(labels),
            fill_value=-1,
            dtype=np.int64,
        )
        if run_null_control
        else None
    )

    fold_rows: list[
        dict[str, Any]
    ] = []

    for fold_index in range(
        CLASSIFICATION_FOLDS
    ):
        held_out_groups = np.flatnonzero(
            fold_assignment
            == fold_index
        )

        test_mask = np.isin(
            groups,
            held_out_groups,
        )

        train_mask = ~test_mask

        training_groups = np.unique(
            groups[
                train_mask
            ]
        )

        testing_groups = np.unique(
            groups[
                test_mask
            ]
        )

        if np.intersect1d(
            training_groups,
            testing_groups,
        ).size:
            raise RuntimeError(
                "Classification group "
                "leakage detected."
            )

        train_features = features[
            train_mask
        ]

        test_features = features[
            test_mask
        ]

        train_labels = labels[
            train_mask
        ]

        test_labels = labels[
            test_mask
        ]

        effective_rank = int(
            min(
                CLASSIFICATION_PCA_RANK,
                train_features.shape[0]
                - 1,
                train_features.shape[1],
            )
        )

        fold_seed = stable_seed(
            GLOBAL_SEED,
            family,
            dataset,
            depth_index,
            fold_index,
            "condition_classification",
        )

        pca = PCA(
            n_components=effective_rank,
            whiten=True,
            svd_solver="randomized",
            random_state=fold_seed,
        )

        train_projected = (
            pca.fit_transform(
                train_features
            )
        )

        test_projected = (
            pca.transform(
                test_features
            )
        )

        (
            predictions,
            classifier_diagnostics,
        ) = fit_logistic_with_retry(
            train_features=(
                train_projected
            ),
            train_labels=(
                train_labels
            ),
            test_features=(
                test_projected
            ),
            seed=fold_seed,
        )

        out_of_fold_predictions[
            test_mask
        ] = predictions

        fold_confusion = (
            confusion_matrix(
                test_labels,
                predictions,
                labels=np.arange(4),
            )
        )

        fold_row: dict[
            str,
            Any,
        ] = {
            "fold": fold_index,
            "test_group_count": int(
                len(
                    testing_groups
                )
            ),
            "test_observation_count": int(
                np.sum(
                    test_mask
                )
            ),
            "effective_pca_rank": (
                effective_rank
            ),
            "pca_explained_variance_sum": float(
                np.sum(
                    pca
                    .explained_variance_ratio_
                )
            ),
            **confusion_metrics(
                fold_confusion
            ),
            "confusion_matrix": (
                fold_confusion.tolist()
            ),
            "classifier_diagnostics": (
                classifier_diagnostics
            ),
        }

        if run_null_control:
            (
                null_predictions,
                null_diagnostics,
            ) = fit_logistic_with_retry(
                train_features=(
                    train_projected
                ),
                train_labels=(
                    null_labels[
                        train_mask
                    ]
                ),
                test_features=(
                    test_projected
                ),
                seed=stable_seed(
                    fold_seed,
                    "null_classifier",
                ),
            )

            null_out_of_fold_predictions[
                test_mask
            ] = null_predictions

            null_fold_confusion = (
                confusion_matrix(
                    null_labels[
                        test_mask
                    ],
                    null_predictions,
                    labels=np.arange(4),
                )
            )

            fold_row[
                "null_control"
            ] = {
                **confusion_metrics(
                    null_fold_confusion
                ),
                "confusion_matrix": (
                    null_fold_confusion
                    .tolist()
                ),
                "classifier_diagnostics": (
                    null_diagnostics
                ),
            }

        fold_rows.append(
            fold_row
        )

        del (
            train_features,
            test_features,
            train_projected,
            test_projected,
            predictions,
            pca,
        )

        gc.collect()

    if bool(
        np.any(
            out_of_fold_predictions
            < 0
        )
    ):
        raise RuntimeError(
            "At least one classification "
            "observation lacks an OOF prediction."
        )

    overall_confusion = (
        confusion_matrix(
            labels,
            out_of_fold_predictions,
            labels=np.arange(4),
        )
    )

    overall_metrics = (
        confusion_metrics(
            overall_confusion
        )
    )

    intervals = (
        group_bootstrap_classification(
            y_true=labels,
            y_pred=(
                out_of_fold_predictions
            ),
            groups=groups,
            replicates=(
                BOOTSTRAP_REPLICATES
            ),
            seed=stable_seed(
                GLOBAL_SEED,
                family,
                dataset,
                depth_index,
                "classification_bootstrap",
            ),
        )
    )

    per_class_f1 = f1_score(
        labels,
        out_of_fold_predictions,
        labels=np.arange(4),
        average=None,
        zero_division=0,
    )

    result: dict[
        str,
        Any,
    ] = {
        "completed": True,
        "analysis_version": (
            ANALYSIS_VERSION
        ),
        "experiment": (
            "grouped_four_way_"
            "condition_classification"
        ),
        "family": family,
        "dataset": dataset,
        "pooling": PRIMARY_POOLING,
        "depth_index": int(
            depth_index
        ),
        "depth_fraction": float(
            depth_fraction
        ),
        "conditions": list(
            CONDITION_NAMES
        ),
        "grouping_rule": (
            "cache_row_index"
        ),
        "group_fold_sizes": (
            group_fold_sizes.tolist()
        ),
        "fold_count": (
            CLASSIFICATION_FOLDS
        ),
        "pca_rank": (
            CLASSIFICATION_PCA_RANK
        ),
        "classifier": (
            "multinomial logistic regression"
        ),
        "primary_metric": (
            "macro_f1"
        ),
        "summary": {
            **overall_metrics,
            **intervals,
            "per_class_f1": {
                condition: float(value)
                for condition, value
                in zip(
                    CONDITION_NAMES,
                    per_class_f1,
                )
            },
            "confusion_matrix": (
                overall_confusion
                .tolist()
            ),
            "chance_accuracy": 0.25,
            "chance_macro_f1": 0.25,
        },
        "fold_rows": (
            fold_rows
        ),
        "null_control_run": (
            run_null_control
        ),
        "completed_at_unix": (
            time.time()
        ),
    }

    if run_null_control:
        if bool(
            np.any(
                null_out_of_fold_predictions
                < 0
            )
        ):
            raise RuntimeError(
                "Null-control OOF predictions "
                "are incomplete."
            )

        null_confusion = (
            confusion_matrix(
                null_labels,
                null_out_of_fold_predictions,
                labels=np.arange(4),
            )
        )

        null_metrics = (
            confusion_metrics(
                null_confusion
            )
        )

        null_intervals = (
            group_bootstrap_classification(
                y_true=null_labels,
                y_pred=(
                    null_out_of_fold_predictions
                ),
                groups=groups,
                replicates=(
                    BOOTSTRAP_REPLICATES
                ),
                seed=stable_seed(
                    GLOBAL_SEED,
                    family,
                    dataset,
                    depth_index,
                    "null_classification_bootstrap",
                ),
            )
        )

        result[
            "null_control_summary"
        ] = {
            **null_metrics,
            **null_intervals,
            "confusion_matrix": (
                null_confusion.tolist()
            ),
        }

    return result


def row_cosine_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first_normalized = (
        row_l2_normalize(
            first
        )
    )

    second_normalized = (
        row_l2_normalize(
            second
        )
    )

    cosine_similarity = np.sum(
        first_normalized
        * second_normalized,
        axis=1,
    )

    distance = (
        1.0
        - np.clip(
            cosine_similarity,
            -1.0,
            1.0,
        )
    )

    require_finite(
        "cosine distance",
        distance,
    )

    return distance.astype(
        np.float64,
        copy=False,
    )


def row_relative_l2_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first_values = np.asarray(
        first,
        dtype=np.float32,
    )

    second_values = np.asarray(
        second,
        dtype=np.float32,
    )

    numerator = np.linalg.norm(
        first_values
        - second_values,
        axis=1,
    )

    denominator = (
        0.5
        * (
            np.linalg.norm(
                first_values,
                axis=1,
            )
            + np.linalg.norm(
                second_values,
                axis=1,
            )
        )
    )

    distance = (
        numerator
        / np.maximum(
            denominator,
            EPSILON,
        )
    )

    require_finite(
        "relative L2 distance",
        distance,
    )

    return distance.astype(
        np.float64,
        copy=False,
    )


def bootstrap_mean_interval(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> list[float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    sample_indices = rng.integers(
        0,
        len(values),
        size=(
            replicates,
            len(values),
        ),
        dtype=np.int32,
    )

    bootstrap_means = np.mean(
        values[
            sample_indices
        ],
        axis=1,
    )

    interval = np.quantile(
        bootstrap_means,
        [
            0.025,
            0.975,
        ],
    ).tolist()

    del sample_indices

    gc.collect()

    return interval


def paired_sign_flip_p_value(
    differences: np.ndarray,
    replicates: int,
    seed: int,
    chunk_size: int = 1000,
) -> float:
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    observed = abs(
        float(
            np.mean(values)
        )
    )

    rng = np.random.default_rng(
        seed
    )

    exceed_count = 0
    completed = 0

    while completed < replicates:
        current_size = min(
            chunk_size,
            replicates
            - completed,
        )

        signs = rng.choice(
            np.array(
                [
                    -1.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            size=(
                current_size,
                len(values),
            ),
            replace=True,
        )

        permuted_means = np.mean(
            signs
            * values[None, :],
            axis=1,
        )

        exceed_count += int(
            np.sum(
                np.abs(
                    permuted_means
                )
                >= observed
            )
        )

        completed += current_size

    return float(
        (
            exceed_count
            + 1
        )
        / (
            replicates
            + 1
        )
    )


def summarize_distance_vector(
    values: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "ci95": (
            bootstrap_mean_interval(
                values=values,
                replicates=(
                    BOOTSTRAP_REPLICATES
                ),
                seed=seed,
            )
        ),
    }


def run_dense_mismatch_unit(
    matched: np.ndarray,
    text_swapped: np.ndarray,
    image_swapped: np.ndarray,
    image_only: np.ndarray,
    family: str,
    dataset: str,
    depth_index: int,
    depth_fraction: float,
) -> dict[str, Any]:
    arrays = {
        "matched": matched,
        "text_swapped": (
            text_swapped
        ),
        "image_swapped": (
            image_swapped
        ),
        "image_only": (
            image_only
        ),
    }

    shapes = {
        key: value.shape
        for key, value
        in arrays.items()
    }

    if len(
        set(
            shapes.values()
        )
    ) != 1:
        raise ValueError(
            "Mismatch arrays have "
            "different shapes: "
            f"{shapes}"
        )

    cosine_text = (
        row_cosine_distance(
            matched,
            text_swapped,
        )
    )

    cosine_image = (
        row_cosine_distance(
            matched,
            image_swapped,
        )
    )

    cosine_image_only = (
        row_cosine_distance(
            matched,
            image_only,
        )
    )

    relative_l2_text = (
        row_relative_l2_distance(
            matched,
            text_swapped,
        )
    )

    relative_l2_image = (
        row_relative_l2_distance(
            matched,
            image_swapped,
        )
    )

    relative_l2_image_only = (
        row_relative_l2_distance(
            matched,
            image_only,
        )
    )

    cosine_asymmetry = (
        cosine_image
        - cosine_text
    )

    relative_l2_asymmetry = (
        relative_l2_image
        - relative_l2_text
    )

    cosine_asymmetry_std = float(
        np.std(
            cosine_asymmetry,
            ddof=1,
        )
    )

    relative_asymmetry_std = float(
        np.std(
            relative_l2_asymmetry,
            ddof=1,
        )
    )

    cosine_dz = (
        float(
            np.mean(
                cosine_asymmetry
            )
            / cosine_asymmetry_std
        )
        if cosine_asymmetry_std > 0
        else 0.0
    )

    relative_l2_dz = (
        float(
            np.mean(
                relative_l2_asymmetry
            )
            / relative_asymmetry_std
        )
        if relative_asymmetry_std > 0
        else 0.0
    )

    return {
        "completed": True,
        "analysis_version": (
            ANALYSIS_VERSION
        ),
        "experiment": (
            "sae_independent_dense_"
            "mismatch_validation"
        ),
        "family": family,
        "dataset": dataset,
        "pooling": PRIMARY_POOLING,
        "depth_index": int(
            depth_index
        ),
        "depth_fraction": float(
            depth_fraction
        ),
        "example_count": int(
            matched.shape[0]
        ),
        "cosine": {
            "text_swapped": (
                summarize_distance_vector(
                    cosine_text,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "cosine_text_bootstrap",
                    ),
                )
            ),
            "image_swapped": (
                summarize_distance_vector(
                    cosine_image,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "cosine_image_bootstrap",
                    ),
                )
            ),
            "image_only": (
                summarize_distance_vector(
                    cosine_image_only,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "cosine_image_only_bootstrap",
                    ),
                )
            ),
            "image_minus_text": {
                **summarize_distance_vector(
                    cosine_asymmetry,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "cosine_asymmetry_bootstrap",
                    ),
                ),
                "cohen_dz": (
                    cosine_dz
                ),
                "sign_flip_p_value": (
                    paired_sign_flip_p_value(
                        cosine_asymmetry,
                        PERMUTATION_REPLICATES,
                        stable_seed(
                            GLOBAL_SEED,
                            family,
                            dataset,
                            depth_index,
                            "cosine_sign_flip",
                        ),
                    )
                ),
            },
            "combined_text_image_mean": float(
                np.mean(
                    np.concatenate(
                        [
                            cosine_text,
                            cosine_image,
                        ]
                    )
                )
            ),
        },
        "relative_l2": {
            "text_swapped": (
                summarize_distance_vector(
                    relative_l2_text,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "relative_l2_text_bootstrap",
                    ),
                )
            ),
            "image_swapped": (
                summarize_distance_vector(
                    relative_l2_image,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "relative_l2_image_bootstrap",
                    ),
                )
            ),
            "image_only": (
                summarize_distance_vector(
                    relative_l2_image_only,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "relative_l2_image_only_bootstrap",
                    ),
                )
            ),
            "image_minus_text": {
                **summarize_distance_vector(
                    relative_l2_asymmetry,
                    stable_seed(
                        GLOBAL_SEED,
                        family,
                        dataset,
                        depth_index,
                        "relative_l2_asymmetry_bootstrap",
                    ),
                ),
                "cohen_dz": (
                    relative_l2_dz
                ),
                "sign_flip_p_value": (
                    paired_sign_flip_p_value(
                        relative_l2_asymmetry,
                        PERMUTATION_REPLICATES,
                        stable_seed(
                            GLOBAL_SEED,
                            family,
                            dataset,
                            depth_index,
                            "relative_l2_sign_flip",
                        ),
                    )
                ),
            },
            "combined_text_image_mean": float(
                np.mean(
                    np.concatenate(
                        [
                            relative_l2_text,
                            relative_l2_image,
                        ]
                    )
                )
            ),
        },
        "completed_at_unix": (
            time.time()
        ),
    }


def run_cross_dataset_retrieval_unit(
    source_llm: np.ndarray,
    source_vlm: np.ndarray,
    target_llm: np.ndarray,
    target_vlm: np.ndarray,
    family: str,
    source_dataset: str,
    target_dataset: str,
    depth_index: int,
    depth_fraction: float,
) -> dict[str, Any]:
    if (
        source_llm.shape
        != source_vlm.shape
    ):
        raise ValueError(
            "Source LLM/VLM "
            "shapes differ."
        )

    if (
        target_llm.shape
        != target_vlm.shape
    ):
        raise ValueError(
            "Target LLM/VLM "
            "shapes differ."
        )

    if (
        source_llm.shape[1]
        != target_llm.shape[1]
    ):
        raise ValueError(
            "Source and target hidden "
            "dimensions differ within one family."
        )

    unit_seed = stable_seed(
        GLOBAL_SEED,
        family,
        source_dataset,
        target_dataset,
        depth_index,
        "cross_dataset_transfer",
    )

    projection = fit_projection_pair(
        vlm_train=source_vlm,
        llm_train=source_llm,
        vlm_test=target_vlm,
        llm_test=target_llm,
        rank=ALIGNMENT_PCA_RANK,
        seed=unit_seed,
    )

    (
        aligned_target_vlm,
        aligned_target_llm,
        _,
    ) = mapped_test_representations(
        projection=projection,
        seed=unit_seed,
        shuffle_training_pairs=False,
    )

    (
        shuffled_target_vlm,
        shuffled_target_llm,
        _,
    ) = mapped_test_representations(
        projection=projection,
        seed=unit_seed,
        shuffle_training_pairs=True,
    )

    (
        no_alignment_target_vlm,
        no_alignment_target_llm,
    ) = no_alignment_test_representations(
        vlm_train=source_vlm,
        llm_train=source_llm,
        vlm_test=target_vlm,
        llm_test=target_llm,
    )

    rank_vectors = {
        "aligned": (
            paired_ranks_from_similarity(
                aligned_target_vlm
                @ aligned_target_llm.T
            )
        ),
        "shuffled_alignment": (
            paired_ranks_from_similarity(
                shuffled_target_vlm
                @ shuffled_target_llm.T
            )
        ),
        "no_alignment": (
            paired_ranks_from_similarity(
                no_alignment_target_vlm
                @ no_alignment_target_llm.T
            )
        ),
    }

    intervals = (
        bootstrap_retrieval_intervals(
            rank_vectors=rank_vectors,
            replicates=(
                BOOTSTRAP_REPLICATES
            ),
            seed=stable_seed(
                unit_seed,
                "cross_dataset_bootstrap",
            ),
        )
    )

    summary_rows = [
        {
            "baseline": baseline,
            **retrieval_metrics_from_ranks(
                ranks
            ),
            **intervals[
                baseline
            ],
        }
        for baseline, ranks
        in rank_vectors.items()
    ]

    candidate_pool = int(
        target_llm.shape[0]
    )

    return {
        "completed": True,
        "analysis_version": (
            ANALYSIS_VERSION
        ),
        "experiment": (
            "zero_target_fit_"
            "cross_dataset_retrieval"
        ),
        "family": family,
        "source_dataset": (
            source_dataset
        ),
        "target_dataset": (
            target_dataset
        ),
        "pooling": (
            PRIMARY_POOLING
        ),
        "depth_index": int(
            depth_index
        ),
        "depth_fraction": float(
            depth_fraction
        ),
        "source_example_count": int(
            source_llm.shape[0]
        ),
        "target_candidate_pool": (
            candidate_pool
        ),
        "target_data_used_for_fitting": (
            False
        ),
        "summary_rows": (
            summary_rows
        ),
        "neighborhood_metrics": (
            neighborhood_overlap_metrics(
                representation_a=(
                    aligned_target_vlm
                ),
                representation_b=(
                    aligned_target_llm
                ),
                k_values=(
                    RETRIEVAL_K_VALUES
                ),
            )
        ),
        "chance_metrics": {
            "chance_recall_at_1": float(
                1.0
                / candidate_pool
            ),
            "chance_recall_at_5": float(
                5.0
                / candidate_pool
            ),
            "chance_recall_at_10": float(
                10.0
                / candidate_pool
            ),
            "chance_mrr": float(
                harmonic_number(
                    candidate_pool
                )
                / candidate_pool
            ),
        },
        "completed_at_unix": (
            time.time()
        ),
    }


primary_index = (
    PRIMARY_CACHE_INDEX.copy()
)

primary_index = primary_index[
    primary_index[
        "pooling"
    ]
    == PRIMARY_POOLING
].copy()

primary_index = (
    primary_index.sort_values(
        [
            "family",
            "dataset",
        ],
        ignore_index=True,
    )
)

if len(primary_index) != 9:
    raise RuntimeError(
        "Primary validation requires "
        "exactly nine family–dataset files."
    )

RUN_SPECIFICATION = {
    "analysis_version": (
        ANALYSIS_VERSION
    ),
    "primary_pooling": (
        PRIMARY_POOLING
    ),
    "paired_retrieval": {
        "primary_metric": "MRR",
        "secondary_metrics": [
            "R@1",
            "R@5",
            "R@10",
            "median rank",
            "mean rank",
            "held-out kNN overlap",
        ],
        "folds": (
            RETRIEVAL_FOLDS
        ),
        "candidate_pool": (
            RETRIEVAL_CANDIDATE_POOL
        ),
        "alignment": (
            "train_only_whitened_pca_procrustes"
        ),
        "controls": [
            "shuffled_pairs",
            "direct_no_map",
            "analytic_chance",
        ],
    },
    "condition_classification": {
        "primary_metric": (
            "macro-F1"
        ),
        "secondary_metrics": [
            "accuracy",
            "balanced accuracy",
            "per-class F1",
            "confusion matrix",
        ],
        "folds": (
            CLASSIFICATION_FOLDS
        ),
        "group_fold_sizes": [
            103,
            103,
            102,
            102,
            102,
        ],
        "split_unit": (
            "base-example cache row index"
        ),
        "negative_control": (
            "within-group label permutation"
        ),
    },
    "dense_mismatch": {
        "primary_metric": (
            "matched-to-swapped "
            "cosine distance"
        ),
        "robustness_metric": (
            "relative L2 distance"
        ),
        "asymmetry_test": (
            "paired sign-flip "
            "permutation test"
        ),
    },
    "cross_dataset_transfer": {
        "target_fit_allowed": False,
        "primary_metric": "MRR",
    },
    "pca_rank": (
        ALIGNMENT_PCA_RANK
    ),
    "depth_policy": (
        "all eight cached relative depths"
    ),
    "bootstrap_replicates": (
        BOOTSTRAP_REPLICATES
    ),
    "permutation_replicates": (
        PERMUTATION_REPLICATES
    ),
    "created_at_unix": (
        time.time()
    ),
}

RUN_SPECIFICATION_PATH = (
    AUDIT_DIR
    / "primary_validation_specification.json"
)

atomic_json_write(
    RUN_SPECIFICATION,
    RUN_SPECIFICATION_PATH,
)


total_primary_depth_units = (
    len(primary_index)
    * CONTRACT.expected_depth_count
)

print("=" * 72)
print(
    "PRIMARY HELD-OUT VALIDATION"
)
print("=" * 72)

print(
    f"Primary files:          "
    f"{len(primary_index)}"
)

print(
    f"Family–dataset–depths:  "
    f"{total_primary_depth_units}"
)

print(
    f"Retrieval folds:        "
    f"{RETRIEVAL_FOLDS}"
)

print(
    "Classification folds: "
    f"{CLASSIFICATION_FOLDS} "
    "(group sizes "
    "103/103/102/102/102)"
)

print(
    f"PCA rank:               "
    f"{ALIGNMENT_PCA_RANK}"
)

print(
    f"Checkpoint root:        "
    f"{EVALUATION_ROOT}"
)

print()

primary_progress = tqdm(
    total=total_primary_depth_units,
    desc=(
        "Primary family–dataset–depth units"
    ),
    unit="unit",
)

for _, cache_row in (
    primary_index.iterrows()
):
    family = str(
        cache_row[
            "family"
        ]
    )

    dataset = str(
        cache_row[
            "dataset"
        ]
    )

    cache_path = Path(
        cache_row[
            "path"
        ]
    ).resolve()

    with np.load(
        cache_path,
        allow_pickle=False,
    ) as archive:
        depth_fractions = np.asarray(
            archive[
                "depth_fractions"
            ],
            dtype=np.float64,
        )

        llm_train_all = np.asarray(
            archive[
                "llm_train"
            ],
            dtype=np.float32,
        )

        vlm_train_all = np.asarray(
            archive[
                "vlm_train"
            ],
            dtype=np.float32,
        )

        evaluation_arrays = {
            condition_name: np.asarray(
                archive[
                    array_key
                ],
                dtype=np.float32,
            )
            for (
                condition_name,
                array_key,
            ) in zip(
                CONDITION_NAMES,
                CONDITION_KEYS,
            )
        }

    if (
        llm_train_all.shape[:2]
        != (
            CONTRACT
            .expected_train_examples,
            CONTRACT
            .expected_depth_count,
        )
    ):
        raise RuntimeError(
            "Unexpected llm_train shape "
            f"in {cache_path.name}: "
            f"{llm_train_all.shape}"
        )

    if (
        vlm_train_all.shape
        != llm_train_all.shape
    ):
        raise RuntimeError(
            "llm_train and vlm_train "
            "shapes differ in "
            f"{cache_path.name}."
        )

    null_depth_index = int(
        np.argmin(
            np.abs(
                depth_fractions
                - NULL_CONTROL_TARGET_DEPTH
            )
        )
    )

    for (
        depth_index,
        depth_fraction,
    ) in enumerate(
        depth_fractions
    ):
        retrieval_checkpoint_path = (
            RETRIEVAL_CHECKPOINT_DIR
            / unit_filename(
                family,
                dataset,
                f"depth_{depth_index}",
            )
        )

        classification_checkpoint_path = (
            CLASSIFICATION_CHECKPOINT_DIR
            / unit_filename(
                family,
                dataset,
                f"depth_{depth_index}",
            )
        )

        mismatch_checkpoint_path = (
            MISMATCH_CHECKPOINT_DIR
            / unit_filename(
                family,
                dataset,
                f"depth_{depth_index}",
            )
        )

        if (
            read_completed_json(
                retrieval_checkpoint_path
            )
            is None
        ):
            retrieval_result = (
                run_paired_retrieval_unit(
                    llm_all=(
                        llm_train_all[
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    vlm_all=(
                        vlm_train_all[
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    family=family,
                    dataset=dataset,
                    depth_index=(
                        depth_index
                    ),
                    depth_fraction=float(
                        depth_fraction
                    ),
                )
            )

            atomic_json_write(
                retrieval_result,
                retrieval_checkpoint_path,
            )

        if (
            read_completed_json(
                classification_checkpoint_path
            )
            is None
        ):
            classification_result = (
                run_condition_classification_unit(
                    condition_arrays={
                        condition_name: (
                            evaluation_arrays[
                                condition_name
                            ][
                                :,
                                depth_index,
                                :,
                            ]
                        )
                        for condition_name
                        in CONDITION_NAMES
                    },
                    family=family,
                    dataset=dataset,
                    depth_index=(
                        depth_index
                    ),
                    depth_fraction=float(
                        depth_fraction
                    ),
                    run_null_control=(
                        depth_index
                        == null_depth_index
                    ),
                )
            )

            atomic_json_write(
                classification_result,
                classification_checkpoint_path,
            )

        if (
            read_completed_json(
                mismatch_checkpoint_path
            )
            is None
        ):
            mismatch_result = (
                run_dense_mismatch_unit(
                    matched=(
                        evaluation_arrays[
                            "matched"
                        ][
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    text_swapped=(
                        evaluation_arrays[
                            "text_swapped"
                        ][
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    image_swapped=(
                        evaluation_arrays[
                            "image_swapped"
                        ][
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    image_only=(
                        evaluation_arrays[
                            "image_only"
                        ][
                            :,
                            depth_index,
                            :,
                        ]
                    ),
                    family=family,
                    dataset=dataset,
                    depth_index=(
                        depth_index
                    ),
                    depth_fraction=float(
                        depth_fraction
                    ),
                )
            )

            atomic_json_write(
                mismatch_result,
                mismatch_checkpoint_path,
            )

        primary_progress.update(1)

    del (
        llm_train_all,
        vlm_train_all,
        evaluation_arrays,
    )

    gc.collect()

primary_progress.close()


if RUN_CROSS_DATASET_TRANSFER:
    families = sorted(
        primary_index[
            "family"
        ].unique().tolist()
    )

    datasets = sorted(
        primary_index[
            "dataset"
        ].unique().tolist()
    )

    ordered_dataset_pairs = [
        (
            source_dataset,
            target_dataset,
        )
        for source_dataset
        in datasets
        for target_dataset
        in datasets
        if (
            source_dataset
            != target_dataset
        )
    ]

    cross_unit_count = (
        len(families)
        * len(
            ordered_dataset_pairs
        )
        * CONTRACT
        .expected_depth_count
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ZERO-TARGET-FIT "
        "CROSS-DATASET TRANSFER"
    )

    print("=" * 72)

    print(
        "Ordered source→target pairs: "
        f"{len(ordered_dataset_pairs)}"
    )

    print(
        "Family–pair–depth units:     "
        f"{cross_unit_count}"
    )

    cross_progress = tqdm(
        total=cross_unit_count,
        desc=(
            "Cross-dataset transfer units"
        ),
        unit="unit",
    )

    for family in families:
        family_rows = primary_index[
            primary_index[
                "family"
            ]
            == family
        ]

        family_cache: dict[
            str,
            dict[
                str,
                np.ndarray,
            ],
        ] = {}

        for _, row in (
            family_rows.iterrows()
        ):
            dataset = str(
                row[
                    "dataset"
                ]
            )

            path = Path(
                row[
                    "path"
                ]
            ).resolve()

            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                family_cache[
                    dataset
                ] = {
                    "depth_fractions": (
                        np.asarray(
                            archive[
                                "depth_fractions"
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "llm_train": (
                        np.asarray(
                            archive[
                                "llm_train"
                            ],
                            dtype=np.float32,
                        )
                    ),
                    "vlm_train": (
                        np.asarray(
                            archive[
                                "vlm_train"
                            ],
                            dtype=np.float32,
                        )
                    ),
                }

        reference_depths = (
            family_cache[
                datasets[0]
            ][
                "depth_fractions"
            ]
        )

        for dataset in datasets[1:]:
            if not np.array_equal(
                family_cache[
                    dataset
                ][
                    "depth_fractions"
                ],
                reference_depths,
            ):
                raise RuntimeError(
                    "Depth fractions differ "
                    "across datasets for family "
                    f"{family}."
                )

        for (
            source_dataset,
            target_dataset,
        ) in ordered_dataset_pairs:
            source_data = family_cache[
                source_dataset
            ]

            target_data = family_cache[
                target_dataset
            ]

            for (
                depth_index,
                depth_fraction,
            ) in enumerate(
                reference_depths
            ):
                checkpoint_path = (
                    CROSS_DATASET_CHECKPOINT_DIR
                    / unit_filename(
                        family,
                        (
                            "source_"
                            f"{source_dataset}"
                        ),
                        (
                            "target_"
                            f"{target_dataset}"
                        ),
                        f"depth_{depth_index}",
                    )
                )

                if (
                    read_completed_json(
                        checkpoint_path
                    )
                    is None
                ):
                    transfer_result = (
                        run_cross_dataset_retrieval_unit(
                            source_llm=(
                                source_data[
                                    "llm_train"
                                ][
                                    :,
                                    depth_index,
                                    :,
                                ]
                            ),
                            source_vlm=(
                                source_data[
                                    "vlm_train"
                                ][
                                    :,
                                    depth_index,
                                    :,
                                ]
                            ),
                            target_llm=(
                                target_data[
                                    "llm_train"
                                ][
                                    :,
                                    depth_index,
                                    :,
                                ]
                            ),
                            target_vlm=(
                                target_data[
                                    "vlm_train"
                                ][
                                    :,
                                    depth_index,
                                    :,
                                ]
                            ),
                            family=family,
                            source_dataset=(
                                source_dataset
                            ),
                            target_dataset=(
                                target_dataset
                            ),
                            depth_index=(
                                depth_index
                            ),
                            depth_fraction=float(
                                depth_fraction
                            ),
                        )
                    )

                    atomic_json_write(
                        transfer_result,
                        checkpoint_path,
                    )

                cross_progress.update(1)

        del family_cache

        gc.collect()

    cross_progress.close()


def load_checkpoint_payloads(
    directory: Path,
) -> list[
    dict[str, Any]
]:
    payloads = []

    for path in sorted(
        directory.glob(
            "*.json"
        )
    ):
        payload = read_completed_json(
            path
        )

        if payload is None:
            raise RuntimeError(
                "Incomplete or invalid "
                f"checkpoint: {path}"
            )

        payloads.append(
            payload
        )

    return payloads


retrieval_payloads = (
    load_checkpoint_payloads(
        RETRIEVAL_CHECKPOINT_DIR
    )
)

classification_payloads = (
    load_checkpoint_payloads(
        CLASSIFICATION_CHECKPOINT_DIR
    )
)

mismatch_payloads = (
    load_checkpoint_payloads(
        MISMATCH_CHECKPOINT_DIR
    )
)

cross_dataset_payloads = (
    load_checkpoint_payloads(
        CROSS_DATASET_CHECKPOINT_DIR
    )
    if RUN_CROSS_DATASET_TRANSFER
    else []
)


retrieval_summary_rows = []
retrieval_fold_rows = []

for payload in retrieval_payloads:
    common = {
        "family": (
            payload[
                "family"
            ]
        ),
        "dataset": (
            payload[
                "dataset"
            ]
        ),
        "pooling": (
            payload[
                "pooling"
            ]
        ),
        "depth_index": (
            payload[
                "depth_index"
            ]
        ),
        "depth_fraction": (
            payload[
                "depth_fraction"
            ]
        ),
        "candidate_pool": (
            payload[
                "candidate_pool"
            ]
        ),
    }

    for row in payload[
        "summary_rows"
    ]:
        retrieval_summary_rows.append(
            {
                **common,
                **row,
                **payload[
                    "neighborhood_metrics"
                ],
                **payload[
                    "chance_metrics"
                ],
            }
        )

    for row in payload[
        "fold_rows"
    ]:
        retrieval_fold_rows.append(
            {
                **common,
                **row,
            }
        )

retrieval_summary_df = (
    pd.DataFrame(
        retrieval_summary_rows
    )
    .sort_values(
        [
            "family",
            "dataset",
            "depth_index",
            "baseline",
        ],
        ignore_index=True,
    )
)

retrieval_fold_df = (
    pd.DataFrame(
        retrieval_fold_rows
    )
    .sort_values(
        [
            "family",
            "dataset",
            "depth_index",
            "fold",
            "baseline",
        ],
        ignore_index=True,
    )
)


classification_summary_rows = []
classification_fold_rows = []

for payload in (
    classification_payloads
):
    common = {
        "family": (
            payload[
                "family"
            ]
        ),
        "dataset": (
            payload[
                "dataset"
            ]
        ),
        "pooling": (
            payload[
                "pooling"
            ]
        ),
        "depth_index": (
            payload[
                "depth_index"
            ]
        ),
        "depth_fraction": (
            payload[
                "depth_fraction"
            ]
        ),
    }

    summary = payload[
        "summary"
    ]

    classification_summary_rows.append(
        {
            **common,
            "group_fold_sizes_json": json.dumps(
                payload[
                    "group_fold_sizes"
                ]
            ),
            "macro_f1": (
                summary[
                    "macro_f1"
                ]
            ),
            "macro_f1_ci95_low": (
                summary[
                    "macro_f1_ci95"
                ][0]
            ),
            "macro_f1_ci95_high": (
                summary[
                    "macro_f1_ci95"
                ][1]
            ),
            "accuracy": (
                summary[
                    "accuracy"
                ]
            ),
            "accuracy_ci95_low": (
                summary[
                    "accuracy_ci95"
                ][0]
            ),
            "accuracy_ci95_high": (
                summary[
                    "accuracy_ci95"
                ][1]
            ),
            "balanced_accuracy": (
                summary[
                    "balanced_accuracy"
                ]
            ),
            "balanced_accuracy_ci95_low": (
                summary[
                    "balanced_accuracy_ci95"
                ][0]
            ),
            "balanced_accuracy_ci95_high": (
                summary[
                    "balanced_accuracy_ci95"
                ][1]
            ),
            "matched_f1": (
                summary[
                    "per_class_f1"
                ][
                    "matched"
                ]
            ),
            "text_swapped_f1": (
                summary[
                    "per_class_f1"
                ][
                    "text_swapped"
                ]
            ),
            "image_swapped_f1": (
                summary[
                    "per_class_f1"
                ][
                    "image_swapped"
                ]
            ),
            "image_only_f1": (
                summary[
                    "per_class_f1"
                ][
                    "image_only"
                ]
            ),
            "confusion_matrix_json": json.dumps(
                summary[
                    "confusion_matrix"
                ]
            ),
            "null_control_run": (
                payload[
                    "null_control_run"
                ]
            ),
            "null_macro_f1": (
                payload.get(
                    "null_control_summary",
                    {},
                ).get(
                    "macro_f1"
                )
            ),
            "null_accuracy": (
                payload.get(
                    "null_control_summary",
                    {},
                ).get(
                    "accuracy"
                )
            ),
        }
    )

    for fold_row in payload[
        "fold_rows"
    ]:
        classification_fold_rows.append(
            {
                **common,
                "fold": (
                    fold_row[
                        "fold"
                    ]
                ),
                "test_group_count": (
                    fold_row[
                        "test_group_count"
                    ]
                ),
                "test_observation_count": (
                    fold_row[
                        "test_observation_count"
                    ]
                ),
                "macro_f1": (
                    fold_row[
                        "macro_f1"
                    ]
                ),
                "accuracy": (
                    fold_row[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": (
                    fold_row[
                        "balanced_accuracy"
                    ]
                ),
                "effective_pca_rank": (
                    fold_row[
                        "effective_pca_rank"
                    ]
                ),
                "pca_explained_variance_sum": (
                    fold_row[
                        "pca_explained_variance_sum"
                    ]
                ),
                "null_macro_f1": (
                    fold_row.get(
                        "null_control",
                        {},
                    ).get(
                        "macro_f1"
                    )
                ),
                "null_accuracy": (
                    fold_row.get(
                        "null_control",
                        {},
                    ).get(
                        "accuracy"
                    )
                ),
            }
        )

classification_summary_df = (
    pd.DataFrame(
        classification_summary_rows
    )
    .sort_values(
        [
            "family",
            "dataset",
            "depth_index",
        ],
        ignore_index=True,
    )
)

classification_fold_df = (
    pd.DataFrame(
        classification_fold_rows
    )
    .sort_values(
        [
            "family",
            "dataset",
            "depth_index",
            "fold",
        ],
        ignore_index=True,
    )
)


mismatch_summary_rows = []

for payload in mismatch_payloads:
    cosine = payload[
        "cosine"
    ]

    relative_l2 = payload[
        "relative_l2"
    ]

    mismatch_summary_rows.append(
        {
            "family": (
                payload[
                    "family"
                ]
            ),
            "dataset": (
                payload[
                    "dataset"
                ]
            ),
            "pooling": (
                payload[
                    "pooling"
                ]
            ),
            "depth_index": (
                payload[
                    "depth_index"
                ]
            ),
            "depth_fraction": (
                payload[
                    "depth_fraction"
                ]
            ),
            "cosine_text_mean": (
                cosine[
                    "text_swapped"
                ][
                    "mean"
                ]
            ),
            "cosine_text_ci95_low": (
                cosine[
                    "text_swapped"
                ][
                    "ci95"
                ][0]
            ),
            "cosine_text_ci95_high": (
                cosine[
                    "text_swapped"
                ][
                    "ci95"
                ][1]
            ),
            "cosine_image_mean": (
                cosine[
                    "image_swapped"
                ][
                    "mean"
                ]
            ),
            "cosine_image_ci95_low": (
                cosine[
                    "image_swapped"
                ][
                    "ci95"
                ][0]
            ),
            "cosine_image_ci95_high": (
                cosine[
                    "image_swapped"
                ][
                    "ci95"
                ][1]
            ),
            "cosine_image_only_mean": (
                cosine[
                    "image_only"
                ][
                    "mean"
                ]
            ),
            "cosine_combined_mismatch": (
                cosine[
                    "combined_text_image_mean"
                ]
            ),
            "cosine_image_minus_text": (
                cosine[
                    "image_minus_text"
                ][
                    "mean"
                ]
            ),
            "cosine_asymmetry_ci95_low": (
                cosine[
                    "image_minus_text"
                ][
                    "ci95"
                ][0]
            ),
            "cosine_asymmetry_ci95_high": (
                cosine[
                    "image_minus_text"
                ][
                    "ci95"
                ][1]
            ),
            "cosine_asymmetry_cohen_dz": (
                cosine[
                    "image_minus_text"
                ][
                    "cohen_dz"
                ]
            ),
            "cosine_asymmetry_p": (
                cosine[
                    "image_minus_text"
                ][
                    "sign_flip_p_value"
                ]
            ),
            "relative_l2_text_mean": (
                relative_l2[
                    "text_swapped"
                ][
                    "mean"
                ]
            ),
            "relative_l2_image_mean": (
                relative_l2[
                    "image_swapped"
                ][
                    "mean"
                ]
            ),
            "relative_l2_image_only_mean": (
                relative_l2[
                    "image_only"
                ][
                    "mean"
                ]
            ),
            "relative_l2_combined_mismatch": (
                relative_l2[
                    "combined_text_image_mean"
                ]
            ),
            "relative_l2_image_minus_text": (
                relative_l2[
                    "image_minus_text"
                ][
                    "mean"
                ]
            ),
            "relative_l2_asymmetry_p": (
                relative_l2[
                    "image_minus_text"
                ][
                    "sign_flip_p_value"
                ]
            ),
        }
    )

mismatch_summary_df = (
    pd.DataFrame(
        mismatch_summary_rows
    )
    .sort_values(
        [
            "family",
            "dataset",
            "depth_index",
        ],
        ignore_index=True,
    )
)


cross_dataset_rows = []

for payload in (
    cross_dataset_payloads
):
    common = {
        "family": (
            payload[
                "family"
            ]
        ),
        "source_dataset": (
            payload[
                "source_dataset"
            ]
        ),
        "target_dataset": (
            payload[
                "target_dataset"
            ]
        ),
        "pooling": (
            payload[
                "pooling"
            ]
        ),
        "depth_index": (
            payload[
                "depth_index"
            ]
        ),
        "depth_fraction": (
            payload[
                "depth_fraction"
            ]
        ),
        "target_candidate_pool": (
            payload[
                "target_candidate_pool"
            ]
        ),
        "target_data_used_for_fitting": (
            payload[
                "target_data_used_for_fitting"
            ]
        ),
    }

    for row in payload[
        "summary_rows"
    ]:
        cross_dataset_rows.append(
            {
                **common,
                **row,
                **payload[
                    "neighborhood_metrics"
                ],
                **payload[
                    "chance_metrics"
                ],
            }
        )

cross_dataset_df = (
    pd.DataFrame(
        cross_dataset_rows
    )
)

if not cross_dataset_df.empty:
    cross_dataset_df = (
        cross_dataset_df.sort_values(
            [
                "family",
                "source_dataset",
                "target_dataset",
                "depth_index",
                "baseline",
            ],
            ignore_index=True,
        )
    )


retrieval_summary_path = (
    EVALUATION_ROOT
    / "paired_retrieval_summary.csv"
)

retrieval_fold_path = (
    EVALUATION_ROOT
    / "paired_retrieval_fold_metrics.csv"
)

classification_summary_path = (
    EVALUATION_ROOT
    / "condition_classification_summary.csv"
)

classification_fold_path = (
    EVALUATION_ROOT
    / "condition_classification_fold_metrics.csv"
)

mismatch_summary_path = (
    EVALUATION_ROOT
    / "dense_mismatch_summary.csv"
)

cross_dataset_path = (
    EVALUATION_ROOT
    / "cross_dataset_retrieval_summary.csv"
)

atomic_csv_write(
    retrieval_summary_df,
    retrieval_summary_path,
)

atomic_csv_write(
    retrieval_fold_df,
    retrieval_fold_path,
)

atomic_csv_write(
    classification_summary_df,
    classification_summary_path,
)

atomic_csv_write(
    classification_fold_df,
    classification_fold_path,
)

atomic_csv_write(
    mismatch_summary_df,
    mismatch_summary_path,
)

if not cross_dataset_df.empty:
    atomic_csv_write(
        cross_dataset_df,
        cross_dataset_path,
    )


expected_primary_units = (
    9
    * CONTRACT.expected_depth_count
)

expected_retrieval_rows = (
    expected_primary_units
    * 3
)

expected_classification_rows = (
    expected_primary_units
)

expected_mismatch_rows = (
    expected_primary_units
)

expected_cross_dataset_units = (
    3
    * 6
    * CONTRACT.expected_depth_count
)

expected_cross_dataset_rows = (
    expected_cross_dataset_units
    * 3
)


def dataframe_numeric_values_finite(
    dataframe: pd.DataFrame,
) -> bool:
    numeric = dataframe.select_dtypes(
        include=[
            np.number
        ]
    )

    if numeric.empty:
        return True

    return bool(
        np.isfinite(
            numeric.to_numpy()
        ).all()
    )


integrity_checks = {
    "retrieval_checkpoint_units": (
        len(
            retrieval_payloads
        )
        == expected_primary_units
    ),
    "retrieval_summary_rows": (
        len(
            retrieval_summary_df
        )
        == expected_retrieval_rows
    ),
    "classification_checkpoint_units": (
        len(
            classification_payloads
        )
        == expected_primary_units
    ),
    "classification_summary_rows": (
        len(
            classification_summary_df
        )
        == expected_classification_rows
    ),
    "mismatch_checkpoint_units": (
        len(
            mismatch_payloads
        )
        == expected_primary_units
    ),
    "mismatch_summary_rows": (
        len(
            mismatch_summary_df
        )
        == expected_mismatch_rows
    ),
    "cross_dataset_checkpoint_units": (
        (
            len(
                cross_dataset_payloads
            )
            == expected_cross_dataset_units
        )
        if RUN_CROSS_DATASET_TRANSFER
        else True
    ),
    "cross_dataset_summary_rows": (
        (
            len(
                cross_dataset_df
            )
            == expected_cross_dataset_rows
        )
        if RUN_CROSS_DATASET_TRANSFER
        else True
    ),
    "retrieval_numeric_finite": (
        dataframe_numeric_values_finite(
            retrieval_summary_df
        )
    ),
    "classification_numeric_finite": (
        dataframe_numeric_values_finite(
            classification_summary_df.drop(
                columns=[
                    "null_macro_f1",
                    "null_accuracy",
                ],
                errors="ignore",
            )
        )
    ),
    "mismatch_numeric_finite": (
        dataframe_numeric_values_finite(
            mismatch_summary_df
        )
    ),
    "cross_dataset_numeric_finite": (
        dataframe_numeric_values_finite(
            cross_dataset_df
        )
        if not cross_dataset_df.empty
        else True
    ),
}

aligned_retrieval = (
    retrieval_summary_df[
        retrieval_summary_df[
            "baseline"
        ]
        == "aligned"
    ]
)

shuffled_retrieval = (
    retrieval_summary_df[
        retrieval_summary_df[
            "baseline"
        ]
        == "shuffled_alignment"
    ]
)

null_classification_rows = (
    classification_summary_df[
        classification_summary_df[
            "null_control_run"
        ]
        == True
    ]
)

control_diagnostics = {
    "aligned_mean_mrr": float(
        aligned_retrieval[
            "mrr"
        ].mean()
    ),
    "shuffled_mean_mrr": float(
        shuffled_retrieval[
            "mrr"
        ].mean()
    ),
    "null_mean_macro_f1": (
        float(
            null_classification_rows[
                "null_macro_f1"
            ].mean()
        )
        if not null_classification_rows.empty
        else None
    ),
}

all_integrity_checks_pass = bool(
    all(
        integrity_checks.values()
    )
)

integrity_report = {
    "completed": (
        all_integrity_checks_pass
    ),
    "analysis_version": (
        ANALYSIS_VERSION
    ),
    "integrity_checks": (
        integrity_checks
    ),
    "control_diagnostics": (
        control_diagnostics
    ),
    "expected_counts": {
        "primary_depth_units": (
            expected_primary_units
        ),
        "retrieval_summary_rows": (
            expected_retrieval_rows
        ),
        "classification_summary_rows": (
            expected_classification_rows
        ),
        "mismatch_summary_rows": (
            expected_mismatch_rows
        ),
        "cross_dataset_summary_rows": (
            expected_cross_dataset_rows
        ),
    },
    "output_files": {
        "retrieval_summary": str(
            retrieval_summary_path
        ),
        "retrieval_fold_metrics": str(
            retrieval_fold_path
        ),
        "classification_summary": str(
            classification_summary_path
        ),
        "classification_fold_metrics": str(
            classification_fold_path
        ),
        "dense_mismatch_summary": str(
            mismatch_summary_path
        ),
        "cross_dataset_retrieval": (
            str(
                cross_dataset_path
            )
            if not cross_dataset_df.empty
            else None
        ),
        "run_specification": str(
            RUN_SPECIFICATION_PATH
        ),
    },
    "completed_at_unix": (
        time.time()
    ),
}

integrity_report_path = (
    AUDIT_DIR
    / "primary_validation_integrity.json"
)

atomic_json_write(
    integrity_report,
    integrity_report_path,
)

if not all_integrity_checks_pass:
    failed_checks = [
        key
        for key, value
        in integrity_checks.items()
        if value is not True
    ]

    raise RuntimeError(
        "Primary validation integrity verification failed:\n"
        + "\n".join(
            failed_checks
        )
    )

PRIMARY_VALIDATION_READY = True


print(
    "\n"
    + "=" * 72
)

print(
    "PRIMARY VALIDATION SUMMARY"
)

print("=" * 72)

print(
    "Retrieval units:          "
    f"{len(retrieval_payloads)} "
    f"/ {expected_primary_units}"
)

print(
    "Classification units:     "
    f"{len(classification_payloads)} "
    f"/ {expected_primary_units}"
)

print(
    "Dense mismatch units:     "
    f"{len(mismatch_payloads)} "
    f"/ {expected_primary_units}"
)

print(
    "Cross-dataset units:      "
    f"{len(cross_dataset_payloads)} "
    f"/ {expected_cross_dataset_units}"
)

print(
    "Integrity verification:  "
    f"{'PASS' if all_integrity_checks_pass else 'FAIL'}"
)

print(
    "Primary validation ready: "
    f"{PRIMARY_VALIDATION_READY}"
)

print(
    "\nControl diagnostics:"
)

print(
    json.dumps(
        control_diagnostics,
        indent=2,
        ensure_ascii=False,
    )
)

late_depth_retrieval = (
    aligned_retrieval
    .sort_values(
        "depth_fraction"
    )
    .groupby(
        [
            "family",
            "dataset",
        ],
        as_index=False,
    )
    .tail(1)
    [
        [
            "family",
            "dataset",
            "depth_fraction",
            "mrr",
            "recall_at_1",
            "recall_at_5",
            "knn_overlap_at_20",
        ]
    ]
)

late_depth_classification = (
    classification_summary_df
    .sort_values(
        "depth_fraction"
    )
    .groupby(
        [
            "family",
            "dataset",
        ],
        as_index=False,
    )
    .tail(1)
    [
        [
            "family",
            "dataset",
            "depth_fraction",
            "macro_f1",
            "accuracy",
            "balanced_accuracy",
        ]
    ]
)

late_depth_mismatch = (
    mismatch_summary_df
    .sort_values(
        "depth_fraction"
    )
    .groupby(
        [
            "family",
            "dataset",
        ],
        as_index=False,
    )
    .tail(1)
    [
        [
            "family",
            "dataset",
            "depth_fraction",
            "cosine_combined_mismatch",
            "cosine_image_minus_text",
            "cosine_asymmetry_p",
        ]
    ]
)

if display is not None:
    print(
        "\nPaired representation retrieval:"
    )

    display(
        late_depth_retrieval
        .reset_index(
            drop=True
        )
    )

    print(
        "\nCondition classification:"
    )

    display(
        late_depth_classification
        .reset_index(
            drop=True
        )
    )

    print(
        "\nDense mismatch:"
    )

    display(
        late_depth_mismatch
        .reset_index(
            drop=True
        )
    )

else:
    print(
        late_depth_retrieval
        .to_string(
            index=False
        )
    )

    print(
        late_depth_classification
        .to_string(
            index=False
        )
    )

    print(
        late_depth_mismatch
        .to_string(
            index=False
        )
    )

print(
    "\nGenerated files:"
)

for key, value in (
    integrity_report[
        "output_files"
    ].items()
):
    print(
        f"- {key}: {value}"
    )

print(
    "- integrity_report: "
    f"{integrity_report_path}"
)

print(
    "\nPrimary validation completed successfully."
)
