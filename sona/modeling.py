"""Train-only clustering, classifier comparison, persistence and prediction."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .constants import CLUSTER_FEATURES, MODEL_FEATURES, MOODS, MOOD_PROTOTYPES, SCHEMA_VERSION
from .preprocessing import make_preprocessor, validate_features


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "imbalanced-learn", "xgboost"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _log(message: str) -> None:
    print(f"[Sona] {message}", flush=True)


def assign_mood_names(centroids: np.ndarray) -> dict[int, str]:
    """Find the minimum-distance one-to-one matching to named mood prototypes.

    Centroids are in original [0, 1] audio-feature units. Sorting them first makes
    ties deterministic independently of arbitrary KMeans cluster numbering.
    The mood names describe clusters; they are not observed listener emotions.
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.shape != (5, 3) or not np.isfinite(centroids).all():
        raise ValueError("Mood assignment requires five finite, three-feature centroids.")
    order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    prototypes = np.array([MOOD_PROTOTYPES[mood] for mood in MOODS])
    cost = ((centroids[order, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    rows, columns = linear_sum_assignment(cost)
    return {int(order[row]): MOODS[column] for row, column in zip(rows, columns)}


def _labels(cluster_ids: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    return np.array([MOODS.index(mapping[int(cluster)]) for cluster in cluster_ids], dtype=int)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(MOODS))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=MOODS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _estimators(seed: int, extended: bool) -> tuple[dict[str, Any], list[str]]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, n_jobs=1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    skipped = []
    if extended:
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=150, random_state=seed
        )
        try:
            from xgboost import XGBClassifier
        except ImportError:
            skipped.append("xgboost: install Sona's research extra to include this classifier")
        else:
            models["xgboost"] = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                tree_method="hist", n_jobs=4, random_state=seed,
            )
    return models, skipped


def _classifier_pipeline(estimator: Any, y_train: np.ndarray, seed: int, smote: bool) -> Any:
    preprocessing = make_preprocessor()
    steps = [("preprocessing", preprocessing)]
    if smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as SamplingPipeline
        except ImportError as exc:
            raise ValueError("--smote requires imbalanced-learn; install Sona's research extra.") from exc
        smallest_class = int(np.bincount(y_train, minlength=5).min())
        if smallest_class < 2:
            raise ValueError("SMOTE requires at least two training tracks in every mood cluster; increase --sample.")
        # imbalanced-learn rejects nested Pipeline instances as intermediate
        # steps. Flatten the same raw/column transforms without changing fit scope.
        steps = list(preprocessing.steps)
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, smallest_class - 1))))
        steps.append(("classifier", estimator))
        return SamplingPipeline(steps)
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _load_training_data(csv_path: str | Path, sample: int | None, seed: int) -> tuple[pd.DataFrame, dict]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Supply the SpotifyFeatures.csv path with --data.")
    frame = pd.read_csv(path)
    required = ["track_id"] + MODEL_FEATURES + CLUSTER_FEATURES
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Training CSV is missing columns: " + ", ".join(missing))
    counts = {"input_rows": len(frame), "comedy_removed": 0, "duplicate_tracks_removed": 0, "missing_rows_removed": 0}
    if "genre" in frame:
        comedy = frame["genre"].astype(str).str.strip().str.casefold().eq("comedy")
        counts["comedy_removed"] = int(comedy.sum())
        frame = frame.loc[~comedy].copy()
    # Track IDs are normalized before dropping duplicates to prevent leakage
    # through repeated rows under different genre labels.
    frame["track_id"] = frame["track_id"].astype("string").str.strip().replace("", pd.NA)
    before = len(frame)
    frame = frame.dropna(subset=required)
    counts["missing_rows_removed"] = before - len(frame)
    before = len(frame)
    frame = frame.drop_duplicates(subset="track_id", keep="first")
    counts["duplicate_tracks_removed"] = before - len(frame)
    counts["unique_tracks_available"] = len(frame)
    if sample is not None:
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 50:
            raise ValueError("--sample must be an integer of at least 50.")
        frame = frame.sample(n=min(sample, len(frame)), random_state=seed)
    if len(frame) < 50:
        raise ValueError("At least 50 valid, distinct non-Comedy tracks are required for a 90/5/5 split.")
    canonical = validate_features(frame, include_cluster=True)
    for column in canonical.columns:
        frame[column] = canonical[column]
    counts["rows_used"] = len(frame)
    counts["genre_column_present"] = "genre" in frame
    return frame, counts


def train(
    csv_path: str | Path,
    output_dir: str | Path = "artifacts",
    sample: int | None = None,
    seed: int = 42,
    extended: bool = False,
    smote: bool = False,
) -> dict[str, Any]:
    """Train Sona and save the selected model and honest held-out evaluation.

    Deduplication precedes the random 90/5/5 split. Clustering, scaling,
    categorical encoding and optional SMOTE fit only on the training split.
    Validation macro F1 selects the classifier; the test split is scored once.
    """
    _log(f"Reading {csv_path}")
    frame, data_counts = _load_training_data(csv_path, sample, seed)
    training, heldout = train_test_split(frame, test_size=0.10, random_state=seed)
    validation, test = train_test_split(heldout, test_size=0.50, random_state=seed)
    data_counts.update(train_rows=len(training), validation_rows=len(validation), test_rows=len(test))
    _log(f"Unique tracks: {len(frame):,}; train / validation / test: {len(training):,} / {len(validation):,} / {len(test):,}")
    if len(training[CLUSTER_FEATURES].drop_duplicates()) < 5:
        raise ValueError("Training data needs at least five distinct valence/energy/acousticness profiles.")
    cluster_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=5, n_init=20, random_state=seed)),
    ])
    with threadpool_limits(limits=1):
        cluster_pipeline.fit(training[CLUSTER_FEATURES])
        clusters = {
            "train": cluster_pipeline.predict(training[CLUSTER_FEATURES]),
            "validation": cluster_pipeline.predict(validation[CLUSTER_FEATURES]),
            "test": cluster_pipeline.predict(test[CLUSTER_FEATURES]),
        }
    centers = cluster_pipeline.named_steps["scaler"].inverse_transform(
        cluster_pipeline.named_steps["kmeans"].cluster_centers_
    )
    mapping = assign_mood_names(centers)
    y_train = _labels(clusters["train"], mapping)
    y_validation = _labels(clusters["validation"], mapping)
    y_test = _labels(clusters["test"], mapping)
    if len(np.unique(y_train)) != 5:
        raise ValueError("KMeans did not produce five populated training clusters; use more varied tracks.")

    models, skipped = _estimators(seed, extended)
    comparisons: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in models.items():
        _log(f"Training {name}...")
        pipeline = _classifier_pipeline(estimator, y_train, seed, smote)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            pipeline.fit(training[MODEL_FEATURES], y_train)
            prediction = pipeline.predict(validation[MODEL_FEATURES])
        comparisons[name] = {"validation": _score(y_validation, prediction), "fit_seconds": round(time.perf_counter() - started, 3)}
        fitted[name] = pipeline
        _log(f"{name}: validation macro F1 = {comparisons[name]['validation']['macro_f1']:.4f}")
    selected = max(comparisons, key=lambda name: (
        comparisons[name]["validation"]["macro_f1"], comparisons[name]["validation"]["accuracy"]
    ))
    model = fitted[selected]
    with threadpool_limits(limits=1):
        test_predictions = model.predict(test[MODEL_FEATURES])
    versions = _versions()
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "selected_model": selected,
        "selection_metric": "validation_macro_f1",
        "seed": seed,
        "candidates": comparisons,
        "skipped_candidates": skipped,
        "test": _score(y_test, test_predictions),
        "data": data_counts,
        "features": {"classifier": MODEL_FEATURES, "clustering": CLUSTER_FEATURES, "interactions": ["dance_loud", "speech_tempo"]},
        "mood_mapping": {str(cluster): mood for cluster, mood in mapping.items()},
        "mood_prototypes": MOOD_PROTOTYPES,
        "mood_assignment": "Minimum total squared Euclidean distance, one-to-one, between training cluster centroids in original feature units and the original notebook's human-named centroids.",
        "label_caveat": "Labels are unsupervised audio-profile groups with heuristic mood names, not listener-reported emotions. Confidence is an uncalibrated classifier probability.",
        "preprocessing_fit_scope": "training split only; validation selects model; test used only after selection",
        "split": "random 90/5/5 after track-id deduplication; not stratified because labels are learned from the training split",
        "smote": smote,
        "smote_caveat": "When enabled, SMOTE interpolates the scaled numeric and one-hot categorical training features only." if smote else None,
        "versions": versions,
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "cluster_pipeline": cluster_pipeline,
        "cluster_to_mood": mapping,
        "mood_names": MOODS,
        "metadata": {"model_features": MODEL_FEATURES, "cluster_features": CLUSTER_FEATURES, "versions": versions, "selected_model": selected, "seed": seed, "smote": smote},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "sona.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    profiles = pd.DataFrame(centers, columns=CLUSTER_FEATURES)
    profiles.insert(0, "cluster_id", range(5))
    profiles.insert(1, "mood", [mapping[cluster] for cluster in range(5)])
    for split, ids in clusters.items():
        profiles[f"{split}_count"] = np.bincount(ids, minlength=5)
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    split_manifest = pd.concat([
        pd.DataFrame({"source_row": part.index, "track_id": part["track_id"].to_numpy(), "split": split})
        for split, part in (("train", training), ("validation", validation), ("test", test))
    ], ignore_index=True)
    split_manifest.to_csv(output / "split_manifest.csv", index=False)
    selected_test = test.iloc[:100]
    identity = [column for column in ("track_id", "track_name", "artist_name") if column in test]
    examples = selected_test[identity + MODEL_FEATURES].copy()
    examples["cluster_mood"] = [MOODS[label] for label in y_test[:len(selected_test)]]
    examples = pd.concat([examples, predict_df(bundle, selected_test)], axis=1)
    examples.to_csv(output / "heldout_predictions.csv", index=False)
    _save_confusion(metrics["test"]["confusion_matrix"], selected, output / "confusion_matrix.png")
    _log(f"Selected {selected}; test macro F1 = {metrics['test']['macro_f1']:.4f}; artifacts saved to {output}")
    return metrics


def _save_confusion(matrix: list[list[int]], selected: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    image = ax.imshow(values, cmap="Purples")
    ax.set(xticks=np.arange(5), yticks=np.arange(5), xticklabels=MOODS, yticklabels=MOODS,
           xlabel="Predicted mood", ylabel="Cluster-derived mood", title=f"Sona · held-out test set\n{selected.replace('_', ' ')}")
    threshold = values.max() / 2
    for row in range(5):
        for column in range(5):
            ax.text(column, row, str(values[row, column]), ha="center", va="center", color="white" if values[row, column] > threshold else "#281d42")
    fig.colorbar(image, ax=ax, label="Tracks", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a Sona artifact. Only load joblib files from a trusted source."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Model not found: {artifact}. Train it first with python -m sona train --data PATH.")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Sona model schema. Retrain the model with this version of Sona.")
    if not {"model", "cluster_pipeline", "metadata", "mood_names", "cluster_to_mood"}.issubset(bundle):
        raise ValueError("Incomplete Sona model bundle; retrain the model.")
    trained_version = bundle["metadata"].get("versions", {}).get("scikit-learn")
    current_version = importlib.metadata.version("scikit-learn")
    if trained_version and trained_version != current_version:
        warnings.warn(f"Model was trained with scikit-learn {trained_version}; current version is {current_version}. Use the training version or retrain for reliable persistence.", RuntimeWarning, stacklevel=2)
    return bundle


def predict_df(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict from raw audio features, preserving input order and index labels."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION or "model" not in bundle:
        raise ValueError("Expected a supported Sona model bundle returned by load_bundle().")
    normalized = validate_features(frame)
    columns = ["mood", "confidence"] + [f"probability_{mood}" for mood in MOODS]
    if len(normalized) == 0:
        return pd.DataFrame(index=frame.index, columns=columns)
    model = bundle["model"]
    with threadpool_limits(limits=1):
        raw_probabilities = model.predict_proba(normalized)
    probabilities = np.zeros((len(frame), len(MOODS)), dtype=float)
    for column, class_id in enumerate(model.classes_):
        probabilities[:, int(class_id)] = raw_probabilities[:, column]
    winners = probabilities.argmax(axis=1)
    output = pd.DataFrame({"mood": [MOODS[winner] for winner in winners], "confidence": probabilities.max(axis=1)}, index=frame.index)
    for index, mood in enumerate(MOODS):
        output[f"probability_{mood}"] = probabilities[:, index]
    return output
