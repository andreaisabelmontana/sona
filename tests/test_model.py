"""End-to-end checks use one small training run, then real persisted inference."""

import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sona.constants import CLUSTER_FEATURES, INTERACTION_FEATURES, MODEL_FEATURES, MOODS, NUMERIC_FEATURES
from sona.model import load_bundle, predict_df, train
from sona.preprocessing import RawFeatureTransformer


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_tracks):
    directory = tmp_path_factory.mktemp("sona_training")
    csv_path = directory / "tracks.csv"
    # One track can appear under several genres in the source catalog.
    repeated = synthetic_tracks.iloc[:30].copy()
    repeated["genre"] = "Another listing"
    pd.concat([synthetic_tracks, repeated], ignore_index=True).to_csv(csv_path, index=False)
    fitted_rows = []
    cluster_fit_values = []
    original_fit = StandardScaler.fit
    original_cluster_fit = KMeans.fit

    def audited_fit(scaler, features, *args, **kwargs):
        if isinstance(features, pd.DataFrame):
            fitted_rows.append((tuple(features.columns), tuple(features.index)))
        return original_fit(scaler, features, *args, **kwargs)

    def audited_cluster_fit(clusterer, features, *args, **kwargs):
        cluster_fit_values.append(np.asarray(features).copy())
        return original_cluster_fit(clusterer, features, *args, **kwargs)

    output_dir = directory / "artifacts"
    with (
        patch.object(StandardScaler, "fit", audited_fit),
        patch.object(KMeans, "fit", audited_cluster_fit),
    ):
        train(csv_path, output_dir=output_dir, seed=42, extended=False, smote=False)
    return {
        "bundle": load_bundle(output_dir / "sona.joblib"),
        "output_dir": output_dir,
        "fitted_rows": fitted_rows,
        "cluster_fit_values": cluster_fit_values,
        "source": synthetic_tracks,
    }


def test_persisted_model_predicts_probabilities_and_preserves_index(trained):
    frame = trained["source"].iloc[[0, 70, 110, 180, 230]].copy()
    frame.index = pd.Index([9, 4, 4, 100, 2], name="original_row")
    before = frame.copy(deep=True)
    result = predict_df(trained["bundle"], frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_index_equal(result.index, frame.index)
    assert set(result["mood"]).issubset(MOODS)
    probabilities = result[[f"probability_{mood}" for mood in MOODS]]
    assert np.isfinite(probabilities.to_numpy()).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all().all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(result["confidence"], probabilities.max(axis=1))
    assert result["mood"].tolist() == probabilities.idxmax(axis=1).str.removeprefix("probability_").tolist()


def test_inference_accepts_equivalent_raw_categories(trained):
    numeric = trained["source"].iloc[:8].copy()
    numeric["key"] = [0, 1, 3, 6, 8, 10, 11, 2]
    numeric["mode"] = [0, 1] * 4
    numeric["time_signature"] = 4
    human = numeric.copy()
    human["key"] = ["C", "Db", "Eb", "Gb", "Ab", "Bb", "B", "D"]
    human["mode"] = ["Minor", "Major"] * 4
    human["time_signature"] = "4/4"
    pd.testing.assert_frame_equal(
        predict_df(trained["bundle"], numeric), predict_df(trained["bundle"], human)
    )


def test_cluster_features_and_metadata_cannot_change_prediction(trained):
    frame = trained["source"].iloc[::40].copy()
    baseline = predict_df(trained["bundle"], frame[MODEL_FEATURES])
    frame[CLUSTER_FEATURES] = [0.0, 1.0, 0.0]
    frame["genre"] = "unseen genre"
    frame["popularity"] = 999
    frame["track_name"] = "metadata is not a predictor"
    pd.testing.assert_frame_equal(baseline, predict_df(trained["bundle"], frame))
    assert set(CLUSTER_FEATURES).isdisjoint(MODEL_FEATURES)


@pytest.mark.parametrize("case", ["missing", "out_of_range", "non_finite"])
def test_inference_rejects_invalid_input(trained, case):
    frame = trained["source"].iloc[:1].copy()
    if case == "missing":
        frame = frame.drop(columns="tempo")
    elif case == "out_of_range":
        frame["danceability"] = 1.01
    else:
        frame["tempo"] = np.inf
    with pytest.raises(ValueError):
        predict_df(trained["bundle"], frame)


def test_artifact_serialization_roundtrip(trained, tmp_path):
    frame = trained["source"].iloc[::25]
    expected = predict_df(trained["bundle"], frame)
    second_path = tmp_path / "roundtrip.joblib"
    joblib.dump(trained["bundle"], second_path)
    reloaded = load_bundle(second_path)
    pd.testing.assert_frame_equal(expected, predict_df(reloaded, frame))


def test_training_produces_readable_evaluation_artifacts(trained):
    directory = trained["output_dir"]
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert isinstance(report, dict) and report
    profiles = pd.read_csv(directory / "cluster_profiles.csv")
    assert len(profiles) == 5
    assert set(CLUSTER_FEATURES).issubset(profiles.columns)
    assert (directory / "confusion_matrix.png").stat().st_size > 1000


def test_all_scalers_fit_the_same_training_rows_only(trained):
    recorded = trained["fitted_rows"]
    assert recorded, "Audit must observe actual scaler fits, not just report fields."
    fit_indices = [set(indices) for _, indices in recorded]
    assert all(indices == fit_indices[0] for indices in fit_indices)
    assert 0 < len(fit_indices[0]) < len(trained["source"])
    assert any(set(columns) == set(CLUSTER_FEATURES) for columns, _ in recorded)
    assert any(set(columns).isdisjoint(CLUSTER_FEATURES) for columns, _ in recorded)


def test_duplicate_tracks_are_isolated_and_holdouts_never_fit_scalers(trained):
    manifest = pd.read_csv(trained["output_dir"] / "split_manifest.csv")
    assert len(manifest) == len(trained["source"])
    assert manifest["track_id"].is_unique
    assert set(manifest["split"]) == {"train", "validation", "test"}
    training_rows = set(manifest.loc[manifest["split"].eq("train"), "source_row"])
    holdout_rows = set(manifest.loc[manifest["split"].ne("train"), "source_row"])
    assert training_rows.isdisjoint(holdout_rows)
    for _, fit_rows in trained["fitted_rows"]:
        assert set(fit_rows) == training_rows
        assert set(fit_rows).isdisjoint(holdout_rows)

    source = trained["source"].loc[sorted(training_rows)]
    bundle = trained["bundle"]
    cluster_scaler = bundle["cluster_pipeline"].named_steps["scaler"]
    np.testing.assert_allclose(cluster_scaler.mean_, source[CLUSTER_FEATURES].mean())
    training_in_fit_order = trained["source"].loc[
        manifest.loc[manifest["split"].eq("train"), "source_row"]
    ]
    assert len(trained["cluster_fit_values"]) == 1
    np.testing.assert_allclose(
        trained["cluster_fit_values"][0],
        cluster_scaler.transform(training_in_fit_order[CLUSTER_FEATURES]),
    )
    feature_scaler = (
        bundle["model"].named_steps["preprocessing"]
        .named_steps["columns"].named_transformers_["numeric"]
    )
    transformed = RawFeatureTransformer().transform(source)
    np.testing.assert_allclose(
        feature_scaler.mean_, transformed[NUMERIC_FEATURES + INTERACTION_FEATURES].mean()
    )
    metrics = json.loads((trained["output_dir"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data"]["duplicate_tracks_removed"] == 30
    assert metrics["data"]["rows_used"] == len(trained["source"])
    assert sum(map(sum, metrics["test"]["confusion_matrix"])) == manifest["split"].eq("test").sum()


def test_incomplete_artifact_reports_actionable_error(tmp_path):
    path = tmp_path / "incomplete.joblib"
    joblib.dump({"schema_version": "unsupported"}, path)
    with pytest.raises(ValueError, match="schema|Retrain|retrain"):
        load_bundle(path)
