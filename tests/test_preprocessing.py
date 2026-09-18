"""Raw input validation is shared by training and all inference entry points."""

import numpy as np
import pandas as pd
import pytest

from sona.constants import CLUSTER_FEATURES, MODEL_FEATURES
from sona.preprocessing import RawFeatureTransformer, validate_features


def test_equivalent_spotify_and_human_categories_normalize_identically(raw_tracks):
    numeric = raw_tracks.copy()
    numeric["key"] = [0, 1, 3, 6, 8, 10, 11, -1]
    numeric["mode"] = [0, 1] * 4
    numeric["time_signature"] = [3, 4] * 4
    text = numeric.copy()
    text["key"] = ["C", "Db", "E♭", "F#", "G♯", "Bb", "B", "Unknown"]
    text["mode"] = ["minor", " MAJOR "] * 4
    text["time_signature"] = ["3/4", "4/4"] * 4
    pd.testing.assert_frame_equal(validate_features(numeric), validate_features(text))


@pytest.mark.parametrize("column", MODEL_FEATURES)
def test_missing_required_feature_is_explained(raw_tracks, column):
    with pytest.raises(ValueError, match=column):
        validate_features(raw_tracks.drop(columns=column))


@pytest.mark.parametrize("column,value", [
    ("danceability", -0.01), ("danceability", 1.01),
    ("instrumentalness", 2), ("liveness", -1), ("speechiness", 1.1),
    ("tempo", -1), ("tempo", np.inf), ("loudness", np.nan),
    ("key", 12), ("key", "H"), ("mode", 2),
    ("time_signature", "4/8"), ("time_signature", 8),
])
def test_invalid_values_fail_with_feature_context(raw_tracks, column, value):
    raw_tracks[column] = value
    with pytest.raises(ValueError, match=column):
        validate_features(raw_tracks)


def test_cluster_features_required_only_when_constructing_labels(raw_tracks):
    prediction_input = raw_tracks.drop(columns=CLUSTER_FEATURES)
    assert list(validate_features(prediction_input)) == MODEL_FEATURES
    with pytest.raises(ValueError, match="valence"):
        validate_features(prediction_input, include_cluster=True)
    raw_tracks["energy"] = 2
    with pytest.raises(ValueError, match="energy"):
        validate_features(raw_tracks, include_cluster=True)


def test_validation_preserves_index_and_does_not_mutate_input(raw_tracks):
    raw_tracks.index = pd.Index([9, 4, 4, 22, 0, 61, 17, 99], name="source_row")
    before = raw_tracks.copy(deep=True)
    transformed = RawFeatureTransformer().fit_transform(raw_tracks)
    pd.testing.assert_frame_equal(raw_tracks, before)
    pd.testing.assert_index_equal(transformed.index, before.index)
    assert set(CLUSTER_FEATURES).isdisjoint(transformed.columns)
    np.testing.assert_allclose(
        transformed["dance_loud"], before["danceability"] * before["loudness"]
    )
    np.testing.assert_allclose(
        transformed["speech_tempo"], before["speechiness"] * before["tempo"]
    )


def test_ambiguous_duplicate_columns_are_rejected(raw_tracks):
    ambiguous = pd.concat([raw_tracks, raw_tracks[["tempo"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        validate_features(ambiguous)
