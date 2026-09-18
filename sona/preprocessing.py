"""One raw-input transformation shared by training, batch prediction and the UI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CATEGORICAL_FEATURES, CLUSTER_FEATURES, INTERACTION_FEATURES,
    MODEL_FEATURES, NUMERIC_FEATURES,
)

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KEY_ALIASES = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
                "CB": "B", "FB": "E", "E#": "F", "B#": "C"}
_UNIT_FEATURES = {"danceability", "instrumentalness", "liveness", "speechiness", *CLUSTER_FEATURES}


def _integer(value: object) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if np.isfinite(number) and number.is_integer():
        return int(number)
    return None


def _category(value: object, column: str) -> str:
    raw = str(value).strip()
    numeric = _integer(value)
    if column == "key":
        if numeric is not None and -1 <= numeric <= 11:
            return "Unknown" if numeric == -1 else _KEY_NAMES[numeric]
        text = raw.replace("♯", "#").replace("♭", "b").upper()
        text = _KEY_ALIASES.get(text, text)
        if text in _KEY_NAMES:
            return text
        if text in {"UNKNOWN", "NONE", "N/A"}:
            return "Unknown"
        raise ValueError("expected a pitch name (C, C#, Db, …) or a Spotify key integer -1–11")
    if column == "mode":
        if numeric in (0, 1):
            return "Minor" if numeric == 0 else "Major"
        if raw.lower() in {"minor", "major"}:
            return raw.title()
        raise ValueError("expected Major/Minor or 1/0")
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) != 2 or _integer(parts[1]) != 4:
            raise ValueError("expected a Spotify meter integer 0–7 or N/4 (for example 4/4)")
        numeric = _integer(parts[0])
    if numeric is not None and 0 <= numeric <= 7:
        return str(numeric)
    raise ValueError("expected a Spotify meter integer 0–7 or N/4 (for example 4/4)")


def validate_features(frame: pd.DataFrame, *, include_cluster: bool = False) -> pd.DataFrame:
    """Return canonical values without modifying the caller's data or index.

    Extra metadata columns are ignored. Missing required fields, NaN/inf,
    invalid audio ranges and unsupported categorical values raise ValueError.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Audio features must be supplied as a pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names; each feature must appear once.")
    required = MODEL_FEATURES + (CLUSTER_FEATURES if include_cluster else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required audio feature columns: " + ", ".join(missing))
    result = frame.loc[:, required].copy()
    for column in NUMERIC_FEATURES + (CLUSTER_FEATURES if include_cluster else []):
        values = pd.to_numeric(result[column], errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        if invalid.any():
            rows = list(frame.index[invalid][:5])
            raise ValueError(f"{column} must contain finite numbers; invalid row labels: {rows}.")
        if column in _UNIT_FEATURES and ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must be between 0 and 1 inclusive.")
        if column == "tempo" and (values < 0).any():
            raise ValueError("tempo must be non-negative, measured in beats per minute.")
        result[column] = values.astype(float)
    for column in CATEGORICAL_FEATURES:
        canonical = []
        for row, value in result[column].items():
            if pd.isna(value):
                raise ValueError(f"{column} is missing at row {row!r}; provide a raw feature value.")
            try:
                canonical.append(_category(value, column))
            except ValueError as exc:
                raise ValueError(f"Invalid {column} value {value!r} at row {row!r}: {exc}.") from exc
        result[column] = canonical
    return result


class RawFeatureTransformer(TransformerMixin, BaseEstimator):
    """Validate raw fields and construct the notebook's two interactions."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "RawFeatureTransformer":
        validate_features(X)
        self.n_features_in_ = len(MODEL_FEATURES)
        self.feature_names_in_ = np.array(MODEL_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = validate_features(X)
        values["dance_loud"] = values["danceability"] * values["loudness"]
        values["speech_tempo"] = values["speechiness"] * values["tempo"]
        if not np.isfinite(values[NUMERIC_FEATURES + INTERACTION_FEATURES].to_numpy()).all():
            raise ValueError("Audio features produce non-finite interaction values; check their scale.")
        return values

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.array(MODEL_FEATURES + INTERACTION_FEATURES, dtype=object)


def make_preprocessor() -> Pipeline:
    """Build an unfitted transformer; all learned statistics fit on training rows."""
    columns = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES + INTERACTION_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("raw", RawFeatureTransformer()), ("columns", columns)])
