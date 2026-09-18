"""Small, deterministic fixtures; tests never download data or call Spotify."""

import numpy as np
import pandas as pd
import pytest

from sona.constants import CLUSTER_FEATURES, MOOD_PROTOTYPES


@pytest.fixture(scope="session")
def synthetic_tracks():
    """Five well-separated clusters with realistic, varied classifier inputs."""
    rng = np.random.default_rng(19)
    rows = []
    for group, center in enumerate(MOOD_PROTOTYPES.values()):
        for offset in range(50):
            cluster = np.clip(np.asarray(center) + rng.normal(0, 0.012, 3), 0, 1)
            rows.append({
                "track_id": f"synthetic-{group}-{offset:03d}",
                "track_name": f"Song {group}-{offset}",
                "artist_name": f"Artist {group}",
                "genre": "Test",
                "popularity": 50,
                "duration_ms": 180_000,
                **dict(zip(CLUSTER_FEATURES, cluster)),
                "danceability": np.clip(0.12 + 0.15 * group + rng.normal(0, 0.02), 0, 1),
                "instrumentalness": rng.uniform(0, 1),
                "liveness": rng.uniform(0.05, 0.4),
                "loudness": -24 + 4 * group + rng.normal(0, 0.5),
                "speechiness": rng.uniform(0.02, 0.2),
                "tempo": 65 + 24 * group + rng.normal(0, 2),
                "key": offset % 12,
                "mode": offset % 2,
                "time_signature": 4,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_tracks(synthetic_tracks):
    return synthetic_tracks.iloc[:8].copy()
