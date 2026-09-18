"""Public feature and label definitions for Sona."""

CLUSTER_FEATURES = ["valence", "energy", "acousticness"]
NUMERIC_FEATURES = [
    "danceability", "instrumentalness", "liveness", "loudness", "speechiness", "tempo"
]
CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INTERACTION_FEATURES = ["dance_loud", "speech_tempo"]
MOODS = ["Calm", "Content", "Energetic", "Happy", "Mellow"]
SCHEMA_VERSION = "1.0"

# Original notebook's human-named cluster centroids, used only as semantic
# prototypes, not ground-truth labels. Values follow CLUSTER_FEATURES order.
MOOD_PROTOTYPES = {
    "Calm": [0.173639, 0.184282, 0.877450],
    "Content": [0.274675, 0.513595, 0.219184],
    "Energetic": [0.401377, 0.821940, 0.060605],
    "Happy": [0.758946, 0.741482, 0.134775],
    "Mellow": [0.639244, 0.442294, 0.671929],
}
