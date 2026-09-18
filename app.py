"""Sona's local song-mood explorer. Run: streamlit run app.py."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from sona import load_bundle, predict_df

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "artifacts" / "sona.joblib"

st.set_page_config(page_title="Sona · Find the feeling", page_icon="◒", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 3rem;}
h1 {letter-spacing: -0.045em;}
[data-testid="stMetricValue"] {color: #baf27c;}
div[data-testid="stTabs"] button {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)
st.caption("S O N A  /  A MUSIC MOOD LAB")
st.title("Find the feeling behind every track.")
st.markdown("Explore a song, shape a sound, or discover the moods in your own library.")
st.caption("Five musical moods · Audio features + machine learning · Runs locally")

if not MODEL.exists():
    st.info("Train Sona once to start exploring. Run these commands from the project folder:")
    st.code("python -m pip install -e .\npython -m sona train --data data/SpotifyFeatures.csv --output artifacts", language="bash")
    st.stop()


@st.cache_resource
def get_model(modified: float):
    return load_bundle(MODEL)


@st.cache_data
def get_catalog():
    frame = pd.read_csv(ROOT / "data" / "SpotifyFeatures.csv")
    return frame.loc[frame["genre"].ne("Comedy")].drop_duplicates("track_id").reset_index(drop=True)


bundle = get_model(MODEL.stat().st_mtime)


def show_prediction(frame):
    try:
        prediction = predict_df(bundle, frame).iloc[0]
    except ValueError as error:
        st.error(str(error))
        return
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.caption("PREDICTED MUSICAL MOOD")
        st.header(prediction["mood"])
        st.metric("Model probability", f"{prediction['confidence']:.0%}")
    with right:
        probabilities = pd.DataFrame({
            "Mood": ["Calm", "Content", "Energetic", "Happy", "Mellow"],
            "Probability": [prediction[f"probability_{mood}"] for mood in ["Calm", "Content", "Energetic", "Happy", "Mellow"]],
        })
        st.bar_chart(probabilities, x="Mood", y="Probability", color="#BAF27C", horizontal=True)
    st.caption("Moods are descriptive names for audio-feature clusters. Probabilities are model estimates, not a measure of a listener's emotions.")


explore, create, batch, about = st.tabs(["Explore a song", "Shape a sound", "Your library", "Inside the model"])

with explore:
    st.subheader("Start with a song you know.")
    st.write("Search the included historical Spotify feature catalog. No account or API key needed.")
    catalog = get_catalog()
    query = st.text_input("Song or artist", placeholder="Try Adele, Coldplay, or your favorite song")
    if query.strip():
        matched = catalog.loc[
            catalog["track_name"].fillna("").str.contains(query.strip(), case=False, regex=False)
            | catalog["artist_name"].fillna("").str.contains(query.strip(), case=False, regex=False)
        ].copy()
        term = query.strip().casefold()
        artist = matched["artist_name"].fillna("").str.casefold()
        title = matched["track_name"].fillna("").str.casefold()
        matched["search_rank"] = (artist.eq(term).astype(int) * 4 + title.eq(term).astype(int) * 4
                                  + artist.str.startswith(term).astype(int) * 2 + title.str.startswith(term).astype(int))
        matched = matched.sort_values(["search_rank", "popularity"], ascending=False).head(80)
    else:
        matched = catalog.sort_values("popularity", ascending=False).head(20)
    if matched.empty:
        st.info("No match in this catalog. Try another artist, or use Shape a sound.")
    else:
        selected = st.selectbox(
            "Choose a track", matched.index.tolist(),
            format_func=lambda i: f"{catalog.at[i, 'track_name']} — {catalog.at[i, 'artist_name']}",
        )
        song = catalog.loc[[selected]]
        st.caption(f"{song.iloc[0]['genre']} · {song.iloc[0]['tempo']:.0f} BPM · Historical catalog")
        show_prediction(song)

with create:
    st.subheader("What does your sound feel like?")
    st.write("Adjust the audio features to explore how the classifier responds.")
    with st.form("manual_features"):
        first, second, third = st.columns(3)
        with first:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.65, 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.05, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
        with second:
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.06, 0.01)
            loudness = st.slider("Loudness (dB)", -60.0, 5.0, -7.0, 0.5)
            tempo = st.slider("Tempo (BPM)", 1.0, 250.0, 120.0, 1.0)
        with third:
            key = st.selectbox("Musical key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
            mode = st.selectbox("Mode", ["Major", "Minor"])
            signature = st.selectbox("Time signature", ["4/4", "3/4", "5/4", "1/4", "0/4"])
        submitted = st.form_submit_button("Discover the mood", type="primary")
    if submitted:
        st.session_state["manual_song"] = dict(
            danceability=danceability, instrumentalness=instrumentalness, liveness=liveness,
            speechiness=speechiness, loudness=loudness, tempo=tempo, key=key, mode=mode,
            time_signature=signature,
        )
    if "manual_song" in st.session_state:
        show_prediction(pd.DataFrame([st.session_state["manual_song"]]))

with batch:
    st.subheader("Give your library a mood.")
    st.write("Upload a CSV of audio features. Predictions are processed here and are not saved to the repository.")
    example_path = ROOT / "data" / "example_tracks.csv"
    if example_path.exists():
        st.download_button("Download example CSV", example_path.read_bytes(), file_name="sona-example.csv", mime="text/csv")
    st.caption("Required: danceability, instrumentalness, liveness, loudness, speechiness, tempo, key, mode, time_signature. Song and artist columns are optional.")
    uploaded = st.file_uploader("Audio-feature CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            if len(frame) > 10000:
                st.warning("Please upload at most 10,000 tracks per batch.")
            else:
                predictions = predict_df(bundle, frame)
                result = pd.concat([frame.drop(columns=predictions.columns, errors="ignore"), predictions], axis=1)
                st.success(f"{len(result):,} tracks analyzed")
                st.dataframe(result, hide_index=True)
                st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"), file_name="sona-predictions.csv", mime="text/csv")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read these tracks: {error}")

with about:
    st.subheader("From audio features to five moods.")
    st.write("Sona groups songs using valence, energy and acousticness, then learns to predict those groups from the remaining audio features. Mood names are matched to reference cluster profiles from the original project.")
    st.write("Repeated track IDs are removed before splitting. Clustering and preprocessing learn from training data only; model selection uses validation data, and the final score uses a held-out test set.")
    st.info("This is an experiment in predicting cluster labels. It does not listen to audio, analyze lyrics, or measure human emotions. Catalog predictions can include training examples.")
    profiles = ROOT / "artifacts" / "cluster_profiles.csv"
    if profiles.exists():
        st.dataframe(pd.read_csv(profiles), hide_index=True)
    report = ROOT / "artifacts" / "metrics.json"
    if report.exists():
        with st.expander("Measured training and evaluation report"):
            st.json(json.loads(report.read_text(encoding="utf-8")))
    chart = ROOT / "artifacts" / "confusion_matrix.png"
    if chart.exists():
        st.image(str(chart), caption="Held-out test predictions against cluster labels")

st.divider()
st.caption("SONA · A little science. A lot of feeling.")
