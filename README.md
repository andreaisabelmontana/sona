# Sona

**Find the feeling behind every track.**

Sona is a runnable song-mood machine learning project, recreated from [Moodify](https://github.com/cayetana-h/Moodify) with a new name, a reusable training pipeline, and an interactive app. It discovers five audio-profile groups — **Calm, Content, Energetic, Happy, Mellow** — and learns to predict them from musical features.

This repository includes the historical feature dataset, a trained model, measured evaluation reports, an executable notebook, and tests. No Spotify account or API credentials are required.

## Run the app

Use Python 3.12 for the supplied model. From this project folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the recorded environment and launch:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m streamlit run app.py
```

Open the local address printed by Streamlit. The app lets you:

- Search songs and artists in the included catalog.
- Adjust audio features and see mood probabilities.
- Upload your own feature CSV and download predictions.
- Inspect learned cluster profiles and held-out evaluation results.

For a smaller installation, `python -m pip install -e .` installs only the core and app dependencies. Retrain the model if your scikit-learn version differs from the saved model's version. Only load model files you trust.

## Train and predict

Train on all eligible unique tracks:

```bash
python -m sona train --data data/SpotifyFeatures.csv --output artifacts
```

Run a shorter experiment, or compare the optional neural-network and XGBoost models with training-only SMOTE:

```bash
python -m sona train --data data/SpotifyFeatures.csv --output artifacts-quick --sample 10000
python -m pip install -e '.[research]'
python -m sona train --data data/SpotifyFeatures.csv --output artifacts-research --extended --smote
```

Score a CSV:

```bash
python -m sona predict --model artifacts/sona.joblib --input data/example_tracks.csv --output predictions.csv
```

The classifier accepts raw `danceability`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `key`, `mode`, and `time_signature`. Use note names or Spotify key integers, Major/Minor or 1/0, and meters such as 4/4 or 4. Metadata columns are optional and preserved in prediction exports. See the [example CSV](data/example_tracks.csv).

Python API:

```python
import pandas as pd
from sona import load_bundle, predict_df

model = load_bundle("artifacts/sona.joblib")
predictions = predict_df(model, pd.read_csv("data/example_tracks.csv"))
print(predictions)
```

## The experiment

1. Remove Comedy entries and repeated track IDs from 232,725 source rows.
2. Split the 167,101 eligible unique tracks into 90% training, 5% validation, and 5% test data.
3. Fit standardization and five-cluster KMeans on training `valence`, `energy`, and `acousticness` only.
4. Match the learned centers to reference mood profiles from the original notebook.
5. Predict the cluster labels from the other audio features, with shared scaling, categorical encoding, and two interaction features.
6. Compare Logistic Regression, Random Forest, and Histogram Gradient Boosting using validation macro F1. Evaluate the selected model on the held-out test split.

The optional research mode adds MLP and XGBoost. SMOTE is opt-in and applies only to training features. Sona fixes duplicate-track overlap and inconsistent inference preprocessing in the original project. It is a recreation with corrected evaluation, so its results are not directly comparable with the upstream accuracy claim.

These labels are **audio-feature clusters, not human-validated emotional judgments**. Sona consumes feature tables, not audio files or lyrics. Predicted probabilities are uncalibrated model estimates. Catalog exploration may show tracks that were used for training.

See [methodology](docs/METHODOLOGY.md), [dataset provenance](data/README.md), and [measured results](artifacts/metrics.json).

### Measured full-data results

Seed 42; 150,390 training tracks, 8,355 validation tracks, and 8,356 test tracks.

| Candidate | Validation macro F1 |
| --- | ---: |
| Logistic Regression | 0.5181 |
| Random Forest | 0.5625 |
| Histogram Gradient Boosting (selected) | **0.5668** |

The selected model achieves **60.24% test accuracy** and **0.5758 test macro F1** against the cluster-derived labels. The included notebook's smaller 10,000-track experiment has separate results. The corrected split provides a more conservative estimate than a row-level split with repeated songs.

## Explore and verify

Open [the notebook](notebooks/sona_exploration.ipynb) in Jupyter or VS Code using this Python environment. It runs a 10,000-track experiment and writes to `artifacts-notebook/`, preserving the main model. The recorded environment includes the notebook kernel; install your preferred notebook frontend separately if needed.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests cover input validation, category normalization, persistence, separation of training and test data, and consistent preprocessing. GitHub Actions runs the suite on pushes and pull requests.

## Files

| Path | Contents |
| --- | --- |
| `app.py` | Local Streamlit explorer |
| `sona/` | Training, preprocessing, prediction API and CLI |
| `notebooks/sona_exploration.ipynb` | Guided experiment with executed outputs |
| `data/` | Original feature dataset and example inputs |
| `artifacts/` | Trained model, metrics, cluster profiles, split manifest and sample predictions |
| `tests/` | Behavioral and leakage regression tests |

## Attribution and privacy

Recreated from **Moodify by cayetana-h and contributors**, revision [`f0596bb`](https://github.com/cayetana-h/Moodify/tree/f0596bbf19567b2400924b3c172dfb806eabae02). Original attribution and Apache-2.0 license are retained in [NOTICE](NOTICE) and [LICENSE](LICENSE). The separate dataset provenance limits are recorded in [data/README.md](data/README.md).

The GitHub repository is private. Its URL works for you and collaborators you grant access to; it is not an anonymous public sharing link. The app runs locally unless you choose to deploy it. Streamlit usage telemetry is disabled in this project.
