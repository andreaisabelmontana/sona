# Sona methodology

Sona recreates and extends the experiment in [Moodify](https://github.com/cayetana-h/Moodify/tree/f0596bbf19567b2400924b3c172dfb806eabae02), by cayetana-h and contributors. The implementation and notebook have been rewritten, with changes to data splitting, preprocessing, inference, and reporting. The original Apache-2.0 license and attribution are retained in `LICENSE` and `NOTICE`.

## What the model learns

The dataset has audio features and track metadata, but no human mood annotations. Sona first constructs five clusters from `valence`, `energy`, and `acousticness`, then learns to predict those cluster assignments using other audio features. The names **Calm**, **Content**, **Energetic**, **Happy**, and **Mellow** are interpretive labels for clusters. They are not verified emotions or measurements of a listener's response.

The supervised model uses six numeric inputs: `danceability`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, and `tempo`. It also uses `key`, `mode`, and `time_signature`, and constructs `danceability * loudness` and `speechiness * tempo` interaction features. The three features that define the clustering are excluded from supervised inputs, preserving the original experiment's central question: how well can the remaining features recover the cluster labels? Track names, artist names, genre, popularity, and identifiers are not prediction inputs.

Sona does not analyze an audio file. It operates on precomputed feature values. Predictions and class probabilities indicate agreement with the learned cluster labels, not certainty about a song's emotional meaning.

## Data and evaluation

1. Read the bundled CSV and clean required feature values. Exclude rows in the Comedy genre and remove repeated track IDs before splitting. Repeated listings of the same track must not appear in both training and test data.
2. Optionally select a reproducible subset for a quick experiment. The notebook uses 10,000 cleaned tracks; full training uses the available cleaned dataset.
3. Randomly split 90%/5%/5% into training, validation, and test sets. The split is not stratified because cluster labels do not exist yet. The validation set is used to compare candidate classifiers; the test set is reserved for the final selected classifier.
4. Fit the clustering scaler and five-cluster KMeans on the training split only. Assign validation and test pseudo-labels using that fitted clustering pipeline.
5. Match cluster centers one-to-one to the original project's documented centers. This keeps naming interpretable when KMeans assigns different numeric cluster IDs.
6. Fit the supervised preprocessing and classifiers on the training split. By default, compare Logistic Regression, Random Forest, and histogram-based Gradient Boosting. Select a model using validation macro F1, so each mood contributes equally to the selection metric. The selected model remains fitted on training data; validation tracks are not added for a second fit.
7. Report test accuracy, macro F1, weighted F1, per-class precision/recall/F1, and a confusion matrix. Store model-selection details and dataset counts alongside the result.

The original notebook's reference cluster centers are shown below. Values are means in the original feature units, not standardized coordinates. They are naming references; Sona's fitted centers can differ after deduplication, splitting, or sampling.

| Mood | Valence | Energy | Acousticness |
| --- | ---: | ---: | ---: |
| Happy | 0.758946 | 0.741482 | 0.134775 |
| Calm | 0.173639 | 0.184282 | 0.877450 |
| Content | 0.274675 | 0.513595 | 0.219184 |
| Mellow | 0.639244 | 0.442294 | 0.671929 |
| Energetic | 0.401377 | 0.821940 | 0.060605 |

## Changes from Moodify

| Area | Original experiment | Sona recreation |
| --- | --- | --- |
| Packaging | A large notebook and machine-specific requirements | Reusable Python package, portable dependencies, application, and executable notebook |
| Repeated tracks | Row-level random split | Deduplication by track ID before splitting |
| Cluster fitting | Clustering fit before the split | Clustering scaler and KMeans fit on training data only |
| Cluster names | A fixed map from numeric cluster IDs | One-to-one matching of learned centers to documented references |
| Inference | Raw inputs passed to a model trained on scaled features | Persisted preprocessing applied with the selected classifier |
| Categories | Manual dummy columns and inconsistent mode conventions | Shared preprocessing and category normalization |
| Model evaluation | Multiple notebook experiments and manually copied results | Stored validation comparison and final holdout reports |
| External services | Optional Spotify calls and interactive prompts in the notebook | Local dataset and feature input; notebook runs without network or credentials |

The original notebook also explored polynomial/interaction features, SMOTE, MLP, gradient boosting, XGBoost, and extensive hyperparameter searches. Sona retains two interaction features and offers optional extended-model and SMOTE experiments. SMOTE, when enabled, operates inside the training pipeline and interpolates both scaled numeric and one-hot categorical features; the latter can create fractional category indicators. It is disabled by default. Sona provides a compact reproducible experiment; it does not claim an exact numerical reproduction of every upstream experiment. In particular, its stricter splitting and deduplication make a direct comparison with the original reported 76.31% accuracy inappropriate.

## Reading the artifacts

- `sona.joblib` contains the fitted model and associated preprocessing information for `load_bundle` and `predict_df`.
- `metrics.json` records model selection, holdout metrics, dataset counts, feature definitions, mood mapping, and package versions.
- `cluster_profiles.csv` records the learned cluster profiles for interpretation.
- `confusion_matrix.png` visualizes the selected model's holdout predictions.
- `heldout_predictions.csv` contains up to 100 held-out examples for inspection.
- `split_manifest.csv` records each sampled track's source row and split, making train/validation/test membership auditable.

The notebook writes to `artifacts-notebook/` so its small experiment can be inspected separately from the main trained artifacts. A fixed seed makes the split and fitting repeatable within the same software environment; versions are recorded because numerical results can vary across library releases.

## Limits

High agreement with pseudo-labels does not establish that the labels match human judgments. The naming references are heuristic, and the model can inherit sampling biases and feature limitations from the source dataset. Deduplicating track IDs prevents exact-track overlap, but this is not an artist-disjoint evaluation; an artist's different tracks may occur across splits. A future validation study should collect human annotations and evaluate separately by listener, artist, genre, and collection period.

Dataset provenance and the limits of the available licensing record are documented in [`data/README.md`](../data/README.md).
