"""Command-line entry point: python -m sona train|predict."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .modeling import load_bundle, predict_df, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sona", description="Learn audio-profile moods and predict them from raw song features.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and compare classifiers, saving a complete model bundle")
    training.add_argument("--data", required=True, type=Path, help="Path to SpotifyFeatures.csv")
    training.add_argument("--output", default=Path("artifacts"), type=Path)
    training.add_argument("--sample", type=int, help="Use a reproducible sample of unique tracks (at least 50)")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--extended", action="store_true", help="Also compare MLP and XGBoost, if installed")
    training.add_argument("--smote", action="store_true", help="Oversample training features only (requires imbalanced-learn)")
    prediction = commands.add_parser("predict", help="Predict moods from a CSV of raw audio features")
    prediction.add_argument("--model", required=True, type=Path, help="Trusted Sona joblib artifact")
    prediction.add_argument("--input", required=True, type=Path)
    prediction.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train(args.data, args.output, args.sample, args.seed, args.extended, args.smote)
        else:
            frame = pd.read_csv(args.input)
            predictions = predict_df(load_bundle(args.model), frame)
            # Replace previous prediction fields if this CSV is being rescored.
            result = pd.concat([frame.drop(columns=predictions.columns, errors="ignore"), predictions], axis=1)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"[Sona] Wrote {len(result):,} predictions to {args.output}", flush=True)
    except (ValueError, TypeError, FileNotFoundError, ImportError, pd.errors.ParserError) as exc:
        print(f"Sona: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
