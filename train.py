from model import normalize_record, NUMERICAL_FEATURES
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pathlib import Path
from typing import Any
import argparse
import pickle
import csv

DEFAULT_DATA = Path(__file__).resolve().parent / 'data' / 'student_dropout_risk_2026.csv'
DEFAULT_MODEL = Path(__file__).resolve().parent / "pipeline.bin"

def read_training_rows(path: Path) -> tuple[list[dict[str, Any]], list[int], dict[str, float]]:
    raw_rows: list[dict[str, str]] = []
    targets: list[int] = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            raw_rows.append(row)
            targets.append(int(row['dropout']))

    defaults = {}

    for feature in NUMERICAL_FEATURES:
        values = sorted(float(row[feature]) for row in raw_rows if row[feature] != "")

        middle = len(values) // 2

        if len(values) % 2:
            defaults[feature] = values[middle]
        else:
            defaults[feature] = (values[middle - 1] + values[middle]) / 2

    records = [normalize_record(row, defaults) for row in raw_rows]
    return records, targets, defaults

def build_pipeline():
    return make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    )

def train(data_path: Path, model_path: Path) -> None:
    records, targets, defaults = read_training_rows(data_path)
    pipeline = build_pipeline()
    pipeline.fit(records, targets)

    with model_path.open('wb') as model_file:
        pickle.dump((pipeline, defaults), model_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Drop Out predict model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.data, args.model)