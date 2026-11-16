from urllib.request import Request, urlopen
from model import FEATURES
from pathlib import Path
import json
import csv
import sys

url = "http://localhost:9696/predict"
data_path = Path(__file__).resolve().parent / "data" / "student_dropout_risk_2026.csv"

expected_dropout = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with data_path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    row = next((row for row in reader if int(row["dropout"]) == expected_dropout), None)

if row is None:
    raise ValueError("No matching student found")

student = {
    feature: row[feature]
    for feature in FEATURES
}

request = Request(
    url,
    data=json.dumps(student).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urlopen(request) as response:
    result = json.load(response)

assert set(result) == {"dropout_probability", "dropout"}
assert 0 <= result["dropout_probability"] <= 1
assert isinstance(result["dropout"], bool)

print(f"Actual dropout: {bool(expected_dropout)}")
print(f"Predicted dropout: {result['dropout']}")
print(f"Dropout probability: {result['dropout_probability']:.6f}")
print(f"Prediction correct: {result['dropout'] == bool(expected_dropout)}")
