from urllib.request import Request, urlopen
from model import FEATURES
from pathlib import Path
import json
import csv
import sys

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "student_dropout_risk_2026.csv"

with DEFAULT_DATA.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    row = next(reader)

student = {
    feature: row[feature]
    for feature in FEATURES
}

request = Request(
    URL,
    data=json.dumps(student).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request) as response:
    assert response.status == 200
    result = json.load(response)

assert set(result) == {"dropout_probability", "dropout"}

assert 0 <= result["dropout_probability"] <= 1
assert isinstance(result["dropout"], bool)

assert result["dropout"] == (result["dropout_probability"] >= 0.41)

print("Prediction API test passed.")
print(json.dumps(result, indent=2))
