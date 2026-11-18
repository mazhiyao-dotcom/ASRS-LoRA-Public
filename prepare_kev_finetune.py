import csv
import json
from pathlib import Path
from collections import defaultdict

# =========================
# Configuration
# =========================

DATA_PATH = Path("data/asrs/train.csv")

OUTPUT_DIR = Path("data/asrs/kev_finetune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_OUTPUT = OUTPUT_DIR / "train_full.jsonl"
SMOKE_OUTPUT = OUTPUT_DIR / "train_smoke_100.jsonl"

# =========================
# Load training data
# =========================

with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

reports = defaultdict(list)

for row in rows:
    reports[row["acn"]].append(row)

print(f"Total reports: {len(reports)}")
print(f"Total classification labels: {len(rows)}")

# =========================
# Convert to Kev format
# =========================

def convert_report(acn, report_rows):

    report_text = report_rows[0]["report"]

    if len(report_rows) != 12:
        raise ValueError(
            f"ACN {acn}: expected 12 factors, got {len(report_rows)}"
        )

    questions = {}

    for row in report_rows:

        factor = row["factor"]
        label = row["label"].strip().upper()

        if label not in ("YES", "NO"):
            raise ValueError(f"Invalid label: {label}")

        if row["report"] != report_text:
            raise ValueError(f"ACN {acn}: inconsistent report text")

        if factor in questions:
            raise ValueError(f"ACN {acn}: duplicate factor {factor}")

        questions[factor] = {
            "type": "choice",
            "instructions": (
                f"Does {factor} significantly contribute to the event "
                "described in this aviation safety report?"
            ),
            "criteria": {
                "YES": (
                    f"{factor} significantly contributed to the event."
                ),
                "NO": (
                    f"{factor} did not significantly contribute to the event."
                )
            },
            "label": label
        }

    return {
        "state": report_text,
        "questions": questions
    }


converted = [
    convert_report(acn, report_rows)
    for acn, report_rows in reports.items()
]

# =========================
# Save JSONL
# =========================

def save_jsonl(path, records):

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


save_jsonl(FULL_OUTPUT, converted)
save_jsonl(SMOKE_OUTPUT, converted[:100])

print("\n========== DATA PREPARATION ==========")
print(f"Full training reports: {len(converted)}")
print(f"Smoke test reports: {min(100, len(converted))}")

print(f"\nFull dataset saved: {FULL_OUTPUT}")
print(f"Smoke dataset saved: {SMOKE_OUTPUT}")

print("\nData preparation completed!")