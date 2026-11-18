import csv
import json
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

# =========================
# 1. Configuration
# =========================

DATA_PATH = "data/asrs/val.csv"
API_URL = "http://127.0.0.1:8009/v1/systemone"

MODEL_NAME = "Kev-0.8B"
LIMIT_REPORTS = None

OUTPUT_DIR = Path(
    "outputs/experiments/Kev-0.8B_Baseline"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Load validation data
# =========================

with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

reports = defaultdict(list)

for row in rows:
    reports[row["acn"]].append(row)

selected_reports = list(reports.items())[:LIMIT_REPORTS]

print(f"Model: {MODEL_NAME}")
print(f"Reports to evaluate: {len(selected_reports)}")


# =========================
# 3. Run Kev inference
# =========================

predictions = []
latencies = []

start_time = time.time()

for index, (acn, report_rows) in enumerate(selected_reports, start=1):

    report = report_rows[0]["report"]

    questions = {}

    for row in report_rows:

        factor = row["factor"]

        questions[factor] = {
            "type": "choice",
            "instructions": (
                f"Does {factor} significantly contribute to the event "
                "described in this aviation safety report?"
            ),
            "criteria": {
                "YES": f"{factor} significantly contributed to the event.",
                "NO": f"{factor} did not significantly contribute to the event."
            }
        }

    payload = {
        "state": report,
        "questions": questions
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    answers = result["answers"]

    latency = result.get("latency_ms", 0)
    latencies.append(latency)

    for row in report_rows:

        factor = row["factor"]
        true_label = row["label"].strip().upper()

        answer = answers[factor]
        prediction = answer["choice"].strip().upper()

        yes_probability = answer.get(
            "probabilities", {}
        ).get("YES")

        predictions.append({
            "acn": acn,
            "factor": factor,
            "true_label": true_label,
            "prediction": prediction,
            "yes_probability": yes_probability
        })

    print(
        f"[{index}/{len(selected_reports)}] "
        f"ACN={acn} | "
        f"Factors={len(report_rows)} | "
        f"Latency={latency:.1f} ms",
        flush=True
    )


# =========================
# 4. Calculate metrics
# =========================

def calculate_class_metrics(true_labels, pred_labels, target):

    tp = sum(
        t == target and p == target
        for t, p in zip(true_labels, pred_labels)
    )

    fp = sum(
        t != target and p == target
        for t, p in zip(true_labels, pred_labels)
    )

    fn = sum(
        t == target and p != target
        for t, p in zip(true_labels, pred_labels)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return precision, recall, f1


true_labels = [r["true_label"] for r in predictions]
pred_labels = [r["prediction"] for r in predictions]

if not all(x in ("YES", "NO") for x in true_labels + pred_labels):
    raise ValueError("Unexpected label found; expected YES or NO.")

accuracy = sum(
    t == p for t, p in zip(true_labels, pred_labels)
) / len(predictions)

yes_p, yes_r, yes_f1 = calculate_class_metrics(
    true_labels, pred_labels, "YES"
)

no_p, no_r, no_f1 = calculate_class_metrics(
    true_labels, pred_labels, "NO"
)

macro_precision = (yes_p + no_p) / 2
macro_recall = (yes_r + no_r) / 2
macro_f1 = (yes_f1 + no_f1) / 2

total_time = time.time() - start_time

metrics = {
    "model": MODEL_NAME,
    "reports": len(selected_reports),
    "samples": len(predictions),
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "yes_f1": yes_f1,
    "average_latency_ms_per_report": (
        sum(latencies) / len(latencies)
    ),
    "total_wall_time_seconds": total_time
}


# =========================
# 5. Save results
# =========================

with open(OUTPUT_DIR / "predictions.csv", "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "acn",
            "factor",
            "true_label",
            "prediction",
            "yes_probability"
        ]
    )

    writer.writeheader()
    writer.writerows(predictions)


with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# =========================
# 6. Print results
# =========================

print("\n========== RESULTS ==========")

print(f"Reports: {len(selected_reports)}")
print(f"Samples: {len(predictions)}")

print(f"Accuracy: {accuracy:.2%}")
print(f"Macro Precision: {macro_precision:.2%}")
print(f"Macro Recall: {macro_recall:.2%}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"YES F1: {yes_f1:.4f}")

print(
    f"Average latency: "
    f"{metrics['average_latency_ms_per_report']:.1f} ms/report"
)

print(f"Total time: {total_time:.1f} seconds")

print(f"\nResults saved to: {OUTPUT_DIR}")