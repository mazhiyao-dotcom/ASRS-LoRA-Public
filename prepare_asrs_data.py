import argparse
import csv
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/asrs")
DEFAULT_INPUT = DATA_DIR / "ASRS_clean_core.csv"

LABELS = [
    "Situational Awareness",
    "Communication Breakdown",
    "Distraction",
    "Confusion",
    "Workload",
    "Time Pressure",
    "Troubleshooting",
    "Training / Qualification",
    "Human-Machine Interface",
    "Other / Unknown",
    "Physiological - Other",
    "Fatigue",
]


def parse_factors(value):
    if not value:
        return set()
    return {
        factor.strip()
        for factor in value.split(";")
        if factor.strip()
    }


def load_reports(path):
    reports = []
    seen_acns = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "ACN",
            "Report_1_Narrative",
            "Person_1_Human_Factors",
            "Person_2_Human_Factors",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Missing required CSV columns")

        for row in reader:
            acn = row["ACN"].strip()
            narrative = row["Report_1_Narrative"].strip()

            person1 = parse_factors(
                row["Person_1_Human_Factors"]
            )
            person2 = parse_factors(
                row["Person_2_Human_Factors"]
            )

            factors = person1 | person2

            if not acn or not narrative or not factors:
                continue

            if acn in seen_acns:
                raise ValueError(f"Duplicate ACN: {acn}")

            unknown = factors - set(LABELS)
            if unknown:
                raise ValueError(
                    f"Unknown labels for ACN {acn}: {unknown}"
                )

            seen_acns.add(acn)

            reports.append({
                "acn": acn,
                "report": narrative,
                "factors": factors,
            })

    return reports


def split_reports(reports, seed=42):
    rng = random.Random(seed)
    rng.shuffle(reports)

    n = len(reports)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = reports[:n_train]
    val = reports[n_train:n_train + n_val]
    test = reports[n_train + n_val:]

    return train, val, test


def save_dataset(reports, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["acn", "report", "factor", "label"]
        )
        writer.writeheader()

        for item in reports:
            for factor in LABELS:
                label = (
                    "YES"
                    if factor in item["factors"]
                    else "NO"
                )

                writer.writerow({
                    "acn": item["acn"],
                    "report": item["report"],
                    "factor": factor,
                    "label": label,
                })

                counts[label] += 1

    print(
        f"{path}: {len(reports)} reports, "
        f"{sum(counts.values())} samples, "
        f"YES={counts['YES']}, NO={counts['NO']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )
    args = parser.parse_args()

    reports = load_reports(args.input)

    print(f"Valid reports: {len(reports)}")
    print(f"Human Factors categories: {len(LABELS)}")

    train, val, test = split_reports(reports)

    save_dataset(train, DATA_DIR / "train.csv")
    save_dataset(val, DATA_DIR / "val.csv")
    save_dataset(test, DATA_DIR / "test.csv")

    print("Data preparation completed.")


if __name__ == "__main__":
    main()