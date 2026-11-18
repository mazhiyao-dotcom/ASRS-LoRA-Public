import argparse
import csv
import re
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from collections import defaultdict


MODEL_PATH = "models/Qwen3-1.7B"
DATA_PATH = "data/asrs/external_test/ASRS_HumanFactors_202401-202501_ExternalTest_Labeled.csv"
OUTPUT_DIR = Path(
    "outputs/experiments/Qwen3-1.7B_LoRA_ExternalTest_202401-202501"
)
PROMPT_PATH = "prompts/io_asrs.yaml"

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

def parse_prediction(text):
    text = text.strip().upper()
    match = re.match(r"^(YES|NO)\b", text)
    return match.group(1) if match else "INVALID"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Loading validation data...")

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = rows[:args.limit]

    print(f"Evaluating {len(rows)} samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(
    model,
    "outputs/asrs_lora/final_adapter"
)


    model.eval()

    results = []

    for i, row in enumerate(rows, 1):
        report = row["report"]
        factor = row["factor"]
        true_label = row["label"].strip().upper()

                # Limit report length while preserving the classification question
        MAX_INPUT_TOKENS = 2048

        prompt_template = PROMPTS[factor]

        # Calculate token budget for the prompt and chat template
        empty_prompt = prompt_template.format(context="")
        empty_messages = [{"role": "user", "content": empty_prompt}]

        prompt_overhead = len(
            tokenizer.apply_chat_template(
                empty_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False
            )
        )

        report_budget = MAX_INPUT_TOKENS - prompt_overhead - 64

        if report_budget <= 0:
            raise ValueError("Prompt exceeds maximum input length")

        report_tokens = tokenizer.encode(
            report,
            add_special_tokens=False
        )

        if len(report_tokens) > report_budget:
            report = tokenizer.decode(
                report_tokens[:report_budget],
                skip_special_tokens=True
            )

        prompt = prompt_template.format(context=report)


        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
                # Verify the final input length
        # Final safety check: truncate the report, not the chat template
        actual_input_tokens = inputs["input_ids"].shape[-1]

        if actual_input_tokens > MAX_INPUT_TOKENS:
            excess = actual_input_tokens - MAX_INPUT_TOKENS
            report_tokens = report_tokens[:max(0, len(report_tokens) - excess - 64)]

            report = tokenizer.decode(
                report_tokens,
                skip_special_tokens=True
            )

            prompt = prompt_template.format(context=report)
            messages = [{"role": "user", "content": prompt}]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True
            ).to("cuda")

            if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
                raise ValueError("Input still exceeds 2048 tokens")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        prediction = parse_prediction(response)

        results.append({
            "acn": row["acn"],
            "factor": factor,
            "true_label": true_label,
            "prediction": prediction,
            "raw_response": response
        })

        print(
            f"[{i}/{len(rows)}] "
            f"{factor}: True={true_label}, Pred={prediction}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual predictions
    prediction_path = OUTPUT_DIR / "predictions.csv"

    with open(prediction_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Overall metrics
    y_true = [r["true_label"] for r in results]
    y_pred = [r["prediction"] for r in results]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["NO", "YES"],
        average="macro",
        zero_division=0
    )

    # YES-class metrics
    yes_precision, yes_recall, yes_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["YES"],
        average=None,
        zero_division=0
    )

    # Metrics for each human factor
    factor_results = defaultdict(list)

    for r in results:
        factor_results[r["factor"]].append(r)

    factor_metrics = {}

    for factor, rows in factor_results.items():
        ft = [r["true_label"] for r in rows]
        fp = [r["prediction"] for r in rows]

        p, rec, f, _ = precision_recall_fscore_support(
            ft,
            fp,
            labels=["YES"],
            average=None,
            zero_division=0
        )

        factor_metrics[factor] = {
            "accuracy": accuracy_score(ft, fp),
            "precision_yes": float(p[0]),
            "recall_yes": float(rec[0]),
            "f1_yes": float(f[0])
        }

    metrics = {
        "experiment": "Qwen3-1.7B LoRA",
        "model": "Qwen/Qwen3-1.7B",
        "dataset": DATA_PATH,
        "samples": len(results),
        "reports": len(set(r["acn"] for r in results)),
        "fine_tuned": True,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "yes_precision": float(yes_precision[0]),
        "yes_recall": float(yes_recall[0]),
        "yes_f1": float(yes_f1[0]),
        "invalid_predictions": sum(p == "INVALID" for p in y_pred),
        "human_factors": factor_metrics
    }
        # Save experiment information
    info_path = OUTPUT_DIR / "experiment_info.txt"

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Experiment: Qwen3-1.7B LoRA Fine-tuned\n")
        f.write("Model: Qwen/Qwen3-1.7B\n")
        f.write("Fine-tuning: LoRA\n")
        f.write("Adapter: outputs/asrs_lora/final_adapter\n")
        f.write("Dataset: ASRS External Test (2024-01 to 2025-01)\n")
        f.write(f"Classification samples: {len(results)}\n")
        f.write(f"Accident reports: {len(set(r['acn'] for r in results))}\n")
        f.write("Prompt: prompts/io_asrs.yaml\n")
        f.write("Prompt version: Significantly Contribute (V2)\n")
        f.write("Maximum input length: 2048 tokens\n")
        f.write("Decoding: Greedy (do_sample=False)\n")

    # Save evaluation metrics
    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n===== LORA FINE-TUNED RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"YES F1: {yes_f1[0]:.4f}")
    print(f"Invalid predictions: {metrics['invalid_predictions']}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()