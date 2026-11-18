import argparse
import csv
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ASRS LoRA training configuration
DEFAULT_PROMPTS = Path("prompts/io_asrs.yaml")
DEFAULT_OUTPUT = Path("outputs/asrs_lora")
DEFAULT_MAX_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
    )

    return parser.parse_args()


# Load ASRS Human Factors prompt templates.
def load_prompt_templates(path):
    with path.open("r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)

    if not isinstance(templates, dict) or not templates:
        raise ValueError(
            f"Expected a nonempty YAML mapping in: {path}"
        )

    for factor, template in templates.items():
        if (
            not isinstance(factor, str)
            or not isinstance(template, str)
            or "{context}" not in template
        ):
            raise ValueError(
                f"Invalid prompt or missing {{context}} for factor: {factor!r}"
            )

    return templates


# Load labeled ASRS training data.
# Expected CSV columns: acn, report, factor, label
def load_training_rows(path, templates):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing labeled CSV: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {"report", "factor", "label"}

        if (
            not reader.fieldnames
            or not required.issubset(set(reader.fieldnames))
        ):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Training CSV is empty")

    for line, row in enumerate(rows, start=2):

        if not row["report"] or not row["report"].strip():
            raise ValueError(
                f"Empty report on CSV line {line}"
            )

        if (
            not row["factor"]
            or row["factor"].strip() not in templates
        ):
            raise ValueError(
                f"Unknown ASRS factor on CSV line {line}: "
                f"{row['factor']!r}"
            )

        if (
            not row["label"]
            or row["label"].strip().lower() not in {"yes", "no"}
        ):
            raise ValueError(
                f"Expected YES/NO on CSV line {line}: "
                f"{row['label']!r}"
            )

    return rows


# Use an end-of-response token after YES or NO.
def answer_end_token_id(tokenizer):

    vocab = tokenizer.get_vocab()

    if "<|im_end|>" in vocab:
        return vocab["<|im_end|>"]

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer has neither <|im_end|> nor eos_token_id"
        )

    return tokenizer.eos_token_id


# ASRS training dataset.
class ASRSDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        rows,
        tokenizer,
        templates,
        max_length
    ):

        if max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        if tokenizer.chat_template is None:
            raise ValueError(
                "This implementation requires a chat template"
            )

        self.samples = []

        end_id = answer_end_token_id(tokenizer)

        truncated_count = 0

        for index, row in enumerate(rows, start=1):

            factor = row["factor"].strip()
            answer = row["label"].strip().upper()

            prompt_template = templates[factor]

            chat_kwargs = {}

            if "enable_thinking" in str(tokenizer.chat_template):
                chat_kwargs["enable_thinking"] = False

            # Build a chat prompt while preserving the generation header.
            def build_prompt(context):
                prompt_text = prompt_template.replace(
                    "{context}",
                    context
                )

                prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_kwargs,
                )

                return tokenizer(
                    prompt,
                    add_special_tokens=False
                )["input_ids"]

            answer_ids = tokenizer(
                answer,
                add_special_tokens=False
            )["input_ids"]

            if not answer_ids:
                raise ValueError(
                    f"Empty answer tokens for sample {index}"
                )

            # Reserve space for the answer and end token.
            allowed_prompt_length = (
                max_length - len(answer_ids) - 1
            )

            if allowed_prompt_length <= 0:
                raise ValueError(
                    "max_length is too small"
                )

            report = row["report"]

            prompt_ids = build_prompt(report)

            # Truncate the report rather than the chat generation header.
            if len(prompt_ids) > allowed_prompt_length:

                truncated_count += 1

                report_ids = tokenizer(
                    report,
                    add_special_tokens=False
                )["input_ids"]

                low = 0
                high = len(report_ids)

                while low < high:

                    mid = (low + high + 1) // 2

                    shortened_report = tokenizer.decode(
                        report_ids[:mid],
                        skip_special_tokens=True
                    )

                    candidate_ids = build_prompt(
                        shortened_report
                    )

                    if len(candidate_ids) <= allowed_prompt_length:
                        low = mid
                    else:
                        high = mid - 1

                shortened_report = tokenizer.decode(
                    report_ids[:low],
                    skip_special_tokens=True
                )

                prompt_ids = build_prompt(
                    shortened_report
                )

                if len(prompt_ids) > allowed_prompt_length:
                    raise ValueError(
                        f"Unable to fit sample {index} "
                        f"within {max_length} tokens"
                    )

            input_ids = (
                prompt_ids
                + answer_ids
                + [end_id]
            )

            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
                + [end_id]
            )

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

        print(
            f"Prepared {len(self.samples)} samples. "
            f"Truncated {truncated_count} long samples."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():

    config = parse_args()

    # Load ASRS prompts.
    templates = load_prompt_templates(
        DEFAULT_PROMPTS
    )

    print(
        f"Loaded {len(templates)} ASRS Human Factors prompts."
    )

    # Load training data.
    rows = load_training_rows(
        config.train_data,
        templates
    )

    print(
        f"Loaded {len(rows)} labeled training rows."
    )

    # Check GPU availability.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. "
            "Training requires a configured NVIDIA GPU."
        )

    print(
        f"Training GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Base model: {config.model}"
    )

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        config.model
    )

    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer lacks pad_token and eos_token"
            )

        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset.
    dataset = ASRSDataset(
        rows=rows,
        tokenizer=tokenizer,
        templates=templates,
        max_length=DEFAULT_MAX_LENGTH,
    )

    # Load base model.
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    # Configure LoRA.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Training configuration.
    args = TrainingArguments(
        output_dir=str(DEFAULT_OUTPUT),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        seed=42,
    )

    # Initialize Trainer.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
    )

    # Train LoRA adapter.
    trainer.train()

    # Save trained adapter.
    final_path = (
        DEFAULT_OUTPUT / "final_adapter"
    )

    final_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        final_path
    )

    tokenizer.save_pretrained(
        final_path
    )

    print(
        f"LoRA adapter saved to: {final_path}"
    )


if __name__ == "__main__":
    main()