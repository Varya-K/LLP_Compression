import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot

from data_categories import subcategories


COLUMN_NAMES = ["question", "A", "B", "C", "D", "answer"]
DATASET_BASE_URL = "https://huggingface.co/datasets/Varya-K/MMLU_data/resolve/main/test"


def parse_args():
    parser = argparse.ArgumentParser(description="AWQ INT4 quantization with llmcompressor")

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF model id"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Qwen3-8B-AWQ-INT4",
        help="Where to save quantized model"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Max sequence length for calibration"
    )

    return parser.parse_args()


def load_calibration_dataset(limit: int):
    datasets = []

    for subcat in subcategories:
        ds = load_dataset(
            "csv",
            data_files=f"{DATASET_BASE_URL}/{subcat}_test.csv",
            split="train",
            column_names=COLUMN_NAMES,
        )
        datasets.append(ds)

    ds = concatenate_datasets(datasets)
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))

    def preprocess(example):
        prompt = (
            f"Question: {example['question']}\n"
            f"A. {example['A']}\n"
            f"B. {example['B']}\n"
            f"C. {example['C']}\n"
            f"D. {example['D']}\n"
            f"Answer:"
        )
        return {"text": prompt}

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    return ds


def main():
    args = parse_args()

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )

    print("Loading calibration dataset...")
    ds = load_calibration_dataset(
        limit=args.num_calibration_samples
    )

    print(f"Calibration samples: {len(ds)}")

    recipe = [
        AWQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["lm_head"],
        )
    ]

    print("Starting AWQ INT4 quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(ds),
    )

    print(f"AWQ INT4 quantization finished successfully.")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
