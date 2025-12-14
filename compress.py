import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Qwen3-8B-FP8-Dynamic",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    MODEL_ID = args.model_id
    SAVE_DIR = args.output_dir

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    # Квантицируем все линейные слои, за исключением выходного lm_head
    print("Applying FP8 Dynamic quantization...")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"]
    )

    oneshot(model=model, recipe=recipe)

    print(f"Saving compressed model to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Compression finished successfully.")


if __name__ == "__main__":
    main()
