from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from evaluate_models import evaluate_mmlu

BASELINE_MODEL_PATH = "Qwen/Qwen3-8B"
TUNED_COMPRESSED_MODEL_PATH = "Varya-K/Qwen3-8B-AWQ-INT4-TUNED"

if __name__ == "__main__":
    baseline_acc = 0.7542105263157894
    baseline_size = 15622.588134765625
    print(f"Baseline accuracy: {baseline_acc:.6f}")
    print(f"Baseline size (MB): {baseline_size:.4f}")   

    print("Evaluating tuned compressed model...")
    tuned_comp_acc, tuned_comp_size = evaluate_mmlu(TUNED_COMPRESSED_MODEL_PATH)
    print(f"Tuned compressed accuracy: {tuned_comp_acc:.6f}")
    print(f"Tuned compressed size (MB): {tuned_comp_size:.4f}")

    compression_ratio = baseline_size / tuned_comp_size
    performance_drop = (baseline_acc - tuned_comp_acc) / baseline_acc
    score = compression_ratio / (1 + performance_drop)

    print("\n==== RESULTS ====")
    print(f"Baseline accuracy: {baseline_acc:.6f}")
    print(f"Tenude compressed accuracy: {tuned_comp_acc:.6f}")
    print(f"Baseline size (MB): {baseline_size:.4f}")
    print(f"Tuned compressed size (MB): {tuned_comp_size:.4f}")
    print(f"Compression ratio: {compression_ratio:.4f}")
    print(f"Performance drop: {performance_drop:.6f}")

    print(f"Score: {score:.6f}")