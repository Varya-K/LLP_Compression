from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASELINE_MODEL_PATH = "Qwen/Qwen3-8B"
COMPRESSED_MODEL_PATH = "Varya-K/Qwen3-8B-AWQ-INT4"

def evaluate_mmlu(model_path, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model_size = get_model_size_in_MB(model)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    lm.seqlen = 2048

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=0,
        batch_size=batch_size,
        limit=100
    )

    return results["results"]["mmlu"]["acc,none"], model_size

def get_model_size_in_MB(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_bytes = param_size + buffer_size
    return total_size_bytes / 1024 ** 2


if __name__ == "__main__":
    print("Evaluating baseline model...")
    baseline_acc, baseline_size = evaluate_mmlu(BASELINE_MODEL_PATH)
    print(f"Baseline accuracy: {baseline_acc:.6f}")
    print(f"Baseline size (MB): {baseline_size:.4f}")

    print("Evaluating compressed model...")
    comp_acc, comp_size = evaluate_mmlu(COMPRESSED_MODEL_PATH)
    print(f"Compressed accuracy: {comp_acc:.6f}")
    print(f"Compressed size (MB): {comp_size:.4f}")

    compression_ratio = baseline_size / comp_size
    performance_drop = (baseline_acc - comp_acc) / baseline_acc
    score = compression_ratio / (1 + performance_drop)

    print("\n==== RESULTS ====")
    print(f"Baseline accuracy: {baseline_acc:.6f}")
    print(f"Compressed accuracy: {comp_acc:.6f}")
    print(f"Baseline size (MB): {baseline_size:.4f}")
    print(f"Compressed size (MB): {comp_size:.4f}")
    print(f"Compression ratio: {compression_ratio:.4f}")
    print(f"Performance drop: {performance_drop:.6f}")

    print(f"Score: {score:.6f}")