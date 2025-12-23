import torch
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B-AWQ-INT4 with LoRA")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="Varya-K/Qwen3-8B-AWQ-INT4",
        help="Path or HuggingFace model ID"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Qwen3-8B-AWQ-INT4-TUNED",
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    return model, tokenizer


def setup_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return get_peft_model(model, lora_config)


def format_example(example):
    question = example["question"]
    choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(example["choices"])])
    answer = example["answer"]
    
    prompt = f"Answer the multiple choice question: {question}. Choices: {choices}. Solve the task step-by-step and choose the correct answer."
    formatted_answer = f"\nCorrect answer: {answer}."
    
    return {"text": prompt + formatted_answer}


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def prepare_dataset(tokenizer):
    dataset = load_dataset("cais/mmlu", 'all')["auxiliary_train"]
    
    subset_size = int(len(dataset) * 0.005)
    dataset = dataset.select(range(subset_size))
    
    dataset = dataset.map(format_example)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset


def setup_training(model, tokenizer, dataset, args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
        bf16=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    return trainer


def main():
    args = parse_args()
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    model = setup_lora(model)
    
    print("Loading training dataset...")
    dataset = prepare_dataset(tokenizer)
    
    trainer = setup_training(model, tokenizer, dataset, args)
    
    print("Starting LoRA training...")
    trainer.train()
    
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()