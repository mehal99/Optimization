import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import wandb
import gc
from peft import LoraConfig, get_peft_model

gc.collect()
torch.cuda.empty_cache()

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_data(item):
    return {
        'prompt': 'Instruct: ' + item['prompt'] + '\n',
        'chosen': 'Output: ' + item['chosen'],
        'rejected': 'Output: ' + item['rejected']
    }

# Define a function to apply LoRA to a model
def apply_lora(model, device):
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # Scaling factor for LoRA
        target_modules=["c_attn"],  # Target layers to apply LoRA to
        lora_dropout=0.1,  # Dropout for regularization
        bias="none"  # Bias handling in LoRA
    )
    model = get_peft_model(model, lora_config).to(device)
    return model

def train(model, ref_model, train_dataset, eval_dataset, tokenizer, beta, training_args):
    model.train()
    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=1024,
        max_prompt_length=512
    )

    dpo_trainer.train()

def evaluate(model, ref_model, train_dataset, eval_dataset, tokenizer, beta, training_args):
    model.eval()
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=1024,
        max_prompt_length=512
    )
    dpo_trainer.evaluate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="dpo-HF")
    parser.add_argument("--wandb_run", type=str, default="lora-eval-gpt2")
    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login(key="5ac851d8254d920a4610373723ddd35d19cbd2e8")
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Apply LoRA to both the model and the reference model
    model = apply_lora(base_model, device)
    ref_model = apply_lora(base_model, device)

    train_dataset = load_dataset(args.dataset_name, split="train[:80%]")
    val_dataset = load_dataset(args.dataset_name, split="train[80%:]")

    train_dataset = train_dataset.map(preprocess_data)
    val_dataset = val_dataset.map(preprocess_data)

    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        output_dir='./results',
        logging_steps=10,
        remove_unused_columns=False,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=100,
        evaluation_strategy="steps",
        eval_steps=10
    )

    for _ in range(args.epochs):
        train(model, ref_model, train_dataset, val_dataset, tokenizer, args.beta, training_args)
        evaluate(model, ref_model, train_dataset, val_dataset, tokenizer, args.beta, training_args)

    model.save_pretrained("model-HF-DPO.pt")

if __name__ == "__main__":
    main()
