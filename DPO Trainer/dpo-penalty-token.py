import argparse
import random
import numpy as np
from datasets import load_dataset
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import wandb
import pandas as pd
import gc
import os

gc.collect()
torch.cuda.empty_cache()

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def load_bad_words(tokenizer):
#     df = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv')
#     bad_responses = df[df['toxic'] == 1]['comment_text'].tolist() 
#     #Extract bad words instead of complete bad responses (memory issues)
#     bad_words_set = set()
#     for response in bad_responses:
#         words = response.strip().split()
#         bad_words_set.update(words)
#     bad_words = [word for word in bad_words_set if word]
    
#     #Seeting a maximum number of bad words (memory issues)
#     max_bad_words = 300  
#     if len(bad_words) > max_bad_words:
#         bad_words = random.sample(bad_words, max_bad_words)
    
#     print("Bad words set: ", bad_words)
#     #Tokenize bad words
#     bad_token_ids = []
#     for word in bad_words:
#         token_ids = tokenizer.encode(word, add_special_tokens=False)
#         bad_token_ids.extend(token_ids)
#     #Remove duplicates
#     bad_token_ids = list(set(bad_token_ids))
    
#     return bad_token_ids


def load_bad_words(tokenizer):
    # df = pd.read_csv('bad_words.csv', header=None, names=['bad_words'])
    df = pd.read_csv('data\\fixed_normal_words.csv', header=0, names=['normal_words'])
    bad_words = df['normal_words'].tolist()
    bad_words = [word.lower() for word in bad_words if isinstance(word, str)]
    bad_words = list(set(bad_words))
    # max_bad_words = 300
    # if len(bad_words) > max_bad_words:
    #     bad_words = random.sample(bad_words, max_bad_words)
    # print("Bad words set: ", bad_words)  
    print("Normal words set: ", bad_words)  
    # Tokenize bad words
    bad_token_ids = []
    for word in bad_words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        bad_token_ids.extend(token_ids)
    
    # Remove duplicates
    bad_token_ids = list(set(bad_token_ids))
    return bad_token_ids

def preprocess_data(item):
    return {
        'prompt': 'Instruct: ' + item['prompt'] + '\n',
        'chosen': 'Output: ' + item['chosen'],
        'rejected': 'Output: ' + item['rejected']
    }

def apply_lora(model):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    return model

def compute_penalty(model, input_ids, attention_mask, bad_token_ids, lambda_val, prompt_length):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  #[batch_size, seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1) 
    response_probs = probs[:, prompt_length - 1:-1, :]
    if not isinstance(bad_token_ids, torch.Tensor):
        bad_token_ids = torch.tensor(bad_token_ids).to(input_ids.device)
    
    # Compute probabilities assigned to bad tokens
    bad_token_probs = response_probs.index_select(dim=-1, index=bad_token_ids) # Shape: [batch_size, response_seq_len, num_bad_tokens]
    #Sum up the probabilities assigned to bad tokens at each position in the response 
    total_bad_probs = bad_token_probs.sum(dim=-1)  # Shape: [batch_size, seq_len]

    # average probability assigned to bad tokens across all positions and sequences
    avg_bad_prob = total_bad_probs.mean()

    # penalty
    penalty = lambda_val * avg_bad_prob

    return penalty, avg_bad_prob

class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, bad_token_ids=None, lambda_val=0.1, step_size=0.01, threshold=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.bad_token_ids = bad_token_ids
        self.lambda_val = lambda_val
        self.step_size = step_size
        self.threshold = threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        
        dpo_loss = super().compute_loss(model, inputs, return_outputs=False)
        
        prompt_inputs = self.tokenizer(inputs['prompt'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        prompt_input_ids = prompt_inputs['input_ids'].to(model.device)
        prompt_attention_mask = prompt_inputs['attention_mask'].to(model.device)
        
        if prompt_input_ids.dim() == 1:
            prompt_input_ids = prompt_input_ids.unsqueeze(0)
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)

        #chosen response
        chosen_response = inputs['chosen']
        chosen_inputs = self.tokenizer(chosen_response, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        chosen_input_ids = chosen_inputs['input_ids'].to(model.device)
        chosen_attention_mask = chosen_inputs['attention_mask'].to(model.device)

        # prompt + chosen response
        input_ids = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, chosen_attention_mask], dim=1)
        
        # Compute penalty
        prompt_length = prompt_input_ids.size(1)
        penalty, avg_bad_prob = compute_penalty(model=model, input_ids=input_ids, attention_mask=attention_mask, bad_token_ids=self.bad_token_ids, lambda_val=self.lambda_val, prompt_length=prompt_length)

        #Check constraint violation and update lambda
        self.lambda_val = max(self.lambda_val + self.step_size*(avg_bad_prob.item()-self.threshold), 0)
        # penalty = self.lambda_val * (avg_bad_prob.item() - self.threshold)/avg_bad_prob

        # constraint_violation = avg_bad_prob.item() > self.threshold
        # if constraint_violation:
        #     self.lambda_val += self.step_size
        # else:
        #     self.lambda_val = max(0.0, self.lambda_val - self.step_size)

        total_loss = dpo_loss + penalty

        wandb.log({
            'dpo_loss': dpo_loss.item(),
            'penalty': penalty,
            'total_loss': total_loss.item(),
            'lambda_val': self.lambda_val,
            'avg_bad_prob': avg_bad_prob.item()
        })

        if return_outputs:
            return total_loss, None
        else:
            return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lambda_penalty", type=float, default=0.1)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="dpo_penalty_2")
    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login(key="5ac851d8254d920a4610373723ddd35d19cbd2e8")  
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model = apply_lora(base_model)

    # Data Loading and preprecessing
    train_dataset = load_dataset(args.dataset_name, split="train[:80%]")
    val_dataset = load_dataset(args.dataset_name, split="train[80%:]")
    train_dataset = train_dataset.map(preprocess_data)
    val_dataset = val_dataset.map(preprocess_data)    

    # Load bad dataset (words) -> Tokenize bad words
    bad_token_ids = load_bad_words(tokenizer)

    output_dir = os.path.abspath('results')
    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_steps=10,
        remove_unused_columns=False,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=100,
        eval_steps=10,
        evaluation_strategy="steps",
        logging_dir='./logs',
        logging_strategy="steps"
    )

    model.train()
    ref_model.eval()
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        bad_token_ids=bad_token_ids,
        lambda_val=args.lambda_penalty, 
        step_size=args.step_size,
        threshold=args.threshold,  
        max_length=1024,
        max_prompt_length=512
    )

    trainer.train()
    model.save_pretrained(f"model-HF-Constrained-DPO-{args.lambda_penalty}-{args.threshold}-{args.step_size}.pt")

if __name__ == "__main__":
    main()
