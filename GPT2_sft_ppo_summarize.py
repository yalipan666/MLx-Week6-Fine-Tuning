"""
Integrated SFT and PPO Summarization Script
- SFT: Supervised fine-tuning of GPT-2 for Reddit TL;DR summarization
- PPO: Reinforcement learning (RLHF) with reward model
"""

# =============================
# 1. Imports and Setup
# =============================
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
import evaluate

# PPO/RLHF imports
from trl import (
    PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model, AutoModelForCausalLMWithValueHead
)
from trl.core import LengthSampler

# =============================
# 2. Utility Functions
# =============================
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# =============================
# 3. SFT: Supervised Fine-Tuning
# =============================
class TLDRDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }

# SFT Training setup
output_dir = "./gpt2-supervised-summarize-checkpoint"
train_batch_size = 16
gradient_accumulation_steps = 1
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 500
max_input_length = 550
save_steps = 1000
num_train_epochs = 5
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

data_path = "CarperAI/openai_summarize_tldr"
train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    max_length=max_input_length,
)
dev_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "valid",
    max_length=max_input_length,
)

# Metric for SFT
rouge = evaluate.load("rouge")
def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

# SFT Trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_accumulation_steps=1,
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_checkpointing=True,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=save_steps,
    max_steps=29000,
    load_best_model_at_end=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# To train:
# trainer.train()
# trainer.save_model(output_dir)

# =============================
# 4. PPO: RLHF Training
# =============================
# PPO model and reward setup
SFT_MODEL_PATH = output_dir  # Use SFT checkpoint as PPO starting point
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_MODEL_PATH, is_trainable=True)
ref_model = create_reference_model(ppo_model)

# Reward model (OpenAssistant)
from transformers import AutoModelForSequenceClassification
rw_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rw_tokenizer = AutoTokenizer.from_pretrained(rw_model_name)
rw_model = AutoModelForSequenceClassification.from_pretrained(rw_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# PPO dataset preparation (tokenized)
def build_dataset(model_name, dataset_name, input_min_text_length, input_max_text_length):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess(split):
        split = split.filter(
            lambda x: input_min_text_length < len(x["prompt"]) <= input_max_text_length,
            batched=False
        )
        def tokenize(sample):
            prompt = f"{sample['prompt']}\n\n"
            inputs = tokenizer(prompt, truncation=True, max_length=1024)
            sample["input_ids"] = inputs["input_ids"]
            sample["query"] = tokenizer.decode(inputs["input_ids"], skip_special_tokens=True)
            return sample
        split = split.map(tokenize, batched=False)
        split.set_format(type="torch")
        return split
    dataset["train"] = preprocess(dataset["train"])
    dataset["valid"] = preprocess(dataset["valid"])
    dataset["test"]  = preprocess(dataset["test"])
    return dataset

ppo_dataset = build_dataset(
    model_name="gpt2",
    dataset_name="CarperAI/openai_summarize_tldr",
    input_min_text_length=200,
    input_max_text_length=1000
)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# PPO Trainer setup
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    ppo_epochs=1,
    mini_batch_size=4,
    batch_size=16,
)
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=ppo_dataset["train"],
    data_collator=collator
)

# PPO Training loop (example, not full loop)
output_min_length = 100
output_max_length = 512
output_length_sampler = LengthSampler(output_min_length, output_max_length)
generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}
max_ppo_steps = 12

# Uncomment to run PPO training
# for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     if step >= max_ppo_steps:
#         break
#     prompt_tensors = batch["input_ids"]
#     summary_tensors = []
#     for prompt_tensor in prompt_tensors:
#         max_new_tokens = output_length_sampler()
#         generation_kwargs["max_new_tokens"] = max_new_tokens
#         summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
#         summary_tensors.append(summary.squeeze()[-max_new_tokens:])
#     batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in summary_tensors]
#     reward_texts = [q + "\n" + r for q, r in zip(batch["query"], batch["response"])]
#     reward_tensors = []
#     for text in reward_texts:
#         inputs = rw_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(rw_model.device)
#         with torch.no_grad():
#             logits = rw_model(**inputs).logits
#             reward = torch.sigmoid(logits).item()
#         reward_tensors.append(torch.tensor(reward))
#     stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
#     ppo_trainer.log_stats(stats, batch, reward_tensors)
#     print(f'objective/kl: {stats["objective/kl"]}')
#     print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
#     print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
#     print("-" * 80)

# =============================
# 5. Main Execution Block
# =============================
if __name__ == "__main__":
    print("This script integrates SFT and PPO training for Reddit summarization.")
    print("To train SFT, uncomment trainer.train(). To train PPO, uncomment the PPO training loop.") 