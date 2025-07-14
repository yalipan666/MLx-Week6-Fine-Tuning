"""
************* step 1: fine-tuning gpt-2 for summarization  ***************
"""


"""
dataset
"""


from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        dataset = load_dataset(train_path, split=split)
        self.examples = [sample["prompt"] + sample["label"] for sample in dataset]
        self.examples = self.examples[:2000] if "valid" in split else self.examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx], truncation=True, max_length=self.max_length, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]), # mask for a real valid token or padding
            "labels": torch.tensor(enc["input_ids"]),  # teacher forcing, during training, the model is given the correct previous token as input at each step, rather than its own previous prediction.
        }
    


"""
model & tokenizer setup
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
model.resize_token_embeddings(len(tokenizer))  # adjust token count, as the special EOS is just added into vocab
model.config.pad_token_id = tokenizer.eos_token_id



"""
training loop (huggingface trainer)
"""

from transformers import TrainingArguments, Trainer, default_data_collator
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    # batch_decode: It’s the reverse of tokenization: from numbers (token IDs) back to strings
    preds = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=True) 
    labels = tokenizer.batch_decode(eval_preds.label_ids, skip_special_tokens=True)
    return rouge.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="/notebooks/gpt2-supervised-summarize-checkpoint",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=50,
    gradient_accumulation_steps=1,
    fp16=True,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()




"""
************* step 2: training a reward model vs using an off-the shelf one  ***************
"""

"""
dataset
"""
def create_comparison_dataset(path, split):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in dataset:
        if sample["chosen"] == sample["rejected"]:
            continue
        if len(sample["chosen"].split()) < 5 or len(sample["rejected"].split()) < 5:
            continue
        pairs.append({
            "chosen": sample["prompt"] + "\n" + sample["chosen"],
            "rejected": sample["prompt"] + "\n" + sample["rejected"],
        })
    return pairs



"""
model architecture: gpt-2 with a value head
"""
from transformers import AutoModelForCausalLM
import torch.nn as nn

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]



from reward_model.reward_model import GPTRewardModel
# Load the base model (from SFT checkpoint)
model = GPTRewardModel("/notebooks/gpt2-supervised-summarize-checkpoint/checkpoint-29000")

# Freeze the first 70% of transformer layers to save memory
layers = model.transformer.h
num_layers = len(layers)
num_unfrozen = int(0.3 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)


"""
dataset formatting: pairwise comparison
"""
class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        # tokenize chosen and rejected summarie








"""
************* step 3: reinforcement learning: PPO  ***************
"""

"""
load the fine-tuned gpt-2 model
"""
model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


"""
setting up the PPO model
"""
from trl import AutoModelForCausalLMWithValueHead

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_MODEL_PATH, is_trainable=True)
ref_model = create_reference_model(ppo_model)

"""
loading the reward model
"""
rw_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2").to("cuda")
rw_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")


full_gen_input = f"{input_text}\n{generated_summary}"
full_ref_input = f"{input_text}\n{reference_summary}"

gen_inputs = reward_tokenizer(full_gen_input, return_tensors="pt").to("cuda")
ref_inputs = reward_tokenizer(full_ref_input, return_tensors="pt").to("cuda")

gen_reward = sigmoid(reward_model(**gen_inputs).logits).item()
ref_reward = sigmoid(reward_model(**ref_inputs).logits).item()

normalized_reward = gen_reward / ref_reward if ref_reward != 0 else 0.0


"""
PPO training loop
"""
for step, batch in enumerate(ppo_trainer.dataloader):
    if step >= max_ppo_steps:
        break

    # Generate responses
    summaries = ppo_trainer.generate(batch["input_ids"], **generation_kwargs)

    # Compute reward scores
    reward_texts = [q + "\n" + r for q, r in zip(batch["query"], decoded_summaries)]
    reward_scores = [sigmoid(rw_model(**rw_tokenizer(t, return_tensors="pt").to("cuda")).logits).item() for t in reward_texts]

    # PPO step
    stats = ppo_trainer.step(batch["input_ids"], summaries, reward_scores)
    ppo_trainer.log_stats(stats, batch, reward_scores)



"""
evaluation: did PPO improve the model
"""
compare_results["reward_before"] = get_rewards(reference_summaries)
compare_results["reward_after"] = get_rewards(ppo_summaries)

Counter({'B': 12, 'A': 6, 'Neither': 2})







