import os
import torch
import numpy as np
import wandb
from typing import Optional

from functools import partial
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    TrainingArguments,
    Trainer,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    PreTrainedTokenizer,
    EvalPrediction,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset, Dataset

CHOICE_TOKENS = (9454, 2753)

def apply_template(examples:dict, tokenizer:PreTrainedTokenizer) -> dict:
    """
    Apply chat template to the examples using the tokenizer.
    """
    chat_list = [
        {"role" : "system", "content" : examples["system"]},
        {"role" : "user", "content" : examples["instruction"]},
    ]
    chat = tokenizer.apply_chat_template(chat_list, tokenize=True, add_generation_prompt=True)
    examples["input_ids"] = chat
    del examples["system"], examples["instruction"]
    return examples

def compute_loss(outputs:CausalLMOutputWithPast, labels:int, num_items_in_batch:Optional[int]=None) -> torch.Tensor:
    """
    Compute the loss for the model outputs.
    """
    prob = outputs.logits[..., -1, :] # (batch_size, n_vocab)
    target = torch.zeros_like(prob)
    target[..., CHOICE_TOKENS[0]] = labels
    target[..., CHOICE_TOKENS[1]] = 1-labels
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(prob, target)
    
    return loss

def preprocess_logits_for_metrics(logits:torch.FloatTensor, labels:torch.FloatTensor) -> torch.Tensor:
    return torch.softmax(logits[..., -1, :], dim=-1)

def compute_metrics(eval_pred:EvalPrediction) -> dict:
    """
    Compute evaluation metrics.
    """
    prob = eval_pred.predictions
    labels = eval_pred.label_ids
    targets = np.zeros_like(prob)
    targets[..., CHOICE_TOKENS[0]] = labels
    targets[..., CHOICE_TOKENS[1]] = 1-labels
    mae = np.mean(np.sum(np.abs(prob - targets), axis=-1))
    mse = np.mean(np.sum((prob - targets) ** 2, axis=-1))
    return {"MSE":mse, "MAE":mae}

# settings
dataset_path = "dataset_training"

# Initialize wandb
wandb.init(project="or_judgement")


peft_config = LoraConfig(
    # target_modules="all-linear",
)

base_model:Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
tokenizer:Qwen2Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = get_peft_model(base_model, peft_config)
model.train()

# load datasets and preprocess
train_dataset = load_dataset("json", data_dir=dataset_path, data_files="or_judgement_train.json").map(partial(apply_template, tokenizer=tokenizer))
eval_dataset = {
    "answer":load_dataset("json", data_dir=dataset_path, data_files="eval_answer.json").map(partial(apply_template, tokenizer=tokenizer))["train"],
    "toxic":load_dataset("json", data_dir=dataset_path, data_files="eval_toxic.json").map(partial(apply_template, tokenizer=tokenizer))["train"],
}

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=AdamW,
    lr=5e-5,
    loraplus_lr_ratio=16,
)

output_path = "result_finetuning/lora/qwen"
i=0
while os.path.exists(output_path+f"_{i}"):
    i+=1
output_path += f"_{i}"

training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=5e-5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    logging_steps=50,
    logging_dir="log",
    log_level="info",
    warmup_steps=20,
    save_steps=20,
    eval_steps=20,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_answer_loss",
    bf16=True,
    report_to="wandb",
    label_names=["labels"],
    eval_on_start=True,
    # resume_from_checkpoint=True,
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    compute_loss_func=compute_loss,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

# save model
model.save_pretrained("model_finetuning/lora/qwen")
tokenizer.save_pretrained("model_finetuning/lora/qwen")

# finish wandb
wandb.finish()