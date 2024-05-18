#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# /home/jovyan/.imgenv-optimistic-hellman-0/bin/python3 -u /home/jovyan/gpt13/13b_pretrain_119000/train_new.py > /home/jovyan/gpt13/13b_pretrain_119000/train_log.txt 2>&1

import torch

import os
import time
import random
import json
from pynvml import *
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer
from sftDataSet import SFTDataset

from datasets import load_dataset
from datasets import Dataset

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import bitsandbytes as bnb

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model, PeftModel

from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer
import os

# In[ ]:


os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_CACHE'] = 'artifacts/cache/'

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    #return trainable_params, all_param, 100 * trainable_params / all_param
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

res_dir = '/home/aegotovtsev/neural_2024/artifacts/result/'
loss_fn = '/home/aegotovtsev/neural_2024/artifacts/result_loss.json'

TOKENIZER_PATH = '/home/aegotovtsev/neural_2024/artifacts/model'
MODEL_PATH = '/home/aegotovtsev/neural_2024/artifacts/model'

# In[ ]:


print("loading tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH, device_map = 'auto')
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# In[ ]:

print("loading model")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=False, device_map = 'auto',  # cuda:0
    low_cpu_mem_usage=True, torch_dtype=torch.float32, offload_state_dict=True )

# In[ ]:


"""for module in base_model.modules():
    if isinstance(module, bnb.nn.Linear8bitLt):
        module.state.memory_efficient_backward = True"""

for param in base_model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

#model.gradient_checkpointing_enable()  # reduce number of stored activations
#model.model.decoder.project_in = lambda x: x.requires_grad_(True)
base_model.gradient_checkpointing_enable()  # reduce number of stored activations
base_model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

# In[ ]:


config = LoraConfig(
    fan_in_fan_out=True,
    r=16, # 8
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_fc", "c_proj", 'lm_head'],
)

# lora_checkpoint = '/home/jovyan/gpt13/13b_pretrain_119000/outputs_instr_psih12/checkpoint-38000'

model = get_peft_model(base_model, config)

# model = PeftModel.from_pretrained(base_model, lora_checkpoint, is_trainable=True, torch_dtype=torch.float16)  # torch_dtype=torch.float16

model.lm_head = CastOutputToFloat(model.lm_head)

print_trainable_parameters(model)

# In[ ]:


instruct_dataset_train = []
# for dataset_fn, diap in train_datasets:
#     with open(dataset_fn, 'r') as file:
#         instruct_dataset = file.read().strip()

#     instruct_dataset = instruct_dataset.split('<|endoftext|>')
#     instruct_dataset = [t.strip() + '<|endoftext|>' for t in instruct_dataset if len(t) > 0]
    
#     if diap:
#         instruct_dataset = instruct_dataset[diap[0]: diap[1]]
#     instruct_dataset_train += instruct_dataset


instruct_dataset_train = SFTDataset(tokenizer = tokenizer, sft_dataset = "data/dataset.jsonl")
print(len(instruct_dataset_train), 'len(instruct_dataset)')
print('non_trainnig_sample_0:\n' + instruct_dataset_train[0]['text'])
print('training_sample_0:\n' + instruct_dataset_train[0]['response'])
print(instruct_dataset_train[0]['text'] + instruct_dataset_train[0]['response'])
instruct_dataset_test = []
# for dataset_fn, diap in test_datasets:
#     with open(dataset_fn, 'r') as file:
#         instruct_dataset = file.read().strip()

#     instruct_dataset = instruct_dataset.split('<|endoftext|>')
#     instruct_dataset = [t.strip() + '<|endoftext|>' for t in instruct_dataset if len(t) > 0]
    
#     if diap:
#         instruct_dataset = instruct_dataset[diap[0]: diap[1]]
        
#     instruct_dataset_test += instruct_dataset

instruct_dataset_test = SFTDataset(tokenizer = tokenizer, sft_dataset = "data/dataset.jsonl")
print(len(instruct_dataset_test), 'len(instruct_dataset_test)')

# instruct_dataset_train = instruct_dataset_train[:10]
# instruct_dataset_test = instruct_dataset_test[:10]

# ds_train = Dataset.from_dict({"content": instruct_dataset_train})
# ds_train = ds_train.map(lambda example: tokenizer(example['content']), batched=True)
# ds_train = ds_train.filter(lambda x: len(x['input_ids']) < 2048)

# ds_test = Dataset.from_dict({"content": instruct_dataset_test})
# ds_test = ds_test.map(lambda example: tokenizer(example['content']), batched=True)
# ds_test = ds_test.filter(lambda x: len(x['input_ids']) < 2048)
ds_test = instruct_dataset_test
# ds1 = sorted(ds_train, key=lambda x: -len(x['input_ids']))

ds1 = instruct_dataset_train
#ds1 = sorted(instruct_dataset_train, key = lambda x: -len(x['input_ids']))
#print('longest example', len(ds1[0]['input_ids']))

# num_tokens = len(tokenizer)
# tokenizer.pad_token = tokenizer.eos_token

print(' || '.join(f'{key} -> {value}' for key, value in tokenizer.special_tokens_map.items()))

# #batch_size = 6
# #epochs = 3
# #acc_steps = 1
# #save_steps = 5000
# #max_steps = len(ds1)*epochs//batch_size


# In[ ]:


# from torch.utils.data import DataLoader

def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
# loader = DataLoader(ds1, batch_size=2, collate_fn=collate_fn)

# from itertools import islice

# for x in islice(loader, 1): 
#     print(x)

# In[ ]:


# это позволит сохранять только веса LORA
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class SaveLoRACallback(TrainerCallback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.save_lora_weights(kwargs['model'], self.output_dir, state.global_step)

    @staticmethod
    def save_lora_weights(model, output_dir, step):
        lora_dir = os.path.join(output_dir, f"lora_weights_step_{step}")
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        model.save_pretrained(lora_dir)

save_lora_callback = SaveLoRACallback(output_dir=os.path.join(res_dir, "trained"))


# In[ ]:


# TYT

batch_size = 4
epochs = 6
acc_steps = 4
save_steps = 300
logging_steps = 150
print_trainable_parameters(model)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=ds1, 
    # eval_dataset=ds_test,
    eval_dataset=None,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=acc_steps,
        do_train=True,
        warmup_steps=250,
        learning_rate=3e-4,
        bf16=True,
        logging_steps=logging_steps,
        output_dir=res_dir,
        logging_strategy='steps',
        save_strategy='steps',
        save_steps=save_steps,
        # do_eval=True,
        report_to = ["tensorboard"],
        # evaluation_strategy='steps',
        prediction_loss_only=True,
        per_device_eval_batch_size=batch_size,
        # eval_accumulation_steps=acc_steps,
        # eval_steps=save_steps, 
        num_train_epochs=epochs
    ),
    data_collator=collate_fn,
    callbacks=[save_lora_callback]
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

t0 = time.time()
trainer.train()
t1 = time.time()

with open(loss_fn, 'w') as f:
    json.dump(trainer.state.log_history, f)

print('train finished', t1-t0)

# model.save_pretrained(res_dir)


