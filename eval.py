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


import os

# model = get_peft_model(base_model, config)
model = PeftModel.from_pretrained(base_model, os.environ["LORA_CHECKPOINT"], is_trainable=False, torch_dtype=torch.float16)  # torch_dtype=torch.float16

# model = PeftModel.from_pretrained(base_model, lora_checkpoint, is_trainable=True, torch_dtype=torch.float16)  # torch_dtype=torch.float16

model.lm_head = CastOutputToFloat(model.lm_head)


############################# EVAL ############################################
############################# EVAL ############################################
############################# EVAL ############################################

with torch.no_grad():
    model.eval()
    template = "Ты - бот-анекдот. Ты рассказываешь анекдоты, о которых тебя попросит пользователь.\n\nU:{query}\n\nA:"
    for request in [
        "Расскажи анекдот про звонок в бордель.",
        "Расскажи анекдот про доктора Кошку и мышку, которую покусали коты.",
        "Расскажи анекдот про армян", # не из трейна
        "Расскажи анекдот про суп", # не из трейна
        "Расскажи анекдот про нюанс", # не из трейна
        "Расскажи анекдот про старца который жил на горе", # не из трейна
        "Расскажи анекдот про мужика который мешал бетон", # не из трейна
    ]:
        encoded_input = tokenizer(template.format(query=request), return_tensors='pt', \
                                add_special_tokens=False).to('cuda:0')
        output = model.generate(
            **encoded_input,
            num_beams=2,
            do_sample=True,
            max_new_tokens=500,
            repetition_penalty=1.5,
        )
        print(request)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print()
        print()


