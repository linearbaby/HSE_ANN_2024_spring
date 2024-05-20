#!/usr/bin/env python
# coding: utf-8

import torch

import pandas as pd
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


print("loading tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH, device_map = 'auto')
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

print("loading model")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=False, device_map = 'auto',  # cuda:0
    low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, offload_state_dict=True )


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

import os
base_path = "/home/aegotovtsev/neural_2024/artifacts/result"

result = {}

for lora_model in [
    "checkpoint_best",  "checkpoint_best_v2",  "checkpoint_best_v3"
]:
    result[lora_model] = {}
    print(f"evaling {lora_model}")
    # model = get_peft_model(base_model, config)
    model = PeftModel.from_pretrained(base_model, os.path.join(base_path, lora_model), is_trainable=False, torch_dtype=torch.bfloat16)  # torch_dtype=torch.float16

    ############################# EVAL ############################################
    ############################# EVAL ############################################
    ############################# EVAL ############################################

    with torch.no_grad():
        model.eval()
        template = "Ты - бот-анекдот. Ты рассказываешь анекдоты, о которых тебя попросит пользователь.\n\nU:{query}\n\nA:"
        for idx, request in enumerate([
            'Расскажи анекдот про боцмана, который спасает тонущий корабль с помощью резиновой уточки.',
            'Расскажи анекдот про старца, который никогда не спорил и однажды выиграл в споре, заключив пари на луну.',
            'Расскажи анекдот про мужика, который пришёл в магазин и купил утюг для ловли рыбы.',
            'Расскажи анекдот про Вовочку, который использовал школьный микроскоп для приготовления завтрака.',
            'Расскажи анекдот про армянское радио, отвечающее на вопрос о том, почему курица пересекла дорогу.',
            'Расскажи анекдот про медведя, который решил стать вегетарианцем и пришёл в лесной ресторан за салатом.',
            'Расскажи анекдот про хохла и москаля, которые поспорили, чей борщ вкуснее.',
            'Расскажи анекдот про грузина, который в ресторане заказал компот из шашлыков.',
            'Расскажи анекдот про зайца, который решил обмануть волка, выдав себя за лесного духа.',
            'Расскажи анекдот про чукчу, который впервые увидел телевизор и подумал, что это волшебное окно.',

            'Расскажи анекдот про блондинку и машину.',
            'Расскажи анекдот про директора и секретаршу.',
            'Расскажи анекдот про мужа, который опоздал домой.',
            'Расскажи анекдот про рыбалку и пьяного рыбака.',
            'Расскажи анекдот про студента на экзамене.',
            'Расскажи анекдот про гаишника и водителя.',
            'Расскажи анекдот про тещу и зятя.',
            'Расскажи анекдот про Новый год и Деда Мороза.',
            'Расскажи анекдот про Петю и Васю на охоте.',
            'Расскажи анекдот про врача и пациента.',
            'Расскажи анекдот про учителя и ученика.',

            'Расскажи анекдот про охотника и рыбака, которые спорили о том, чье хобби лучше.',
            'Расскажи анекдот про дворника, который нашёл что-то интересное на своей территории.',
            'Расскажи анекдот про сантехника, который пришел чинить кран и нашел там нечто неожиданное.',
            'Расскажи анекдот про генерала, который решил проверить солдат на смекалку.',
            'Расскажи анекдот про милиционера, который поймал преступника с помощью хитрости.',
            'Расскажи анекдот про профессора, который задал студентам очень сложный вопрос.',
            'Расскажи анекдот про жену, которая обнаружила любовницу мужа в неожиданной ситуации.',
            'Расскажи анекдот про монашку, которая решила испытать свою веру.',
            'Расскажи анекдот про программиста, который пытался объяснить компьютеру, что такое любовь.',
            'Расскажи анекдот про английского лорда, который удивил своего слугу странным поступком.'
        ]):
            print(f"encoded anek {idx}")
            encoded_input = tokenizer(template.format(query=request), return_tensors='pt', \
                                    add_special_tokens=False).to('cuda:0')
            output = model.generate(
                **encoded_input,
                num_beams=2,
                do_sample=True,
                max_new_tokens=500,
                repetition_penalty=1.5,
            )
            result[lora_model][str(idx)] = tokenizer.decode(output[0], skip_special_tokens=True)

pd.DataFrame.from_dict(result).T.to_csv("check_endpoints.csv")
