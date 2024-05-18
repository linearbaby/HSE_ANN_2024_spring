import json
import os
from typing import Dict, List, Tuple, Any, Union
import pandas as pd

import transformers
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import re, random
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM = "Ты - бот-анекдот. Ты рассказываешь анекдоты, о которых тебя попросит пользователь."
ROLE_MAPPING = {
    "system": ("", ""),
    "user": ("\n\nU:", ""),
    "assistant": ("\n\nA:", ""),
}
BOT_TOKEN = "[bot_token]"

@dataclass
class DatasetConfig:
    max_seq_length: int = 2048
    role_mapping: Dict[str, Tuple] = field(default_factory=lambda: ROLE_MAPPING.copy())
    bot_token: str = BOT_TOKEN

@dataclass
class RawDialog:
    """
    Represents a raw dialog with system, context, and response.
    """
    system: str
    context: str
    response: str
    # last_prompt: str

result = []
context = []


config = DatasetConfig(
            max_seq_length=2048,
            role_mapping=ROLE_MAPPING,
            bot_token="[bot_token]",
        )


def load_json_file(file_path):
    # Open and read the JSON file, automatically closing it after reading
    with open(file_path, "r") as file:
        data_dict = json.load(file)
    return data_dict

# def get_role_prefix(role):
#     #print('get_role_prefix -> ' + config.role_mapping[role][0])
#     return config.role_mapping[role][0]
# def get_mapped_role(role, content):
#     prefix, postfix = config.role_mapping.get(role, ("",""))
#     #print(f'prefix -> |{prefix}| \ncontent -> |{content}| \npostfix -> |{postfix}|')
#     return f"{prefix}{content}{postfix}"

def _slice_dialog(system_prompt_tokens: List[int], 
                  context_tokens: List[int], 
                  response_tokens: List[int], 
                  last_prompt_tokens: List[int] = None):
    
    if len(system_prompt_tokens) + len(context_tokens) + len(response_tokens) > config.max_seq_length:
        return None
    return system_prompt_tokens + context_tokens + response_tokens

def _get_labels(input_tokens: List[int], response_tokens: List[int]):
    label_tokens = [-100 for _ in range(len(input_tokens))]
    label_tokens[-len(response_tokens) :] = response_tokens

    assert len(label_tokens) == len(input_tokens)

    return label_tokens

class SFTDataset(Dataset):
    def __init__(
        self,
        sft_dataset: str,
        tokenizer,
        config = config,
    ):
        """
        Initialize the SFTDataset instance.

        Parameters:
            sft_dataset (str): The path to the directory containing the SFT dataset.
            tokenizer (PreTrainedTokenizerFast): The tokenizer used to tokenize the data.
            config (SFTDatasetConfig): The configuration for the SFT dataset.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config
        self.eos_token_idx = self.tokenizer.eos_token_id
        self.processed_examples = self._prepare_data(sft_dataset)

    def _prepare_data(self, sft_dataset) -> List[Dict[str, torch.LongTensor]]:
        dataset = []
        df = pd.read_json(sft_dataset, lines=True)

        prep_dataset = self._prep(df)
        examples = self._token_dialogs(prep_dataset)
        return examples

    def _prep(self, dataset) -> List[RawDialog]:
        print('splitting texts')
        result = []
        
        for idx, row in dataset.iterrows():
            context = ROLE_MAPPING["user"][0] + row["prompt"] + ROLE_MAPPING["user"][1] + ROLE_MAPPING["assistant"][0]
            answer = row["text"]
            
            result.append(
                RawDialog(
                    system=DEFAULT_SYSTEM,
                    context=context,
                    response=answer,
                )
            )

        return result
    

    def _token_dialogs(self, results: List[RawDialog]):
        print('tokenization texts')
        examples = []
        for sample in results:
        # fmt: off
            system_prompt_tokens = self.tokenizer(sample.system, add_special_tokens=False)["input_ids"]
            context_tokens = self.tokenizer(sample.context, add_special_tokens=False)["input_ids"]
            response_tokens = self.tokenizer(sample.response, add_special_tokens=False)["input_ids"]
            # last_prompt_tokens = tokenizer(sample.last_prompt, add_special_tokens=False)["input_ids"]
            # fmt: on

            response_tokens = response_tokens + [self.tokenizer.eos_token_id]

            input_tokens = _slice_dialog(
                system_prompt_tokens=system_prompt_tokens,
                context_tokens=context_tokens,
                response_tokens=response_tokens,
                # last_prompt_tokens=last_prompt_tokens,
            )

            if input_tokens is not None:
                labels = _get_labels(input_tokens, response_tokens)

                assert len(input_tokens) == len(labels) <= config.max_seq_length, (
                    len(input_tokens),
                    len(labels),
                )

                examples.append(
                    {
                        "labels": torch.LongTensor(labels),
                        "input_ids": torch.LongTensor(input_tokens),
                        "attention_mask": torch.ones(
                            len(input_tokens), dtype=torch.long
                        ),
                        "text": sample.system + " " + sample.context,
                        "response":  sample.response
                    }
                )
        
        return examples


    def __len__(self):
            return len(self.processed_examples)

    def __getitem__(self, item):
            return self.processed_examples[item]