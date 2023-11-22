from dataclasses import dataclass
from typing import List, Tuple
from database.database import database

import torch

LLM_PATH = r'D:\huggingface\hflchinese-macbert-large'
LEVEL_TOKEN_FORMAT = '{:0>2}'
LABEL_TOKEN_FORMAT = '{}'


@dataclass
class promptConfig():
    name: str
    lr: float = 1e-4
    weight_decay: float = 5e-4
    num_epochs: int = 50
    seed: int = 24
    dropout: float = 0.2
    device: torch.device = torch.device('cuda')
    prompt_init_shape: int = 64
    prompt_gru_hidden_state: int = 1024
    prompt_gru_layer: int = 1
    prompt_seq_length: int = 64
    mask_str: str = "[MASK]"
    slice_num: int = 100
    mask_hidden_features: Tuple[int] = (512, 256)
    cut_hidden_features: Tuple[int] = (64, 128, 256)
    random_state: Tuple[int] = (0, 1, 2, 3, 4)


ADNI = database('ADNI', 'ADNI')
PPMI = database('PPMI', 'PPMI')
ADNI_fMRI = database('ADNI_fMRI', 'ADNI_90_120_fMRI')
OCD_fMRI = database('OCD_fMRI', 'OCD_90_200_fMRI')
FTD_fMRI = database('FTD_fMRI', 'FTD_90_200_fMRI')

ADNI_config = promptConfig('ADNI')
PPMI_config = promptConfig('PPMI')
ADNI_fMRI_config = promptConfig('ADNI_fMRI')
OCD_fMRI_config = promptConfig('OCD_fMRI')
FTD_fMRI_config = promptConfig('FTD_fMRI')
