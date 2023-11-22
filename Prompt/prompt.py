import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from Prompt.utils import prompt_reflect, mask_slice_reflect


class prompt(nn.Module):
    def __init__(self, init_shape: int = 64, gru_hidden_state: int = 1024, gru_layers: int = 1, seq_length: int = 64,
                 device=torch.device('cuda'), prompt_num: int = 1):
        super().__init__()
        self.prompt_init = nn.Parameter(torch.empty((init_shape, init_shape)).to(device))
        self.mask_slice_init = nn.Parameter(torch.empty((init_shape * init_shape)).to(device))
        # 初始化为正态分布
        nn.init.normal_(self.prompt_init)
        nn.init.normal_(self.mask_slice_init)

        self.gru = nn.GRU(init_shape, gru_hidden_state, num_layers=gru_layers)
        self.prompt_num = prompt_num
        self.prompt_length = seq_length
        self.generate_prompt = nn.Linear(gru_hidden_state * init_shape,
                                         self.prompt_num * self.prompt_length)
        self.relu = nn.ReLU()

        self.generate_mask_slice = nn.Linear(init_shape * init_shape, self.prompt_num * 2)
        self.device = device
        self.to(self.device)

    def forward(self):
        prompt, _ = self.gru(self.prompt_init)
        prompt = self.generate_prompt(prompt.view(-1))
        prompt = prompt.view(self.prompt_num, self.prompt_length)

        mask_slice = self.generate_mask_slice(self.mask_slice_init)
        mask_slice = mask_slice.view(self.prompt_num, 2)
        return prompt, mask_slice

    def returnPrompt(self, tokenizer):
        prompt, mask_slice = self.forward()
        prompt = prompt_reflect(prompt, tokenizer, device=self.device)
        mask, slice = mask_slice_reflect(mask_slice, prompt.shape[-1], device=self.device)

        masktokenid = tokenizer.mask_token_id
        clstokenid = tokenizer.cls_token_id
        septokenid = tokenizer.sep_token_id

        mask_tensor = torch.tensor([masktokenid], dtype=torch.long, device=self.device)
        cls_tensor = torch.tensor([clstokenid], dtype=torch.long, device=self.device)
        sep_tensor = torch.tensor([septokenid], dtype=torch.long, device=self.device)

        prompt = prompt.view(-1)
        mask_pos = mask.item()
        slice_pos = slice.item()
        prompt = torch.cat(
            (cls_tensor, prompt[:mask_pos], mask_tensor, prompt[mask_pos:], sep_tensor)
        )

        before_prompt = prompt[:slice_pos + 1]
        after_prompt = prompt[slice_pos + 1:]
        return tokenizer.decode(before_prompt), tokenizer.decode(after_prompt), mask_pos
