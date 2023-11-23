import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from Prompt.utils import prompt_reflect, mask_slice_reflect


class prompt(nn.Module):
    def __init__(self, seq_length: int = 64,
                 device=torch.device('cuda'), prompt_num: int = 1):
        super().__init__()
        self.prompt_num = prompt_num
        self.prompt_length = seq_length
        self.device = device
        self.prompt, self.mask_slice = self.reParameterize()
        self.to(self.device)

    def forward(self):
        return self.prompt, self.mask_slice

    def returnPrompt(self, tokenizer):
        prompt, mask_slice = self.prompt, self.mask_slice
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

    def reParameterize(self):
        prompt = torch.randn((self.prompt_num, self.prompt_length))
        mask_slice = torch.randn((self.prompt_num, 2))
        return prompt.to(self.device), mask_slice.to(self.device)
