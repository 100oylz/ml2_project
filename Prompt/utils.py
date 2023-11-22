import torch
from transformers import PreTrainedTokenizer


def prompt_reflect(prompt, tokenizer, device):
    max_vals, _ = torch.max(prompt, dim=1, keepdim=True)
    min_vals, _ = torch.min(prompt, dim=1, keepdim=True)
    prompt_slice = max_vals - min_vals
    prompt_slice[prompt_slice == 0] = 1e-18
    prompt = (prompt - min_vals) / prompt_slice
    prompt = prompt * tokenizer.vocab_size
    prompt = prompt.clamp(0, tokenizer.vocab_size - 1).to(device)
    return prompt.long()


def mask_slice_reflect(mask_slice, prompt_length, device):
    mask, slice = mask_slice[:, 0], mask_slice[:, 1]
    mask, slice = torch.abs_(mask) * prompt_length, torch.abs_(slice) * (prompt_length + 1)
    mask, slice = mask.long().to(device), slice.long().to(device)
    mask, slice = mask % (prompt_length), slice % (prompt_length + 1)
    return mask.long(), slice.long()


def concate_prompt_data(prompt, data, mask, slice, tokenizer: PreTrainedTokenizer, device):
    batch_size = data.shape[0]
    data_length = data.shape[-1]
    masktokenid = tokenizer.mask_token_id
    clstokenid = tokenizer.cls_token_id
    septokenid = tokenizer.sep_token_id

    mask_tensor = torch.tensor([masktokenid], dtype=torch.long, device=device)
    cls_tensor = torch.tensor([clstokenid], dtype=torch.long, device=device)
    sep_tensor = torch.tensor([septokenid], dtype=torch.long, device=device)

    prompt = prompt.view(-1)
    mask_pos = mask.item()
    slice_pos = slice.item()
    prompt = torch.cat(
        (cls_tensor, prompt[:mask_pos], mask_tensor, prompt[mask_pos:], sep_tensor)
    )

    before_prompt = prompt[:slice_pos + 1]
    after_prompt = prompt[slice_pos + 1:]

    before_prompt = torch.stack([before_prompt] * batch_size, dim=0)
    after_prompt = torch.stack([after_prompt] * batch_size, dim=0)

    mask = mask + data_length + 1 if mask_pos >= slice_pos else mask + 1

    concat_data = torch.cat(
        (before_prompt, data, after_prompt), dim=1
    )
    return concat_data, mask
