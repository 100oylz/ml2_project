from transformers import AutoModel, PreTrainedModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Tuple


def getLLM(path) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    获取预训练语言模型和分词器。

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        包含加载的语言模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return model, tokenizer

