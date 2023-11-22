import torch.nn

from Database.database import database
from Database.dataset import dataset
from Database.utils import split_train_valid_test
from Prompt.prompt import prompt
from Prompt.mlp import mlp
from Prompt.utils import concate_prompt_data, mask_slice_reflect, prompt_reflect
from Prompt.llm import getLLM
from config import *
from logger import logConfig

import numpy as np
import random
from transformers import PreTrainedTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def setup_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def encode(data: torch.Tensor, tokenizer: PreTrainedTokenizer):
    shape = data.shape
    if (len(shape) == 2):
        encode_tensor_list = []
        for i in range(shape[0]):
            encode_item_list = []
            for j in range(shape[1]):
                encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j])
                encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                encode_item_list.append(encode_item)
            # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
            encode_tensor_list.append(encode_item_list)
        return torch.tensor(encode_tensor_list, dtype=torch.long)
    elif (len(shape) == 3):
        encode_tensor_tensor_list = []
        for i in range(shape[0]):
            encode_tensor_list = []
            for j in range(shape[1]):
                encode_item_list = []
                for k in range(shape[2]):
                    encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j, k])
                    encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                    encode_item_list.append(encode_item)
                # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
                encode_tensor_list.append(encode_item_list)
            encode_tensor_tensor_list.append(encode_tensor_list)
        return torch.tensor(encode_tensor_tensor_list, dtype=torch.long)
    else:
        raise NotImplementedError("len(shape)!=2 or len(shape)!=3 Not Implement!")


def train(db: database, db_cfg: promptConfig):
    setup_seed(db_cfg.seed)
    model, tokenizer = getLLM(LLM_PATH)
    model.eval()
    mask_hidden_size = model.config.hidden_size
    data, label, labelmap = db.discrete(db_cfg.slice_num)
    device = db_cfg.device
    for random_state in db_cfg.random_state:
        logger = logConfig(JOURNALPATH, TASKFORMAT, db_cfg.add_terminal, db_cfg.name, random_state)

        traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_train_valid_test(data, label,
                                                                                                   randomstate=random_state)
        prompt_num = 1

        prompt_model = prompt(db_cfg.prompt_init_shape, db_cfg.prompt_gru_hidden_state, db_cfg.prompt_gru_layer,
                              db_cfg.prompt_seq_length, device)
        cut_model = mlp(db_cfg.prompt_seq_length + prompt_num * 2, db_cfg.cut_hidden_features, mask_hidden_size,
                        db_cfg.dropout)
        mask_model = mlp(mask_hidden_size, db_cfg.mask_hidden_features, len(labelmap), db_cfg.dropout)

        model.to(device)
        prompt_model.to(device)
        cut_model.to(device)
        mask_model.to(device)

        train_dataset = dataset(traindata, trainlabel)
        valid_dataset = dataset(validdata, validlabel)

        train_dataset.data = encode(train_dataset.data, tokenizer)
        valid_dataset.data = encode(valid_dataset.data, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=db_cfg.batch_size, num_workers=1, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=db_cfg.batch_size, num_workers=1)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(
            list(prompt_model.parameters()) + list(cut_model.parameters()) + list(mask_model.parameters()),
            lr=db_cfg.lr, weight_decay=db_cfg.weight_decay)

        best_valid_loss = float('inf')
        for epoch in range(1, db_cfg.num_epochs + 1):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            valid_epoch_loss = 0.0
            valid_correct_predictions = 0
            valid_total_samples = 0
            correct_predictions, epoch_loss, total_samples = batchProcess(correct_predictions, criterion, cut_model,
                                                                          device, epoch_loss, mask_model, model,
                                                                          prompt_model, tokenizer, total_samples,
                                                                          train_loader, optimizer, train=True)
            prompt_model.eval()
            mask_model.eval()
            cut_model.eval()
            valid_correct_predictions, valid_epoch_loss, valid_total_samples = batchProcess(valid_correct_predictions,
                                                                                            criterion, cut_model,
                                                                                            device, valid_epoch_loss,
                                                                                            mask_model, model,
                                                                                            prompt_model, tokenizer,
                                                                                            valid_total_samples,
                                                                                            valid_loader, optimizer,
                                                                                            train=False)
            prompt_model.train()
            mask_model.train()
            cut_model.train()

            # 计算epoch平均损失和准确率
            epoch_loss /= len(train_loader)
            accuracy = correct_predictions / total_samples
            valid_epoch_loss /= len(valid_loader)
            valid_accuracy = valid_correct_predictions / valid_total_samples

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                # 保存模型
                torch.save(prompt_model, f'checkpoint/promptModel/{db_cfg.name}_{random_state}_promptModel.pt')
                torch.save(mask_model, f'checkpoint/maskModel/{db_cfg.name}_{random_state}_maskModel.pt')
                logger.info(f'Epoch [{epoch}/{db_cfg.num_epochs}],Valid Loss: {valid_epoch_loss:.4f}!')
                prompt_model.eval()

                beforePrompt, afterPrompt, mask = prompt_model.returnPrompt(tokenizer)

                logger.info(f'beforePrompt:{beforePrompt},afterPrompt:{afterPrompt},mask:{mask}')
                prompt_model.train()

            logger.info(
                f'Epoch [{epoch}/{db_cfg.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')


def batchProcess(correct_predictions, criterion, cut_model, device, epoch_loss, mask_model, model, prompt_model,
                 tokenizer, total_samples, train_loader, optimizer, train: bool = False):
    for i, batch in enumerate(train_loader):
        data = batch['data']
        label = batch['label']

        data = data.to(device)
        label = label.to(device)

        shape = data.shape

        prompt_data, mask_slice = prompt_model()
        cut_input = torch.cat((prompt_data.view(-1), mask_slice.view(-1)), dim=0)
        prompt_data = prompt_reflect(prompt_data, tokenizer, device)
        prompt_length = prompt_data.shape[-1]
        mask, slice = mask_slice_reflect(mask_slice, prompt_length, device)
        prompt_data, mask_pos = concate_prompt_data(prompt_data, data, mask, slice, tokenizer, device)
        attention_mask = torch.ones_like(prompt_data, device=device)
        cut_output = cut_model(cut_input)

        if (len(shape) == 2):
            with torch.no_grad():
                out = model(prompt_data, attention_mask)
            mask_data = out.last_hidden_state[:, mask_pos, :]
            mask_data=mask_data.squeeze()
            output = mask_model(mask_data)
        elif (len(shape == 3)):
            mask_data_list = []
            for i in range(shape[0]):
                promptitem = prompt_data[i, :]
                attention_mask_item = attention_mask[i, :]

                out = model(promptitem, attention_mask_item)
                mask_item_list = []
                for j in range(shape[1]):
                    mask_item = out.last_hidden_state[i, mask_pos[j].item(), :]
                    mask_item_list.append(mask_item)
                mask_data = torch.stack(mask_item_list, dim=0)
                output = mask_model(mask_data)
                mask_data_list.append(output)
            output = torch.stack(mask_data_list, dim=0)
        output = output.view(output.shape[0], output.shape[-1])
        loss = criterion(output, label.long())

        # 计算准确率
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

        if (len(shape) == 2):
            eps = 1e-5
            p_score = 1 - correct_predictions / shape[0]
            alpha = eps + p_score ** 2
            cut_loss = torch.nn.functional.mse_loss(torch.stack([cut_output] * shape[0], dim=0), mask_data)
            loss += alpha * cut_loss
        if (train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 累积epoch损失
        epoch_loss += loss.item()
    return correct_predictions, epoch_loss, total_samples


if __name__ == '__main__':
    # train(ADNI, ADNI_config)
    train(ADNI_fMRI, ADNI_fMRI_config)
