from torch.utils.data import Dataset
import torch


def data_2_text(data):
    data = data.dropna(subset=['keyword', 'text'])
    keyword = data['keyword'].values
    text = data['text'].values
    tokenizer_text = []
    for k, t in zip(keyword, text):
        tokenizer_text.append(str(k) + " [SEP] " + str(t))
    return tokenizer_text


def data_2_target(data):
    data = data.dropna(subset=['keyword', 'text'])
    return data['target'].values


class BERTDataset(Dataset):
    def __init__(self, text, target, tokenizer):
        self.target = target
        self.dict = tokenizer(text, padding=True, truncation=True)

    def __len__(self):
        return len(self.dict['input_ids'])

    def __getitem__(self, ids):
        if (self.target is None):
            return {
                'input_ids': torch.tensor(self.dict['input_ids'][ids], dtype=torch.long),
                'token_type_ids': torch.tensor(self.dict['token_type_ids'][ids], dtype=torch.long),
                'attention_mask': torch.tensor(self.dict['attention_mask'][ids], dtype=torch.long),
            }
        else:
            return {
                'input_ids': torch.tensor(self.dict['input_ids'][ids], dtype=torch.long),
                'token_type_ids': torch.tensor(self.dict['token_type_ids'][ids], dtype=torch.long),
                'attention_mask': torch.tensor(self.dict['attention_mask'][ids], dtype=torch.long),
                'labels': torch.tensor(self.target[ids], dtype=torch.long)
            }
