from torch.utils.data import Dataset
import torch

def add_category(text):
    output_list=[]
    for t in text:
        if "http" in t:
            output_list.append("[CATEGORY_1] "+str(t))
        else:
            output_list.append("[CATEGORY_0] "+str(t))
    return output_list
 
def add_keyword(text,keyword):
    output_list=[]
    for k,t in zip(keyword,text):
        output_list.append(str(k)+" [SEP] "+str(t))
    return output_list
 
def data_2_text(data):
    data = data.dropna(subset=['keyword', 'text'])
    keyword = data['keyword'].values
    text = data['text'].values
    text = add_keyword(text,keyword)
    tokenizer_text = add_category(text)
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
            'input_ids' : torch.tensor(self.dict['input_ids'][ids], dtype=torch.long),
            'token_type_ids' : torch.tensor(self.dict['token_type_ids'][ids], dtype=torch.long),
            'attention_mask' : torch.tensor(self.dict['attention_mask'][ids], dtype=torch.long),
        }
        else :
            return {
            'input_ids' : torch.tensor(self.dict['input_ids'][ids], dtype=torch.long),
            'token_type_ids' : torch.tensor(self.dict['token_type_ids'][ids], dtype=torch.long),
            'attention_mask' : torch.tensor(self.dict['attention_mask'][ids], dtype=torch.long),
            'labels' : torch.tensor(self.target[ids], dtype=torch.long)
        }