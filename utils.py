from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
import requests
from bs4 import BeautifulSoup
import re

from torch.utils.data import Dataset
import torch



SHORTENED_URL_PATTERN = r"(https?://t\.co/[a-zA-Z0-9]*)"
PHOTO_URL_PATTERN = r"https?://twitter\.com/[^/]*/status/[0-9]*/photo/[0-9]*"
VIDEO_URL_PATTERN = r"https?://twitter\.com/[^/]*/status/[0-9]*/video/[0-9]*"

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

def data_2_text(data, titles):
    keyword = data['keyword'].values
    text = data['text'].values
    titles = titles['page_title'].values

    text = add_keyword(text, keyword)
    text = add_category(text)
    text = add_url_title(text, titles)
    text = list(map(replace_http_tag, text))
    return text

def data_2_target(data):
    return data['target'].values

def add_url_title(texts, titles):
    output_list=[]
    for text, title in zip(texts, titles):
        output_list.append(str(text)+" [SEP] "+str(title.replace("|", " ")))
    return output_list

def replace_http_tag(text: str, replacement_tag: str = "") -> str:
    return re.sub(SHORTENED_URL_PATTERN, replacement_tag, text)

def get_link_page_title(text: str) -> [str]:
    title_array = []

    matches = re.findall(SHORTENED_URL_PATTERN, text)

    for url in matches:
        try:
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=(4.0, 7.5))
            print(res.url)

            if re.match(PHOTO_URL_PATTERN, res.url):
                title_array.append("PHOTO")
            elif re.match(VIDEO_URL_PATTERN, res.url):
                title_array.append("VIDEO")
            else:
                title = BeautifulSoup(res.text, "lxml").title
                title_array.append(title.string.replace("|", "-") if title is not None else "")
        except Exception as inst:
            print(f"exception on {url} \n\t: {type(inst)}")
            title_array.append("")
    return title_array

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