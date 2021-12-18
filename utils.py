from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

from torch.utils.data import Dataset
import torch
import requests

from urllib.request import urlopen

from bs4 import BeautifulSoup
from lxml import html
from mechanize import Browser
import re

SHORTENED_URL_PATTERN = "(http://(t\.co)/\S*)"


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


def replace_http_tag(input: str, replacement_tag: str = "") -> str:
    result = re.sub(SHORTENED_URL_PATTERN, replacement_tag, input)

    return result


def get_link_page_title(input: str) -> [str]:
    return_array = []

    matches = re.findall(SHORTENED_URL_PATTERN, input)

    for match in matches:

        url = match[0]

        try:
            soup = BeautifulSoup(urlopen(url, timeout=15), features="lxml")
            title = soup.title

            if title is None:
                return_array.append("")
                continue

            return_array.append(title.get_text())
        except HTTPError:
            return_array.append("")
        except UnicodeEncodeError:
            return_array.append("")
        except URLError:
            return_array.append("")
        except IncompleteRead:
            return_array.append("")
        except Exception:
            return_array.append("")

    return return_array


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
