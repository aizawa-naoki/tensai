import csv
import os

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import load_metric
from utils import data_2_text, data_2_target, BERTDataset


def __print_result_to_csv(expected: [], actual: []) -> None:
    output_pathname = "output.csv"

    print(len(expected))
    print(len(actual))

    if os.path.exists(output_pathname):
        os.remove(output_pathname)

    with open(output_pathname, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["id", "expected", "actual", "error"])

        for i in range(len(expected)):
            difference = abs(expected[i] - actual[i])
            writer.writerow([i, expected[i], actual[i], difference])

        ### hyper paramerters


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 12

# saved model paths and data path
model_path = "./models/21_12_07_02h/run_5"
tokenizer_path = "./models/21_12_07_02h/tokenizer"
base_path = './dataset/'
###


model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

train_data = pd.read_csv(base_path + 'train.csv')
print(train_data.head())

_, eval_text, _, eval_target = train_test_split(data_2_text(train_data), data_2_target(train_data), random_state=0,
                                                test_size=0.15, stratify=data_2_target(train_data))
eval_dataset = BERTDataset(eval_text, eval_target, tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

print("start evaluation")
metric = load_metric("accuracy")
model.eval()
preds = []
for batch in tqdm(eval_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    preds.extend(predictions.detach().numpy().tolist())
    metric.add_batch(predictions=predictions, references=batch["labels"])
metric.compute()

__print_result_to_csv(data_2_target(train_data), preds)
