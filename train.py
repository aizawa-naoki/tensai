# https://www.kaggle.com/landonhd/disaster-tweets-classification-with-bert

import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from datasets import load_metric
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
from utils import data_2_text, data_2_target, BERTDataset

base_path = './dataset/'  # should end with "/"
batch_size = 24
num_epochs = 5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_data, test_data = pd.read_csv(base_path + 'train.csv'), pd.read_csv(base_path + 'test.csv')

train_text, eval_text, train_target, eval_target = train_test_split(data_2_text(train_data), data_2_target(train_data),
                                                                    random_state=0,
                                                                    test_size=0.15, stratify=data_2_target(train_data))

test_text = data_2_text(test_data)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
train_dataset = BERTDataset(train_text, train_target, tokenizer)
eval_dataset = BERTDataset(eval_text, eval_target, tokenizer)
test_dataset = BERTDataset(test_text, None, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.to(device)

num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

save_folder = f"models/{datetime.now().strftime('%y_%m_%d_%Hh')}"
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    if epoch == 0:
        tokenizer.save_pretrained(f"{save_folder}/tokenizer")
    model.save_pretrained(f"{save_folder}/run_{epoch + 1}")
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print(f"eval_acc of {epoch + 1} epoch:{metric.compute()}")

"""
preds = []
model.eval()
for batch in test_dataloader:
    batch = {k : v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    preds.append(torch.argmax(logits, dim=-1))


flat_preds = [item.cpu().numpy() for batch in preds for item in batch]
flat_preds = np.array(flat_preds)
sub = pd.read_csv(base_path + 'sample_submission.csv')
sub['target'] = flat_preds
sub.to_csv('submission.csv', index=False)
"""
