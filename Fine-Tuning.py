import pandas as pd
import numpy as np
import torch
import transformers
from transformers.pipelines.base import Dataset

print("1")

df_train = pd.read_csv('archive/train.csv', encoding="utf-8")
df_test = pd.read_csv('archive/test.csv', encoding="utf-8")

print("2")

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("3")

train_comments = df_train['comment_text'].tolist()

print("4")

test_comments = df_test['comment_text'].tolist()

print("5")

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print("6")

train_tokenized = tokenizer(train_comments, padding="max_length", truncation=True, max_length=128)
test_tokenized = tokenizer(test_comments, padding="max_length", truncation=True, max_length=128)

print("7")

class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

print("8")

train_dataset = TextClassifierDataset(train_tokenized, labels)
test_dataset = TextClassifierDataset(test_tokenized, labels)

print("9")

model = transformers.AutoModel.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=6)

print("10")

training_arguments = transformers.TrainingArguments(ouput_dir=".", evaluation_strategy="epoch", per_device_train_batch_size=2, per_device_eval_batch_size=2)

print("11")

trainer = transformers.Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=test_dataset)

print("12")

trainer.train()