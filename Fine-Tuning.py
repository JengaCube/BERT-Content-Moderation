import pandas as pd
import numpy as np
import torch
import transformers

df_train = pd.read_csv('Dataset/train.csv', encoding="utf-8")
df_test = pd.read_csv('Dataset/test.csv', encoding="utf-8")

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_comments = df_train['comment_text'].tolist()

test_comments = df_test['comment_text'].tolist()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_tokenized = tokenizer(train_comments, padding="max_length", truncation=True, max_length=1024)
test_tokenized = tokenizer(test_comments, padding="max_length", truncation=True, max_length=1024)

class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idk]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextClassifierDataset(train_tokenized, labels)
test_dataset = TextClassifierDataset(test_tokenized, labels)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=6)
