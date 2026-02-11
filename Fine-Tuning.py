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