import pandas as pd
import torch
import transformers
from transformers.pipelines.base import Dataset

df_train = pd.read_csv('archive/train.csv', encoding="utf-8")
df_test = pd.read_csv('archive/test.csv', encoding="utf-8")

not_labels = ['id', 'comment_text']

label_columns = [col for col in df_train.columns if col not in not_labels]

df_labels_train = df_train[label_columns]
labels_list_train = df_labels_train.values.tolist()
labels_list_train = torch.tensor(labels_list_train).float()

df_labels_test = pd.read_csv('archive/test_labels.csv', encoding="utf-8")
labels_list_test = df_labels_test.values.tolist()
labels_list_test = torch.tensor(labels_list_test).float()

train_comments = df_train['comment_text'].tolist()

test_comments = df_test['comment_text'].tolist()

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_tokenized = tokenizer(train_comments, padding="max_length", truncation=True, max_length=128)
test_tokenized = tokenizer(test_comments, padding="max_length", truncation=True, max_length=128)

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

train_dataset = TextClassifierDataset(train_tokenized, labels_list_train)
test_dataset = TextClassifierDataset(test_tokenized, labels_list_test)

model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=6)

training_arguments = transformers.TrainingArguments(output_dir=".", eval_strategy="epoch", per_device_train_batch_size=16, per_device_eval_batch_size=16)

trainer = transformers.Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=test_dataset)


trainer.train()

trainer.save_model("SavedModel/Model.pth")