import pandas as pd
import torch
import transformers
from transformers.pipelines.base import Dataset
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('archive/train.csv', encoding="utf-8")

not_labels = ['id', 'comment_text']

label_columns = [col for col in df_train.columns if col not in not_labels]

df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=42)

df_labels_train = df_train[label_columns]
labels_list_train = df_labels_train.values.tolist()
labels_list_train = torch.tensor(labels_list_train).float()

df_labels_eval = df_eval[label_columns]
labels_list_eval = df_labels_eval.values.tolist()
labels_list_eval = torch.tensor(labels_list_eval).float()


train_comments = df_train['comment_text'].tolist()
eval_comments = df_train['comment_text'].tolist()

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_tokenized = tokenizer(train_comments, padding="max_length", truncation=True, max_length=128)
eval_tokenized = tokenizer(eval_comments, padding="max_length", truncation=True, max_length=128)

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
eval_dataset = TextClassifierDataset(eval_tokenized, labels_list_eval)

model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=6)

training_arguments = transformers.TrainingArguments(
output_dir="Dump",
eval_strategy="epoch",
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
learning_rate=2e-5,
num_train_epochs=4,
weight_decay=0.01
)

trainer = transformers.Trainer(model=model, args=training_arguments, train_dataset=train_dataset, eval_dataset=eval_dataset)

trainer.train()

trainer.save_model("Outputs/SavedModel/Model.pth")