import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("SavedModel/Model.pth").to(DEVICE)
model.eval()

df_test = pd.read_csv("archive/test.csv")
df_labels = pd.read_csv("archive/test_labels.csv")

mask = ~(df_labels[LABELS] == -1).any(axis=1)
texts = df_test.loc[mask, "comment_text"].fillna("").tolist()
y_true = df_labels.loc[mask, LABELS].values

preds = []
batch_size = 16

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)
        preds.extend((probs >= 0.5).int().cpu().numpy())

y_pred = pd.DataFrame(preds, columns=LABELS)

print(classification_report(y_true, y_pred.values, target_names=LABELS, zero_division=0))

y_pred.to_csv("bert_predictions.csv", index=False)