import torch
import transformers
import pandas as pd
from Metrics import metrics

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = transformers.AutoModelForSequenceClassification.from_pretrained("SavedModel/Model.pth")

df_test = pd.read_csv('archive-lite/test.csv', encoding="UTF-8")
df_labels = pd.read_csv("archive-lite/test_labels.csv")

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

mask = ~(df_labels[LABELS] == -1).any(axis=1)
text_list = df_test.loc[mask, "comment_text"].fillna("").tolist()
y_true = df_labels.loc[mask, LABELS].values

headers = ["comment", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df = pd.DataFrame(columns=headers)

model.eval()

for x in range(0, 18):
    preds = []
    rows = []
    threshold = 0.1 + (0.05 * x)
    print(threshold)
    for text in text_list:

        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]


        binary_preds = (probs >= threshold).astype(int)
        preds.append(binary_preds)

        row = [text] + probs.tolist()
        rows.append(row)


    df = pd.DataFrame(rows, columns=["comment_text"] + LABELS)

    df.to_csv('Outputs/output.csv', index=False)

    metrics(preds, y_true)