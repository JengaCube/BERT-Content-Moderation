import torch
import transformers
import pandas as pd

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = transformers.AutoModelForSequenceClassification.from_pretrained("SavedModel/Model.pth")

df_test = pd.read_csv('archive/test.csv', encoding="UTF-8")

text_list = df_test["comment_text"].tolist()

headers = ["comment", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df = pd.DataFrame(columns=headers)

model.eval()
for text in text_list:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.tolist()
    flattened = [item for sublist in probabilities for item in sublist]
    flattened.insert(0, text)
    new_row = pd.DataFrame([flattened], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)



print(df)
df.to_csv('Outputs/output.csv', index=False)