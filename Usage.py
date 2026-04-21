import torch
import transformers
import pandas as pd
from Metrics import metrics

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


model = transformers.AutoModelForSequenceClassification.from_pretrained("SavedModel/Model")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

headers = ["comment", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

mode = 3
while mode >= 3:
    print("(1) Metrics eval")
    print("(2) Manual Input")
    mode = int(input(""))
    if mode == 1:
        df_test = pd.read_csv('archive/test.csv', encoding="UTF-8")
        df_labels = pd.read_csv("archive/test_labels.csv")
        mask = ~(df_labels[LABELS] == -1).any(axis=1)
        text_list = df_test.loc[mask, "comment_text"].fillna("").tolist()
        y_true = df_labels.loc[mask, LABELS].values
    elif mode >= 3:
        print("Invalid mode chosen. Please choose option 1 or 2.")

df = pd.DataFrame(columns=headers)

model.eval()

if mode == 1:

    for x in range(0, 18):
        preds = []
        rows = []
        threshold = 0.1 + (0.05 * x)
        print(threshold)
        for text in text_list:

            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

            inputs = inputs.to(device)

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

elif mode == 2:
    preds = []
    rows = []
    threshold = 0.95

    while mode == 2:
        text = str(input("Please enter you model prompt"))
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        binary_preds = (probs >= threshold).astype(int)
        preds.append(binary_preds)

        row = [text] + binary_preds.tolist()
        rows.append(row)



        print(text, binary_preds)
        print("Would you like to exit the script?")
        print("(1) Exit")
        print("(2) New Prompt")
        mode = int(input(":"))

    df = pd.DataFrame(rows, columns=["comment_text"] + LABELS)

print("Model output.csv can be found in the Outputs folder")