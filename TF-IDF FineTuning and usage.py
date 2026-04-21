import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from Metrics import metrics


data = pd.read_csv("archive/train.csv")

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

X = data["comment_text"]
y = data[LABELS]

df_test = pd.read_csv('archive/test.csv', encoding="UTF-8")
df_labels = pd.read_csv("archive/test_labels.csv")

mask = ~(df_labels[LABELS] == -1).any(axis=1)
text_list = df_test.loc[mask, "comment_text"].fillna("").tolist()
y_true = df_labels.loc[mask, LABELS].values

vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
train_vectorised = vectorizer.fit_transform(X)
test_vectorised = vectorizer.transform(text_list)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(train_vectorised, y)

predictions = model.predict(test_vectorised)

output = pd.DataFrame(predictions, columns=y.columns)
output.to_csv("Outputs/predictions.csv", index=False)

metrics(predictions, y_true)