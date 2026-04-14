import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("archive/train.csv")

X = data["comment_text"]
y = data[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

output = pd.DataFrame(predictions, columns=y.columns)
output.to_csv("Outputs/predictions.csv", index=False)