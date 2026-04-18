import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Metrics import metrics

KEYWORDS = {
    'toxic': [
        'idiot', 'stupid', 'moron', 'loser', 'dumb', 'hate', 'horrible',
        'disgusting', 'pathetic', 'trash', 'garbage', 'worthless', 'awful'
    ],
    'severe_toxic': [
        'kill yourself', 'kys', 'die', 'worthless piece', 'piece of shit',
        'go to hell', 'drop dead'
    ],
    'obscene': [
        'fuck', 'shit', 'bitch', 'ass', 'cunt', 'bastard', 'piss',
        'cock', 'dick', 'pussy', 'whore', 'slut'
    ],
    'threat': [
        'i will kill', 'going to kill', 'will hurt', 'you will die',
        'i will find you', 'watch your back', 'you are dead'
    ],
    'insult': [
        'ugly', 'fat', 'pathetic', 'retard', 'idiot', 'moron', 'dumb',
        'stupid', 'loser', 'freak', 'pig', 'clown', 'joke'
    ],
    'identity_hate': [
        'nigger', 'nigga', 'faggot', 'fag', 'chink', 'kike', 'spic',
        'wetback', 'tranny', 'dyke', 'cracker', 'gook'
    ]
}

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def predict_single(text):
    text_lower = text.lower()
    return [
        int(any(kw in text_lower for kw in KEYWORDS[label]))
        for label in LABELS
    ]


def predict_all(comments):
    predictions = [predict_single(c) for c in comments]
    return np.array(predictions)


def evaluate(y_true, y_pred, model_name="Rule-based baseline"):
    results = {}

    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, i], y_pred[:, i], labels=[0, 1]
        ).ravel()

        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall    = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1        = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate

        results[label] = {
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
            'fpr':       round(fpr, 4),
            'fp_count':  int(fp),
            'tp_count':  int(tp),
        }

    df = pd.DataFrame(results).T
    macro_f1 = round(df['f1'].mean(), 4)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(df.to_string())
    print(f"\n  Macro-F1: {macro_f1}")
    print(f"  Mean FPR: {round(df['fpr'].mean(), 4)}")
    print(f"{'='*55}\n")

    return df, macro_f1

df_test        = pd.read_csv('archive/test.csv', encoding='utf-8')
df_labels = pd.read_csv('archive/test_labels.csv', encoding='utf-8')

mask = ~(df_labels[LABELS] == -1).any(axis=1)
text_list = df_test.loc[mask, "comment_text"].fillna("").tolist()
y_true = df_labels.loc[mask, LABELS].values

preds = predict_all(text_list)

metrics(preds, y_true)