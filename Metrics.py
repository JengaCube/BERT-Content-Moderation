import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def metrics(preds, y_true):
    y_pred = pd.DataFrame(preds, columns=LABELS)

    print(classification_report(y_true, y_pred.values, target_names=LABELS, zero_division=0))

    y_pred.to_csv("bert_predictions.csv", index=False)

    print("\nFalse Positive Rate (per label):")

    fpr_results = {}

    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred.values[:, i]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_results[label] = fpr

        print(f"{label}: {fpr:.4f}")