import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def metrics(preds, y_true):
    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    y_pred = pd.DataFrame(preds, columns=LABELS)

    print(classification_report(y_true, y_pred.values, target_names=LABELS, zero_division=0))

    y_pred.to_csv("bert_predictions.csv", index=False)

    print("\nFalse Positive Rate (per label):")

    fpr_results = {}

    for i, label in enumerate(LABELS):
        cm = confusion_matrix(y_true[:, i],y_pred.values[:, i],labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_results[label] = fpr

        print(f"{label}: {fpr:.4f}")

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')

        ax.set_title(f"Confusion Matrix - {label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        for x in range(2):
            for y in range(2):
                ax.text(y, x, cm[x, y], ha='center', va='center', color='black')

        fig.colorbar(im)

        plt.savefig(f"Outputs/confusion_matrix_{label}.png", dpi=300, bbox_inches='tight')
        plt.close()