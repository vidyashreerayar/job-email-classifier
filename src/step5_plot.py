import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Numbers copied from the step 5 run (5 fold cross validation on train, then validation).
models = ["Baseline", "Logistic Reg.", "Linear SVM", "Naive Bayes", "Decision Tree", "Random Forest"]
train_f1 = [0.31, 0.92, 0.99, 0.55, 1.00, 0.99]
cv_f1 = [0.31, 0.68, 0.69, 0.42, 0.68, 0.71]
cv_std = [0.00, 0.16, 0.15, 0.08, 0.11, 0.13]
cv_acc = [0.84, 0.87, 0.92, 0.86, 0.89, 0.93]
classes = ["Other", "Positive", "Rejection"]
cm = np.array([[62, 0, 0], [3, 1, 0], [4, 0, 3]])
recall = [1.00, 0.25, 0.43]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
x = np.arange(len(models)); w = 0.38
ax[0, 0].bar(x - w/2, train_f1, w, label="train F1")
ax[0, 0].bar(x + w/2, cv_f1, w, yerr=cv_std, capsize=3, label="cross validation F1")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(models, rotation=30, ha="right")
ax[0, 0].set_title("Overfitting check: train vs cross validation (macro F1)"); ax[0, 0].legend()

ax[0, 1].bar(models, cv_acc, color="tab:gray")
ax[0, 1].axhline(0.84, color="red", ls="--", label="always guess Other")
ax[0, 1].set_ylim(0.7, 1.0); ax[0, 1].tick_params(axis="x", rotation=30)
ax[0, 1].set_title("Accuracy looks high for every model"); ax[0, 1].legend()

ax[1, 0].imshow(cm, cmap="Blues")
ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(classes)
ax[1, 0].set_yticks(range(3)); ax[1, 0].set_yticklabels(classes)
for i in range(3):
    for j in range(3):
        ax[1, 0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 30 else "black")
ax[1, 0].set_xlabel("predicted"); ax[1, 0].set_ylabel("true")
ax[1, 0].set_title("Validation confusion matrix (Random Forest, 73 emails)")

ax[1, 1].bar(classes, recall, color=["tab:green", "tab:orange", "tab:red"])
ax[1, 1].set_ylim(0, 1.05); ax[1, 1].set_title("Validation recall per class")
for i, r in enumerate(recall):
    ax[1, 1].text(i, r + 0.02, f"{r:.2f}", ha="center")
plt.tight_layout(); plt.savefig("docs/step5_models.png", dpi=120)
