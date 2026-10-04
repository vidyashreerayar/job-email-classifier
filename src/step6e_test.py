import sys, os, joblib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
import step4_prepare  # needed so joblib can load the custom transformers
from step4_prepare import load

# 6e Evaluate on the test set (done once)
# Why once: every extra look at the test set leaks information into our choices and makes the score too good.
FLAG = "models/test_run.flag"
if os.path.exists(FLAG) and "--again" not in sys.argv:
    sys.exit("The test set was already used once. Run with --again only to redraw the same result.")

LABELS = ["JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"]
SHORT = ["Other", "Positive", "Rejection"]
model = joblib.load("models/step6_final.joblib")
X = load("test")
y = pd.read_csv("data/processed/test_labels.csv").set_index("email_id").loc[X.email_id, "label"].values
pred = model.predict(X)

acc = accuracy_score(y, pred)
f1 = f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
base = accuracy_score(y, np.full(len(y), "JOB_OTHER"))
# Why bootstrap: with only a few Positive emails, one number is shaky. Resampling the test set 1000 times shows how shaky.
rng = np.random.default_rng(42)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    boot.append(f1_score(y[i], pred[i], average="macro", labels=LABELS, zero_division=0))
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"TEST set, n = {len(y)} (used once)")
print(f"accuracy {acc:.2f} | accuracy of always guessing Other {base:.2f}")
print(f"macro F1 {f1:.2f} | 95% range {lo:.2f} to {hi:.2f}")
print(classification_report(y, pred, labels=LABELS, zero_division=0))
cm = confusion_matrix(y, pred, labels=LABELS)
print(cm)
print("NOTE: labels come from keyword rules, so this measures agreement with the rules, not with a human.")

# Chart: the model decision (numbers typed from the step 6 run) and the test result
names = ["Logistic\nRegression", "Linear\nSVM", "Random\nForest", "Voting\nensemble"]
cv = [0.71, 0.70, 0.71, 0.70]
sd = [0.09, 0.10, 0.08, 0.08]
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].bar(names, cv, yerr=sd, capsize=4, color=["tab:green", "tab:gray", "tab:gray", "tab:gray"])
ax[0, 0].axhline(0.31, color="red", ls="--", label="baseline (always Other) 0.31")
ax[0, 0].set_ylim(0, 1); ax[0, 0].legend(loc="lower right")
ax[0, 0].set_title("Model choice: cross validation macro F1 (tied, so simplest wins)")

ax[0, 1].imshow(cm, cmap="Blues")
ax[0, 1].set_xticks(range(3)); ax[0, 1].set_xticklabels(SHORT)
ax[0, 1].set_yticks(range(3)); ax[0, 1].set_yticklabels(SHORT)
for r in range(3):
    for c in range(3):
        ax[0, 1].text(c, r, cm[r, c], ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
ax[0, 1].set_xlabel("predicted"); ax[0, 1].set_ylabel("true")
ax[0, 1].set_title(f"Test confusion matrix (n = {len(y)})")

p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
x = np.arange(3); w = 0.27
ax[1, 0].bar(x - w, p, w, label="precision"); ax[1, 0].bar(x, r, w, label="recall"); ax[1, 0].bar(x + w, f, w, label="F1")
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(SHORT); ax[1, 0].set_ylim(0, 1.05); ax[1, 0].legend()
ax[1, 0].set_title("Test scores per class")

ax[1, 1].hist(boot, bins=30)
ax[1, 1].axvline(f1, color="black", label=f"test macro F1 {f1:.2f}")
ax[1, 1].axvline(lo, color="red", ls="--", label=f"95% range {lo:.2f} to {hi:.2f}"); ax[1, 1].axvline(hi, color="red", ls="--")
ax[1, 1].legend(); ax[1, 1].set_title("How shaky is the score? (1000 resamples)")

fig.text(0.5, 0.01, "Why Logistic Regression: tied on score within noise, simplest, fast, readable top words, easy to retrain. Labels come from keyword rules.", ha="center")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("docs/step6_model_choice_and_test.png", dpi=120)
open(FLAG, "w").write("done")
