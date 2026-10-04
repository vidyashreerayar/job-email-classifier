import warnings, joblib
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from step4_prepare import build_prep, load
warnings.filterwarnings("ignore")

def get(name):
    x = load(name)
    y = pd.read_csv(f"data/processed/{name}_labels.csv").set_index("email_id").loc[x.email_id, "label"].values
    return x, y

# Tune on train + validation together. The test set stays locked until 6e.
Xtr, ytr = get("train"); Xva, yva = get("val")
X = pd.concat([Xtr, Xva], ignore_index=True); y = np.concatenate([ytr, yva])
print("tuning on", len(X), "emails:", pd.Series(y).value_counts().to_dict())
cv = StratifiedKFold(5, shuffle=True, random_state=42)
LABELS = ["JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"]
def pipe(clf): return Pipeline([("prep", build_prep()), ("clf", clf)])
text_grid = {"prep__features__text__ngram_range": [(1, 1), (1, 2)], "prep__features__text__min_df": [1, 2]}

# 6a Grid search
# Why: the grids are small (16 combinations each), so trying every one is cheap and easy to explain.
# Why macro F1: it counts the rare Positive and Rejection classes as much as Other. Accuracy would not.
lr = GridSearchCV(pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                  {**text_grid, "clf__C": [0.1, 1, 10, 100]}, cv=cv, scoring="f1_macro").fit(X, y)
svm = GridSearchCV(pipe(LinearSVC(class_weight="balanced", max_iter=10000)),
                   {**text_grid, "clf__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="f1_macro").fit(X, y)

# 6b Randomized search
# Why: the Random Forest has many settings, and sampling 12 random combinations is cheaper than trying all of them.
rf = RandomizedSearchCV(pipe(RandomForestClassifier(class_weight="balanced", random_state=42)),
                        {**text_grid, "clf__n_estimators": [100, 200, 400], "clf__max_depth": [None, 10, 20],
                         "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2"]},
                        n_iter=12, cv=cv, scoring="f1_macro", random_state=42).fit(X, y)

# 6c Ensemble methods
# Why: three different model types voting together can fix each other's mistakes. It uses the default text settings.
ens = pipe(VotingClassifier([("lr", clone(lr.best_estimator_.named_steps["clf"])),
                             ("svm", clone(svm.best_estimator_.named_steps["clf"])),
                             ("rf", clone(rf.best_estimator_.named_steps["clf"]))], voting="hard"))
es = cross_val_score(ens, X, y, cv=cv, scoring="f1_macro")

cands = {
    "Logistic Regression": (lr.best_score_, lr.cv_results_["std_test_score"][lr.best_index_], lr.best_estimator_),
    "Linear SVM": (svm.best_score_, svm.cv_results_["std_test_score"][svm.best_index_], svm.best_estimator_),
    "Random Forest": (rf.best_score_, rf.cv_results_["std_test_score"][rf.best_index_], rf.best_estimator_),
    "Voting ensemble": (es.mean(), es.std(), ens),
}
print(f"{'model':22} {'cv macro F1':>12}")
for n, (s, sd, _) in cands.items():
    print(f"{n:22} {s:6.2f} +/- {sd:.2f}")
print("LR params:", lr.best_params_); print("SVM params:", svm.best_params_); print("RF params:", rf.best_params_)

# Pick the winner. If several models are within 0.02 of the best, they are tied within noise, so take the simplest one.
top = max(v[0] for v in cands.values())
winner = [n for n in cands if cands[n][0] >= top - 0.02][0]
print("FINAL MODEL:", winner)
final = cands[winner][2]
if winner == "Voting ensemble":
    final.fit(X, y)

# 6d Analyze the best model and its errors
pred = cross_val_predict(final, X, y, cv=cv)
print(classification_report(y, pred, zero_division=0))
print(confusion_matrix(y, pred, labels=LABELS))
wrong = np.where(pred != y)[0]
print("wrong:", len(wrong), "of", len(y), "| first 15 (true -> predicted | subject)")
for i in wrong[:15]:
    print(f"{y[i][4:]:10} -> {pred[i][4:]:10} | {str(X.subject[i])[:60]}")
clf = final.named_steps["clf"]
names = final.named_steps["prep"].named_steps["features"].get_feature_names_out()
if hasattr(clf, "coef_"):
    for i, c in enumerate(clf.classes_):
        print(c, "top words:", [names[j].replace("text__", "") for j in np.argsort(clf.coef_[i])[-10:][::-1]])
elif hasattr(clf, "feature_importances_"):
    print("top features:", [names[j].replace("text__", "") for j in np.argsort(clf.feature_importances_)[-15:][::-1]])
joblib.dump(final, "models/step6_final.joblib")
print("saved models/step6_final.joblib")
