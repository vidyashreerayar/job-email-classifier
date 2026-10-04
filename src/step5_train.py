import joblib, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from step4_prepare import build_prep, load

def get(name):
    x = load(name)
    y = pd.read_csv(f"data/processed/{name}_labels.csv").set_index("email_id").loc[x.email_id, "label"].values
    return x, y

Xtr, ytr = get("train")
Xva, yva = get("val")
models = {
    "Baseline (always Other)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "Naive Bayes": ComplementNB(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = {}
print(f"{'model':26} {'train F1':>8} {'cv F1 (mean +/- std)':>24} {'cv acc':>7}")
for name, clf in models.items():
    pipe = Pipeline([("prep", build_prep()), ("clf", clf)])
    r = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["f1_macro", "accuracy"])
    train_f1 = f1_score(ytr, pipe.fit(Xtr, ytr).predict(Xtr), average="macro")
    scores[name] = r["test_f1_macro"].mean()
    print(f"{name:26} {train_f1:8.2f} {scores[name]:15.2f} +/- {r['test_f1_macro'].std():.2f} {r['test_accuracy'].mean():7.2f}")
best = max(scores, key=scores.get)
print("best on cross validation:", best)
final = Pipeline([("prep", build_prep()), ("clf", models[best])]).fit(Xtr, ytr)
pv = final.predict(Xva)
print("VALIDATION (not test), n =", len(yva))
print(classification_report(yva, pv, zero_division=0))
print(confusion_matrix(yva, pv, labels=["JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"]))
joblib.dump(final, "models/step5_best.joblib")
