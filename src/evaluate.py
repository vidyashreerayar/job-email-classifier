import re, html, joblib
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()

d = pd.read_csv("data/processed/human_labeled_emails.csv").dropna(subset=["human_label"])
d["text"] = d.subject.fillna("") + " " + d.body.map(clean)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
d.human_label.value_counts().plot.bar(ax=ax[0], title="Class counts")
d.text.str.len().plot.hist(ax=ax[1], bins=20, title="Text length")
plt.tight_layout(); plt.savefig("docs/exploration.png")

Xtr, Xte, ytr, yte = train_test_split(d.text, d.human_label, test_size=0.2, stratify=d.human_label, random_state=42)
print("train", len(Xtr), "test", len(Xte))
pipe = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)), ("clf", LogisticRegression(max_iter=1000))])
grid = [
    {"clf": [LogisticRegression(max_iter=1000, class_weight="balanced")], "clf__C": [0.1, 1, 10, 100],
     "tfidf__ngram_range": [(1, 1), (1, 2)]},
    {"clf": [ComplementNB()], "clf__alpha": [0.1, 0.5, 1.0], "tfidf__ngram_range": [(1, 1), (1, 2)]},
]
gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(4, shuffle=True, random_state=0), scoring="f1_macro")
gs.fit(Xtr, ytr)
print("best:", gs.best_params_, "cv f1:", round(gs.best_score_, 3))
pred = gs.predict(Xte)
print(classification_report(yte, pred, zero_division=0))
print(confusion_matrix(yte, pred))
final = gs.best_estimator_.fit(d.text, d.human_label)
joblib.dump(final, "models/model.joblib")
