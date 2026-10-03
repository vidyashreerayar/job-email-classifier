import re, html, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()

d = pd.read_csv("data/processed/human_labeled_emails.csv")
d["text"] = d.subject.fillna("") + " " + d.body.map(clean)
lab, pool = d[d.human_label.notna()], d[d.human_label.isna()]
print(lab.human_label.value_counts())
mk = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                           LogisticRegression(max_iter=1000, class_weight="balanced"))
k = min(5, lab.human_label.value_counts().min())
if k >= 2:
    pred = cross_val_predict(mk(), lab.text, lab.human_label, cv=StratifiedKFold(k, shuffle=True, random_state=0))
    print(classification_report(lab.human_label, pred, zero_division=0))
    print(confusion_matrix(lab.human_label, pred))
m = mk().fit(lab.text, lab.human_label)
joblib.dump(m, "models/model.joblib")
p = m.predict_proba(pool.text)
out = pool[["email_id", "subject"]].copy()
out["predicted"] = m.classes_[p.argmax(axis=1)]
out["confidence"] = p.max(axis=1).round(3)
out.to_csv("data/processed/model_predicted_emails.csv", index=False)
print(out.predicted.value_counts())
