import html, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

RAW = "data/raw/emails.csv"
LAB = "data/processed/human_labeled_emails.csv"
KEYS = {"p": "JOB_POSITIVE", "r": "JOB_REJECTION", "o": "JOB_OTHER"}
POS = r"interview|assessment|next step|invitation|einladung"

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()

raw = pd.read_csv(RAW)
raw["text"] = raw.subject.fillna("") + " " + raw.body.map(clean)
old = pd.read_csv(LAB)
labels = dict(old.dropna(subset=["human_label"])[["email_id", "human_label"]].values)

def save():
    out = raw[["email_id", "subject", "body"]].copy()
    out["human_label"] = out.email_id.map(labels)
    out.to_csv(LAB, index=False)

while True:
    lab = raw.email_id.isin(labels)
    pool = raw[~lab]
    if pool.empty:
        break
    ys = raw[lab].email_id.map(labels)
    if ys.nunique() >= 2:
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(raw[lab].text, ys)
        unc = 1 - model.predict_proba(pool.text).max(axis=1)
        batch = pool.assign(u=unc).nlargest(7, "u")
    else:
        batch = pool.sample(7, random_state=0)
    rest = pool[~pool.email_id.isin(batch.email_id)]
    if (ys == "JOB_POSITIVE").sum() < 5:
        hits = rest[rest.text.str.contains(POS, case=False, regex=True)]
        extra = hits.head(3)
    else:
        extra = rest.sample(min(3, len(rest)))
    batch = pd.concat([batch, extra])
    print(f"\nLabeled so far: {len(labels)} | {ys.value_counts().to_dict()}")
    for _, row in batch.iterrows():
        print("\n" + "-" * 60)
        print("SUBJECT:", row.subject)
        print("BODY:", clean(row.body)[:300])
        a = ""
        while a not in list(KEYS) + ["s", "q"]:
            a = input("p / r / o / s / q > ").strip().lower()
        if a == "q":
            save()
            raise SystemExit
        if a in KEYS:
            labels[row.email_id] = KEYS[a]
            save()
save()
