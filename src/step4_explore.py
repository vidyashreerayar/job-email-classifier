import re, html
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()

d = pd.read_csv("data/processed/train.csv")
d["text"] = d.subject.fillna("") + " " + d.body.map(clean)
d["len"] = d.text.str.len()
de = d.text.str.contains(r"\b(bewerbung|ihre|deine|vielen dank|wir haben)\b", case=False)
print("rows:", len(d), "| median length:", int(d.len.median()), "| max:", d.len.max())
print("share German:", round(de.mean(), 2), "| duplicate subjects:", int(d.subject.duplicated().sum()))
stop = set("the and you your for with that this are have from will der die und das ist nicht sie ihre wir ein eine mit zu von auf den dem des bei als f m d w".split())
words = Counter(w for t in d.text.str.lower() for w in re.findall(r"[a-zäöüß]{4,}", t) if w not in stop)
print(words.most_common(25))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(d.len.clip(upper=6000), bins=30); ax[0].set_title("Text length (capped at 6000)")
pd.Series({"German": de.sum(), "Other language": (~de).sum()}).plot.bar(ax=ax[1], title="Language (rough)")
plt.tight_layout(); plt.savefig("docs/step4_explore.png")
