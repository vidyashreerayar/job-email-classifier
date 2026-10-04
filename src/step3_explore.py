import re, html
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t)
    t = re.sub(r"\b\w*\d\w*\b", " ", t)
    t = re.sub(r"\b[a-zA-Z]{18,}\b", " ", t)
    t = re.sub(r"\bnull\b|[\u200b-\u200f\u00ad\u034f]", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

d = pd.read_csv("data/processed/train.csv")
raw = d.body.fillna("")
d["links"] = raw.str.count(r"https?://")
d["text"] = raw.map(clean)
d["words"] = d.text.str.split().str.len()
low = d.text.str.lower()
de = low.str.count(r"\b(?:und|nicht|ihre|deine|wir|sie|bewerbung|der|die|das)\b")
en = low.str.count(r"\b(?:and|the|you|your|we|application|for|with|have)\b")
d["lang"] = (de > en).map({True: "German", False: "English"})

stop = set("the and you your for with that this are have from will der die und das ist nicht sie ihre wir ein eine mit zu von auf den dem des bei als thank vielen dank please best regards".split())
words = Counter(w for t in low for w in re.findall(r"[a-zäöüß]{4,}", t) if w not in stop)
top = words.most_common(15)[::-1]
cum = pd.Series([c for _, c in words.most_common(200)]).cumsum() / sum(words.values())

groups = {
    "interview / assessment": r"interview|vorstellungsgespr|assessment|einladung|next step|meet and greet",
    "rejection phrases": r"absage|leider (?:nicht|müssen|haben|können)|andere[nr]? (?:bewerber|kandidaten)|regret to inform|not moving forward|other candidates|not selected",
    "application received": r"thank you for (?:your )?appl|eingegangen|eingangsbestätigung|received your application|vielen dank für (?:ihre|deine) bewerbung",
    "job alert / ads": r"job alert|new jobs?|stellen|webinar|bootcamp|unsubscribe|abmelden",
}
share = {k: int(low.str.contains(v, regex=True).sum()) for k, v in groups.items()}

print("rows:", len(d), "| median words:", int(d.words.median()), "| max words:", int(d.words.max()))
print("German share:", round((d.lang == "German").mean(), 2), "| almost empty bodies:", int((d.words < 5).sum()))
print("bodies with links:", round((d.links > 0).mean(), 2), "| signal phrase counts:", share)

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].hist(d.words.clip(upper=800), bins=30); ax[0, 0].set_title("Histogram: body length in words (cap 800)")
ax[0, 1].barh([w for w, _ in top], [c for _, c in top]); ax[0, 1].set_title("Bar: top 15 body words")
for lang, col in (("English", "tab:blue"), ("German", "tab:orange")):
    s = d[d.lang == lang]; ax[0, 2].scatter(s.words.clip(upper=800), s.links, alpha=0.5, c=col, label=lang)
ax[0, 2].set_title("Scatter: body words vs links"); ax[0, 2].set_xlabel("words"); ax[0, 2].set_ylabel("links"); ax[0, 2].legend()
d.lang.value_counts().plot.bar(ax=ax[1, 0], title="Bar: language (rough)", rot=0)
ax[1, 1].plot(range(1, 201), cum.values); ax[1, 1].set_title("Line: share of all words covered by top N"); ax[1, 1].set_xlabel("top N words")
ax[1, 2].barh(list(share), list(share.values())); ax[1, 2].set_title("Bar: emails containing signal phrases")
plt.tight_layout(); plt.savefig("docs/step4_explore.png", dpi=120)
