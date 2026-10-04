import re, html
import pandas as pd
from step4_prepare import clean_text

# Rule labels from the BODY only. Order matters: rejection wins over positive, positive wins over other.
REJ = re.compile("absage|leider nicht|leider m.ssen|leider haben wir|leider k.nnen|andere[nr]? (bewerber|kandidaten)|nicht weiter ber.cksichtigen|nicht ber.cksichtigen|engere wahl|gegen ihre bewerbung|gegen sie entschieden|entscheidung gegen|vielleicht klappt es|regret to inform|not moving forward|not be moving forward|will not be proceeding|decided to proceed with|not selected|unfortunately.{0,80}(application|position|candidate)", re.I)
POS = re.compile("invite you to|would like to invite|einladung zum (online|vorstellungs|interview|gespr)|schedule (an?|your) interview|complete your (one.way )?video interview|online.assessment|coding challenge|meet and greet|offer letter|job offer|upload your documents|unterlagen hochladen|vorstellungsgespr.ch", re.I)
NEWS = re.compile("unsubscribe|abmelden|webinar|bootcamp|coaching|jobbooster|job alert|jobs, die du nicht verpassen|if (your profile|there is a fit|selected|we see)|falls (sie|ihr profil)|sollten wir", re.I)

def rule(body):
    t = clean_text(html.unescape(str(body)))
    if REJ.search(t): return "JOB_REJECTION"
    if POS.search(t) and not NEWS.search(t): return "JOB_POSITIVE"
    return "JOB_OTHER"

for name in ("train", "val", "test"):
    d = pd.read_csv(f"data/processed/{name}.csv")
    d["label"] = d.body.map(rule)
    d[["email_id", "label"]].to_csv(f"data/processed/{name}_labels.csv", index=False)
    print(name, len(d), d.label.value_counts().to_dict())
    if name == "train":
        for lab, g in d.groupby("label"):
            print(lab, "examples:", list(g.subject.fillna("").str[:45].head(4)))
