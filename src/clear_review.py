import sys, re, time, base64, html
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
APPLY = "--apply" in sys.argv
gm = build("gmail", "v1", credentials=Credentials.from_authorized_user_file("token_modify.json")).users()
ids = {l["name"]: l["id"] for l in gm.labels().list(userId="me").execute()["labels"]}
REJ = re.compile("absage|leider nicht|leider m.ssen|leider haben wir|leider k.nnen|andere[nr]? (bewerber|kandidaten)|nicht weiter ber.cksichtigen|nicht ber.cksichtigen|engere wahl|gegen ihre bewerbung|gegen sie entschieden|entscheidung gegen|vielleicht klappt es|regret to inform|not moving forward|not be moving forward|will not be proceeding|decided to proceed with|not selected", re.I)
POS = re.compile("interview|vorstellungsgespr|assessment|einladung|upload|hochladen|offer letter|job offer|angebot|next step|n.chster schritt|meet and greet|coding test|video", re.I)
def text(p):
    d = p.get("body", {}).get("data")
    t = base64.urlsafe_b64decode(d).decode("utf-8", "ignore") if d else ""
    return t + " ".join(text(x) for x in p.get("parts", []))
def get(i):
    for _ in range(8):
        try:
            time.sleep(0.4)
            return gm.messages().get(userId="me", id=i, format="full").execute()
        except Exception:
            time.sleep(30)
    raise RuntimeError("too many retries")
rev = ids["JobMail/Review"]
msgs = gm.messages().list(userId="me", labelIds=[rev], maxResults=200).execute().get("messages", [])
count = {"Positive": 0, "Rejection": 0, "Other": 0}
for m in msgs:
    msg = get(m["id"])
    subj = next((h["value"] for h in msg["payload"]["headers"] if h["name"].lower() == "subject"), "")
    body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text(msg["payload"]))))
    hit = REJ.search(body)
    dest = "Rejection" if hit else None
    if not hit:
        hit = None if re.search("share request|events happening|passw", subj, re.I) else POS.search(subj)
        dest = "Positive" if hit else "Other"
    count[dest] += 1
    print(f"{dest:9} | {(hit.group(0) if hit else '-'):18} | {subj[:55]}")
    if APPLY:
        gm.messages().modify(userId="me", id=m["id"], body={"addLabelIds": [ids["JobMail/" + dest]], "removeLabelIds": [rev]}).execute()
print(count)
if APPLY:
    gm.labels().delete(userId="me", id=rev).execute()
    print("Review label deleted")
