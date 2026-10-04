import sys, re, time, base64, html
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
APPLY = "--apply" in sys.argv
gm = build("gmail", "v1", credentials=Credentials.from_authorized_user_file("token_modify.json")).users()
ids = {l["name"]: l["id"] for l in gm.labels().list(userId="me").execute()["labels"]}
RX = re.compile("absage|leider nicht|leider m.ssen|leider haben wir|leider k.nnen|andere[nr]? (bewerber|kandidaten)|nicht weiter ber.cksichtigen|nicht ber.cksichtigen|engere wahl|gegen ihre bewerbung|gegen sie entschieden|entscheidung gegen|vielleicht klappt es", re.I)
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
n = 0
for src in ("JobMail/Other", "JobMail/Review"):
    tok = None
    while True:
        r = gm.messages().list(userId="me", labelIds=[ids[src]], maxResults=100, pageToken=tok).execute()
        for m in r.get("messages", []):
            msg = get(m["id"])
            hit = RX.search(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text(msg["payload"])))))
            if hit and not re.search("Automatische Antwort|Neues Profil", next((h["value"] for h in msg["payload"]["headers"] if h["name"].lower() == "subject"), "")):
                n += 1
                subj = next((h["value"] for h in msg["payload"]["headers"] if h["name"].lower() == "subject"), "")
                print(f"{src[8:]:6} | {hit.group(0):28} | {subj[:55]}")
                if APPLY:
                    gm.messages().modify(userId="me", id=m["id"], body={"addLabelIds": [ids["JobMail/Rejection"]], "removeLabelIds": [ids[src]]}).execute()
        tok = r.get("nextPageToken")
        if not tok:
            break
print("matches:", n)
