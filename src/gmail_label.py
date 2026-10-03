import os, sys, base64, re, html, joblib
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN, APPLY, MIN_CONF = "token_modify.json", "--apply" in sys.argv, 0.5
NAMES = {"JOB_POSITIVE": "JobMail/Positive", "JOB_REJECTION": "JobMail/Rejection", "JOB_OTHER": "JobMail/Other"}

def clean(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()

def body(part):
    data = part.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
    return " ".join(body(p) for p in part.get("parts", []))

creds = Credentials.from_authorized_user_file(TOKEN, SCOPES) if os.path.exists(TOKEN) else None
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        creds = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES).run_local_server(port=0)
    open(TOKEN, "w").write(creds.to_json())
gm = build("gmail", "v1", credentials=creds).users()

have = {l["name"]: l["id"] for l in gm.labels().list(userId="me").execute()["labels"]}
if APPLY:
    for n in NAMES.values():
        if n not in have:
            have[n] = gm.labels().create(userId="me", body={"name": n}).execute()["id"]
ours = {have[n] for n in NAMES.values() if n in have}

model = joblib.load("models/model.joblib")
ids = gm.messages().list(userId="me", q="in:inbox newer_than:7d", maxResults=20).execute().get("messages", [])
for m in ids:
    msg = gm.messages().get(userId="me", id=m["id"], format="full").execute()
    if ours & set(msg.get("labelIds", [])):
        continue
    subj = next((h["value"] for h in msg["payload"]["headers"] if h["name"].lower() == "subject"), "")
    p = model.predict_proba([subj + " " + clean(body(msg["payload"]))])[0]
    cls, conf = model.classes_[p.argmax()], p.max()
    ok = conf >= MIN_CONF
    print(f"{cls:14} {conf:.2f} {'LABEL' if ok else 'skip '} | {subj[:60]}")
    if APPLY and ok:
        gm.messages().modify(userId="me", id=m["id"], body={"addLabelIds": [have[NAMES[cls]]]}).execute()
