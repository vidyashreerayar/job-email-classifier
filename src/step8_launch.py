import sys, os, re, time, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib, pandas as pd
import step4_prepare  # the custom transformers must be importable for joblib
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

APPLY, ALL = "--apply" in sys.argv, "--all" in sys.argv
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
NAMES = {"JOB_POSITIVE": "JobMail/Positive", "JOB_REJECTION": "JobMail/Rejection", "JOB_OTHER": "JobMail/Other"}

# Why one saved model: the same pipeline (cleaning, features, classifier) that was tested in step 6e runs on live mail.
model = joblib.load("models/step6_final.joblib")

creds = Credentials.from_authorized_user_file("token_modify.json", SCOPES) if os.path.exists("token_modify.json") else None
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        creds = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES).run_local_server(port=0)
    open("token_modify.json", "w").write(creds.to_json())
gm = build("gmail", "v1", credentials=creds).users()

def body_text(p):
    d = p.get("body", {}).get("data")
    t = base64.urlsafe_b64decode(d).decode("utf-8", "ignore") if d else ""
    return t + " ".join(body_text(x) for x in p.get("parts", []))

def get(i):
    # Why retry: Gmail limits requests per minute, so we wait and try again instead of failing.
    for _ in range(8):
        try:
            time.sleep(0.4)
            return gm.messages().get(userId="me", id=i, format="full").execute()
        except HttpError as e:
            if e.resp.status in (403, 429, 500, 503):
                time.sleep(30)
            else:
                raise
    raise RuntimeError("too many retries")

have = {l["name"]: l["id"] for l in gm.labels().list(userId="me").execute()["labels"]}
ours = {have[n] for n in NAMES.values() if n in have}
q = "in:inbox" if ALL else "in:inbox newer_than:14d"
ids, tok = [], None
while True:
    r = gm.messages().list(userId="me", q=q, maxResults=100, pageToken=tok).execute()
    ids += [m["id"] for m in r.get("messages", [])]
    tok = r.get("nextPageToken")
    if not tok:
        break

rows = []
for i in ids:
    msg = get(i)
    if ours & set(msg.get("labelIds", [])):
        continue  # already sorted, skip
    subj = next((h["value"] for h in msg["payload"]["headers"] if h["name"].lower() == "subject"), "")
    rows.append({"email_id": i, "subject": subj, "body": body_text(msg["payload"])})

if not rows:
    print("no new emails to sort")
    sys.exit()
df = pd.DataFrame(rows)
proba = model.predict_proba(df[["subject", "body"]])
df["label"] = model.classes_[proba.argmax(axis=1)]
df["confidence"] = proba.max(axis=1).round(3)
for _, r in df.iterrows():
    print(f"{r.label[4:]:10} {r.confidence:.2f} | {r.subject[:55]}")
print(df.label.value_counts().to_dict(), "| mode:", "APPLY" if APPLY else "dry run")

if APPLY:
    for n in NAMES.values():
        if n not in have:
            have[n] = gm.labels().create(userId="me", body={"name": n}).execute()["id"]
    for _, r in df.iterrows():
        gm.messages().modify(userId="me", id=r.email_id, body={"addLabelIds": [have[NAMES[r.label]]]}).execute()
    # Monitoring: keep a small log (ids and labels only, no email content) so drift can be checked later.
    os.makedirs("logs", exist_ok=True)
    log = df[["email_id", "label", "confidence"]].assign(date=pd.Timestamp.now().date())
    log.to_csv("logs/predictions.csv", mode="a", header=not os.path.exists("logs/predictions.csv"), index=False)
