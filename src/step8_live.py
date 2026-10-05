import os, sys, time, base64, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib, pandas as pd
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import step4_prepare  # the custom transformers must be importable for joblib
from step8_monitor import batch_stats

POLL, MODEL = 300, "models/step6_final.joblib"
NAMES = {"JOB_POSITIVE": "JobMail/Positive", "JOB_REJECTION": "JobMail/Rejection", "JOB_OTHER": "JobMail/Other"}
def say(m): print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)

creds = Credentials.from_authorized_user_file("token_modify.json", ["https://www.googleapis.com/auth/gmail.modify"])
gm = build("gmail", "v1", credentials=creds).users()  # the login token refreshes itself when it expires
have = {l["name"]: l["id"] for l in gm.labels().list(userId="me").execute()["labels"]}
for n in NAMES.values():
    if n not in have:
        have[n] = gm.labels().create(userId="me", body={"name": n}).execute()["id"]

def body_text(p):
    d = p.get("body", {}).get("data")
    t = base64.urlsafe_b64decode(d).decode("utf-8", "ignore") if d else ""
    return t + " ".join(body_text(x) for x in p.get("parts", []))
def get(i):
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

def sort_new(model, days):
    # Only emails with none of our three labels come back, so one search call is enough per check.
    q = f"in:inbox newer_than:{days}d -label:JobMail-Positive -label:JobMail-Rejection -label:JobMail-Other"
    ids, tok = [], None
    while True:
        r = gm.messages().list(userId="me", q=q, maxResults=100, pageToken=tok).execute()
        ids += [m["id"] for m in r.get("messages", [])]
        tok = r.get("nextPageToken")
        if not tok:
            break
    if not ids:
        return 0
    rows = []
    for i in ids:
        m = get(i)
        subj = next((h["value"] for h in m["payload"]["headers"] if h["name"].lower() == "subject"), "")
        rows.append({"email_id": i, "subject": subj, "body": body_text(m["payload"])})
    df = pd.DataFrame(rows)
    proba = model.predict_proba(df[["subject", "body"]])
    df["label"] = model.classes_[proba.argmax(axis=1)]
    for _, r in df.iterrows():
        gm.messages().modify(userId="me", id=r.email_id, body={"addLabelIds": [have[NAMES[r.label]]]}).execute()
        say(f"{r.label[4:]:10} | {r.subject[:60]}")
    os.makedirs("logs", exist_ok=True)
    batch_stats(df, model).assign(date=pd.Timestamp.now().date()).to_csv(
        "logs/monitor_log.csv", mode="a", header=not os.path.exists("logs/monitor_log.csv"), index=False)
    return len(df)

model, mtime, first, last_mon = joblib.load(MODEL), os.path.getmtime(MODEL), True, 0
say(f"live service started, checking Gmail every {POLL} seconds")
while True:
    try:
        if os.path.getmtime(MODEL) != mtime:  # a retrain deployed a new model
            model, mtime = joblib.load(MODEL), os.path.getmtime(MODEL)
            say("reloaded the new model")
        n = sort_new(model, 14 if first else 2)  # first check catches up on 14 days, later checks look at 2 days
        first = False
        if n:
            say(f"labeled {n} new emails")
        if time.time() - last_mon > 6 * 3600:  # health check every 6 hours, can show a macOS alert
            subprocess.run([sys.executable, "src/step8_monitor.py"], cwd=ROOT)
            last_mon = time.time()
    except Exception as e:
        say(f"error, will retry: {e!r}")
    time.sleep(POLL)
