import os, sys, base64, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import step4_prepare  # the custom transformers must be importable for joblib
from step4_prepare import load
from step2b_labels import rule
from step8_backup import backup

MIN_NEW = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 50
LABELS = ["JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"]

# Step 1: collect fresh data. An email is new if it is in no split file yet.
known = set()
for n in ("train", "val", "test"):
    known |= set(pd.read_csv(f"data/processed/{n}.csv").email_id)
creds = Credentials.from_authorized_user_file("token_modify.json", ["https://www.googleapis.com/auth/gmail.modify"])
if not creds.valid:
    creds.refresh(Request())
gm = build("gmail", "v1", credentials=creds).users()
ids, tok = [], None
while True:
    r = gm.messages().list(userId="me", q="in:inbox", maxResults=100, pageToken=tok).execute()
    ids += [m["id"] for m in r.get("messages", [])]
    tok = r.get("nextPageToken")
    if not tok:
        break
new_ids = [i for i in ids if i not in known]
print(len(new_ids), "new emails (need", MIN_NEW, "to retrain)")
if len(new_ids) < MIN_NEW:
    print("not enough new emails yet, nothing changed")
    sys.exit(0)

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
rows = []
for i in new_ids:
    m = get(i)
    subj = next((h["value"] for h in m["payload"]["headers"] if h["name"].lower() == "subject"), "")
    rows.append({"email_id": i, "subject": subj, "body": body_text(m["payload"]), "label": np.nan})
new = pd.DataFrame(rows)
new["rule"] = new["body"].map(rule)  # same keyword rules as before, applied to the body only

# Back up first, then add the new emails to the split files (same 70/10/20 idea as step 2)
print("backup saved:", backup())
cols = ["email_id", "subject", "body", "label"]
new[cols].to_csv("data/raw/emails.csv", mode="a", header=False, index=False)
tmp, test = train_test_split(new, test_size=0.2, random_state=42)
train, val = train_test_split(tmp, test_size=0.125, random_state=42)
for name, part in (("train", train), ("val", val), ("test", test)):
    pd.concat([pd.read_csv(f"data/processed/{name}.csv"), part[cols]], ignore_index=True).to_csv(f"data/processed/{name}.csv", index=False)
    lab = pd.read_csv(f"data/processed/{name}_labels.csv")
    pd.concat([lab, pd.DataFrame({"email_id": part["email_id"], "label": part["rule"]})], ignore_index=True).to_csv(f"data/processed/{name}_labels.csv", index=False)
    print(name, "+", len(part))

# Step 2: retrain with the same tuned settings on train + val
def get_xy(names):
    X = pd.concat([load(n) for n in names], ignore_index=True)
    y = pd.concat([pd.read_csv(f"data/processed/{n}_labels.csv") for n in names]).set_index("email_id").loc[X.email_id, "label"].values
    return X, y
Xtr, ytr = get_xy(["train", "val"])
Xte, yte = get_xy(["test"])
current = joblib.load("models/step6_final.joblib")
candidate = clone(current).fit(Xtr, ytr)

# Step 3: score both models on the updated test set, deploy only if the new one did not get worse
res = {}
for name, m in (("current model", current), ("new model", candidate)):
    p = m.predict(Xte)
    res[name] = f1_score(yte, p, average="macro", labels=LABELS, zero_division=0)
    print(f"{name:14} macro F1 {res[name]:.3f} | accuracy {accuracy_score(yte, p):.3f} | test n = {len(yte)}")
deploy = bool(res["new model"] >= res["current model"])
if deploy:
    joblib.dump(candidate, "models/step6_final.joblib")
    print("DEPLOYED the new model, backup saved:", backup())
else:
    print("KEPT the current model, the new one scored lower. Investigate why before retrying.")
os.makedirs("logs", exist_ok=True)
pd.DataFrame([{"date": pd.Timestamp.now().date(), "new_emails": len(new), "current_f1": round(res["current model"], 3),
               "new_f1": round(res["new model"], 3), "deployed": deploy}]).to_csv(
    "logs/retrain_log.csv", mode="a", header=not os.path.exists("logs/retrain_log.csv"), index=False)
