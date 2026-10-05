import os, re, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import joblib, pandas as pd
from step4_prepare import TextCleaner, CountFeatures, clean_text, load
from step2b_labels import rule

def batch_stats(df, model):
    # Numbers only, no email text is stored.
    proba = model.predict_proba(df[["subject", "body"]])
    feat = CountFeatures().transform(TextCleaner().transform(df[["subject", "body"]].fillna("")))
    vocab = model.named_steps["prep"].named_steps["features"].named_transformers_["text"].vocabulary_
    def oov(b):
        t = re.findall(r"(?u)\b\w\w+\b", clean_text(b).lower())
        return sum(w not in vocab for w in t) / max(len(t), 1)
    return pd.DataFrame({"email_id": df.email_id.values, "label": model.classes_[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1).round(3), "rule": df.body.map(rule).values,
        "words": feat.words.values, "german": (feat.lang == "German").values, "oov": df.body.map(oov).round(3).values})

if __name__ == "__main__":
    model = joblib.load("models/step6_final.joblib")
    base = batch_stats(load("val"), model)
    bshare = base.label.value_counts(normalize=True)
    print(f"baseline (validation, n={len(base)}): confidence {base.confidence.mean():.2f}, German {base.german.mean():.2f}, unseen words {base.oov.mean():.2f}")
    path = "logs/monitor_log.csv"
    log = pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else pd.DataFrame()
    recent = log[log.date >= pd.Timestamp.now() - pd.Timedelta(days=14)] if len(log) else log
    if len(recent) < 20:
        sys.exit(f"monitor: only {len(recent)} new emails logged in 14 days, need 20 to check")
    agree, conf, de = (recent.label == recent.rule).mean(), recent.confidence.mean(), recent.german.mean()
    unseen, empty, sh = recent.oov.mean(), (recent.words < 5).mean(), recent.label.value_counts(normalize=True)
    rows = [("agreement with keyword rules (alert below 0.80)", agree, agree < 0.80),
            ("mean confidence (alert if 0.10 under baseline)", conf, conf < base.confidence.mean() - 0.10),
            ("German share (alert if 0.15 from baseline)", de, abs(de - base.german.mean()) > 0.15),
            ("unseen words (alert if 0.10 over baseline)", unseen, unseen > base.oov.mean() + 0.10),
            ("empty bodies (alert above 0.05)", empty, empty > 0.05)]
    for c in model.classes_:
        rows.append((f"{c} share (alert if 0.15 from baseline)", sh.get(c, 0), abs(sh.get(c, 0) - bshare.get(c, 0)) > 0.15))
    for name, v, bad in rows:
        print(f"{'ALERT' if bad else 'ok   '} | {name}: {v:.2f}")
    alerts = [n for n, v, bad in rows if bad]
    if alerts:
        open("logs/ALERT.txt", "w").write("\n".join(alerts))
        subprocess.run(["osascript", "-e", 'display notification "Model check failed, see logs/ALERT.txt" with title "JobMail monitor"'])
