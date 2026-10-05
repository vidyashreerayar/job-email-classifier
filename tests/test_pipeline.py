import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pandas as pd, pytest
from step4_prepare import clean_text
from step2b_labels import rule

def test_clean_text_removes_junk():
    t = clean_text("<p>Hello Vidyashree https://x.com/a1 code ABC123 <b>apply</b></p>")
    assert "http" not in t and "Vidyashree" not in t and "ABC123" not in t
    assert "Hello" in t and "apply" in t

def test_rules_use_body_phrases():
    assert rule("Unfortunately we regret to inform you that we will not be moving forward.") == "JOB_REJECTION"
    assert rule("We would like to invite you to an interview next week.") == "JOB_POSITIVE"
    assert rule("Thank you for applying. We received your application.") == "JOB_OTHER"
    assert rule("Great job offer for you. Unsubscribe here.") == "JOB_OTHER"

def test_splits_are_disjoint_and_complete():
    if not os.path.exists("data/processed/test.csv"):
        pytest.skip("data not available")
    parts = [pd.read_csv(f"data/processed/{n}.csv").email_id for n in ("train", "val", "test")]
    assert sum(map(len, parts)) == len(set().union(*map(set, parts)))
    assert set().union(*map(set, parts)) == set(pd.read_csv("data/raw/emails.csv").email_id)

def test_model_predicts_valid_labels():
    if not os.path.exists("models/step6_final.joblib"):
        pytest.skip("model not available")
    import joblib
    m = joblib.load("models/step6_final.joblib")
    df = pd.DataFrame({"subject": ["Update", "Invitation", "Hello"], "body": [
        "Unfortunately we regret to inform you that we will not be moving forward with your application.",
        "We would like to invite you to an online assessment. Please use the link.",
        "Here are new jobs that match your profile. Unsubscribe anytime."]})
    p = m.predict_proba(df)
    assert set(m.classes_) == {"JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"}
    assert abs(p.sum(axis=1) - 1).max() < 1e-6
    assert m.predict(df)[0] == "JOB_REJECTION"
