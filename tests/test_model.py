import joblib
def test_model_has_three_classes():
    m = joblib.load("models/model.joblib")
    assert set(m.classes_) == {"JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"}
