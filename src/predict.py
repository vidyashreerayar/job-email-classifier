import sys, joblib
m = joblib.load("models/model.joblib")
text = " ".join(sys.argv[1:])
p = m.predict_proba([text])[0]
print(m.classes_[p.argmax()], round(p.max(), 3))
