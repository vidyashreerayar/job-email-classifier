# Job Email Classifier

Gmail job emails into JOB_POSITIVE, JOB_REJECTION, JOB_OTHER.
Pipeline: Gmail ingestion, human labels, TF-IDF + Logistic Regression.
Train: python src/train.py. Predict: python src/predict.py "email text".
Result: about 81% cross validated accuracy on 62 human labels (small, near duplicates).
Archived experiments: src/experiments (heuristics, Snorkel, rejected).
