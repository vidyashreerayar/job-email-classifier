import joblib, pandas as pd
from scipy import stats
import step4_prepare
from step4_prepare import load
model = joblib.load("models/step6_final.joblib")
X = load("test")
y = pd.read_csv("data/processed/test_labels.csv").set_index("email_id").loc[X.email_id, "label"].values
ok = (model.predict(X) == y).astype(float)  # predict only transforms, it never refits
ci = stats.t.interval(0.95, len(ok) - 1, loc=ok.mean(), scale=stats.sem(ok))
print(f"test accuracy {ok.mean():.2f} | 95% confidence interval {ci[0]:.2f} to {ci[1]:.2f}")
