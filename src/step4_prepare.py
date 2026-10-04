import re, html, joblib
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# 4a Data cleaning
# Why SimpleImputer: some emails have an empty subject or body. Dropping them loses data, and "" is the honest value for missing text.
imputer = ColumnTransformer(
    [("fill", SimpleImputer(strategy="constant", fill_value=""), ["subject", "body"])],
    verbose_feature_names_out=False).set_output(transform="pandas")

# 4b Text cleaning (custom transformer: fit returns self, transform does the work, fit_transform comes from TransformerMixin)
# Why a class, not a loose function: it sits inside the Pipeline, so train, val, test and new Gmail emails get identical cleaning.
def clean_text(t):
    t = re.sub(r"<(style|script).*?</\1>", " ", str(t), flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t)
    t = re.sub(r"\b\w*\d\w*\b", " ", t)
    t = re.sub(r"\b[a-zA-Z]{18,}\b", " ", t)
    t = re.sub(r"\bnull\b|[\u200b-\u200f\u00ad\u034f]", " ", t, flags=re.I)
    t = re.sub(r"\b(vidyashree|rayar|frau|herr)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        b = X["body"].astype(str).map(clean_text)   # the body is the main text
        X["text"] = b.where(b.str.len() > 20, X["subject"].astype(str) + " " + b).str.strip()  # subject only as fallback
        return X

# 4c Handling text and categories
# Why TF-IDF: simple, explainable, works with few labels. Word embeddings need far more data than we have.
# Why OneHotEncoder for language: English and German have no order. OrdinalEncoder is used only for the 3 labels.
LABELS = ["JOB_OTHER", "JOB_POSITIVE", "JOB_REJECTION"]
label_encoder = OrdinalEncoder(categories=[LABELS])
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
lang_encoder = OneHotEncoder(handle_unknown="ignore")

# 4d Custom transformer: extra features
# Why: link count and length separate newsletters from personal mails, which words alone cannot show.
class CountFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["links"] = X["body"].astype(str).str.count(r"https?://")
        X["words"] = X["text"].str.split().str.len()
        low = X["text"].str.lower()
        de = low.str.count(r"\b(?:und|nicht|ihre|deine|wir|sie|bewerbung|der|die|das)\b")
        en = low.str.count(r"\b(?:and|the|you|your|we|application|for|with|have)\b")
        X["lang"] = np.where(de > en, "German", "English")
        return X

# 4e Feature scaling
# Why StandardScaler: counts are on very different scales than TF-IDF, and Logistic Regression trains better when they match.
# Why with_mean=False: keeps the sparse text matrix sparse. Min max scaling reacts too strongly to the few huge outliers.
scaler = StandardScaler(with_mean=False)

# 4f Transformation pipeline
# Why Pipeline plus ColumnTransformer: every step is fitted on train only, so nothing leaks from val or test.
features = ColumnTransformer([
    ("text", tfidf, "text"),
    ("num", scaler, ["links", "words"]),
    ("lang", lang_encoder, ["lang"]),
])

def build_prep():
    return Pipeline([("fill", imputer), ("clean", TextCleaner()), ("counts", CountFeatures()), ("features", features)])

def load(name):
    return pd.read_csv(f"data/processed/{name}.csv").astype({"subject": object, "body": object})

if __name__ == "__main__":
    train, val = load("train"), load("val")
    print(TextCleaner().transform(train.head(3).fillna(""))["text"].str[:160].to_string())
    prep = build_prep()
    Xtr = prep.fit_transform(train)
    Xva = prep.transform(val)
    print("train", Xtr.shape, "val", Xva.shape, "(test stays locked until step 9)")
    joblib.dump(prep, "models/prep.joblib")
