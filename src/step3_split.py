import pandas as pd
from sklearn.model_selection import train_test_split

d = pd.read_csv("data/raw/emails.csv")
assert d.email_id.is_unique
tmp, test = train_test_split(d, test_size=0.2, random_state=42)
train, val = train_test_split(tmp, test_size=0.125, random_state=42)
for name, part in (("train", train), ("val", val), ("test", test)):
    part.to_csv(f"data/processed/{name}.csv", index=False)
    print(name, len(part))
