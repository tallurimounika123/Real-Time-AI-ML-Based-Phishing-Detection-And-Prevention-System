import pandas as pd
import numpy as np
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------
# Feature extraction
# ---------------------------
def extract(url):
    url = str(url).lower()
    return [
        len(url),
        url.count('.'),
        int('@' in url),
        int('-' in url),
        int('https' in url),
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        int('login' in url),
        int('verify' in url),
        int('secure' in url)
    ]


# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("urls_dataset.csv")

# Remove rows where url or label is missing
df = df.dropna(subset=["url", "label"])

# Clean label column
df["label"] = df["label"].astype(str).str.strip().str.lower()

print("Unique labels BEFORE conversion:")
print(df["label"].unique())

print("\nLabel value counts:")
print(df["label"].value_counts())

# Convert text labels to numeric automatically
# (works for 0/1 OR phishing/legit)
if df["label"].nunique() == 2:
    df["label"] = pd.factorize(df["label"])[0]

# Convert to numeric (if already 0/1 it stays same)
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# Remove any remaining NaN labels
df = df.dropna(subset=["label"])

# Final safety check
if df["label"].isna().sum() > 0:
    print("❌ Still contains NaN labels!")
    exit()

print("✅ Dataset cleaned successfully")
print("Remaining rows:", len(df))


# ---------------------------
# Prepare features
# ---------------------------
X = np.array([extract(u) for u in df["url"]])
y = df["label"].astype(int).values


# ---------------------------
# Split dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------
# Train model
# ---------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully")