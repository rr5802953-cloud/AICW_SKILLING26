import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# -----------------------------
# Load dataset
# -----------------------------
data_path = "data/custom_signs.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Total samples:", len(df))
print("Total columns:", len(df.columns))

# -----------------------------
# Separate features and labels
# -----------------------------
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("\nSigns in dataset:")
print(y.value_counts())

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# Create AI model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining AI model...")
model.fit(X_train, y_train)

# -----------------------------
# Test model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save trained model
# -----------------------------
os.makedirs("models", exist_ok=True)

model_path = "models/sign_language_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
print("Saved at:", model_path)