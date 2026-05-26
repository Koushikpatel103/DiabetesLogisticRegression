import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("diabetes.csv")

# Dataset info
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Correlation
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='Blues'
)
plt.show()

# Features and target
X = df.drop(
    "Outcome",
    axis=1
)

y = df["Outcome"]

# Replace invalid zeros
cols = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

for col in cols:
    X[col] = X[col].replace(
        0,
        np.nan
    )

    X[col] = X[col].fillna(
        X[col].median()
    )

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# Model
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

# Prediction
y_pred = model.predict(
    X_test
)

# Evaluation
acc = accuracy_score(
    y_test,
    y_pred
)

print(
    "Accuracy:",
    acc
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save model
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved")