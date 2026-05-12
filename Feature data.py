import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv("datas.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values handled successfully!")

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

print("Duplicates removed successfully!")

label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nCategorical columns encoded successfully!")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeatures Shape:", X.shape)
print("Target Shape:",y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

def evaluate_model(model_name, y_test, predictions):

    print(f"\n========== {model_name} ==========")

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues')

    plt.title(f"{model_name} - Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

evaluate_model(
    "Decision Tree",
    y_test,
    dt_predictions
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probabilities
)

auc_score = roc_auc_score(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 5))

plt.plot(fpr, tpr,
         label=f"AUC = {auc_score:.2f}")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df)

plt.figure(figsize=(8, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)

plt.title("Feature Importance")

plt.show()

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_predictions
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")

print("\n========================================")
print("PREDICTIVE MODELING PROJECT COMPLETED")
print("========================================")
