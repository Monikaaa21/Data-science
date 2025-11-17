import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_curve, roc_auc_score
)

# Load dataset
df = pd.read_csv('Diabetes.csv')

# Outlier capping function
def outlier_capping(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].apply(
        lambda x: lower if x < lower else upper if x > upper else x
    )

for col in df.select_dtypes(include=['int', 'float']).columns:
    outlier_capping(df, col)

# Features & target
Y = df['Outcome']
X = df.drop(columns=['Outcome'])

# Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X, Y)

# Predictions
df['Y_pred'] = model.predict(X)
df['predict_proba'] = model.predict_proba(X)[:, 1]

# Metrics
cm = confusion_matrix(Y, df['Y_pred'])
acc = accuracy_score(Y, df['Y_pred'])
rec = recall_score(Y, df['Y_pred'])
prec = precision_score(Y, df['Y_pred'])
f1 = f1_score(Y, df['Y_pred'])
auc = roc_auc_score(Y, df['predict_proba'])

# Results
print("\nConfusion Matrix:\n", cm)
print("Accuracy:", round(acc, 3))
print("Recall:", round(rec, 3))
print("Precision:", round(prec, 3))
print("F1 Score:", round(f1, 3))
print("AUC Score:", round(auc, 3))
