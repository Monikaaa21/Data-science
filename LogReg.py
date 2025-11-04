#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: LOG REG (1).ipynb
Conversion Date: 2025-11-04T09:37:49.802Z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_curve, roc_auc_score
)

# Load data
df = pd.read_csv('Diabetes.csv')

# Basic info
print(df.info())
print(df.describe())
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# Outlier capping
def outlier_capping(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)

for col in df.select_dtypes(include=['int', 'float']).columns:
    outlier_capping(df, col)

# Boxplot visualization
df.boxplot(vert=False)
plt.title("Boxplot after Outlier Capping")
plt.show()

# Prepare data
Y = df['Outcome']
X = df.drop(columns=['Outcome'])

# Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X, Y)

df['Y_pred'] = model.predict(X)
df['predict_proba'] = model.predict_proba(X)[:, 1]

# Evaluation
cm = confusion_matrix(Y, df['Y_pred'])
print("Confusion Matrix:\n", cm)
print("Accuracy:", np.round(accuracy_score(Y, df['Y_pred']), 3))
print("Recall (Sensitivity):", np.round(recall_score(Y, df['Y_pred']), 3))
print("Precision:", np.round(precision_score(Y, df['Y_pred']), 3))
print("F1 Score:", np.round(f1_score(Y, df['Y_pred']), 3))

# ROC Curve
fpr, tpr, _ = roc_curve(Y, df["predict_proba"])
plt.plot(fpr, tpr, color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# AUC Score
auc_score = roc_auc_score(Y, df["predict_proba"])
print("AUC Score:", np.round(auc_score, 3))