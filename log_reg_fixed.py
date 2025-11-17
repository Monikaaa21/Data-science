import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix for Streamlit/servers without display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

df = pd.read_csv('Diabetes.csv')

# Outlier capping function
def outlier_capping(df,column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    Lower_Extreme=Q1-1.5*IQR
    Upper_Extreme=Q3+1.5*IQR
    df[column]=df[column].apply(
        lambda x: Lower_Extreme if x<Lower_Extreme else Upper_Extreme if x>Upper_Extreme else x
    )

for col in df.select_dtypes(['int','float']).columns:
    outlier_capping(df,col)

Y = df['Outcome']
X = df.drop(columns=['Outcome'])

model = LogisticRegression(max_iter=500)
model.fit(X,Y)

df['Y_pred'] = model.predict(X)
df['predict_proba'] = model.predict_proba(X)[:,1]

cm = confusion_matrix(Y, df['Y_pred'])
acc = accuracy_score(Y, df['Y_pred'])
rec = recall_score(Y, df['Y_pred'])
prec = precision_score(Y, df['Y_pred'])
f1 = f1_score(Y, df['Y_pred'])

fpr, tpr, _ = roc_curve(Y, df['predict_proba'])
auc = roc_auc_score(Y, df['predict_proba'])

print("Confusion matrix:\n", cm)
print("Accuracy:", acc)
print("Recall:", rec)
print("Precision:", prec)
print("F1 Score:", f1)
print("AUC:", auc)
