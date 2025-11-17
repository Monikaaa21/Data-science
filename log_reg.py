# ---- Code Cell ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Code Cell ----
df=pd.read_csv('Diabetes.csv')
df

# ---- Code Cell ----
df.info()

# ---- Code Cell ----
df.describe()

# ---- Code Cell ----
df.isnull().sum()

# ---- Code Cell ----
df.isnull().sum().sum()

# ---- Code Cell ----
df.duplicated().sum()

# ---- Code Cell ----
df.boxplot(vert=False)

# ---- Code Cell ----
def outlier_capping(df,column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    Lower_Extreme=Q1-1.5*IQR
    Upper_Extreme=Q3+1.5*IQR
    df[column]=df[column].apply(lambda x: Lower_Extreme if x<Lower_Extreme else Upper_Extreme if x>Upper_Extreme else x)
for col in df.select_dtypes(['int','float']).columns: 
    outlier_capping(df,col)

# ---- Code Cell ----
df.boxplot(vert=False)
plt.show()

# ---- Code Cell ----
df.corr()

# ---- Code Cell ----
Y=df['Outcome']
X=df.drop(columns=['Outcome'])
X.head()

# ---- Code Cell ----
Y.head()

# ---- Code Cell ----
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,Y)

# ---- Code Cell ----
df['Y_pred']=model.predict(X)
df['Y_pred']

# ---- Code Cell ----
model.coef_

# ---- Code Cell ----
model.intercept_


# ---- Code Cell ----
df["predict_proba"] = model.predict_proba(X)[:,1:]

# ---- Code Cell ----
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,df["Y_pred"]) 
print("confusion matrix",'\n',cm)
score = accuracy_score(Y,df["Y_pred"])
print("Accuracy score: ", np.round(score,3))

# ---- Code Cell ----

from sklearn.metrics import recall_score,precision_score,f1_score
r_score = recall_score(Y,df["Y_pred"])
print("Sensitivity score: ", np.round(r_score,3))
sp_score = recall_score(df["Y_pred"],Y)
print("specificity score: ", np.round(sp_score,3))
p_score = precision_score(Y,df["Y_pred"])
print("Precision score: ", np.round(p_score,3))
f1_score = f1_score(Y,df["Y_pred"])
print("F1 score: ", np.round(f1_score,3))

# ---- Code Cell ----
from sklearn.metrics import roc_curve,roc_auc_score
tpr,fpr,dummy = roc_curve(Y,df["predict_proba"])

# ---- Code Cell ----
import matplotlib.pyplot as plt
plt.scatter(tpr,fpr)
plt.plot(tpr,fpr,color='red')
plt.xlabel("True positive Rate")
plt.ylabel("False positive Rate")
plt.show()

# ---- Code Cell ----
auc_score = roc_auc_score(Y,df["predict_proba"])
print("AUC score: ", np.round(auc_score,3))

# ---- Code Cell ----
model.coef_

# ---- Code Cell ----
model.intercept_

# ---- Code Cell ----


