import streamlit as st
import pandas as pd
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression from scratch
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.m, self.n = X.shape

        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear)

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear)
        return np.where(y_pred >= 0.5, 1, 0)

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return sigmoid(linear)

# Streamlit file uploader
st.title("Logistic Regression Without sklearn")
uploaded_file = st.file_uploader("Upload Diabetes.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Dataset Preview:", df.head())

# Outlier capping
def outlier_capping(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: lower if x < lower else upper if x > upper else x)

for col in df.select_dtypes(include=['int', 'float']).columns:
    outlier_capping(df, col)

Y = df["Outcome"]
X = df.drop(columns=["Outcome"])

model = LogisticRegressionScratch(lr=0.01, epochs=6000)
model.fit(X, Y)

df["Y_pred"] = model.predict(X)
df["proba"] = model.predict_proba(X)

TP = np.sum((Y == 1) & (df["Y_pred"] == 1))
TN = np.sum((Y == 0) & (df["Y_pred"] == 0))
FP = np.sum((Y == 0) & (df["Y_pred"] == 1))
FN = np.sum((Y == 1) & (df["Y_pred"] == 0))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

st.subheader("Model Evaluation Metrics")
st.write("Confusion Matrix:", [[TN, FP], [FN, TP]])
st.write("Accuracy:", round(accuracy, 3))
st.write("Precision:", round(precision, 3))
st.write("Recall:", round(recall, 3))
st.write("F1 Score:", round(f1, 3))
