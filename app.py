import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, 
#     f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
# )
# import specific model loading logic or libraries
# e.g., import joblib or pickle

st.title("Machine Learning Model Deployment - Assignment 2")

# 1. Dataset Upload Option [Source 8]
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    data_set = pd.read_csv(uploaded_file)
    st.write("Preview of Test Data (or Download csv file from top right of the table):", data_set.head())

    # 2. Model Selection Dropdown [Source 4, 8]
    st.header("2. Select Model")
    model_choice = st.selectbox(
        "Choose a classification model:",
        [
            "Logistic Regression", 
            "Decision Tree Classifier", 
            "K-Nearest Neighbor Classifier", 
            "Naive Bayes Classifier", 
            "Random Forest (Ensemble)", 
            "XGBoost (Ensemble)"
        ]
    )

    # placeholder for actual prediction logic
    # In practice, you would load your model from the /model folder [Source 5]
    st.info(f"Running predictions using: {model_choice}")


    # Load Iris dataset as fallback
    if uploaded_file is None:
        iris = load_iris()
        X = iris.data
        y = iris.target
    else:    
        # Last column is target
        target_col = data_set.columns[-1]
        
        X = data_set.drop(columns=[target_col])
        y = data_set[target_col]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    st.write("X_train:", X_train.head())
    st.write("y_train:", y_train.head())
    
    # Linear Regression
    linreg = LinearRegression() # multi_class="multinomial", solver="lbfgs", C=5) 
    
    linreg.fit(X_train, y_train) # train

    y_prob  = linreg.predict(X_test)
    # acc = accuracy_score(y_test, y_prob)
    
    conf_matrix = confusion_matrix(y_test, y_prob) 
    print(conf_matrix)

    
    # 3. Display Evaluation Metrics [Source 4, 5, 8]
    st.header("3. Evaluation Metrics")
    
    # These variables should be replaced with actual calculation results
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", acc)
    col1.metric("AUC Score", "0.00")
    col2.metric("Precision", "0.00")
    col2.metric("Recall", "0.00")
    col3.metric("F1 Score", "0.00")
    col3.metric("MCC Score", "0.00")

    # 4. Confusion Matrix [Source 8]
    st.header("4. Confusion Matrix")
    # Example visualization using Seaborn/Matplotlib [Source 5]
    fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, ax=ax) # cm would be your calculated confusion matrix
    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")
