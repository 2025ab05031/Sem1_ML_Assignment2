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

def run_logistic_regression(X, y) :

    st.info(f"Running predictions using: {run_logistic_regression}")
    
    import pickle
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    # Load the saved model
    with open("models/logreg_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
        
    # For demo, reuse entire X,y or use a stored X_test,y_test
    y_pred = loaded_model.predict(X)
    
    st.write("Predictions:", y_pred[:10])          # first 10 predictions
    st.write("True labels:", y[:10].to_list())
    
    # If you have X_test, y_test from train_test_split:
    # y_pred = loaded_model.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("Accuracy:", acc)



    
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
    st.write(X_train.dtypes)
    st.write(X_train.head())
    st.write(y_train.dtype)

    st.write("X_train:", X_train.head())
    st.write("y_train:", y_train.head())

    # Drop non-numeric columns
    X_train_num = X_train.select_dtypes(include=["number"])
    X_test_num  = X_test.select_dtypes(include=["number"])

    
    # 2. Model Selection Dropdown [Source 4, 8]
    st.header("2. Select Model")
    model_choice = st.selectbox(
        "Choose a classification model:",
        [
            "Linear Regression",
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
    
    if st.button("Run"):
        if model_choice == "Linear Regression":
            run_linear_regression(X, y)      # <- call def1 here
        elif model_choice == "Logistic Regression":
            run_logistic_regression()
        else:
            st.info(f"Running predictions using: {model_choice}")
    
    # 3. Display Evaluation Metrics [Source 4, 5, 8]
    st.header("3. Evaluation Metrics")
    
    # These variables should be replaced with actual calculation results
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "0.00")
    col1.metric("AUC Score", "0.00")
    col2.metric("Precision", "0.00")
    col2.metric("Recall", "0.00")
    col3.metric("F1 Score", "0.00")
    col3.metric("MCC Score", "0.00")

    # 4. Confusion Matrix [Source 8]
    # st.header("4. Confusion Matrix")
    # # Example visualization using Seaborn/Matplotlib [Source 5]
    # fig, ax = plt.subplots()
    # # sns.heatmap(cm, annot=True, ax=ax) # cm would be your calculated confusion matrix
    # st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")
