import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.datasets import load_iris 

# Example: load your dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Train logistic regression
logreg = LogisticRegression(max_iter=1000)  # increase max_iter to be safe
logreg.fit(X_train, y_train)

# Save model to pickle
with open("models/logreg_model.pkl", "wb") as f:
    pickle.dump(logreg, f)
