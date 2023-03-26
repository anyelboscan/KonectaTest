# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the data
df = pd.read_excel("credits.xls")

# Select a random sample of 10,000 records
df = df.sample(n=10000, random_state=42)

# Print the first five rows of the dataset
print(df.head())


# Check for missing values
print(df.isnull().sum())

# Check data types of each column
print(df.dtypes)


# Create a new column that represents the ratio of the credit limit to the salary
df["credit_limit_to_salary_ratio"] = df["LIMIT_BAL"] / df["PAY_AMT1"]


# Check the correlation between the new column and the target variable
print(df[["credit_limit_to_salary_ratio", "default.payment.next.month"]].corr())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("default.payment.next.month", axis=1),
                                                    df["default.payment.next.month"],
                                                    test_size=0.2,
                                                    random_state=42)

# Create a logistic regression model
lr_model = LogisticRegression(random_state=42)

# Fit the model to the training data
lr_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = lr_model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Create a pipeline
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf_classifier", RandomForestClassifier(random_state=42))
])

# Fit the pipeline to the training data
rf_pipeline.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = rf_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Define the parameter grid
param_grid = {
    "rf_classifier__n_estimators": [100, 200, 300],
    "rf_classifier__max_depth": [None, 10, 20, 30],
    "rf_classifier__min_samples_split": [2, 5, 10],
    "rf_classifier__min_samples_leaf": [1, 2, 4]
}

# Create a GridSearchCV object
rf_grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
rf_grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", rf_grid_search.best_params_)
print("Best Score:", rf_grid_search.best_score_)

# Create a pipeline for scaling the data and applying the Support Vector Classifier
svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc_classifier", SVC(random_state=42))
])

# Fit the pipeline to the training data
svc_pipeline.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = svc_pipeline.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# To improve the model in the future, we could experiment with different feature engineering techniques and try different models and hyperparameters. We could also try ensemble methods, such as combining multiple models to improve the overall accuracy.

# We could create a class called CreditCardDefaultPredictor that contains methods for loading the data, performing data cleaning and feature engineering, and building and evaluating the models. This would make it easy to reuse the code and experiment with different models and parameters.
