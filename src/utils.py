import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set best parameters and train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Optionally, calculate ROC AUC for binary classification
            roc_auc = None
            if len(set(y_test)) == 2:  # Check if it's a binary classification problem
                y_test_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_prob)

            # Store the results in the report dictionary
            report[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
