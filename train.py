import os
import warnings
import sys
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("max_iter")
    parser.add_argument("solver")
    args = parser.parse_args()
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.npz")
    
    dataset = np.load(data_path)
    
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    y_train = dataset['y_train'][:len(X_train)]
    y_test = dataset['y_test'][:len(X_test)]
                

    max_iter = int(args.max_iter)
    solver = str(args.solver)
    


    with mlflow.start_run():
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        (accuracy, precision, recall, f1) = eval_metrics(y_test, y_pred)

        print(f"Logistic Regression (Max Iterations={max_iter}, solver={solver})")
        print("  accuracy: %s" % accuracy)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(clf, "model")
        with open('model.pkl', 'wb') as f:
            pickle.dump(clf, f)
