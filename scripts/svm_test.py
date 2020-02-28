import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score

# X{array-like, sparse matrix}, shape (n_samples, n_features)

def loadData(path):
    df = pd.read_csv(path)
    dataset = df[['%x', '%y', '%theta', '%iter', '%label']]
    dataset = dataset.to_numpy()
    data = dataset[:, :-1]
    print(data.shape)
    labels = np.reshape(dataset[:, -1], (-1, 1))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def trainModel(X, y):
    # model = QuadraticDiscriminantAnalysis()
    # model = LinearDiscriminantAnalysis()
    model = KNeighborsClassifier()
    # model = LogisticRegression()
    # model = svm.SVC()
    model.fit(X, y)

    return model

def predict(model, X, y_true):
    y_hat = model.predict(X)
    conf_matrix = confusion_matrix(y_true, y_hat)
    accuracy = accuracy_score(y_true, y_hat)
    print("Confusion Matrix")
    print(conf_matrix)
    print("accuracy: %s" % accuracy)

if __name__ == '__main__':
    path = '/home/conor/catkin_ws/src/unity_controller/data/sim_data.csv'
    X_train, X_test, y_train, y_test = loadData(path)
    model = trainModel(X_train, y_train)
    predict(model, X_test, y_test)
