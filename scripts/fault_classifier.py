import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix
from nn import NN
import timeit
import multiprocessing

class FaultClassifier():
    def __init__(self, num_features, num_labels, lookback, num_epochs, batch_size):
        self.lookback = lookback
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_labels = num_labels
        self.seed = 0.7
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.lstm_model1 = None
        self.lstm_model2 = None
        self.cnn_model1 = None
        self.cnn_model2 = None
        self.scaler = None
        self.encoder = None

    # helper functions
    def temporalize(self, x, y):
        X = []
        Y = []

        samples = x.shape[0] - self.lookback

        for i in range(samples - self.lookback):
            X.append(x[i:i+self.lookback, :])
            Y.append(y[i+self.lookback])

        return np.array(X), np.reshape(np.array(Y), (np.array(Y).shape[0], -1))

    def train_test_split(self, x, y):
        shuffled_a = np.empty(x.shape, dtype=x.dtype)
        shuffled_b = np.empty(y.shape, dtype=y.dtype)
        permutation = np.random.permutation(len(x))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = x[old_index]
            shuffled_b[new_index] = y[old_index]

        split = int(shuffled_a.shape[0]*self.seed)
        self.train_x = shuffled_a[0:split]
        self.train_y = shuffled_b[0:split]
        self.test_x = shuffled_a[split:]
        self.test_y = shuffled_b[split:]

    def loadData(self, path):
        df = pd.read_csv(path)
        dataset = df[['%velocity','%steering','%x','%y','%theta','%iteration','%time','%delay','%label']]
        dataset = dataset.to_numpy()
        data = dataset[:, :-1]
        print(data.shape)
        labels = np.reshape(dataset[:, -1], (-1, 1))

        for i in range(data.shape[0]):
            x_noise = np.random.normal(0.0, 0.004, 1)
            y_noise = np.random.normal(0.0, 0.004, 1)
            theta_noise = np.random.normal(0.0, 0.004, 1)
            data[i, 2] += x_noise
            data[i, 3] += y_noise
            data[i, 4] += theta_noise

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.encoder = OneHotEncoder(sparse=False)
        normalized_data = self.scaler.fit_transform(data)
        onehot_labels = self.encoder.fit_transform(labels)
        sequence_x, sequence_y = self.temporalize(normalized_data, onehot_labels)
        self.train_test_split(sequence_x, sequence_y)

    def trainModel(self, model_path, weights_path, dropout):
        # LSTM Model
        self.lstm_model = NN(self.train_x, self.train_y, self.test_x, self.test_y,
                                num_features=self.num_features, num_labels=self.num_labels,
                                lookback=self.lookback, num_epochs=self.num_epochs, batch_size=self.batch_size)
        self.lstm_model.buildModel(model_path, weights_path, dropout)

    def predict(self):
        start_time1 = timeit.default_timer()

        x = np.reshape(self.test_x[0, :, :], (1, self.lookback, self.num_features))
        lstm_y_hat_test = self.lstm_model.model.predict(x=x, batch_size=self.batch_size, verbose=1)
        elapsed1 = timeit.default_timer() - start_time1

        print('Finished in %s second(s)' % round((elapsed1), 3))

        lstm_y_hat_eval = self.lstm_model.model.evaluate(x=self.test_x, y=self.test_y, batch_size=self.batch_size, verbose=1)
        print(lstm_y_hat_eval)

        lstm_y_hat = self.lstm_model.model.predict(x=self.test_x, batch_size=self.batch_size, verbose=1)

        lstm_y_pred = np.zeros((lstm_y_hat.shape[0], self.num_labels))
        for i in range(lstm_y_hat.shape[0]):
            if np.argmax(lstm_y_hat[i, :]) == 0:
                lstm_y_pred[i, :] = [1, 0, 0]
            elif np.argmax(lstm_y_hat[i, :]) == 1:
                lstm_y_pred[i, :] = [0, 1, 0]
            else:
                lstm_y_pred[i, :] = [0, 0, 1]
        lstm_y_pred = self.encoder.inverse_transform(lstm_y_pred)
        y_true = self.encoder.inverse_transform(self.test_y)
        print(y_true.shape)
        lstm_conf_matrix = confusion_matrix(y_true, lstm_y_pred)
        print("LSTM Confusion Matrix")
        print(lstm_conf_matrix)

if __name__ == '__main__':
    path = '/home/ace/catkin_ws/src/unity_controller/data/sim_data.csv'
    model_path1 = '/home/ace/catkin_ws/src/unity_controller/data/model1.yaml'
    weights_path1 = '/home/ace/catkin_ws/src/unity_controller/data/model1.h5'
    classifier1 = FaultClassifier(num_features=8, num_labels=3, lookback=10, num_epochs=500, batch_size=128)
    classifier1.loadData(path)
    classifier1.trainModel(model_path1, weights_path1, 0.1)
    classifier1.predict()

    model_path2 = '/home/ace/catkin_ws/src/unity_controller/data/model2.yaml'
    weights_path2 = '/home/ace/catkin_ws/src/unity_controller/data/model2.h5'
    classifier2 = FaultClassifier(num_features=8, num_labels=3, lookback=10, num_epochs=500, batch_size=64)
    classifier2.loadData(path)
    classifier2.trainModel(model_path2, weights_path2, 0.2)
    classifier2.predict()

    model_path3 = '/home/ace/catkin_ws/src/unity_controller/data/model3.yaml'
    weights_path3 = '/home/ace/catkin_ws/src/unity_controller/data/model3.h5'
    classifier3 = FaultClassifier(num_features=8, num_labels=3, lookback=10, num_epochs=500, batch_size=32)
    classifier3.loadData(path)
    classifier3.trainModel(model_path3, weights_path3, 0.3)
    classifier3.predict()

    model_path4 = '/home/ace/catkin_ws/src/unity_controller/data/model4.yaml'
    weights_path4 = '/home/ace/catkin_ws/src/unity_controller/data/model4.h5'
    classifier4 = FaultClassifier(num_features=8, num_labels=3, lookback=10, num_epochs=500, batch_size=16)
    classifier4.loadData(path)
    classifier4.trainModel(model_path4, weights_path4, 0.4)
    classifier4.predict()
