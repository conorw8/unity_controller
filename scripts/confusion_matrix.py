from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np
import random
import pandas as pd
import timeit
import tensorflow as tf
from tensorflow.keras import backend
from keras import optimizers, Sequential
from keras.models import Model
from keras.layers import Dense, CuDNNLSTM
from keras.models import model_from_yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

class Predictor():
    def __init__(self, model_path, weights_path):
        self.model_path = model_path
        self.weights_path = weights_path
        self.model = None
        self.num_features = 8
        self.lookback = 10
        self.batch_size = 128
        self.num_labels = 3
        self.timesteps = 10
        self.scaler = None
        self.encoder = None
        self.seed = 0.7
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

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

    def loadModel(self):
        yaml_file = open(self.model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        self.model.load_weights(self.weights_path)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self):
        start_time1 = timeit.default_timer()

        x = np.reshape(self.test_x[0, :, :], (1, self.lookback, self.num_features))
        lstm_y_hat_test = self.model.predict(x=x, batch_size=self.batch_size, verbose=1)
        elapsed1 = timeit.default_timer() - start_time1

        print('Finished in %s second(s)' % round((elapsed1), 3))

        lstm_y_hat_eval = self.model.evaluate(x=self.test_x, y=self.test_y, batch_size=self.batch_size, verbose=1)
        print(lstm_y_hat_eval)

        lstm_y_hat = self.model.predict(x=self.test_x, batch_size=128, verbose=1)

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
        df_cm = pd.DataFrame(lstm_conf_matrix, index = ["healthy", "left fault", "right fault"],
                          columns = ["healthy", "left fault", "right fault"])
        print(df_cm)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g')
        plt.title("Single LSTM Model Confusion Matrix")
        plt.show()

if __name__=='__main__':
    path = '/home/conor/catkin_ws/src/unity_controller/data/sim_data.csv'
    model_path = '/home/conor/catkin_ws/src/unity_controller/data/model1.yaml'
    weights_path = '/home/conor/catkin_ws/src/unity_controller/data/model1.h5'

    predictor = Predictor(model_path, weights_path)
    predictor.loadData(path)
    predictor.loadModel()
    predictor.predict()
