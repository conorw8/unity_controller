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
        self.num_features = 7
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
        dataset = df[['%velocity','%steering','%x','%y','%theta','%iteration','%time','%label']]
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

def predict(models):
    x = models[0].test_x
    yhat1 = models[0].model.predict(x=x, batch_size=models[0].batch_size, verbose=1)
    yhat2 = models[1].model.predict(x=x, batch_size=models[0].batch_size, verbose=1)
    yhat3 = models[2].model.predict(x=x, batch_size=models[0].batch_size, verbose=1)
    yhat4 = models[3].model.predict(x=x, batch_size=models[0].batch_size, verbose=1)
    y_true = models[0].encoder.inverse_transform(models[0].test_y)

    num_class1 = y_true[(y_true==1).all(axis=1)].shape[0]
    num_class2 = y_true[(y_true==2).all(axis=1)].shape[0]
    num_class3 = y_true[(y_true==3).all(axis=1)].shape[0]
    print(num_class1)

    y_pred1 = np.zeros((yhat1.shape[0], models[0].num_labels))
    for i in range(yhat1.shape[0]):
        if np.argmax(yhat1[i, :]) == 0:
            y_pred1[i, :] = [1, 0, 0]
        elif np.argmax(yhat1[i, :]) == 1:
            y_pred1[i, :] = [0, 1, 0]
        else:
            y_pred1[i, :] = [0, 0, 1]
    y_pred1 = models[0].encoder.inverse_transform(y_pred1)
    conf_matrix1 = confusion_matrix(y_true, y_pred1)
    y1_class1_accuracy = float(conf_matrix1[0,0])/float(num_class1)
    y1_class2_accuracy = float(conf_matrix1[1,1])/float(num_class2)
    y1_class3_accuracy = float(conf_matrix1[2,2])/float(num_class3)
    print(y1_class1_accuracy, y1_class2_accuracy, y1_class3_accuracy)
    df_cm1 = pd.DataFrame(conf_matrix1, index = ["healthy", "left fault", "right fault"],
                      columns = ["healthy", "left fault", "right fault"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm1, annot=True, cmap="YlGnBu", fmt='g')
    plt.title("Single LSTM Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    # plt.show()

    y_pred2 = np.zeros((yhat2.shape[0], models[0].num_labels))
    for i in range(yhat2.shape[0]):
        if np.argmax(yhat2[i, :]) == 0:
            y_pred2[i, :] = [1, 0, 0]
        elif np.argmax(yhat2[i, :]) == 1:
            y_pred2[i, :] = [0, 1, 0]
        else:
            y_pred2[i, :] = [0, 0, 1]
    y_pred2 = models[0].encoder.inverse_transform(y_pred2)
    conf_matrix2 = confusion_matrix(y_true, y_pred2)
    y2_class1_accuracy = float(conf_matrix2[0,0])/float(num_class1)
    y2_class2_accuracy = float(conf_matrix2[1,1])/float(num_class2)
    y2_class3_accuracy = float(conf_matrix2[2,2])/float(num_class3)
    print(y2_class1_accuracy, y2_class2_accuracy, y2_class3_accuracy)
    df_cm2 = pd.DataFrame(conf_matrix2, index = ["healthy", "left fault", "right fault"],
                      columns = ["healthy", "left fault", "right fault"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm2, annot=True, cmap="YlGnBu", fmt='g')
    plt.title("Single LSTM Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    # plt.show()

    y_pred3 = np.zeros((yhat3.shape[0], models[0].num_labels))
    for i in range(yhat3.shape[0]):
        if np.argmax(yhat3[i, :]) == 0:
            y_pred3[i, :] = [1, 0, 0]
        elif np.argmax(yhat3[i, :]) == 1:
            y_pred3[i, :] = [0, 1, 0]
        else:
            y_pred3[i, :] = [0, 0, 1]
    y_pred3 = models[0].encoder.inverse_transform(y_pred3)
    conf_matrix3 = confusion_matrix(y_true, y_pred3)
    y3_class1_accuracy = float(conf_matrix3[0,0])/float(num_class1)
    y3_class2_accuracy = float(conf_matrix3[1,1])/float(num_class2)
    y3_class3_accuracy = float(conf_matrix3[2,2])/float(num_class3)
    print(y3_class1_accuracy, y3_class2_accuracy, y3_class3_accuracy)
    df_cm3 = pd.DataFrame(conf_matrix3, index = ["healthy", "left fault", "right fault"],
                      columns = ["healthy", "left fault", "right fault"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm3, annot=True, cmap="YlGnBu", fmt='g')
    plt.title("Single LSTM Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    # plt.show()

    y_pred4 = np.zeros((yhat4.shape[0], models[0].num_labels))
    for i in range(yhat4.shape[0]):
        if np.argmax(yhat4[i, :]) == 0:
            y_pred4[i, :] = [1, 0, 0]
        elif np.argmax(yhat4[i, :]) == 1:
            y_pred4[i, :] = [0, 1, 0]
        else:
            y_pred4[i, :] = [0, 0, 1]
    y_pred4 = models[0].encoder.inverse_transform(y_pred4)
    conf_matrix4 = confusion_matrix(y_true, y_pred4)
    y4_class1_accuracy = float(conf_matrix4[0,0])/float(num_class1)
    y4_class2_accuracy = float(conf_matrix4[1,1])/float(num_class2)
    y4_class3_accuracy = float(conf_matrix4[2,2])/float(num_class3)
    print(y4_class1_accuracy, y4_class2_accuracy, y4_class3_accuracy)
    df_cm4 = pd.DataFrame(conf_matrix4, index = ["healthy", "left fault", "right fault"],
                      columns = ["healthy", "left fault", "right fault"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm4, annot=True, cmap="YlGnBu", fmt='g')
    plt.title("Single LSTM Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    # plt.show()

    ensemble_yhat = (yhat1 + yhat2 + yhat3 + yhat4) / 4.0
    ensemble_pred = np.zeros((ensemble_yhat.shape[0], models[0].num_labels))
    for i in range(ensemble_yhat.shape[0]):
        if np.argmax(ensemble_yhat[i, :]) == 0:
            ensemble_yhat[i, :] = [1, 0, 0]
        elif np.argmax(ensemble_yhat[i, :]) == 1:
            ensemble_yhat[i, :] = [0, 1, 0]
        else:
            ensemble_yhat[i, :] = [0, 0, 1]
    ensemble_yhat = models[0].encoder.inverse_transform(ensemble_yhat)
    conf_matrix_ensemble = confusion_matrix(y_true, ensemble_yhat)
    y_class1_accuracy = float(conf_matrix_ensemble[0,0])/float(num_class1)
    y_class2_accuracy = float(conf_matrix_ensemble[1,1])/float(num_class2)
    y_class3_accuracy = float(conf_matrix_ensemble[2,2])/float(num_class3)
    print(y_class1_accuracy, y_class2_accuracy, y_class3_accuracy)
    df_cm_ensemble = pd.DataFrame(conf_matrix_ensemble, index = ["healthy", "left fault", "right fault"],
                      columns = ["healthy", "left fault", "right fault"])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm_ensemble, annot=True, cmap="YlGnBu", fmt='g')
    plt.title("Ensemble LSTM Model Confusion Matrix")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()

if __name__=='__main__':
    path = '~/catkin_ws/src/unity_controller/data/sim_data.csv'
    model_path1 = '~/catkin_ws/src/unity_controller/data/model1.yaml'
    weights_path1 = '~/catkin_ws/src/unity_controller/data/model1.h5'

    model_path2 = '~/catkin_ws/src/unity_controller/data/model2.yaml'
    weights_path2 = '~/catkin_ws/src/unity_controller/data/model2.h5'

    model_path3 = '~/catkin_ws/src/unity_controller/data/model3.yaml'
    weights_path3 = '~/catkin_ws/src/unity_controller/data/model3.h5'

    model_path4 = '~/catkin_ws/src/unity_controller/data/model4.yaml'
    weights_path4 = '~/catkin_ws/src/unity_controller/data/model4.h5'

    predictor1 = Predictor(model_path1, weights_path1)
    predictor1.loadData(path)
    predictor1.loadModel()

    predictor2 = Predictor(model_path2, weights_path2)
    predictor2.loadData(path)
    predictor2.loadModel()

    predictor3 = Predictor(model_path3, weights_path3)
    predictor3.loadData(path)
    predictor3.loadModel()

    predictor4 = Predictor(model_path4, weights_path4)
    predictor4.loadData(path)
    predictor4.loadModel()

    models = [predictor1, predictor2, predictor3, predictor4]

    predict(models)
