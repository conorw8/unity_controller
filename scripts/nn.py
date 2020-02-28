import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
import pandas as pd
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, CuDNNLSTM, Dropout, RepeatVector, TimeDistributed, Flatten, Input, ConvLSTM2D
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix

class NN():
    def __init__(self, train_x, train_y, test_x, test_y, num_features, num_labels, lookback, num_epochs, batch_size):
        self.lookback = lookback
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_labels = num_labels
        self.seed = 0.7
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.lstm_model1 = None
        self.lstm_model2 = None
        self.cnn_model1 = None
        self.cnn_model2 = None
        self.scaler = None
        self.encoder = None
        self.model = None

    def buildModel(self):
        # # Multichannel CNN Model
        self.model = Sequential()
        self.model.add(CuDNNLSTM(128, input_shape=(self.lookback,self.num_features), return_sequences=True))
        self.model.add(CuDNNLSTM(128))
        self.model.add(Dense(self.num_labels, activation='softmax'))

        # self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.lookback,self.num_features)))
        # self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Flatten())
        # self.model.add(Dense(self.num_labels, activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(x=self.train_x, y=self.train_y, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)

        # serialize model to YAML
        model_path = '/home/conor/catkin_ws/src/unity_controller/data/model.yaml'
        weights_path = '/home/conor/catkin_ws/src/unity_controller/data/model.h5'
        model_yaml = self.model.to_yaml()
        with open(model_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        self.model.save_weights(weights_path)
        print("Saved model to disk")
