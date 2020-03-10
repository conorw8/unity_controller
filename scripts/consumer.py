from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.layers import Dense, CuDNNLSTM
from keras.models import model_from_yaml

class KerasConsumer():
    def __init__(self, ip, group, model_path, weights_path):
        self.consumer = KafkaConsumer('data', bootstrap_servers=[ip+':9092'],
                                      auto_offset_reset='latest', enable_auto_commit=False,
                                      group_id=group, value_deserializer=lambda x: loads(x.decode('utf-8')))
        self.model_path = model_path
        self.weights_path = weights_path
        self.model = None
        self.num_features = 7
        self.timesteps = 10

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

    def processData(self):
        data = np.empty((self.timesteps, self.num_features))
        sample_count = 0
        true_positive = 0

        for message in self.consumer:
            key, val = message.value.items()[0]
            print(val)
            val = np.reshape(val, (1,self.num_features+1))
            features = val[0, :-1]
            # print(features)
            features = np.reshape(features, (1,self.num_features))
            label = val[0, self.num_features]

            if np.isnan(np.sum(data)):
                data = np.delete(data, 0, 0)
                data = np.append(data, features, axis=0)
            else:
                data = np.delete(data, 0, 0)
                data = np.append(data, features, axis=0)
                #make predictions
                y_hat = self.model.predict(x=np.reshape(data, (1, self.timesteps, self.num_features)))
                sample_count += 1
                # print(y_hat)
                if np.argmax(y_hat) == 0:
                    print("Predicted: Healthy")
                    if label == 1.0:
                        true_positive += 1
                elif np.argmax(y_hat) == 1:
                    print("Predicted: Left Fault")
                    if label == 2.0:
                        true_positive += 1
                else:
                    print("Predicted: Right Fault")
                    if label == 3.0:
                        true_positive += 1

                if true_positive > 0:
                    accuracy = float(true_positive)/float(sample_count) * 100
                    print("Prediction Accuracy: %s" % accuracy)

                if sample_count == 225 - self.timesteps:
                    sys.exit(1)
