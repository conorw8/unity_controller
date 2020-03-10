from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '3.94.90.197'
consumer1 = KafkaConsumer('result1', bootstrap_servers=[ip+':9092'],
                              auto_offset_reset='latest', enable_auto_commit=False,
                              group_id=group, value_deserializer=lambda x: loads(x.decode('utf-8')))
consumer2 = KafkaConsumer('result2', bootstrap_servers=[ip+':9092'],
                              auto_offset_reset='latest', enable_auto_commit=False,
                              group_id=group, value_deserializer=lambda x: loads(x.decode('utf-8')))
consumer3 = KafkaConsumer('result3', bootstrap_servers=[ip+':9092'],
                              auto_offset_reset='latest', enable_auto_commit=False,
                              group_id=group, value_deserializer=lambda x: loads(x.decode('utf-8')))
consumer4 = KafkaConsumer('result4', bootstrap_servers=[ip+':9092'],
                              auto_offset_reset='latest', enable_auto_commit=False,
                              group_id=group, value_deserializer=lambda x: loads(x.decode('utf-8')))
num_labels = 3
sample_count = 0
true_positive = 0

for message1, message2, message3, message4 in zip(consumer1, consumer2, consumer3, consumer4):
    key1, val1 = message1.value.items()[0]
    key2, val2 = message2.value.items()[0]
    key3, val3 = message3.value.items()[0]
    key4, val4 = message4.value.items()[0]

    val1 = np.reshape(val1, (1, num_labels))
    val2 = np.reshape(val2, (1, num_labels))
    val3 = np.reshape(val3, (1, num_labels))
    val4 = np.reshape(val4, (1, num_labels))

    ensemble_yhat = (val1 + val2 + val3 + val4) / 4.0

    if np.argmax(ensemble_yhat) == 0:
        print("Predicted: Healthy")
        true_positive += 1
    elif np.argmax(ensemble_yhat) == 1:
        print("Predicted: Left Fault")
    else:
        print("Predicted: Right Fault")

    if true_positive > 0:
        accuracy = float(true_positive)/float(sample_count) * 100
        print("Prediction Accuracy: %s" % accuracy)
