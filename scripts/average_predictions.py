from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '3.94.90.197'
consumer1 = KafkaConsumer('result1','result2','result3','result4', bootstrap_servers=[ip+':9092'],
                              auto_offset_reset='latest', enable_auto_commit=False,
                              group_id='A', value_deserializer=lambda x: loads(x.decode('utf-8')))
# consumer2 = KafkaConsumer('result2', bootstrap_servers=[ip+':9092'],
#                               auto_offset_reset='latest', enable_auto_commit=False,
#                               group_id='B', value_deserializer=lambda x: loads(x.decode('utf-8')))
# consumer3 = KafkaConsumer('result3', bootstrap_servers=[ip+':9092'],
#                               auto_offset_reset='latest', enable_auto_commit=False,
#                               group_id='C', value_deserializer=lambda x: loads(x.decode('utf-8')))
# consumer4 = KafkaConsumer('result4', bootstrap_servers=[ip+':9092'],
#                               auto_offset_reset='latest', enable_auto_commit=False,
#                               group_id='D', value_deserializer=lambda x: loads(x.decode('utf-8')))
num_labels = 3
sample_count = 0
true_positive = 0
max_iter = 225 - 21
converged = 0.0
results = np.empty((4, 4))

for message in consumer1:
    topic, iter, val = message.topic, message.offset, message.value['result']
    val = np.reshape(val, (1, num_labels))
    # print(topic, iter, val)
    if topic == 'result1':
        results[0, :] = np.concatenate((val, np.reshape([float(iter)], (1, 1))), axis=1)
    elif topic == 'result2':
        results[1, :] = np.concatenate((val, np.reshape([float(iter)], (1, 1))), axis=1)
    elif topic == 'result3':
        results[2, :] = np.concatenate((val, np.reshape([float(iter)], (1, 1))), axis=1)
    else:
        results[3, :] = np.concatenate((val, np.reshape([float(iter)], (1, 1))), axis=1)

    if np.isnan(np.sum(results)):
        print("Not all results have arrived yet.")
    else:
        print(results)
        if results[0, 3] == results[1, 3] == results[2, 3] == results[3, 3]:
            print("Synced")
            ensemble_yhat = (results[0, :-1] + results[1, :-1] + results[2, :-1] + results[3, :-1]) / 4.0
            print("Ensemble Prediction:")
            print(ensemble_yhat)
            sample_count += 1
            if np.argmax(ensemble_yhat) == 0:
                print("Predicted: Healthy")
                true_positive += 1
            elif np.argmax(ensemble_yhat) == 1:
                print("Predicted: Left Fault")
            else:
                print("Predicted: Right Fault")

            prediction = ensemble_yhat[0]*100
            print("Predicted Probability: %s" % prediction)
            if prediction >= 99.0 and converged == 0.0:
                converged = sample_count

            print(converged)
            print(sample_count)
            if true_positive > 0:
                accuracy = float(true_positive)/float(sample_count) * 100
                print("Prediction Accuracy: %s" % accuracy)

            if sample_count == max_iter:
                sys.exit(1)
