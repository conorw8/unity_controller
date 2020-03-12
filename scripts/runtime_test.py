from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np
import rospy

rospy.init_node('runtime', anonymous = True)
print(float(rospy.get_time()))

ip = '127.0.0.1'
consumer1 = KafkaConsumer('result1', bootstrap_servers=[ip+':9092'],
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
runtimes = []
results = np.empty((4, 4))

for message in consumer1:
    topic, iter, val = message.topic, message.offset, message.value['result']
    val = np.reshape(val, (1, num_labels+1))
    feature = val[0, :-1]
    feature = np.reshape(feature, (1,num_labels))
    timestamp = val[0,-1]

    runtime = (float(rospy.get_time()) - timestamp)*1000
    print("Runtime of Ensemble: %s ms" % runtime)
    runtimes.append([runtime, 1])

    prediction = feature[0, 2]*100
    print("Predicted Probability: %s" % prediction)
    if prediction >= 99.0 and converged == 0.0:
        converged = sample_count

    print(converged)
    print(sample_count)

    sample_count += 1

    if sample_count == 200:
        runtimes = np.array(runtimes)
        # f = open('/home/ace/catkin_ws/src/unity_controller/data/runtime.csv', 'a')
        # np.savetxt(f, runtimes, delimiter=",")
        sys.exit(1)
