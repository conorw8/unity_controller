from consumer import KerasConsumer
from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '3.94.90.197'
group = 'Z'
model_path = '/home/ubuntu/unity_controller/data/model4.yaml'
weights_path = '/home/ubuntu/unity_controller/data/model4.h5'
# model_path = '/home/ace/catkin_ws/src/unity_controller/data/model4.yaml'
# weights_path = '/home/ace/catkin_ws/src/unity_controller/data/model4.h5'
topic = 'result4'

consumer1 = KerasConsumer(ip, group, model_path, weights_path, topic)
consumer1.loadModel()
consumer1.processData()
