from consumer import KerasConsumer
from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '127.0.0.1'
group = 'T'
# model_path = '/home/ubuntu/unity_controller/data/model1.yaml'
# weights_path = '/home/ubuntu/unity_controller/data/model1.h5'
model_path = '/home/conor/catkin_ws/src/unity_controller/data/model1.yaml'
weights_path = '/home/conor/catkin_ws/src/unity_controller/data/model1.h5'
topic = 'result1'

consumer1 = KerasConsumer(ip, group, model_path, weights_path, topic)
consumer1.loadModel()
consumer1.processData()
