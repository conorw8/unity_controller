from consumer import KerasConsumer
from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '18.212.18.94'
group = 'X'
model_path = '~/unity_controller/data/model3.yaml'
weights_path = '~/unity_controller/data/model3.h5'
# model_path = '~/catkin_ws/src/unity_controller/data/model3.yaml'
# weights_path = '~/catkin_ws/src/unity_controller/data/model3.h5'
topic = 'result3'

consumer1 = KerasConsumer(ip, group, model_path, weights_path, topic)
consumer1.loadModel()
consumer1.processData()
