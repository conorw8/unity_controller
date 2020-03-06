from consumer import KerasConsumer
from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '54.161.204.248'
group = 'X'
model_path = '/home/ubuntu/unity_controller/data/model3.yaml'
weights_path = '/home/ubuntu/unity_controller/data/model3.h5'

consumer1 = KerasConsumer(ip, group, model_path, weights_path)
consumer1.loadModel()
consumer1.processData()
