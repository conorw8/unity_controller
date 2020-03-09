from consumer import KerasConsumer
from kafka import KafkaConsumer
from json import loads
import sys
import numpy as np

ip = '54.152.215.103'
group = 'Z'
model_path = '/home/ubuntu/unity_controller/data/model4.yaml'
weights_path = '/home/ubuntu/unity_controller/data/model4.h5'

consumer1 = KerasConsumer(ip, group, model_path, weights_path)
consumer1.loadModel()
consumer1.processData()
