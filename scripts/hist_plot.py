import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ace/catkin_ws/src/unity_controller/data/runtime.csv")
data = data.to_numpy()
runtimes = np.reshape(data[:,0], (-1,1))
labels = np.reshape(data[:,1], (-1,1))
cloud_runtimes = data[(labels==0).all(axis=1)]
local_runtimes = data[(labels==1).all(axis=1)]

print(cloud_runtimes)
print(local_runtimes.shape)

plt.hist(cloud_runtimes[:,0], bins=50, range=(0, 150), color='blue', label='Cloud Ensemble Model')
plt.hist(local_runtimes[:,0], bins=50, range=(0, 150), color='orange', label='Local LSTM Model')
plt.title('Cloud Ensemble Runtime vs. Single Local LSTM Runtime')
plt.legend()
plt.xlabel('runtime (ms)')
plt.ylabel('number of runtimes')
plt.show()
