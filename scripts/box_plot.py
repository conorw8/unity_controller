import numpy as np
import pandas as pd
import math
import sys
from matplotlib import pyplot as plt

data = pd.read_csv("~/catkin_ws/src/unity_controller/data/prediction_time.csv")
data = data.to_numpy()
collectn_1 = (data[:, 0] + 10)*100
collectn_2 = (data[:, 1] + 10)*100
collectn_3 = (data[:, 2] + 10)*100

## combine these different collections into a list
data_to_plot = [collectn_1, collectn_2, collectn_3]
print(data_to_plot)

# Create a figure instance
fig = plt.figure(1, figsize=(7, 7))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

## Custom x-axis labels
ax.set_xticklabels(['Healthy', 'Left Fault', 'Right Fault'])
# ax.set_ylim(96,100)
ax.set_title('Ensemble Classification Convergence')
plt.ylabel('ms to converge to 99% probability')
plt.show()
