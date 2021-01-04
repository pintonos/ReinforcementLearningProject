import sys
import csv
import numpy as np
import pandas as pd
import os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

csv_in_path = 'model/dqn-cartpole-v1.out'
png_out_path = 'model/dqnagent.png'

data = pd.read_csv(csv_in_path)

plt.scatter(data['step'], data['score'])
plt.title('Cartpole-v1')
plt.xlabel('step')
plt.ylabel('score')

plt.axhline(y = 195, color = 'r', linestyle = '-') 

# print avg. score of last 100 episodes
last_100 = data["score"][-100:]
print('avg score of last 100 episodes:', sum(last_100)/100)

plt.savefig(png_out_path)
plt.close()
