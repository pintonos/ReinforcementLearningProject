import sys
import csv
import numpy as np
import pandas as pd
import os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

csv_in_path = 'results/dqn-acrobot.csv'
png_out_path = 'results/dqn-acrobot.png'

data = pd.read_csv(csv_in_path)

plt.scatter(data['step'], data['score'])
plt.title('DQN Acrobot-v1')
plt.xlabel('episodes')
plt.ylabel('score')

# print avg. score of last 100 episodes
last_100 = data["score"][-100:]
avg_last_100 = sum(last_100)/100
print('avg score of last 100 episodes:', avg_last_100)
plt.axhline(y=avg_last_100, color='r', linestyle='-')
plt.text(-175, avg_last_100-5, avg_last_100)

plt.grid(True)
plt.savefig(png_out_path)
plt.close()
