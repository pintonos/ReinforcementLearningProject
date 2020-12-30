import sys
import csv
import numpy as np
import pandas as pd
import os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

csv_in_path = 'model/log.out'
png_out_path = 'dqnagent.png'

data = pd.read_csv(csv_in_path)

plt.scatter(data['step'], data['score'])
plt.title('Scatter plot of ' + csv_in_path)
plt.xlabel('step')
plt.ylabel('score')

plt.savefig(png_out_path)
plt.close()
