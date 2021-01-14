import numpy as np
import pandas as pd
import matplotlib
import argparse
matplotlib.use('Agg')
from matplotlib import pyplot as plt


""" interface of plot_csv.py

usage: plot_csv.py [-h] --csv CSV --png PNG

plot_csv

optional arguments:
  -h, --help  show this help message and exit
  --csv CSV   input csv file
  --png PNG   output png file
"""


argparser = argparse.ArgumentParser(description="plot_csv")
argparser.add_argument("--csv", required=True, type=str, 
  help="input csv file")
argparser.add_argument("--png", required=True, type=str, 
  help="output png file")
args = argparser.parse_args()

# scatter data
data = pd.read_csv(args.csv)

# smoothed average of last 10 episodes
rewards_smoothed = pd.Series.rolling(pd.Series(data['score']), 10).mean()
rewards_smoothed = [elem for elem in rewards_smoothed]

plt.scatter(data['step'], data['score'])
plt.plot(rewards_smoothed, color='orange')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)

# add avg. score of last 100 episodes
last_100 = data["score"][-100:]
avg_last_100 = sum(last_100)/100
plt.axhline(y=avg_last_100, color='r', linestyle='-')
plt.text(-175, avg_last_100-5, avg_last_100)

plt.savefig(args.png, dpi=600)
plt.close()
